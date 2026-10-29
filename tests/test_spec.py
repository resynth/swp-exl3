"""Step S1: speculative verification on the fake backend.

The invariant asserted here (and only here) is exactness: speculative greedy output must equal
non-speculative greedy output for every draft pattern, including when the stage must clear and replay
because a lazy rewind is illegal."""
import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_pipeline import PROMPT, pipe
from swp_exl3 import swp
from swp_exl3.coordinator import SpecConfig, ngram_drafts


def reference(cuts, prompt, n, **kw):
    return pipe(list(cuts), **kw)[0].generate(prompt, max_new=n)[0]


class T(unittest.TestCase):
    def _scripted(self, cuts, prompt, n, pattern, **kw):
        ref = reference(cuts, prompt, n, **kw)

        def drafter(context, k):
            i = len(context) - len(prompt)          # index of the next true token
            lead = pattern(i, k)
            return [ref[i + j] if j < lead else (ref[i + j] + 1) % 64 for j in range(min(k, len(ref) - i))]

        p = pipe(list(cuts), **kw)[0]
        got, st = p.generate(prompt, max_new=n, spec=SpecConfig(k=3, drafter=drafter))
        self.assertEqual(got, ref, (cuts, pattern.__name__, st))
        return st

    def test_all_accept_two_and_three_stages(self):
        for cuts in ([4], [2, 5]):
            st = self._scripted(cuts, PROMPT, 14, lambda i, k: k)
            self.assertGreater(st["spec_rounds"], 0)
            self.assertEqual(st["spec_accepted"], st["spec_drafted"], cuts)

    def test_all_reject(self):
        for cuts in ([4], [2, 5]):
            st = self._scripted(cuts, PROMPT, 12, lambda i, k: 0)
            self.assertGreater(st["spec_rounds"], 0)
            self.assertEqual(st["spec_accepted"], 0, cuts)

    def test_mixed_acceptance(self):
        for cuts in ([4], [2, 5]):
            st = self._scripted(cuts, PROMPT, 16, lambda i, k: i % 3)
            self.assertGreater(st["spec_rounds"], 0)
            self.assertGreater(st["spec_accepted"], 0)
            self.assertLess(st["spec_accepted"], st["spec_drafted"], cuts)

    def test_illegal_rewind_falls_back_to_replay(self):
        # The toy can only rewind one row, but the drafter asks for 3 and rejects every draft: the next
        # round's lazy rewind is illegal, so the coordinator must clear+replay and still be exact.
        ref = reference([4], PROMPT, 10)

        def drafter(context, k):
            i = len(context) - len(PROMPT)
            return [ref[i + j] if j == 0 else (ref[i + j] + 1) % 64 for j in range(min(k, len(ref) - i))]

        p = pipe([4], spec_k=3, max_history=1)[0]
        got, st = p.generate(PROMPT, max_new=10, spec=SpecConfig(k=3, drafter=drafter))
        self.assertEqual(got, ref)
        self.assertGreater(st["spec_rounds"], 0)

    def test_verify_logits_match_sequential(self):
        for cuts in ([], [4], [2, 5]):
            p = pipe(list(cuts))[0]
            sc = p.spec_check(PROMPT, rows=12, k=3, klog=8)
            self.assertEqual(sc["rows"], 12, cuts)
            self.assertTrue(sc["top1"], (cuts, sc))
            self.assertLess(sc["max_kl"], 1e-3, (cuts, sc))

    def test_server_rejects_rewind_without_verify_history(self):
        p = pipe([4])[0]
        p.feed(PROMPT[:6])
        with self.assertRaises(swp.SwpError) as e:
            p._relay(2, [7], "none", "last", None, "decode", rewrite=True)
        self.assertEqual(e.exception.code, "unsupported")
        # after a verify forward the same rewind is legal (within last_history)
        p.reset()
        p.feed(PROMPT[:6])
        p._relay(6, PROMPT[:6][-1:] + [40, 41, 42], "none", "all", None, "decode",
                 rewrite=False, extra={"mode": "verify", "draft_ids": [40, 41, 42]})
        p._relay(8, [43], "none", "last", None, "decode", rewrite=True)     # 2 rows back: legal

    def test_prefix_reuse_after_spec_with_cache_ahead(self):
        # A partial accept leaves rejected rows in the stage caches. The next request reuses the committed
        # prefix, so generate() must drop those rows (rollback at pos == len(history)) before prefilling.
        ref = reference([4], PROMPT, 12)

        def drafter(context, k):
            i = len(context) - len(PROMPT)
            return [ref[i]] + [(ref[i + j] + 1) % 64 for j in range(1, min(k, len(ref) - i))]

        p = pipe([4])[0]
        out, _ = p.generate(PROMPT, max_new=8, spec=SpecConfig(k=3, drafter=drafter))
        self.assertEqual(out, ref[:8])
        cont = PROMPT + out[:-1] + [7, 8]
        newref = reference([4], cont, 6)
        got, st = p.generate(cont, max_new=6)
        self.assertEqual(st["reused_prefix"], len(PROMPT) + len(out) - 1)
        self.assertEqual(got, newref)

    def test_mtp_head_drafts_are_consumed_and_exact(self):
        # The fake head returns the reference continuation as its "MTP" drafts; the coordinator must use
        # them (including the off-by-one basis after a plain decode/verify) and stay byte-exact.
        ref = reference([4], PROMPT, 16)

        def plan(pos, k):
            i = pos - len(PROMPT)
            return ref[i:i + k] if 0 <= i < len(ref) else []

        p, _ = pipe([4], mtp_plan=plan)
        self.assertTrue(p.head_mtp)
        got, st = p.generate(PROMPT, max_new=12, spec=SpecConfig(k=3, drafter=lambda c, k: []))
        self.assertEqual(got, ref[:12])
        self.assertGreater(st["spec_rounds"], 0)
        self.assertGreater(st["spec_mtp_drafts"], 0)
        self.assertEqual(st["spec_accepted"], st["spec_drafted"])

    def test_mtp_off_falls_back_to_ngram(self):
        # No MTP drafter on the head: the 2-gram opt-in must still drive an n-gram round and stay exact.
        p, _ = pipe([4])
        self.assertFalse(p.head_mtp)
        got, st = p.generate(PROMPT, max_new=10, spec=SpecConfig(k=3, match=2))
        self.assertEqual(got, reference([4], PROMPT, 10))
        self.assertGreater(st["spec_ngram_rounds"], 0)
        self.assertEqual(st["spec_mtp_rounds"], 0)

    def test_mtp_fallback_uses_mtp_k_when_ngram_k_is_larger(self):
        # The two windows are independent: an n-gram window of 3 must not make the MTP fallback ask for 3.
        ref = reference([4], PROMPT, 18)
        asked = []

        def plan(pos, k):
            asked.append(int(k))
            i = pos - len(PROMPT)
            return ref[i:i + k] if 0 <= i < len(ref) else []

        p, _ = pipe([4], mtp_plan=plan)
        got, st = p.generate(PROMPT, max_new=14,
                             spec=SpecConfig(k=3, mtp_k=2, drafter=lambda c, k: []))
        self.assertEqual(got, ref[:14])
        self.assertGreater(st["spec_mtp_rounds"], 0)
        self.assertGreater(st["spec_mtp_drafted"], 0)
        self.assertTrue(asked)
        self.assertLessEqual(max(asked), 2, asked)

    def test_ngram_hit_discards_pending_mtp(self):
        ref = reference([4], PROMPT, 16)

        def plan(pos, k):
            i = pos - len(PROMPT)
            return ref[i:i + k] if 0 <= i < len(ref) else []

        def drafter(context, k):
            i = len(context) - len(PROMPT)
            return [ref[i + j] for j in range(min(k, len(ref) - i))]

        p, _ = pipe([4], mtp_plan=plan)
        got, st = p.generate(PROMPT, max_new=12,
                             spec=SpecConfig(k=3, mtp_k=2, drafter=drafter))
        self.assertEqual(got, ref[:12])
        self.assertGreater(st["spec_ngram_rounds"], 0)
        self.assertEqual(st["spec_mtp_rounds"], 0)
        self.assertEqual(st["spec_mtp_drafts"], 0)

    def test_source_counters_sum_to_totals(self):
        st = self._scripted([2, 5], PROMPT, 16, lambda i, k: i % 3)
        self.assertEqual(st["spec_ngram_rounds"] + st["spec_mtp_rounds"] + st["spec_plain_rounds"],
                         st["spec_rounds"])
        self.assertEqual(st["spec_ngram_drafted"] + st["spec_mtp_drafted"], st["spec_drafted"])
        self.assertEqual(st["spec_ngram_accepted"] + st["spec_mtp_accepted"], st["spec_accepted"])
        self.assertEqual(st["spec_lookup_hits"] > 0, st["spec_ngram_rounds"] > 0)

    def test_use_mtp_false_stays_exact_and_plain(self):
        ref = reference([4], PROMPT, 12)
        p, _ = pipe([4], mtp_plan=lambda pos, k: [])
        got, st = p.generate(PROMPT, max_new=12,
                             spec=SpecConfig(k=3, mtp_k=2, use_mtp=False, drafter=lambda c, k: []))
        self.assertEqual(got, ref[:12])
        self.assertEqual(st["spec_mtp_rounds"], 0)
        self.assertEqual(st["spec_mtp_drafts"], 0)
        self.assertGreater(st["spec_plain_rounds"], 0)

    def test_near_tie_classifier(self):
        from swp_exl3.coordinator import compare_topk
        # reference top-1 (5) vs verify top-1 (7), 0.05 logits apart -> near-tie
        c = compare_topk([7, 5, 3], [9.95, 10.0, 8.0], [5, 7, 3], [10.0, 9.95, 8.0])
        self.assertFalse(c["top1_equal"])
        self.assertTrue(c["near_tie"])
        # 0.5 logits apart -> a real mismatch
        c = compare_topk([7, 5, 3], [9.5, 10.0, 8.0], [5, 7, 3], [10.0, 9.5, 8.0])
        self.assertFalse(c["near_tie"])
        # verify top-1 outside the reference top-k cannot be a near-tie
        c = compare_topk([11, 5, 3], [10.0, 9.0, 8.0], [5, 7, 3], [10.0, 9.5, 8.0])
        self.assertFalse(c["near_tie"])

    def test_smoke_near_tie_budget(self):
        from swp_exl3.cli import spec_gate
        ok, budget = spec_gate({"rows": 20, "bad_top1": 0, "near_tie_flips": 2, "max_kl": 1e-3})
        self.assertTrue(ok)
        self.assertEqual(budget, 2)
        ok, _ = spec_gate({"rows": 20, "bad_top1": 0, "near_tie_flips": 3, "max_kl": 1e-3})
        self.assertFalse(ok, "a systematic near-tie pattern must fail")
        ok, _ = spec_gate({"rows": 20, "bad_top1": 1, "near_tie_flips": 0, "max_kl": 1e-3})
        self.assertFalse(ok)
        ok, _ = spec_gate({"rows": 20, "bad_top1": 0, "near_tie_flips": 0, "max_kl": 2e-2})
        self.assertFalse(ok)

    def test_drafts_clamped_to_cache_at_the_end(self):
        # cache 17 = prompt(11) + 6 rows: max_new 7 plain needs exactly 17. A full k=3 verify at
        # start 15 would need 19 rows, so the coordinator must clamp the window instead of failing.
        ref = reference([4], PROMPT, 7, cache_tokens=17)

        def drafter(context, k):
            i = len(context) - len(PROMPT)
            return [ref[i + j] for j in range(min(k, len(ref) - i))]

        p, _ = pipe([4], cache_tokens=17)
        got, st = p.generate(PROMPT, max_new=7, spec=SpecConfig(k=3, drafter=drafter))
        self.assertEqual(got, ref)
        self.assertGreater(st["spec_rounds"], 0)

    def test_draft_room_leaves_a_row_for_the_next_verify(self):
        p, _ = pipe([4], cache_tokens=17)
        p.history = list(range(11))
        self.assertEqual(p._draft_room(), 5)          # 11 + 1 + 5 == 17
        self.assertEqual(p._next_draft_room(3), 1)    # next verify starts at 15 and needs 15+1+1
        p.history = list(range(16))
        self.assertEqual(p._draft_room(), 0)
        self.assertEqual(p._next_draft_room(0), 0)

    def test_mtp_fallback_stays_exact_with_a_small_cache(self):
        ref = reference([4], PROMPT, 7, cache_tokens=17)
        calls = []

        def plan(pos, k):
            calls.append((pos, k))
            i = pos - len(PROMPT)
            return [ref[i + j] for j in range(min(k, len(ref) - i))] if 0 <= i < len(ref) else []

        p, _ = pipe([4], cache_tokens=17, mtp_plan=plan)
        got, st = p.generate(PROMPT, max_new=7,
                             spec=SpecConfig(k=3, mtp_k=2, drafter=lambda c, k: []))
        self.assertEqual(got, ref)
        self.assertGreater(st["spec_mtp_rounds"], 0)
        self.assertTrue(calls)
        self.assertLessEqual(max(k for _, k in calls), 2)

    def test_ngram_drafts(self):
        self.assertEqual(ngram_drafts([1, 2, 3, 4, 1, 2, 3], 2, 3), [4, 1])
        self.assertEqual(ngram_drafts([1, 2, 3], 2, 2), [])
        self.assertEqual(ngram_drafts([1, 2, 3, 4], 3, 3), [])

    def test_match_is_a_strict_floor(self):
        self.assertEqual(SpecConfig().match, 4)          # validated default floor
        # The last 2 tokens occurred at 0..1, but the last 3 did not: no implicit 2-gram fallback.
        ctx = [1, 2, 3, 4, 1, 2]
        self.assertEqual(SpecConfig(k=2, match=3).propose(ctx, 2), [])
        self.assertEqual(SpecConfig(k=2, match=4).propose(ctx, 2), [])
        self.assertEqual(SpecConfig(k=2, match=0).propose(ctx, 2), [])
        self.assertEqual(SpecConfig(k=2, match=-1).propose(ctx, 2), [])
        self.assertEqual(SpecConfig(k=2, match=99).propose(ctx, 2), [])
        # match=2 is the aggressive opt-in and restores the old behaviour.
        self.assertEqual(SpecConfig(k=2, match=2).propose(ctx, 2), [3, 4])

    def test_repeated_trigram_returns_the_following_tokens(self):
        ctx = [5, 6, 7, 8, 9, 5, 6, 7]
        self.assertEqual(SpecConfig(k=2, match=3).propose(ctx, 2), [8, 9])
        self.assertEqual(SpecConfig(k=3, match=3).propose(ctx, 3), [8, 9, 5])

    def test_proposals_always_match_at_least_the_floor(self):
        # The invariant the strict floor rests on: every non-empty proposal is backed by an occurrence of
        # the full `match`-token suffix (re-scanned here, independently of ngram_drafts).
        rng = np.random.default_rng(1234)
        for _ in range(200):
            ctx = [int(x) for x in rng.integers(0, 8, size=int(rng.integers(1, 30)))]
            for n in range(1, 7):
                d = SpecConfig(k=3, match=n).propose(ctx, 3)
                if not d:
                    continue
                key = tuple(ctx[-n:])
                hits = [i for i in range(len(ctx) - n - 1, -1, -1) if tuple(ctx[i:i + n]) == key]
                self.assertTrue(hits, (ctx, n, d))
                self.assertEqual(d, [int(x) for x in ctx[hits[0] + n:hits[0] + n + 3]], (ctx, n))

    def test_two_gram_only_context_uses_mtp_instead(self):
        # Chosen so the early rounds revisit a bigram but never a trigram (the match=2 run proves the
        # bigram hit is real). The strict floors must leave the draft to MTP instead of taking the weak hit.
        prompt = [57, 51, 39, 27, 32, 16, 31, 23, 15, 62, 0]
        ref = reference([4], prompt, 12)

        def plan(pos, k):
            i = pos - len(prompt)
            return ref[i:i + k] if 0 <= i < len(ref) else []

        def run(**kw):
            p, _ = pipe([4], mtp_plan=plan)
            got, st = p.generate(prompt, max_new=12, spec=SpecConfig(k=3, mtp_k=2, **kw))
            self.assertEqual(got, ref)
            return st

        self.assertGreater(run(match=2)["spec_ngram_rounds"], 0)
        for kw in ({"match": 3}, {}):                    # {} exercises the default floor (4)
            st = run(**kw)
            self.assertEqual(st["spec_ngram_rounds"], 0, kw)
            self.assertEqual(st["spec_lookup_hits"], 0, kw)
            self.assertGreater(st["spec_mtp_rounds"], 0, kw)

    def test_drafter_override_bypasses_the_floor(self):
        ctx = [1, 2, 3, 4, 1, 2]
        d = SpecConfig(k=2, match=99, drafter=lambda c, k: [7, 8]).propose(ctx, 2)
        self.assertEqual(d, [7, 8])

    def test_cli_spec_generate(self):
        import tempfile
        from tests.test_cli import run, toml
        from tests.test_pipeline import start
        d = tempfile.mkdtemp()
        _, specs = start([4])
        p = toml(specs, d, "p.toml")
        ids = ",".join(str(x) for x in PROMPT)
        c, o, e = run(["generate", "--pipeline", p, "--ids", ids, "--max-new", "9", "--spec", "--bench"])
        self.assertEqual(c, 0, e)
        self.assertIn("DECODE", e)
        self.assertIn("spec:", e)

    def test_spec_disabled_is_plain_decode(self):
        p = pipe([4], spec_k=0)[0]
        self.assertEqual(p.spec_k, 0)
        got, st = p.generate(PROMPT, max_new=8, spec=SpecConfig(k=3))
        self.assertEqual(got, reference([4], PROMPT, 8))
        self.assertNotIn("spec_rounds", st)


if __name__ == "__main__":
    unittest.main(verbosity=2)
