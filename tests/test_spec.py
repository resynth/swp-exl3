"""Step S1: speculative verification on the fake backend.

The invariant asserted here (and only here) is exactness: speculative greedy output must equal
non-speculative greedy output for every draft pattern, including when the stage must clear and replay
because a lazy rewind is illegal."""
import os, sys, unittest
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
        p, _ = pipe([4])
        self.assertFalse(p.head_mtp)
        got, st = p.generate(PROMPT, max_new=10, spec=SpecConfig(k=3))
        self.assertEqual(got, reference([4], PROMPT, 10))

    def test_ngram_drafts(self):
        self.assertEqual(ngram_drafts([1, 2, 3, 4, 1, 2, 3], 2, 3), [4, 1])
        self.assertEqual(ngram_drafts([1, 2, 3], 2, 2), [])
        self.assertEqual(ngram_drafts([1, 2, 3, 4], 3, 3), [])

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
