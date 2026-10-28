"""Tests for swp_kyojin_server against Kyojin's REAL HTTP layer (tools/glm/serve.py) and a fake pipeline (no GPU).
Set KYOJIN_ROOT (checkout) and SWP_EXL3_ROOT (the swp-exl3 repo) if they are not at the defaults below.
torch is stubbed only if it is not installed (Kyojin's helper modules import it at the top)."""
import asyncio, json, os, sys, tempfile, threading, unittest
from unittest import mock

KYOJIN = os.path.expanduser(os.environ.get("KYOJIN_ROOT", "~/engines/kyojin"))
SWP = os.path.expanduser(os.environ.get("SWP_EXL3_ROOT", "~/swp-exl3"))
for p in (SWP, os.path.join(KYOJIN, "tools", "glm"), os.path.join(KYOJIN, "tools"), os.path.dirname(os.path.abspath(__file__))):
    sys.path.insert(0, p)
try:
    import torch  # noqa: F401
except ImportError:
    sys.modules["torch"] = mock.MagicMock()

import serve as kserve                                   # noqa: E402  Kyojin's HTTP layer
import swp_kyojin_server as sks                          # noqa: E402
from aiohttp.test_utils import TestClient, TestServer    # noqa: E402
from swp_exl3.backend import FakeBackend                 # noqa: E402
from swp_exl3.coordinator import Pipeline, SpecConfig    # noqa: E402
from swp_exl3.server import StageServer                  # noqa: E402

ALPHA = "abcdefghijklmnopqrstuvwxyz0123456789 \n:.,!?-_'\"()/;<>|[]=+*&%#@^~"[:63] + "\u0001"   # 64 symbols; id 63 = eos
TEMPLATE = "{% for m in messages %}{{ m.role }}: {{ m.content }}|{% endfor %}assistant: "


def prompt_of(msgs):
    return "".join(f"{m['role']}: {m['content']}|" for m in msgs) + "assistant: "


def enc(text):
    return [ALPHA.index(ch) for ch in text]


class Ids(list):
    def flatten(self): return self
    def tolist(self): return list(self)
    def reshape(self, *_): return self
    def numel(self): return len(self)


class FakeTok:
    def encode(self, text, add_bos=False, encode_special_tokens=False, **_):
        return Ids(ALPHA.index(c) for c in text)
    def decode(self, seq, decode_special_tokens=False):
        return "".join(ALPHA[int(i)] for i in seq if int(i) != 63 or decode_special_tokens)


def make_pipeline(cuts=(4,), recurrent=True):
    edges = [0] + list(cuts) + [8]
    specs = []
    for i in range(len(edges) - 1):
        s = StageServer(FakeBackend((edges[i], edges[i + 1]), recurrent=recurrent), f"n{i}", "127.0.0.1:0", log=lambda *a: None)
        s.bind(); threading.Thread(target=s.serve_forever, daemon=True).start()
        specs.append((f"127.0.0.1:{s.port}", (edges[i], edges[i + 1])))
    p = Pipeline(specs, chunk=16, log=lambda *a: None)
    p.connect()
    return p


def fresh_greedy(ids, n, eos=(63,)):
    p = make_pipeline()
    out, _ = p.generate(ids, max_new=n, stop_ids=eos)
    return [t for t in out if t not in eos]


def run(coro):
    return asyncio.run(coro)


class T(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTok()
        self.pipe = make_pipeline()
        self.engine = sks.PipelineEngine(self.pipe, self.tok, ctx=4096, eos_ids=[63], seq_factory=lambda ids: list(ids), log=lambda *a: None)
        self.app = kserve.create_app(self.engine, "fake-model", TEMPLATE)

    async def post(self, client, messages, **kw):
        body = {"model": "fake-model", "messages": messages, "max_tokens": 24, "temperature": 0, **kw}
        return await client.post("/v1/chat/completions", json=body)

    def test_endpoints_and_nonstream_matches_direct_pipeline(self):
        async def go():
            async with TestClient(TestServer(self.app)) as c:
                self.assertEqual((await (await c.get("/v1/models")).json())["data"][0]["id"], "fake-model")
                self.assertEqual((await c.get("/health")).status, 200)
                r = await self.post(c, [{"role": "user", "content": "hi there"}], ignore_eos=True)
                self.assertEqual(r.status, 200, await r.text())
                return (await r.json())["choices"][0]["message"]["content"], await (await c.post("/tokenize", json={"prompt": "abc"})).json()
        content, tokstatus = run(go())
        ids = enc(prompt_of([{"role": "user", "content": "hi there"}]))
        want = "".join(ALPHA[t] for t in fresh_greedy(ids, 24, eos=()))
        self.assertEqual(content.strip(), want.strip())
        self.assertEqual(tokstatus["tokens"], enc("abc"))

    def test_stream_equals_nonstream_and_stop_string(self):
        async def go():
            async with TestClient(TestServer(self.app)) as c:
                msgs = [{"role": "user", "content": "tell me"}]
                full = (await (await self.post(c, msgs, ignore_eos=True)).json())["choices"][0]["message"]["content"]
                r = await self.post(c, msgs, ignore_eos=True, stream=True)
                text = ""
                async for line in r.content:
                    line = line.decode().strip()
                    if line.startswith("data:") and "[DONE]" not in line:
                        d = json.loads(line[5:])
                        text += (d["choices"][0]["delta"].get("content") or "")
                stop = full[6:8]
                cut = (await (await self.post(c, msgs, ignore_eos=True, stop=[stop])).json())["choices"][0]["message"]["content"]
                return full, text, stop, cut
        full, streamed, stop, cut = run(go())
        self.assertEqual(streamed, full)
        self.assertEqual(cut, full[:full.find(stop)])

    def test_multi_turn_reuses_prefix_without_replay(self):
        async def go():
            async with TestClient(TestServer(self.app)) as c:
                for word in ("hello", "abc", "xyz", "go", "hey", "ok", "yes", "run"):
                    m1 = [{"role": "user", "content": word}]
                    await self.engine.reset_prefix_cache()
                    r1 = (await (await self.post(c, m1, ignore_eos=True, max_tokens=12)).json())["choices"][0]["message"]["content"]
                    if r1 and r1 == r1.strip() and "".join(ALPHA[i] for i in self.engine._prev_ids[len(enc(prompt_of(m1))):]) == r1:
                        break
                else:
                    self.skipTest("no fixture word produced an exactly round-trippable reply")
                hist_after_1 = len(self.pipe.history)
                m2 = m1 + [{"role": "assistant", "content": r1}, {"role": "user", "content": "go on"}]
                r2 = (await (await self.post(c, m2, ignore_eos=True, max_tokens=12)).json())["choices"][0]["message"]["content"]
                return r1, r2, hist_after_1, m2
        r1, r2, h1, m2 = run(go())
        phases = {h[0] for h in self.pipe.trace.hops}
        self.assertNotIn("replay", phases, "multi-turn must append, not replay")
        self.assertGreaterEqual(self.engine.last_stats["cached_tokens"], h1)
        self.assertEqual(self.engine.last_stats["replayed_tokens"], 0)
        # same answer as a cold start on the full conversation
        want = "".join(ALPHA[t] for t in fresh_greedy(enc(prompt_of(m2)), 12, eos=()))
        self.assertEqual(r2.strip(), want.strip())

    def test_trim_reports_no_cache_hit_and_counts_the_replay(self):
        async def go():
            async with TestClient(TestServer(self.app)) as c:
                m1 = [{"role": "user", "content": "hello"}]
                first = (await (await self.post(c, m1, ignore_eos=True, max_tokens=12)).json())["choices"][0]["message"]["content"]
                # asking the same turn again trims the previous reply: recurrent state cannot rewind -> replay
                again = (await (await self.post(c, m1, ignore_eos=True, max_tokens=12)).json())["choices"][0]["message"]["content"]
                return first, again
        first, again = run(go())
        self.assertEqual(first.strip(), again.strip())
        st = self.engine.last_stats
        self.assertGreater(st["replayed_tokens"], 0)
        self.assertEqual(st["cached_tokens"], 0, "a replayed prefix was not served from the cache")

    def test_terminal_report_shows_cache_hit_miss_and_rates(self):
        logs = []
        self.engine.log = logs.append
        async def go():
            async with TestClient(TestServer(self.app)) as c:
                m1 = [{"role": "user", "content": "hello"}]
                first = (await (await self.post(c, m1, ignore_eos=True, max_tokens=8)).json())["choices"][0]["message"]
                m2 = m1 + [{"role": "assistant", "content": first["content"]},
                           {"role": "user", "content": "more"}]
                await self.post(c, m2, ignore_eos=True, max_tokens=8)
                await self.post(c, m1, ignore_eos=True, max_tokens=8)     # trim -> replay
        run(go())
        self.assertRegex(logs[0], r"\[req 1\] MISS \| prefill \d+ new \+ 0 cached in .* new tok/s")
        self.assertRegex(logs[1], r"\[req 2\] HIT \d+/\d+ \| prefill \d+ new \+ \d+ cached")
        self.assertRegex(logs[2], r"\[req 3\] MISS \(replayed \d+\)")
        for line in logs[:3]:
            self.assertIn("new tok/s", line)
            self.assertIn("effective", line)
            self.assertIn("decode", line)

    def test_request_line_reports_spec_acceptance(self):
        line = sks.request_line(4, "HIT 10/12", 2, 12, 0.1, 8, 8, 0.5, accepted=6, rejected=6, rounds=3)
        self.assertIn("spec 6/12 (50.0%, 3 rounds)", line)
        self.assertNotIn("spec", sks.request_line(1, "MISS", 10, 10, 0.2, 5, 5, 0.4))

    def test_request_line_reports_source_mix(self):
        st = {"spec_ngram_rounds": 12, "spec_ngram_accepted": 44, "spec_ngram_drafted": 44,
              "spec_mtp_rounds": 4, "spec_mtp_accepted": 9, "spec_mtp_drafted": 12,
              "spec_plain_rounds": 1}
        line = sks.request_line(4, "HIT 10/12", 2, 12, 0.1, 8, 8, 0.5, mix=st)
        self.assertIn("spec n-gram 12r 44/44 | mtp 4r 9/12 | plain 1", line)

    def test_spec_log_writes_jsonl(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "spec.jsonl")
        self.engine.spec_log = path
        async def go():
            async with TestClient(TestServer(self.app)) as c:
                r = await self.post(c, [{"role": "user", "content": "hi"}], ignore_eos=True, max_tokens=4)
                self.assertEqual(r.status, 200)
        run(go())
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(x) for x in f if x.strip()]
        self.assertEqual(len(lines), 1)
        rec = lines[0]
        self.assertEqual(rec["request"], 1)
        self.assertIn("timestamp", rec)
        self.assertIn("prompt_tokens", rec)
        self.assertIn("spec_ngram_rounds", rec)

    def test_spec_log_rotates(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "spec.jsonl")
        self.engine.spec_log = path
        old_max = sks.SPEC_LOG_MAX_BYTES
        sks.SPEC_LOG_MAX_BYTES = 1
        try:
            self.engine._append_spec_log({"request": 1})
            self.engine._append_spec_log({"request": 2})
        finally:
            sks.SPEC_LOG_MAX_BYTES = old_max
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.exists(path + ".1"))
        with open(path + ".1", encoding="utf-8") as f:
            self.assertEqual(json.loads(f.readline())["request"], 1)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.loads(f.readline())["request"], 2)

    def test_cli_parses_spec_mtp_flags(self):
        a = sks.build_parser().parse_args(
            ["--pipeline", "p.toml", "--model-dir", "m", "--spec", "--spec-mtp-k", "4", "--no-mtp",
             "--spec-log", "s.jsonl"])
        self.assertEqual(a.spec_mtp_k, 4)
        self.assertTrue(a.no_mtp)
        self.assertEqual(a.spec_log, "s.jsonl")

    def test_num_draft_is_the_larger_window(self):
        e = sks.PipelineEngine(self.pipe, self.tok, ctx=4096, eos_ids=[63], seq_factory=lambda ids: list(ids),
                               log=lambda *a: None, spec=SpecConfig(k=3, mtp_k=5))
        self.assertEqual(e.num_draft, 5)

    def test_ids_for_splices_exact_ids_and_falls_back(self):
        e = self.engine
        base = enc("user: hi|assistant: ")
        gen = [5, 9, 40, 2, 3]
        e._prev_ids = base + gen
        rest = "|user: ok|assistant: "
        p2 = e._dec(base + gen) + rest
        self.assertEqual(e._ids_for(p2), base + gen + enc(rest))
        p3 = e._dec(base + gen[:3]) + "zzz"                      # the reply was edited after 3 tokens
        self.assertEqual(e._ids_for(p3)[:len(base) + 3], base + gen[:3])
        self.assertEqual(e._dec(e._ids_for(p3)), p3)
        self.assertEqual(e._dec(e._ids_for("totally new")), "totally new")

    def test_cancel_midstream_leaves_engine_usable(self):
        async def go():
            async with TestClient(TestServer(self.app)) as c:
                msgs = [{"role": "user", "content": "long"}]
                r = await self.post(c, msgs, ignore_eos=True, stream=True, max_tokens=200)
                await r.content.readline()
                r.close()                                           # client leaves
                await asyncio.sleep(0.3)
                r2 = await self.post(c, msgs, ignore_eos=True, max_tokens=10)
                return r2.status, (await r2.json())["choices"][0]["message"]["content"]
        status, content = run(go())
        self.assertEqual(status, 200)
        want = "".join(ALPHA[t] for t in fresh_greedy(enc(prompt_of([{"role": "user", "content": "long"}])), 10, eos=()))
        self.assertEqual(content.strip(), want.strip())

    def test_prompt_ids_stays_tensor_like(self):
        ids = self.engine.prompt_ids("abc")
        self.assertEqual(ids.numel(), 3)
        self.assertEqual(ids.tolist(), enc("abc"))

    def test_over_context_gets_400(self):
        e = sks.PipelineEngine(self.pipe, self.tok, ctx=10, eos_ids=[63], seq_factory=lambda ids: list(ids),
                               log=lambda *a: None)
        app = kserve.create_app(e, "fake-model", TEMPLATE)
        async def go():
            async with TestClient(TestServer(app)) as c:
                r = await self.post(c, [{"role": "user", "content": "x" * 40}], ignore_eos=True)
                return r.status
        self.assertEqual(run(go()), 400)

    def test_unsupported_sampler_options_warn_once(self):
        logs = []
        self.engine.log = logs.append
        async def go():
            async with TestClient(TestServer(self.app)) as c:
                for _ in range(2):
                    r = await self.post(c, [{"role": "user", "content": "x"}], ignore_eos=True, max_tokens=4, min_p=0.1)
                    self.assertEqual(r.status, 200)
        run(go())
        self.assertLessEqual(sum("min_p" in l for l in logs), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
