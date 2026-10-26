import os, sys, threading, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swp_exl3 import swp
from swp_exl3.backend import FakeBackend
from swp_exl3.coordinator import Pipeline
from swp_exl3.server import StageServer

PROMPT = [5, 17, 3, 42, 9, 60, 21, 8, 33, 12, 1]
LOG = lambda *a: None


def start(cuts, recurrent=True, secret=None, delays=None, fails=None, n_layers=8, **be_kw):
    edges = [0] + list(cuts) + [n_layers]
    servers, specs = [], []
    for i in range(len(edges) - 1):
        be = FakeBackend((edges[i], edges[i + 1]), n_layers=n_layers, recurrent=recurrent,
                         delay_ms=(delays or [0] * 9)[i], fail_calls=(fails or {}).get(i, ()), **be_kw)
        s = StageServer(be, f"n{i}", "127.0.0.1:0", secret=secret, log=LOG)
        s.bind()
        threading.Thread(target=s.serve_forever, daemon=True).start()
        servers.append(s)
        specs.append((f"127.0.0.1:{s.port}", (edges[i], edges[i + 1])))
    return servers, specs


def pipe(cuts, secret=None, chunk=512, wire="f32", **kw):
    servers, specs = start(cuts, secret=secret, **kw)
    p = Pipeline(specs, secret=secret, chunk=chunk, wire_dtype=wire, log=LOG)
    p.connect()
    return p, servers


class T(unittest.TestCase):
    def ref(self, prompt=PROMPT, n=12, recurrent=True):
        p, _ = pipe([], recurrent=recurrent)
        return p.generate(prompt, max_new=n)[0]

    def test_split_equals_unsplit_and_chunk_invariant(self):
        ref = self.ref()
        self.assertGreaterEqual(len(set(ref)), 4, "toy must be discriminating")
        for cuts in ([4], [1], [7], [2, 5], [1, 3, 6]):
            for chunk in (512, 3):
                p, _ = pipe(cuts, chunk=chunk)
                self.assertEqual(p.generate(PROMPT, max_new=12)[0], ref, (cuts, chunk))

    def test_rollback_replay_recurrent_and_truncate_nonrecurrent(self):
        for rec, how in ((True, "replayed"), (False, "truncated")):
            p, _ = pipe([3, 5], recurrent=rec)
            p.generate(PROMPT, max_new=6)
            alt = PROMPT[:7] + [50, 51, 52]
            before = p.replayed_tokens
            self.assertEqual(p.rollback(7), how)
            self.assertEqual(p.replayed_tokens - before, 7 if rec else 0)
            got, st = p.generate(alt, max_new=8)
            self.assertEqual(got, self.ref(alt, 8, rec), rec)
            self.assertEqual(st["replayed_tokens"], 0, "the prefix was already trimmed before generate()")

    def test_prefix_reuse_on_continuation(self):
        p, _ = pipe([4])
        out, _ = p.generate(PROMPT, max_new=5)
        cont = PROMPT + out[:-1] + [7, 8]
        got, st = p.generate(cont, max_new=6)
        self.assertEqual(st["reused_prefix"], len(PROMPT) + len(out) - 1)
        self.assertEqual(st["replayed_tokens"], 0)
        self.assertEqual(got, self.ref(cont, 6))

    def test_protocol_errors(self):
        p, servers = pipe([4])
        with self.assertRaises(swp.SwpError) as e:
            p.peers[0].rpc(swp.FORWARD, {"session": "s1", "start_pos": 5, "n_rows": 1, "token_ids": [1], "input": "tokens", "want": "hidden"}, want=swp.FORWARD_OUT)
        self.assertEqual(e.exception.code, "pos_mismatch")
        servers[0].b.cache_tokens = 3
        with self.assertRaises(swp.SwpError) as e:      # reserve room for the rows that will be written
            p.peers[0].rpc(swp.FORWARD, {"session": "s1", "start_pos": 0, "n_rows": 4, "token_ids": [1, 2, 3, 4], "input": "tokens", "want": "hidden"}, want=swp.FORWARD_OUT)
        self.assertEqual(e.exception.code, "context_exceeded")
        servers[0].b.cache_tokens = 8192
        q = swp.Peer(p.stages[0].addr)
        with self.assertRaises(swp.SwpError) as e:
            q.rpc(swp.STAGE_CONFIGURE, {"session": "x", "layers": [0, 7], "has_embed": True, "has_head": False}, want=swp.STAGE_READY)
        self.assertEqual(e.exception.code, "contract_mismatch")
        with self.assertRaises(swp.SwpError) as e:
            q.rpc(swp.STAGE_CONFIGURE, {"session": "x", "layers": [0, 4], "has_embed": True, "has_head": False,
                                        "contract": {"hidden_size": 999}}, want=swp.STAGE_READY)
        self.assertIn("hidden_size", e.exception.message)
        with self.assertRaises(swp.SwpError) as e:      # wrong hidden payload size is rejected, not crashed on
            p.peers[1].rpc(swp.FORWARD, {"session": "s1", "start_pos": 0, "n_rows": 2, "token_ids": [1, 2], "input": "hidden", "want": "token"}, b"\0" * 10, want=swp.FORWARD_OUT)
        self.assertEqual(e.exception.code, "bad_request")

    def test_auth(self):
        _, specs = start([4], secret="s3cret")
        Pipeline(specs, secret="s3cret", log=LOG).connect()
        with self.assertRaises(swp.SwpError):
            Pipeline(specs, secret="nope", log=LOG).connect()
        with self.assertRaises(swp.SwpError):
            Pipeline(specs, secret=None, log=LOG).connect()

    def test_recovery_after_dropped_connection_and_engine_error(self):
        ref = self.ref()
        p, _ = pipe([4])
        p.feed(PROMPT[:5])
        p.peers[1].sock.close()                       # simulate a link drop mid-conversation
        got, _ = p.generate(PROMPT, max_new=12)
        self.assertEqual(got, ref)
        p, _ = pipe([4], fails={1: {3}})              # engine error on the 3rd forward of stage 1
        self.assertEqual(p.generate(PROMPT, max_new=12)[0], ref)

    def test_topk_cut_invariance_and_bf16(self):
        ids0, v0 = pipe([], chunk=512)[0].logits_topk(PROMPT, 8)
        for cuts in ([4], [2, 5]):
            ids, v = pipe(cuts)[0].logits_topk(PROMPT, 8)
            self.assertTrue((ids == ids0).all() and np.array_equal(v, v0), cuts)
        ids, v = pipe([4], wire="bf16")[0].logits_topk(PROMPT, 8)     # lossy wire: close, not identical
        self.assertEqual(ids[0], ids0[0])
        self.assertLess(np.abs(v - v0)[:1].max(), 0.05)
        self.assertTrue(np.abs(v - v0).max() > 0)

    def test_sampling_is_seeded(self):
        p, _ = pipe([4])
        a = p.generate(PROMPT, max_new=10, temperature=0.8, top_k=8, seed=7)[0]
        b = pipe([4])[0].generate(PROMPT, max_new=10, temperature=0.8, top_k=8, seed=7)[0]
        self.assertEqual(a, b)

    def test_bench_attribution(self):
        p, _ = pipe([4], delays=[2, 6])
        _, st = p.generate(PROMPT, max_new=20)
        r = p.report(st, out=LOG)
        s0, s1 = r["stages"]
        self.assertGreater(s1["decode_compute_ms_per_tok"], 2 * s0["decode_compute_ms_per_tok"] * 0.9)
        self.assertGreater(r["decode_tok_s"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
