"""Needs the Rust `swp-weightd` binary (from splitwire): set SWP_WEIGHTD=/path/to/swp-weightd, else this test is skipped."""
import json, os, random, shutil, socket, struct, subprocess, sys, tempfile, time, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swp_exl3 import provision, swp

BIN = os.environ.get("SWP_WEIGHTD", "")


def write_st(path, tensors):
    hdr, off, blobs = {}, 0, []
    for name, (dt, shape, data) in tensors.items():
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data); blobs.append(data)
    hdr["__metadata__"] = {"format": "pt"}
    h = json.dumps(hdr).encode(); h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs: f.write(b)


def read_st(path):
    b = open(path, "rb").read(); n = struct.unpack("<Q", b[:8])[0]; h = json.loads(b[8:8 + n])
    return {k: (v["dtype"], v["shape"], b[8 + n + v["data_offsets"][0]: 8 + n + v["data_offsets"][1]]) for k, v in h.items() if k != "__metadata__"}


@unittest.skipUnless(os.path.exists(BIN), "set SWP_WEIGHTD to the swp-weightd binary")
class T(unittest.TestCase):
    def test_pull_stages(self):
        random.seed(3)
        rb = lambda n: bytes(random.getrandbits(8) for _ in range(n))
        P = "model.language_model"
        tmp = tempfile.mkdtemp()
        src, out = os.path.join(tmp, "src"), os.path.join(tmp, "out")
        os.makedirs(src)
        t1, t2, all_t = {}, {}, {}
        t1[f"{P}.embed_tokens.weight"] = ("F16", [10, 8], rb(160))
        for l in range(5):                                   # layer 4 plays the MTP layer (num_hidden_layers = 4)
            d = t1 if l < 3 else t2
            d[f"{P}.layers.{l}.self_attn.q.trellis"] = ("I16", [8, 16], rb(256))
            d[f"{P}.layers.{l}.mlp.experts.0.up.trellis"] = ("I16", [8, 8], rb(128))
        t2[f"{P}.norm.weight"] = ("F16", [8], rb(16)); t2["hc_head.fn"] = ("F32", [4], rb(16)); t2["lm_head.trellis"] = ("I16", [8, 10], rb(160))
        t2[f"{P}.visual.blocks.0.w"] = ("F16", [8], rb(16)); t2["rotary.inv_freq"] = ("F32", [2], rb(8))
        write_st(os.path.join(src, "a.safetensors"), t1); write_st(os.path.join(src, "b.safetensors"), t2)
        json.dump({"num_hidden_layers": 4}, open(os.path.join(src, "config.json"), "w"))
        all_t.update(t1); all_t.update(t2)

        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        env = dict(os.environ, SWP_T_SECRET="hunter2")
        srv = subprocess.Popen([BIN, "serve", "--model", src, "--listen", f"127.0.0.1:{port}", "--secret-env", "SWP_T_SECRET"],
                               env=env, stderr=subprocess.DEVNULL)
        try:
            for _ in range(100):
                try: socket.create_connection(("127.0.0.1", port), timeout=0.2).close(); break
                except OSError: time.sleep(0.1)
            addr = f"127.0.0.1:{port}"
            # middle stage: layers 1..3 only
            p = provision.pull(addr, os.path.join(out, "mid"), 1, 3, False, False, secret="hunter2", log=lambda *a: None)
            got = read_st(p)
            self.assertEqual(set(got), {f"{P}.layers.{l}.{s}" for l in (1, 2) for s in ("self_attn.q.trellis", "mlp.experts.0.up.trellis")})
            self.assertTrue(all(all_t[k] == v for k, v in got.items()), "bytes differ from source")
            self.assertTrue(os.path.exists(os.path.join(out, "mid", "config.json")))
            # last stage: layers 3..4 + head + MTP (extra) + norm/hc_head; no vision, no unknown tensors
            got = read_st(provision.pull(addr, os.path.join(out, "last"), 3, 4, False, True, secret="hunter2", log=lambda *a: None))
            self.assertEqual(set(got), {f"{P}.layers.3.self_attn.q.trellis", f"{P}.layers.3.mlp.experts.0.up.trellis",
                                        f"{P}.layers.4.self_attn.q.trellis", f"{P}.layers.4.mlp.experts.0.up.trellis",
                                        f"{P}.norm.weight", "hc_head.fn", "lm_head.trellis"})
            self.assertTrue(all(all_t[k] == v for k, v in got.items()))
            # first stage gets embedding + unclassified tensors, vision only on request
            got = read_st(provision.pull(addr, os.path.join(out, "first"), 0, 1, True, False, secret="hunter2", log=lambda *a: None))
            self.assertIn(f"{P}.embed_tokens.weight", got); self.assertIn("rotary.inv_freq", got)
            self.assertNotIn(f"{P}.visual.blocks.0.w", got)
            got = read_st(provision.pull(addr, os.path.join(out, "first_v"), 0, 1, True, False, secret="hunter2", include_vision=True, log=lambda *a: None))
            self.assertIn(f"{P}.visual.blocks.0.w", got)
            # tied embeddings: the last stage also needs the embedding table
            got = read_st(provision.pull(addr, os.path.join(out, "tied"), 3, 4, False, True, secret="hunter2", tied_embed=True, log=lambda *a: None))
            self.assertIn(f"{P}.embed_tokens.weight", got)
            with self.assertRaises(swp.SwpError):
                provision.pull(addr, os.path.join(out, "bad"), 0, 1, True, False, secret="wrong", log=lambda *a: None)
        finally:
            srv.terminate(); srv.wait(); shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
