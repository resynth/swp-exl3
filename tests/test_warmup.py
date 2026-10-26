import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swp_exl3.backend import FakeBackend
from swp_exl3 import config


class T(unittest.TestCase):
    def test_warmup_runs_prescribed_rows_and_resets(self):
        b = FakeBackend((4, 8), n_layers=8)                 # non-first stage: gets zero boundary tensors
        seen, orig = [], b.forward

        def spy(start, toks, hidden, wire, want, rows, k):
            seen.append((len(toks), hidden))
            return orig(start, toks, hidden, wire, want, rows, k)

        b.forward = spy
        info = b.warmup()
        self.assertEqual(info["forwards"], 7)
        self.assertEqual([n for n, _ in seen], [1, 1, 1, 16, 64, 256, 512])
        self.assertTrue(all(h is not None for _, h in seen))
        self.assertEqual([len(h) for _, h in seen], [n * 4 * 8 * 4 for n, _ in seen])   # f32 [rows, hc=4, hidden=8]
        self.assertEqual(b.cache_len(), 0)
        self.assertEqual(b.pos, 0)

    def test_warmup_first_stage_uses_tokens(self):
        b = FakeBackend((0, 4), n_layers=8)
        seen, orig = [], b.forward

        def spy(start, toks, hidden, *a):
            seen.append(hidden)
            return orig(start, toks, hidden, *a)

        b.forward = spy
        b.warmup()
        self.assertTrue(all(h is None for h in seen))

    def test_warmup_skips_rows_larger_than_the_cache(self):
        b = FakeBackend((0, 4), n_layers=8)
        b.cache_tokens = 64
        seen, orig = [], b.forward

        def spy(start, toks, hidden, *a):
            seen.append(len(toks))
            return orig(start, toks, hidden, *a)

        b.forward = spy
        self.assertEqual(b.warmup()["forwards"], 5)
        self.assertEqual(seen, [1, 1, 1, 16, 64])

    def test_warmup_leaves_a_clean_state(self):
        fresh, warmed = FakeBackend((0, 4)), FakeBackend((0, 4))
        warmed.warmup()
        a = fresh.forward(0, [5, 6, 7], None, "f32", "token", "last", None)
        b = warmed.forward(0, [5, 6, 7], None, "f32", "token", "last", None)
        self.assertEqual(a.payload, b.payload)

    def test_stage_warmup_config_default_and_off(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "stage.toml")
        base = '[stage]\nnode = "n"\nlisten = "127.0.0.1:1"\nmodel_dir = "/m"\nlayers = [0, 4]\n'
        open(p, "w").write(base)
        self.assertTrue(config.load_stage(p).warmup)
        open(p, "w").write(base + "warmup = false\n")
        self.assertFalse(config.load_stage(p).warmup)


if __name__ == "__main__":
    unittest.main(verbosity=2)
