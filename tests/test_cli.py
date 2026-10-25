import io, json, os, sys, tempfile, types, unittest, contextlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_pipeline import start
from swp_exl3 import cli, config


class FakeTok:
    """Minimal stand-in for exllamav3's Tokenizer, enough for smoke on the numpy fake backend."""
    def __init__(self, vocab=64):
        self.vocab = vocab
        self.pieces = [f"<{i}>" for i in range(vocab)]
        self.config = types.SimpleNamespace(eos_token_id_list=[vocab - 1])
        self.eos_token_id = vocab - 1

    def encode(self, text, encode_special_tokens=True):
        return np.array([ord(c) % (self.vocab - 1) for c in text], dtype=np.int32)

    def get_id_to_piece_list(self):
        return self.pieces

    def single_id(self, s):
        raise ValueError(s)


def run(args):
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            cli.main(args)
        except SystemExit as e:
            code = e.code or 0
    return code, out.getvalue(), err.getvalue()


def toml(specs, d, name):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write('[pipeline]\ninsecure = true\nchunk = 4\n')
        for a, (x, y) in specs:
            f.write(f'[[stage]]\naddr = "{a}"\nlayers = [{x}, {y}]\n')
    return p


class T(unittest.TestCase):
    def test_workflow(self):
        d = tempfile.mkdtemp()
        _, s1 = start([3, 6])
        _, s2 = start([5])
        p1, p2 = toml(s1, d, "a.toml"), toml(s2, d, "b.toml")
        ids = "5,17,3,42,9,60,21,8,33,12,1"
        c, o, e = run(["ping", "--pipeline", p1]); self.assertEqual(c, 0); self.assertEqual(o.count("median RTT"), 3)
        c, o1, e = run(["generate", "--pipeline", p1, "--ids", ids, "--max-new", "9", "--bench", "--json", os.path.join(d, "b.json")])
        self.assertEqual(c, 0); self.assertIn("DECODE", e)
        r = json.load(open(os.path.join(d, "b.json"))); self.assertEqual(len(r["stages"]), 3); self.assertGreater(r["decode_tok_s"], 0)
        c, o2, _ = run(["generate", "--pipeline", p2, "--ids", ids, "--max-new", "9"])
        self.assertEqual(o1, o2, "different cuts must generate identical tokens")   # printed ids
        # cut-invariance workflow: save logits at cut A, at cut B, compare
        fa, fb = os.path.join(d, "la.json"), os.path.join(d, "lb.json")
        self.assertEqual(run(["logits", "--pipeline", p1, "--ids", ids, "--save", fa])[0], 0)
        self.assertEqual(run(["logits", "--pipeline", p2, "--ids", ids, "--save", fb])[0], 0)
        c, o, _ = run(["compare", fa, fb]); self.assertEqual(c, 0, o); self.assertIn("top-1 equal: True", o)

    def test_smoke_golden_pass_fail_update(self):
        d = tempfile.mkdtemp()
        servers, specs = start([4])
        p = toml(specs, d, "p.toml")
        with open(p) as f:
            t = f.read().replace("[pipeline]\n", '[pipeline]\nmodel_dir = "/fake"\n')
        open(p, "w").write(t)
        g = os.path.join(d, "golden.json")
        old = cli._tokenizer
        cli._tokenizer = lambda _: FakeTok()
        try:
            c, o, _ = run(["smoke", "--pipeline", p, "--no-contract", "--golden", g, "--update-golden"])
            self.assertEqual(c, 0, o); self.assertIn("golden written", o); self.assertTrue(os.path.exists(g))
            c, o, _ = run(["smoke", "--pipeline", p, "--no-contract", "--golden", g])
            self.assertEqual(c, 0, o); self.assertIn("SMOKE PASS", o)
            for s in servers:                       # perturb the model: smoke must fail
                s.b.W[:] = np.roll(s.b.W, 1, axis=1)
            c, o, _ = run(["smoke", "--pipeline", p, "--no-contract", "--golden", g])
            self.assertNotEqual(c, 0, o); self.assertIn("SMOKE FAIL", o)
            c, o, _ = run(["smoke", "--pipeline", p, "--no-contract", "--golden", g, "--update-golden"])
            self.assertEqual(c, 0, o)
            c, o, _ = run(["smoke", "--pipeline", p, "--no-contract", "--golden", g])
            self.assertEqual(c, 0, o); self.assertIn("SMOKE PASS", o)
        finally:
            cli._tokenizer = old

    def test_config_validation(self):
        d = tempfile.mkdtemp()
        bad = os.path.join(d, "bad.toml")
        open(bad, "w").write('[pipeline]\ninsecure=true\n[[stage]]\naddr="a:1"\nlayers=[0,4]\n[[stage]]\naddr="b:1"\nlayers=[5,8]\n')
        with self.assertRaises(config.ConfigError): config.load_pipeline(bad)
        open(bad, "w").write('[pipeline]\n[[stage]]\naddr="a:1"\nlayers=[0,4]\n')
        pc = config.load_pipeline(bad)
        with self.assertRaises(config.ConfigError): pc.secret()          # neither secret_env nor insecure

    def test_speed_env(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "tools", "glm"))
        src = ('import torch\nSPEED_ENV = {\n    # comment\n    "EXL3_A": "1",\n    "EXL3_B": "2048",\n}\n'
               'def main():\n    for k, v in SPEED_ENV.items():\n        pass\n')
        open(os.path.join(d, "tools", "glm", "serve.py"), "w").write(src)
        c, o, _ = run(["speed-env", "--kyojin-root", d])
        self.assertEqual(c, 0)
        self.assertIn("# 2 flags", o)
        self.assertIn('[stage.env]', o)
        self.assertIn('EXL3_A = "1"', o)
        self.assertIn('EXL3_B = "2048"', o)
        c, _, _ = run(["speed-env", "--kyojin-root", tempfile.mkdtemp()])
        self.assertNotEqual(c, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
