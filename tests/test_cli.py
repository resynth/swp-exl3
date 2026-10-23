import io, json, os, sys, tempfile, unittest, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_pipeline import start
from swp_exl3 import cli, config


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

    def test_config_validation(self):
        d = tempfile.mkdtemp()
        bad = os.path.join(d, "bad.toml")
        open(bad, "w").write('[pipeline]\ninsecure=true\n[[stage]]\naddr="a:1"\nlayers=[0,4]\n[[stage]]\naddr="b:1"\nlayers=[5,8]\n')
        with self.assertRaises(config.ConfigError): config.load_pipeline(bad)
        open(bad, "w").write('[pipeline]\n[[stage]]\naddr="a:1"\nlayers=[0,4]\n')
        pc = config.load_pipeline(bad)
        with self.assertRaises(config.ConfigError): pc.secret()          # neither secret_env nor insecure


if __name__ == "__main__":
    unittest.main(verbosity=2)
