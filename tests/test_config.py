import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swp_exl3 import config


BASE = ('[stage]\nnode = "n"\nlisten = "127.0.0.1:1"\nmodel_dir = "/m"\n'
        'layers = [0, 4]\ncache_tokens = 8192\n')


class T(unittest.TestCase):
    def _load(self, extra=""):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "stage.toml")
        with open(p, "w") as f:
            f.write(BASE + extra)
        return config.load_stage(p)

    def test_cache_bits_defaults_to_fp16(self):
        self.assertEqual(self._load().cache_bits, 0)

    def test_cache_bits_accepts_q8(self):
        self.assertEqual(self._load("cache_bits = 8\n").cache_bits, 8)

    def test_cache_bits_rejects_other_widths(self):
        with self.assertRaises(config.ConfigError):
            self._load("cache_bits = 4\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
