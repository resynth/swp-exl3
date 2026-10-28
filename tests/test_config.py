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

    def test_mtp_cache_must_cover_target_cache(self):
        with self.assertRaises(config.ConfigError):
            self._load("head = true\nmtp = true\nmtp_cache_tokens = 4096\n")

    def test_mtp_cache_equal_to_target_cache_is_valid(self):
        c = self._load("head = true\nmtp = true\nmtp_cache_tokens = 8192\n")
        self.assertEqual(c.mtp_cache_tokens, 8192)

    def test_mtp_cache_is_ignored_when_mtp_is_off(self):
        c = self._load("mtp_cache_tokens = 4096\n")
        self.assertEqual(c.mtp_cache_tokens, 4096)
        self.assertFalse(c.mtp)

    def test_mtp_requires_head(self):
        with self.assertRaises(config.ConfigError):
            self._load("mtp = true\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
