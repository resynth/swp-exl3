import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CANONICAL = ROOT.parent / "splitwire" / "python" / "swp.py"
VENDORED = ROOT / "swp_exl3" / "swp.py"


class TestSwpVendoredCopy(unittest.TestCase):
    @unittest.skipUnless(CANONICAL.exists(), "splitwire checkout not present (separate repo)")
    def test_vendored_copy_matches_canonical(self):
        self.assertEqual(VENDORED.read_bytes(), CANONICAL.read_bytes(),
                         "swp_exl3/swp.py has drifted from splitwire/python/swp.py; re-copy the canonical file")
