import os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swp_exl3 import doctor


class T(unittest.TestCase):
    def test_find_engine_marker_in_python_sources(self):
        root = tempfile.mkdtemp()
        os.makedirs(os.path.join(root, "architecture"))
        p = os.path.join(root, "architecture", "glm5_next.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write("params['dflash_verify'] = True\n")
        self.assertEqual(doctor.find_engine_marker(root, "dflash_verify"), [p])
        self.assertEqual(doctor.find_engine_marker(root, "not_present_anywhere"), [])

    def test_find_engine_marker_binary_fallback(self):
        root = tempfile.mkdtemp()
        p = os.path.join(root, "exllamav3_ext.so")
        with open(p, "wb") as f:
            f.write(b"\x7fELF\0\0dflash_verify\0\0")
        self.assertEqual(doctor.find_engine_marker(root, "dflash_verify"), [p])

    def test_find_engine_marker_prefers_python_hits(self):
        root = tempfile.mkdtemp()
        py = os.path.join(root, "a.py")
        so = os.path.join(root, "a.so")
        with open(py, "w", encoding="utf-8") as f:
            f.write("dflash_verify\n")
        with open(so, "wb") as f:
            f.write(b"dflash_verify")
        self.assertEqual(doctor.find_engine_marker(root, "dflash_verify"), [py])


if __name__ == "__main__":
    unittest.main(verbosity=2)
