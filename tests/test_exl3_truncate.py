import os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swp_exl3.exl3_backend import Exl3Backend


def backend(pos, mtp_pos, committed, mtp_enabled=True, recurrent=False):
    b = object.__new__(Exl3Backend)
    b.pos, b._mtp_pos, b._mtp_committed = pos, mtp_pos, committed
    b.mtp_enabled, b.recurrent = mtp_enabled, recurrent
    return b


class T(unittest.TestCase):
    def test_truncate_at_current_pos_settles_mtp(self):
        b = backend(10, 13, 10)
        b.truncate(10)
        self.assertEqual((b.pos, b._mtp_pos, b._mtp_committed), (10, 10, 10))

    def test_truncate_below_current_pos(self):
        b = backend(12, 15, 10)
        b.truncate(11)
        self.assertEqual((b.pos, b._mtp_pos, b._mtp_committed), (11, 11, 10))

    def test_truncate_forward_position_is_a_noop(self):
        b = backend(10, 13, 10)
        b.truncate(11)
        self.assertEqual((b.pos, b._mtp_pos, b._mtp_committed), (10, 13, 10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
