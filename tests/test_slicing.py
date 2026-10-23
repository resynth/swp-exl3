import os, sys, unittest
from functools import cached_property
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swp_exl3.slicing import slice_model, SliceError, module_kind

# Fakes mirroring exllamav3's structure for Glm5NextModel (class names matter, they are what slicing keys on).
class M:
    def __init__(self, key, caps=None, layer_idx=None):
        self.key, self.caps, self.layer_idx = key, caps or {}, layer_idx
class Embedding(M): pass
class ExpandStreams(M): pass
class HyperHead(M): pass
class RMSNorm(M): pass
class Linear(M): pass
class TransformerBlock(M): pass

class Cfg:
    layer_map = None

class Model:
    def __init__(self, n=6, kda=(0, 1, 2, 4, 5)):
        self.config = Cfg()
        self.modules = [Embedding("emb"), ExpandStreams("hc_expand")]
        self.first_block_idx = len(self.modules)
        for i in range(n):
            caps = {"recurrent_cache": True} if i in kda else {"kv_cache": True}
            self.modules.append(TransformerBlock(f"layers.{i}", caps, layer_idx=i))
        self.last_kv_module_idx = len(self.modules) - 1
        self.modules += [HyperHead("hc_head"), RMSNorm("norm"), Linear("lm_head", {"logits_output": True})]
        self.logit_layer_idx = len(self.modules) - 1
        self.modules_dict = None
        self.prepared = 0
        self.prepare_layer_map()
    def __iter__(self): return iter(self.modules)
    @cached_property
    def _get_cache_layers(self): return [m for m in self if m.caps.get("kv_cache")]
    def get_cache_layers(self): return self._get_cache_layers
    @cached_property
    def _get_recurrent_layers(self): return [m for m in self if m.caps.get("recurrent_cache")]
    def get_recurrent_layers(self): return self._get_recurrent_layers
    def prepare_layer_map(self):
        self.fwd_modules = [(m, 0, i) for i, m in enumerate(self.modules)]
        self.last_kv_module_idx_instance = (self.last_kv_module_idx, 0)
        self.prepared += 1


class T(unittest.TestCase):
    def warm(self, m):          # populate cached properties as a real Cache() construction would
        m.get_cache_layers(); m.get_recurrent_layers()

    def test_kinds(self):
        m = Model()
        self.assertEqual([module_kind(x) for x in m.modules],
                         ["embed", "expand"] + ["block"] * 6 + ["hc_head", "norm", "lm_head"])

    def test_first_middle_last(self):
        for (a, b, first, last, kept_keys) in (
            (0, 2, True, False, ["emb", "hc_expand", "layers.0", "layers.1"]),
            (2, 4, False, False, ["layers.2", "layers.3"]),
            (4, 6, False, True, ["layers.4", "layers.5", "hc_head", "norm", "lm_head"]),
            (0, 6, True, True, None),
        ):
            m = Model(); self.warm(m)
            info = slice_model(m, a, b, first, last)
            keys = [x.key for x in m.modules]
            if kept_keys:
                self.assertEqual(keys, kept_keys)
            self.assertEqual(info.n_blocks, b - a)
            self.assertEqual(m.modules[m.first_block_idx].layer_idx, a)
            self.assertEqual(m.modules[m.last_kv_module_idx].layer_idx, b - 1)
            self.assertEqual(m.logit_layer_idx, len(m.modules) - 1 if last else None)
            self.assertEqual([x[0].key for x in m.fwd_modules], keys)          # layer map rebuilt
            self.assertEqual(m.last_kv_module_idx_instance, (m.last_kv_module_idx, 0))
            # cached properties were invalidated and now reflect only kept layers
            self.assertEqual(sorted(x.layer_idx for x in m.get_cache_layers() + m.get_recurrent_layers()), list(range(a, b)))
            self.assertEqual(info.has_recurrent, any(i in (0, 1, 2, 4, 5) for i in range(a, b)))

    def test_errors(self):
        for args in ((0, 0), (3, 9), (-1, 2)):
            with self.assertRaises(SliceError):
                slice_model(Model(), *args, True, False)
        m = Model(); m.config.layer_map = [0, 1]
        with self.assertRaises(SliceError):
            slice_model(m, 0, 2, True, False)
        m = Model(); m.modules.pop()                      # no lm_head but last requested
        with self.assertRaises(SliceError):
            slice_model(m, 4, 6, False, True)
        m = Model(); m.modules.insert(4, RMSNorm("stray"))
        with self.assertRaises(SliceError):
            slice_model(m, 0, 3, True, False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
