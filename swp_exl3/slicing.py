"""Prune an exllamav3 Model to a layer range (plan item D1). Pure Python: works on any object with the same
shape as exllamav3's Model, so it is unit-tested with fakes and verified on the real classes by `swp-exl3 doctor`.

Structure it relies on (verified in kyojin 1.5.10 and upstream 1.6.4 sources):
  model.modules          flat list: [Embedding, ExpandStreams?, TransformerBlock x N, HyperHead?, RMSNorm, Linear(lm_head)]
  module.layer_idx       on blocks
  module.caps            dict; the head has caps["logits_output"]
  model.first_block_idx, last_kv_module_idx, logit_layer_idx, modules_dict
  functools.cached_property caches (_get_cache_layers, _get_recurrent_layers, _get_prefetch_layers)
  model.prepare_layer_map()  rebuilds fwd_modules (no layer_map) from modules
"""
from dataclasses import dataclass, field
from functools import cached_property


class SliceError(Exception):
    pass


def module_kind(m):
    n = type(m).__name__
    caps = getattr(m, "caps", None) or {}
    if getattr(m, "layer_idx", None) is not None and "Block" in n:
        return "block"
    if caps.get("logits_output"):
        return "lm_head"
    if n == "Embedding":
        return "embed"
    if n == "ExpandStreams":
        return "expand"
    if n == "HyperHead":
        return "hc_head"
    if n.endswith("Norm"):
        return "norm"
    return "other"


@dataclass
class SliceInfo:
    layers: tuple
    first: bool
    last: bool
    kept: list = field(default_factory=list)
    n_dropped: int = 0
    n_blocks: int = 0
    has_recurrent: bool = False
    n_cache_layers: int = 0


def reset_cached_properties(model):
    for klass in type(model).__mro__:
        for name, val in vars(klass).items():
            if isinstance(val, cached_property):
                model.__dict__.pop(name, None)


def slice_model(model, a, b, first, last):
    """Keep blocks with a <= layer_idx < b. `first` keeps the prolog (embedding, stream expansion);
    `last` keeps the epilog (stream collapse, final norm, lm_head). Call BEFORE creating a Cache or loading."""
    if getattr(model.config, "layer_map", None):
        raise SliceError("models with a layer_map (relayering) are not supported by stage slicing")
    mods = list(model.modules)
    kinds = [module_kind(m) for m in mods]
    blocks = [i for i, k in enumerate(kinds) if k == "block"]
    if not blocks:
        raise SliceError("model has no transformer blocks")
    lo, hi = blocks[0], blocks[-1]
    have = sorted(mods[i].layer_idx for i in blocks)
    if not (0 <= a < b) or a < have[0] or b - 1 > have[-1]:
        raise SliceError(f"layer range [{a},{b}) outside model layers [{have[0]},{have[-1]}]")
    keep = []
    for i, (m, k) in enumerate(zip(mods, kinds)):
        if i < lo:
            keep.append(first)
        elif i > hi:
            keep.append(last)
        elif k == "block":
            keep.append(a <= m.layer_idx < b)
        else:
            raise SliceError(f"unexpected module between blocks: {type(m).__name__} ({getattr(m, 'key', '?')})")
    new = [m for m, kp in zip(mods, keep) if kp]
    nk = [module_kind(m) for m in new]
    if last and "lm_head" not in nk:
        raise SliceError("last stage requested but the model has no lm_head module")
    nb = [i for i, k in enumerate(nk) if k == "block"]
    model.modules = new
    model.modules_dict = None
    model.first_block_idx = nb[0]
    model.last_kv_module_idx = nb[-1]
    li = [i for i, k in enumerate(nk) if k == "lm_head"]
    model.logit_layer_idx = li[0] if li else None
    reset_cached_properties(model)
    model.prepare_layer_map()
    rec = model.get_recurrent_layers()
    return SliceInfo(
        layers=(a, b), first=first, last=last,
        kept=[f"{k}:{getattr(m, 'key', '?')}" for k, m in zip(nk, new)],
        n_dropped=len(mods) - len(new), n_blocks=len(nb),
        has_recurrent=bool(rec), n_cache_layers=len(model.get_cache_layers()),
    )
