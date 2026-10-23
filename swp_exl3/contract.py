"""Model contract (spec A5): what must match between stages, independent of engine build."""
import json


def _hash(obj):
    import blake3
    return blake3.blake3(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def contract_from_config(config):
    """Build the contract from an exllamav3 Config (or any object with the same attributes)."""
    g = lambda n, d=None: getattr(config, n, d)
    hc = g("hc_mult", 0) or 0
    hidden = g("hidden_size")
    layers_sig = [list(g(n)) if g(n) is not None else None for n in ("layer_types", "mlp_layer_types", "indexer_types")]
    return {
        "arch": g("arch_string") or type(config).__name__,
        "n_layers": g("num_hidden_layers"),
        "hidden_size": hidden,
        "vocab_size": g("vocab_size"),
        "norm_eps": g("rms_norm_eps"),
        "hc_mult": hc,
        "layers_hash": _hash(layers_sig),
        # Boundary = residual stream entering the next layer: [hc_mult, hidden] for hyper-connection models.
        "boundary": {"tail": [hc, hidden] if hc else [hidden], "canonical": "residual_pre_layer"},
    }


def contract_hash(c):
    return _hash(c)


def diff_contracts(a, b, path=""):
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if isinstance(x, dict) and isinstance(y, dict):
            out += diff_contracts(x, y, f"{path}{k}.")
        elif x != y:
            out.append(f"{path}{k}: {x!r} != {y!r}")
    return out
