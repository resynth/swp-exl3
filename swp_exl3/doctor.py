"""`swp-exl3 doctor`: verify, on THIS machine's installed exllamav3, every engine API the adapter relies on, and
optionally dry-run the slice (build the model structure, no weights) for a layer range."""
import importlib, importlib.util, inspect, os, sys

OK, WARN, FAIL = "ok  ", "WARN", "FAIL"


def find_engine_marker(root, needle):
    """Files under `root` that contain `needle`: the literal in `.py` sources, or as a byte string in a
    compiled `.so`/`.pyd` fallback (an extension build can carry the marker without the source)."""
    def walk(exts, binary):
        hits = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in ("__pycache__", ".git"))
            for fn in sorted(filenames):
                if not fn.endswith(exts):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    if binary:
                        with open(p, "rb") as f:
                            if needle.encode() in f.read():
                                hits.append(p)
                    else:
                        with open(p, encoding="utf-8", errors="ignore") as f:
                            if needle in f.read():
                                hits.append(p)
                except OSError:
                    continue
        return hits
    return walk(".py", False) or walk((".so", ".pyd"), True)


def run(model_dir=None, layers=None, first=False, last=False, out=print, require_verify_fastpath=False):
    res = []

    def check(name, fn):
        try:
            r = fn()
            res.append((OK if r is not False else FAIL, name, "" if r in (True, None, False) else str(r)))
        except Exception as e:
            res.append((FAIL, name, f"{type(e).__name__}: {e}"))
        s, n, d = res[-1]
        out(f"[{s}] {n}" + (f"  {d}" if d else ""))

    def warn(name, detail):
        res.append((WARN, name, detail)); out(f"[{WARN}] {name}  {detail}")

    out(f"python {sys.version.split()[0]}")
    check("numpy", lambda: importlib.import_module("numpy").__version__)
    check("blake3 (needed for authentication)", lambda: importlib.import_module("blake3") and True)
    torch = importlib.import_module("torch")
    cuda = torch.cuda.is_available()
    out(f"torch {torch.__version__}  cuda/hip available={cuda}" + (f"  device={torch.cuda.get_device_name(0)}  hip={getattr(torch.version,'hip',None)}" if cuda else ""))
    if not cuda:
        warn("GPU", "torch sees no GPU; the stage server needs one")
    ex = importlib.import_module("exllamav3")
    ver = getattr(importlib.import_module("exllamav3.version"), "__version__", "?")
    out(f"exllamav3 {ver} from {ex.__file__}")

    # Kyojin gates its exact R-row verify kernels on this marker; without it a 2..8-row verify silently
    # takes the generic multi-row route (a slowdown, not a crash). Upstream has no marker and no such path.
    def verify_fastpath():
        name = "verify fast path (dflash_verify)"
        engine_root = os.path.dirname(os.path.abspath(ex.__file__))
        hits = find_engine_marker(engine_root, "dflash_verify")
        disabled = os.environ.get("SWP_VERIFY_FASTPATH", "1") == "0"
        kyojin = importlib.util.find_spec("exllamav3.modules.block_graph") is not None
        if hits:
            if disabled:
                warn(name, f"marker present in {len(hits)} file(s) but SWP_VERIFY_FASTPATH=0")
            else:
                rel = ", ".join(os.path.relpath(h, engine_root) for h in hits[:3])
                more = "" if len(hits) <= 3 else f" (+{len(hits) - 3})"
                check(name, lambda: f"marker in {len(hits)} file(s): {rel}{more}")
            return
        msg = ("marker missing on a Kyojin-like engine: verify forwards silently use the generic "
               "multi-row path") if kyojin else "generic multi-row path (expected on upstream exllamav3)"
        if kyojin or require_verify_fastpath:
            res.append((FAIL, name, msg))
            out(f"[{FAIL}] {name}  {msg}")
        else:
            warn(name, msg)
    verify_fastpath()

    def mtp_api():
        name = "MTP drafter API"
        try:
            from exllamav3.architecture.glm5_next import Glm5NextMTPModel
        except Exception as e:                                          # noqa: BLE001 - capability probe
            warn(name, f"MTP unavailable ({type(e).__name__}: {e}); keep mtp=false")
            return
        missing = [m for m in ("attach_to", "sample_from_state") if not hasattr(Glm5NextMTPModel, m)]
        if missing:
            warn(name, f"Glm5NextMTPModel lacks {missing}; keep mtp=false")
            return
        check(name, lambda: "Glm5NextMTPModel.attach_to + sample_from_state")
        sidecar = os.environ.get("EXL3_MTP_EH_FP16")
        if sidecar and sidecar != "0" and os.path.isfile(sidecar):
            out(f"       eh_proj sidecar: {sidecar}")
    mtp_api()

    from exllamav3 import Config, Model, Cache
    check("Config.from_directory + infer_params", lambda: hasattr(Config, "from_directory"))
    for m in ("from_config", "prepare_layer_map", "prepare_inputs", "forward_ls", "load_gen", "load", "get_cache_layers", "get_recurrent_layers"):
        check(f"Model.{m}", lambda m=m: hasattr(Model, m))
    check("Cache(max_num_tokens, max_batch_size, max_history, recurrent_dtype)",
          lambda: [p for p in ("max_num_tokens", "max_batch_size", "max_history", "recurrent_dtype") if p not in inspect.signature(Cache.__init__).parameters] == [] or
          f"missing: {[p for p in ('max_num_tokens','max_batch_size','max_history','recurrent_dtype') if p not in inspect.signature(Cache.__init__).parameters]}")
    check("Cache.get_new_state (recurrent models)", lambda: hasattr(Cache, "get_new_state"))

    def quant_cache():
        q = getattr(ex, "CacheLayer_quant", None)
        if q is None:
            warn("quantized KV cache (cache_bits)", "no CacheLayer_quant in this engine (cache_bits = 0 only)")
            return
        cps = inspect.signature(Cache.__init__).parameters
        qps = inspect.signature(q.__init__).parameters
        has_var = any(p.kind == p.VAR_KEYWORD for p in cps.values())
        missing = [p for p in ("layer_type",) if p not in cps]
        missing += [p for p in ("k_bits", "v_bits") if p not in qps]
        if not has_var:
            missing += [p for p in ("k_bits", "v_bits") if p in qps and p not in cps]
        missing = sorted(set(missing))
        if missing:
            warn("quantized KV cache (cache_bits)",
                 f"Cache/CacheLayer_quant lacks {missing} (cache_bits = 0 only)")
            return
        check("quantized KV cache (cache_bits)", lambda: "CacheLayer_quant available (cache_bits = 8)")
    quant_cache()
    check("load_gen(device=, progressbar=)", lambda: all(p in inspect.signature(Model.load_gen).parameters for p in ("device", "progressbar")))
    check("cache.recurrent_util.advance_recurrent_states", lambda: callable(importlib.import_module("exllamav3.cache.recurrent_util").advance_recurrent_states))
    hc = importlib.import_module("exllamav3.modules.hyperconnections")
    flush = "hc_flush" if hasattr(hc, "hc_flush") else ("HyperConnection.flush_pending" if hasattr(getattr(hc, "HyperConnection", None), "flush_pending") else None)
    check("hyper-connection flush helper", lambda: flush or False)
    check("architecture Glm5NextModel (GLM-5.3-Flash)", lambda: hasattr(importlib.import_module("exllamav3.architecture.glm5_next"), "Glm5NextModel"))
    knobs = ("moe_cpu_offload", "moe_cpu_split", "moe_cpu_threads", "moe_cpu_pinned_arena",
             "draft_moe_cpu_offload", "draft_moe_cpu_threads")
    if importlib.util.find_spec("exllamav3.modules.block_sparse_mlp_cpu"):
        def offload_info():
            ip = importlib.import_module("exllamav3.model.config").InferParams()
            have = [k for k in knobs if hasattr(ip, k)]
            if not have:
                return False
            return "defaults: " + ", ".join(f"{k}={getattr(ip, k)!r}" for k in have)
        check("CPU expert offload (modules.block_sparse_mlp_cpu)", offload_info)
    else:
        warn("CPU expert offload", "no exllamav3.modules.block_sparse_mlp_cpu in this engine (offload unavailable)")

    if model_dir:
        from .contract import contract_from_config
        from .slicing import slice_model
        out(f"\n--- dry run on {model_dir} (no weights are loaded)")
        cfg = Config.from_directory(model_dir)
        c = contract_from_config(cfg)
        out(f"arch={c['arch']} layers={c['n_layers']} hidden={c['hidden_size']} hc_mult={c['hc_mult']} boundary tail={c['boundary']['tail']}"
            f"  -> {4 * __import__('math').prod(c['boundary']['tail']) / 1024:.1f} KiB per row at f32")
        model = Model.from_config(cfg)
        if layers:
            a, b = layers
            info = slice_model(model, a, b, first, last)
            out(f"slice {a}..{b} first={first} last={last}: {info.n_blocks} blocks, {info.n_dropped} modules dropped, "
                f"recurrent={info.has_recurrent}, cache layers={info.n_cache_layers}")
            for k in info.kept[:3] + ["..."] + info.kept[-3:]:
                out(f"    {k}")
            res.append((OK, "dry-run slice", ""))
    fails = [r for r in res if r[0] == FAIL]
    out(f"\n{len(res) - len(fails)} passed, {len(fails)} failed")
    return 1 if fails else 0
