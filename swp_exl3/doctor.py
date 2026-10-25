"""`swp-exl3 doctor`: verify, on THIS machine's installed exllamav3, every engine API the adapter relies on, and
optionally dry-run the slice (build the model structure, no weights) for a layer range."""
import importlib, importlib.util, inspect, sys

OK, WARN, FAIL = "ok  ", "WARN", "FAIL"


def run(model_dir=None, layers=None, first=False, last=False, out=print):
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

    from exllamav3 import Config, Model, Cache
    check("Config.from_directory + infer_params", lambda: hasattr(Config, "from_directory"))
    for m in ("from_config", "prepare_layer_map", "prepare_inputs", "forward_ls", "load_gen", "load", "get_cache_layers", "get_recurrent_layers"):
        check(f"Model.{m}", lambda m=m: hasattr(Model, m))
    check("Cache(max_num_tokens, max_batch_size, max_history, recurrent_dtype)",
          lambda: [p for p in ("max_num_tokens", "max_batch_size", "max_history", "recurrent_dtype") if p not in inspect.signature(Cache.__init__).parameters] == [] or
          f"missing: {[p for p in ('max_num_tokens','max_batch_size','max_history','recurrent_dtype') if p not in inspect.signature(Cache.__init__).parameters]}")
    check("Cache.get_new_state (recurrent models)", lambda: hasattr(Cache, "get_new_state"))
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
