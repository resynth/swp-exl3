#!/usr/bin/env python3
"""Static (AST) check of every exllamav3 API that swp_exl3 relies on, against an exllamav3 source tree.
No torch / GPU needed. Usage:  python tools/check_api_against_source.py /path/to/exllamav3
(e.g. $(python -c "import exllamav3,os;print(os.path.dirname(exllamav3.__file__))") on your Strix and Razer)."""
import ast, os, re, sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
src = lambda rel: open(os.path.join(root, rel), encoding="utf-8").read()
tree = lambda rel: ast.parse(src(rel))
results = []


def check(name, fn):
    try:
        r = fn()
        ok = r is not False
        results.append((ok, name, "" if r in (True, None) else str(r)))
    except Exception as e:
        results.append((False, name, f"{type(e).__name__}: {e}"))


def cls(t, name):
    return next(n for n in ast.walk(t) if isinstance(n, ast.ClassDef) and n.name == name)


def funcs(c):
    return {n.name: n for n in c.body if isinstance(n, ast.FunctionDef)}


def params(f):
    return [a.arg for a in f.args.args + f.args.kwonlyargs]


def decorated(f, deco):
    return any((isinstance(d, ast.Name) and d.id == deco) or (isinstance(d, ast.Attribute) and d.attr == deco) for d in f.decorator_list)


M = tree("model/model.py")
model = cls(M, "Model")
mf = funcs(model)
mls = funcs(cls(tree("model/model_ls.py"), "Model_LSMixin")) if "class Model_LSMixin" in src("model/model_ls.py") else {}
mtxt = src("model/model.py") + src("model/model_ls.py")

for m in ("from_config", "prepare_layer_map", "load_gen", "load", "forward", "get_cache_layers", "get_recurrent_layers"):
    check(f"Model.{m}", lambda m=m: m in mf or m in mls)
check("Model.forward_ls(x, params)", lambda: params(mls["forward_ls"])[:3] == ["self", "x", "params"] or params(mls["forward_ls"]))
for m in ("_get_cache_layers", "_get_recurrent_layers", "_get_prefetch_layers"):
    check(f"Model.{m} is a cached_property (slicing resets it)", lambda m=m: decorated(mf[m], "cached_property"))
for a in ("modules", "first_block_idx", "last_kv_module_idx", "logit_layer_idx", "modules_dict"):
    check(f"Model sets self.{a}", lambda a=a: re.search(rf"self\.{a}\s*[:=]", mtxt) is not None)
check("load_gen(device=, progressbar=)", lambda: all(p in params(mf["load_gen"]) for p in ("device", "progressbar")))
check("prepare_layer_map handles layer_map=None", lambda: "layer_map" in ast.get_source_segment(src("model/model.py"), mf["prepare_layer_map"]))

C = cls(tree("cache/cache.py"), "Cache")
cf = funcs(C)
for p in ("max_num_tokens", "max_batch_size", "max_history", "recurrent_dtype"):
    check(f"Cache.__init__ has {p}", lambda p=p: p in params(cf["__init__"]))
check("Cache.get_new_state", lambda: "get_new_state" in cf)

ru = funcs(tree("cache/recurrent_util.py"))
rt = tree("cache/recurrent_util.py")
rfun = {n.name: n for n in ast.walk(rt) if isinstance(n, ast.FunctionDef)}
check("advance_recurrent_states(input_ids, params, model)", lambda: params(rfun["advance_recurrent_states"]) == ["input_ids", "params", "model"] or params(rfun["advance_recurrent_states"]))
check("prepare_for_recurrence(input_ids, params, model)", lambda: params(rfun["prepare_for_recurrence"]) == ["input_ids", "params", "model"] or params(rfun["prepare_for_recurrence"]))

H = tree("modules/hyperconnections.py")
hnames = {n.name for n in ast.walk(H) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
check("hyper-connection flush helper (hc_flush OR HyperConnection.flush_pending)", lambda: ("hc_flush" in hnames) or ("flush_pending" in hnames) or False)
flush = "hc_flush" if "hc_flush" in hnames else "HyperConnection.flush_pending" if "flush_pending" in hnames else None
for c_ in ("ExpandStreams", "HyperConnection", "HyperHead"):
    check(f"class {c_}", lambda c_=c_: c_ in hnames)
check("hc_pending is used (deferred residual path exists)", lambda: "hc_pending" in src("modules/hyperconnections.py"))

G = src("architecture/glm5_next.py")
check("class Glm5NextModel", lambda: "class Glm5NextModel" in G)
check("Glm5NextModel.prepare_inputs", lambda: "prepare_inputs" in funcs(cls(ast.parse(G), "Glm5NextModel")))
for c_ in ("Embedding(", "ExpandStreams(", "TransformerBlock(", "HyperHead(", "RMSNorm(", "Linear("):
    check(f"Glm5Next builds {c_[:-1]}", lambda c_=c_: c_ in G)
check("Glm5Next recurrent state class (GDNState)", lambda: "recurrent_state_cls" in G)

MT = src("architecture/glm5_next_mtp.py")
mtp = cls(ast.parse(MT), "Glm5NextMTPModel")
mtpf = funcs(mtp)
check("class Glm5NextMTPModel", lambda: "Glm5NextMTPModel" in MT)
for m in ("attach_to", "sample_from_state", "prepare_inputs", "default_load_params"):
    check(f"Glm5NextMTPModel.{m}", lambda m=m: m in mtpf)
check("Glm5NextMTPModel.load_eh_proj_sidecar (optional)",
      lambda: True if "load_eh_proj_sidecar" in mtpf else "absent: sidecar ignored")
check("Glm5NextMTPModel.draft_verifier_params", lambda: "draft_verifier_params" in MT)
check("Model.from_config(component=)", lambda: "component" in params(mf["from_config"]))
check("Model.prefill(input_ids, params)", lambda: "prefill" in mf and params(mf["prefill"])[:3] == ["self", "input_ids", "params"])
check("modules.Embedding", lambda: "class Embedding" in src("modules/embedding.py"))
check("Embedding.load(device)", lambda: "load" in funcs(cls(tree("modules/embedding.py"), "Embedding")))

allsrc = "".join(src(os.path.relpath(os.path.join(d, f), root)) for d, _, fs in os.walk(root) for f in fs if f.endswith(".py"))
for k in ("last_tokens_only", "recurrent_states", "past_len", "batch_shape", "attn_mode", "infer_params",
          "moe_cpu_offload", "export_state_norm_keys", "export_states", "target_hidden"):
    check(f"engine references '{k}'", lambda k=k: k in allsrc)
check("Module.prepare_for_device", lambda: "prepare_for_device" in src("modules/module.py"))
check("Config.from_directory", lambda: "from_directory" in src("model/config.py"))

width = max(len(n) for _, n, _ in results)
bad = 0
for ok, n, d in results:
    bad += not ok
    print(f"[{'ok  ' if ok else 'FAIL'}] {n:<{width}} {d}")
print(f"\n{root}\n  version: {re.search(r'\"([^\"]+)\"', open(os.path.join(root, 'version.py')).read()).group(1)}   hc flush helper: {flush}\n  {len(results) - bad} passed, {bad} failed")
sys.exit(1 if bad else 0)
