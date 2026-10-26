import argparse, json, os, sys


def _tokenizer(model_dir):
    from exllamav3 import Config, Tokenizer
    return Tokenizer.from_config(Config.from_directory(model_dir))


def _encode(tok, text):
    return [int(i) for i in tok.encode(text, encode_special_tokens=True).flatten().tolist()]


def _read_ids_file(path):
    return [int(x) for x in open(path).read().replace(",", " ").split()]


def _ids_arg(a, tok, chat=False, system=None):
    """Token ids from --ids, --ids-file, --prompt-file or --prompt, in that order."""
    if a.ids:
        return [int(x) for x in a.ids.split(",")]
    if getattr(a, "ids_file", None):
        return _read_ids_file(a.ids_file)
    text = open(a.prompt_file).read() if getattr(a, "prompt_file", None) else a.prompt
    return _encode(tok, glm_chat(text, system) if chat else text)


def glm_chat(prompt, system=None):
    """GLM-5.x chat format (copied from exllamav3's Glm5NextModel.default_chat_prompt)."""
    s = "[gMASK]<sop>"
    if system:
        s += f"<|system|>\n{system}"
    return s + f"<|user|>\n{prompt}<|assistant|>\n"


# smoke: fixed prompts, 48 greedy tokens each. Golden generated once from the known-good build.
SMOKE_PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
]
SMOKE_LOGIT_PROMPT = "The capital of France is"
SMOKE_BENCH_PROMPT = (
    "Explain how a decoder-only transformer language model generates text. Cover tokenization, the attention "
    "mechanism, the feed-forward layers, and how sampling chooses the next token. Keep the explanation clear "
    "and suitable for a software engineer who has not studied machine learning before. Use concrete examples "
    "where they help. Then describe the main trade-offs between prefill speed and decode speed, why batching "
    "helps prefill more than decode, and what speculative decoding changes. Finally, suppose the model is "
    "split across two machines connected by a fast local link. Explain why per-token latency is roughly the "
    "sum of the two stages, why a pipeline bubble appears during prefill, and why balancing layers between "
    "the machines matters more for decode than for prefill. Finish with a short summary."
)


def _pipeline(args):
    from .config import load_pipeline, load_stage
    from .coordinator import Pipeline
    pc = load_pipeline(args.pipeline)
    # A local stage lives in this process: its env must be in place before exllamav3 is first imported
    # (the contract/tokenizer import below would otherwise freeze the defaults).
    for path in pc.stage_configs:
        if path:
            for k, v in load_stage(os.path.expanduser(path)).env.items():
                os.environ[k] = str(v)
    os.environ.setdefault("EXL3_PF_HC_FUSE", "0")
    os.environ.setdefault("EXL3_PF_GR_FUSE", "0")
    contract = None
    if pc.model_dir and not args.no_contract:
        from exllamav3 import Config
        from .contract import contract_from_config
        contract = contract_from_config(Config.from_directory(pc.model_dir))
    p = Pipeline(pc.stages, secret=pc.secret(), wire_dtype=pc.wire_dtype, chunk=pc.chunk, contract=contract,
                 stage_configs=pc.stage_configs)
    p.connect()
    return p, pc


def cmd_doctor(a):
    from .doctor import run
    return run(a.model_dir, tuple(a.layers) if a.layers else None, a.first, a.last)


def cmd_stage(a):
    from .config import load_stage
    from .exl3_backend import Exl3Backend
    from .server import StageServer
    c = load_stage(a.config)
    be = Exl3Backend(c)
    print(f"loading layers {c.layers} from {c.model_dir} ...", flush=True)
    print("memory:", be.load(), flush=True)
    if c.warmup:
        be.warmup(log=lambda s: print(s, flush=True))
    StageServer(be, c.node, c.listen, secret=c.secret(), wire_dtypes=c.wire_dtypes, max_chunk=c.max_chunk).serve_forever()


def cmd_pull(a):
    from .provision import pull
    from .config import _secret
    pull(a.server, a.out, a.layers[0], a.layers[1], a.first, a.last, secret=_secret(a.secret_env, a.insecure),
         include_vision=a.vision, tied_embed=a.tied_embed)


def kyojin_speed_env(kyojin_root):
    import ast
    path = os.path.join(os.path.expanduser(kyojin_root), "tools", "glm", "serve.py")
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
    except OSError as e:
        raise SystemExit(f"cannot read {path}: {e}")
    i = src.find("SPEED_ENV = {")
    if i < 0:
        raise SystemExit(f"{path}: no 'SPEED_ENV = {{' block found")
    j = src.find("\n}\n", i)
    if j < 0:
        raise SystemExit(f"{path}: unterminated SPEED_ENV block")
    try:
        d = ast.literal_eval(src[i + len("SPEED_ENV = "):j + 3])
    except (SyntaxError, ValueError) as e:
        raise SystemExit(f"{path}: cannot parse SPEED_ENV block: {e}")
    if not isinstance(d, dict) or not all(isinstance(k, str) and isinstance(v, (str, int, float)) for k, v in d.items()):
        raise SystemExit(f"{path}: SPEED_ENV is not a flat str->scalar dict")
    return d


def cmd_speed_env(a):
    d = kyojin_speed_env(a.kyojin_root)
    print(f"# {len(d)} flags from {os.path.expanduser(a.kyojin_root)}/tools/glm/serve.py")
    print("[stage.env]")
    for k, v in d.items():
        print(f"{k} = {json.dumps(str(v))}")


def _dict_diff(where, t, l):
    return [f"  {where}.{k}: template={t.get(k)!r} live={l.get(k)!r}"
            for k in sorted(set(t) | set(l))
            if type(t.get(k)) is not type(l.get(k)) or t.get(k) != l.get(k)]


def cmd_config_check(a):
    """Compare a machine-local stage config against a repo template so the flags do not drift unnoticed.
    Default compares [stage.env] and [stage.engine]; --all compares every field except the host-local ones."""
    import dataclasses
    from .config import load_stage
    tpl, live = load_stage(a.template), load_stage(a.config)
    if a.all:
        ignore = set(a.ignore) | {"model_dir", "listen", "cache_tokens", "mtp_cache_tokens"}
        rows = []
        for f in dataclasses.fields(live):
            if f.name in ignore:
                continue
            tv, lv = getattr(tpl, f.name), getattr(live, f.name)
            if isinstance(tv, dict) and isinstance(lv, dict):
                rows += _dict_diff(f.name, tv, lv)
            elif type(tv) is not type(lv) or tv != lv:
                rows.append(f"  {f.name}: template={tv!r} live={lv!r}")
        what = f"all fields except {', '.join(sorted(ignore))}"
    else:
        rows = _dict_diff("[stage.env]", tpl.env, live.env) + _dict_diff("[stage.engine]", tpl.engine, live.engine)
        what = "[stage.env] and [stage.engine]"
    if not rows:
        print(f"OK  {a.config} matches {a.template} on {what}")
        return 0
    print(f"DIFF  {a.config} vs {a.template} on {what}")
    print("\n".join(rows))
    return 1


def cmd_ping(a):
    p, _ = _pipeline(a)
    for s, r in zip(p.stages, p.rtt_ms(100)):
        print(f"{s.addr}  layers {s.layers[0]}..{s.layers[1]}  median RTT {r*1e3:.0f} us   (planner: \"link_rtt_ms\": {r:.3f})")


def _stop_ids(tok):
    stops = set(getattr(tok.config, "eos_token_id_list", []) or [tok.eos_token_id])
    for s in ("<|user|>", "<|endoftext|>", "<|observation|>"):
        try:
            stops.add(tok.single_id(s))
        except Exception:
            pass
    return stops


def cmd_generate(a):
    from .coordinator import SpecConfig
    p, pc = _pipeline(a)
    tok = _tokenizer(pc.model_dir) if pc.model_dir else None
    ids = _ids_arg(a, tok, a.chat, a.system)
    vocab, stops = None, set()
    if tok is not None:
        vocab = tok.get_id_to_piece_list()
        stops = _stop_ids(tok)
    stops |= {int(x) for x in (a.stop_id or [])}
    spec = None
    if a.spec:
        spec = SpecConfig(k=a.spec_k, match=a.spec_match, seed=a.seed,
                          accept="argmax" if a.temperature <= 0 else "sample")
        if p.spec_k <= 0:
            print("[spec] a stage advertises spec_k=0; running plain decode", file=sys.stderr)
    print(f"[prompt: {len(ids)} tokens]", file=sys.stderr)
    gen, st = p.generate(ids, a.max_new, a.temperature, a.top_p, pc.topk, stops, a.seed, spec=spec,
                         on_token=lambda t: (print(vocab[t] if vocab else f"{t} ", end="", flush=True)))
    print()
    if a.bench:
        r = p.report(st, out=lambda s: print(s, file=sys.stderr))
        print(f"reused prefix: {st['reused_prefix']} tokens", file=sys.stderr)
        if st.get("spec_rounds") is not None:
            n, d = st["spec_rounds"], max(st["spec_drafted"], 1)
            print(f"spec: {n} rounds, {st['spec_accepted']}/{st['spec_drafted']} drafts accepted "
                  f"({100.0 * st['spec_accepted'] / d:.1f}%), k={min(a.spec_k, p.spec_k)}", file=sys.stderr)
            if st.get("spec_mtp_drafts"):
                print(f"spec timing: verify {st.get('spec_verify_ms', 0):.1f} ms, catch-up "
                      f"{st.get('spec_catchup_ms', 0):.1f} ms, mtp roll {st.get('spec_roll_ms', 0):.1f} ms; "
                      f"{st.get('spec_accepted_per_round', 0):.2f} accepted/round, "
                      f"{st.get('spec_tokens_per_round', 0):.2f} tokens/round, "
                      f"{st.get('spec_mtp_drafts', 0)} MTP drafts, "
                      f"accepted/depth {st.get('spec_depth')}", file=sys.stderr)
        if a.json:
            json.dump(r, open(a.json, "w"), indent=1)


def cmd_logits(a):
    p, pc = _pipeline(a)
    ids = _ids_arg(a, None) if (a.ids or getattr(a, "ids_file", None)) else _ids_arg(a, _tokenizer(pc.model_dir))
    ti, tv = p.logits_topk(ids, a.k)
    json.dump({"stages": [list(s.layers) for s in p.stages], "wire": pc.wire_dtype, "prompt_ids": ids,
               "ids": ti.tolist(), "logits": tv.tolist()}, open(a.save, "w"))
    print(f"saved top-{a.k} logits for {len(ids)} prompt tokens to {a.save}  (cut points: {[s.layers for s in p.stages]})")


def _compare_logits(x, y):
    """Compare two saved top-k logit records. KL is over the shared top-k, renormalised
    (a proxy; full-vocab KL needs full logits)."""
    from .coordinator import compare_topk
    return compare_topk(x["ids"], x["logits"], y["ids"], y["logits"])


def cmd_compare(a):
    x, y = (json.load(open(f)) for f in (a.a, a.b))
    assert x["prompt_ids"] == y["prompt_ids"], "different prompts"
    r = _compare_logits(x, y)
    print(f"A cuts {x['stages']} wire={x['wire']}   B cuts {y['stages']} wire={y['wire']}")
    print(f"top-1 equal: {r['top1_equal']}   shared top-k ids: {r['shared']}/{r['topk']}   max |dlogit|: {r['max_abs']:.5f}   KL(A||B) on shared top-k: {r['kl']:.2e}")
    return 0 if r["top1_equal"] and r["kl"] < a.max_kl else 1


def _golden_path(a):
    return a.golden or os.path.splitext(a.pipeline)[0] + ".golden.json"


def _smoke_generate(p, tok, text, max_new=48, topk=64):
    ids = _encode(tok, text)
    vocab = tok.get_id_to_piece_list()
    gen, st = p.generate(ids, max_new, 0.0, 1.0, topk, _stop_ids(tok), None)
    return {"prompt": text, "prompt_ids": ids, "tokens": [int(t) for t in gen],
            "text": "".join(vocab[t] for t in gen)}, st


def cmd_smoke(a):
    """<90 s correctness gate: ping, two fixed greedy prompts vs golden, one top-64 logit check, one bench line."""
    path = _golden_path(a)
    if not a.update_golden and not os.path.exists(path):
        raise SystemExit(f"{path}: no golden; create it with --update-golden on a known-good build")
    p, pc = _pipeline(a)
    if not pc.model_dir:
        raise SystemExit("smoke needs [pipeline] model_dir for the tokenizer")
    tok = _tokenizer(pc.model_dir)
    p.reset()                                  # same starting state for every run

    print("[ping]", file=sys.stderr)
    for s, r in zip(p.stages, p.rtt_ms(50)):
        print(f"  {s.addr}  layers {s.layers[0]}..{s.layers[1]}  median RTT {r*1e3:.0f} us", file=sys.stderr)

    rec = {"version": 1, "model_dir": pc.model_dir, "wire_dtype": pc.wire_dtype,
           "stages": [list(s.layers) for s in p.stages], "prompts": [], "logits": None}
    # First real prefill after the stage's synthetic warm-up; the same prompt is repeated warm at the end.
    _, st_first = _smoke_generate(p, tok, SMOKE_BENCH_PROMPT, 48, pc.topk)
    for name, text in SMOKE_PROMPTS:
        g, _ = _smoke_generate(p, tok, text, 48, pc.topk)
        g["name"] = name
        rec["prompts"].append(g)
        print(f"[gen] {name}: {json.dumps(g['text'])[:160]}", file=sys.stderr)

    lid = _encode(tok, SMOKE_LOGIT_PROMPT)
    ti, tv = p.logits_topk(lid, 64)
    rec["logits"] = {"prompt": SMOKE_LOGIT_PROMPT, "prompt_ids": lid,
                     "ids": ti.tolist(), "logits": tv.tolist()}

    # Speculative verify must reproduce sequential-decode logits row for row (batched attention/recurrent state).
    # Top-1 equality is the functional gate. The KL tolerance is 1e-2, not 1e-3: the engine's batched GDN/attention
    # kernels round differently from 1-row decode (measured: 1-row verify bit-identical, 2-row KL 3e-5, 3/4-row
    # KL up to 5e-3; a plain 4-row prefill shows the same difference, so it is batching, not this code).
    SPEC_MAX_KL = 1e-2
    spec_ok = True
    if p.spec_k > 0:
        sc = p.spec_check(lid, rows=20, k=min(3, p.spec_k), klog=8)
        spec_ok = sc["top1"] and sc["max_kl"] <= SPEC_MAX_KL
        print(f"spec     {'ok' if spec_ok else 'FAIL'}  {sc['rows']} verify rows vs sequential  "
              f"top-1 mismatches: {sc['bad_top1']}  max KL: {sc['max_kl']:.2e}  (k={sc['k']})", file=sys.stderr)
    else:
        print("[spec] stages advertise spec_k=0; verify-vs-sequential check skipped", file=sys.stderr)
    rec["spec"] = {"rows": 20, "ok": spec_ok}

    g, st = _smoke_generate(p, tok, SMOKE_BENCH_PROMPT, 48, pc.topk)
    print(f"[bench prompt] {len(g['prompt_ids'])} tokens", file=sys.stderr)
    p.report(st, out=lambda s: print(s, file=sys.stderr))
    ft = st_first["prefill_tokens"] * 1e3 / max(st_first["prefill_ms"], 1e-9)
    wt = st["prefill_tokens"] * 1e3 / max(st["prefill_ms"], 1e-9)
    ratio = st_first["prefill_ms"] / max(st["prefill_ms"], 1e-9)
    print(f"[first prefill] {st_first['prefill_tokens']} tok in {st_first['prefill_ms']:.0f} ms -> {ft:.1f} tok/s; "
          f"warm {st['prefill_ms']:.0f} ms -> {wt:.1f} tok/s; first/warm {ratio:.2f}x (target <= 1.20x)", file=sys.stderr)

    if a.update_golden:
        with open(path, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"golden written: {path}  (stages {rec['stages']}, wire {rec['wire_dtype']})")
        return 0

    with open(path) as f:
        old = json.load(f)
    if old.get("version") != 1:
        raise SystemExit(f"{path}: unsupported golden version {old.get('version')!r}; re-create with --update-golden")
    if [x["prompt_ids"] for x in old["prompts"]] != [x["prompt_ids"] for x in rec["prompts"]]:
        raise SystemExit(f"{path}: prompts/tokenizer changed; re-create with --update-golden")

    ok = spec_ok
    for o, n in zip(old["prompts"], rec["prompts"]):
        same = o["tokens"] == n["tokens"]
        ok &= same
        print(f"prompt {n['name']:<8} {'ok' if same else 'MISMATCH'}  {len(n['tokens'])} tokens")
        if not same:
            for i, (x, y) in enumerate(zip(o["tokens"], n["tokens"])):
                if x != y:
                    print(f"  first difference at token {i}: golden {x} ({o['text'][:80]!r}) vs now {y} ({n['text'][:80]!r})")
                    break
    if old["logits"]["prompt_ids"] != rec["logits"]["prompt_ids"]:
        raise SystemExit(f"{path}: logit prompt tokenization changed; re-create with --update-golden")
    r = _compare_logits(old["logits"], rec["logits"])
    lok = r["top1_equal"] and r["kl"] <= a.max_kl
    ok &= lok
    print(f"logits   {'ok' if lok else 'FAIL'}  top-1 equal: {r['top1_equal']}  shared {r['shared']}/{r['topk']}  "
          f"max |dlogit|: {r['max_abs']:.5f}  KL: {r['kl']:.2e} (max {a.max_kl:.0e})")
    print("SMOKE", "PASS" if ok else "FAIL", f"(golden {path})")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="swp-exl3")
    sp = ap.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("doctor", help="verify engine APIs on this machine; optional dry-run slice")
    s.add_argument("--model-dir"); s.add_argument("--layers", nargs=2, type=int); s.add_argument("--first", action="store_true"); s.add_argument("--last", action="store_true")
    s.set_defaults(fn=cmd_doctor)

    s = sp.add_parser("stage", help="load a slice and serve it"); s.add_argument("--config", required=True); s.set_defaults(fn=cmd_stage)

    s = sp.add_parser("pull", help="pull one stage's tensors from swp-weightd")
    s.add_argument("--server", required=True); s.add_argument("--out", required=True); s.add_argument("--layers", nargs=2, type=int, required=True)
    s.add_argument("--first", action="store_true"); s.add_argument("--last", action="store_true")
    s.add_argument("--tied-embed", action="store_true"); s.add_argument("--vision", action="store_true")
    s.add_argument("--secret-env"); s.add_argument("--insecure", action="store_true"); s.set_defaults(fn=cmd_pull)

    def pl(s):
        s.add_argument("--pipeline", required=True); s.add_argument("--no-contract", action="store_true")
    s = sp.add_parser("ping", help="TCP round trip to each stage"); pl(s); s.set_defaults(fn=cmd_ping)

    s = sp.add_parser("generate", help="generate text through the pipeline (prefill/decode tok/s with --bench)"); pl(s)
    s.add_argument("--prompt"); s.add_argument("--prompt-file"); s.add_argument("--ids"); s.add_argument("--ids-file", help="file of comma/space-separated token ids (long prompts)"); s.add_argument("--chat", action="store_true"); s.add_argument("--system")
    s.add_argument("--max-new", type=int, default=128); s.add_argument("--temperature", type=float, default=0.0); s.add_argument("--top-p", type=float, default=1.0)
    s.add_argument("--seed", type=int); s.add_argument("--stop-id", action="append"); s.add_argument("--bench", action="store_true"); s.add_argument("--json")
    s.add_argument("--spec", action="store_true", help="speculative decode with prompt-lookup (n-gram) drafting")
    s.add_argument("--spec-k", type=int, default=3, help="drafts per verify round (capped by the stages' spec_k)")
    s.add_argument("--spec-match", type=int, default=2, help="n-gram match length for prompt-lookup drafting")
    s.set_defaults(fn=cmd_generate)

    s = sp.add_parser("smoke", help="<90 s correctness gate: ping, two greedy prompts, top-k logits, one bench"); pl(s)
    s.add_argument("--update-golden", action="store_true", help="(re)write the golden from this run")
    s.add_argument("--golden", help="golden JSON path (default: <pipeline>.golden.json)")
    s.add_argument("--max-kl", type=float, default=1e-4, help="max KL on the shared top-64 (default 1e-4)")
    s.set_defaults(fn=cmd_smoke)

    s = sp.add_parser("logits", help="save top-k logits for a prompt (cut-invariance check, step 1)"); pl(s)
    s.add_argument("--prompt"); s.add_argument("--ids"); s.add_argument("--ids-file"); s.add_argument("--k", type=int, default=64); s.add_argument("--save", required=True); s.set_defaults(fn=cmd_logits)

    s = sp.add_parser("compare", help="compare two saved logit files (cut-invariance check, step 2)")
    s.add_argument("a"); s.add_argument("b"); s.add_argument("--max-kl", type=float, default=1e-3); s.set_defaults(fn=cmd_compare)

    s = sp.add_parser("speed-env", help="print Kyojin's SPEED_ENV block as a [stage.env] table")
    s.add_argument("--kyojin-root", required=True, help="Kyojin checkout (contains tools/glm/serve.py)")
    s.set_defaults(fn=cmd_speed_env)

    s = sp.add_parser("config-check", help="diff a machine-local stage config against a repo template (env/knob drift)")
    s.add_argument("--config", required=True, help="live stage config on this machine")
    s.add_argument("--template", required=True, help="repo template, e.g. examples/stage-strix.toml")
    s.add_argument("--all", action="store_true", help="compare every field, not just [stage.env] and [stage.engine]")
    s.add_argument("--ignore", action="append", default=[], help="field to ignore with --all (repeatable)")
    s.set_defaults(fn=cmd_config_check)

    a = ap.parse_args(argv)
    if a.cmd == "generate" and not (a.prompt or a.prompt_file or a.ids or a.ids_file):
        ap.error("generate needs --prompt, --prompt-file or --ids")
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
