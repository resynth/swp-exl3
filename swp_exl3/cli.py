import argparse, json, math, os, sys
import numpy as np


def _tokenizer(model_dir):
    from exllamav3 import Config, Tokenizer
    return Tokenizer.from_config(Config.from_directory(model_dir))


def _encode(tok, text):
    return [int(i) for i in tok.encode(text, encode_special_tokens=True).flatten().tolist()]


def glm_chat(prompt, system=None):
    """GLM-5.x chat format (copied from exllamav3's Glm5NextModel.default_chat_prompt)."""
    s = "[gMASK]<sop>"
    if system:
        s += f"<|system|>\n{system}"
    return s + f"<|user|>\n{prompt}<|assistant|>\n"


def _pipeline(args):
    from .config import load_pipeline
    from .coordinator import Pipeline
    pc = load_pipeline(args.pipeline)
    contract = None
    if pc.model_dir and not args.no_contract:
        from exllamav3 import Config
        from .contract import contract_from_config
        contract = contract_from_config(Config.from_directory(pc.model_dir))
    p = Pipeline(pc.stages, secret=pc.secret(), wire_dtype=pc.wire_dtype, chunk=pc.chunk, contract=contract)
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
    StageServer(be, c.node, c.listen, secret=c.secret(), wire_dtypes=c.wire_dtypes, max_chunk=c.max_chunk).serve_forever()


def cmd_pull(a):
    from .provision import pull
    from .config import _secret
    pull(a.server, a.out, a.layers[0], a.layers[1], a.first, a.last, secret=_secret(a.secret_env, a.insecure),
         include_vision=a.vision, tied_embed=a.tied_embed)


def cmd_ping(a):
    p, _ = _pipeline(a)
    for s, r in zip(p.stages, p.rtt_ms(100)):
        print(f"{s.addr}  layers {s.layers[0]}..{s.layers[1]}  median RTT {r*1e3:.0f} us   (planner: \"link_rtt_ms\": {r:.3f})")


def cmd_generate(a):
    p, pc = _pipeline(a)
    tok = _tokenizer(pc.model_dir) if pc.model_dir else None
    if a.ids:
        ids = [int(x) for x in a.ids.split(",")]
    else:
        text = open(a.prompt_file).read() if a.prompt_file else a.prompt
        ids = _encode(tok, glm_chat(text, a.system) if a.chat else text)
    vocab, stops = None, set()
    if tok is not None:
        vocab = tok.get_id_to_piece_list()
        stops = set(getattr(tok.config, "eos_token_id_list", []) or [tok.eos_token_id])
        for s in ("<|user|>", "<|endoftext|>", "<|observation|>"):
            try:
                stops.add(tok.single_id(s))
            except Exception:
                pass
    stops |= {int(x) for x in (a.stop_id or [])}
    print(f"[prompt: {len(ids)} tokens]", file=sys.stderr)
    gen, st = p.generate(ids, a.max_new, a.temperature, a.top_p, pc.topk, stops, a.seed,
                         on_token=lambda t: (print(vocab[t] if vocab else f"{t} ", end="", flush=True)))
    print()
    if a.bench:
        r = p.report(st, out=lambda s: print(s, file=sys.stderr))
        print(f"reused prefix: {st['reused_prefix']} tokens", file=sys.stderr)
        if a.json:
            json.dump(r, open(a.json, "w"), indent=1)


def cmd_logits(a):
    p, pc = _pipeline(a)
    ids = [int(x) for x in a.ids.split(",")] if a.ids else _encode(_tokenizer(pc.model_dir), a.prompt)
    ti, tv = p.logits_topk(ids, a.k)
    json.dump({"stages": [list(s.layers) for s in p.stages], "wire": pc.wire_dtype, "prompt_ids": ids,
               "ids": ti.tolist(), "logits": tv.tolist()}, open(a.save, "w"))
    print(f"saved top-{a.k} logits for {len(ids)} prompt tokens to {a.save}  (cut points: {[s.layers for s in p.stages]})")


def cmd_compare(a):
    x, y = (json.load(open(f)) for f in (a.a, a.b))
    assert x["prompt_ids"] == y["prompt_ids"], "different prompts"
    ia, va, ib, vb = np.array(x["ids"]), np.array(x["logits"]), np.array(y["ids"]), np.array(y["logits"])
    common = [i for i in ia if i in set(ib)]
    da = dict(zip(ia, va)); db = dict(zip(ib, vb))
    diffs = [abs(da[i] - db[i]) for i in common]
    # KL over the shared top-k, renormalised (a proxy; full-vocab KL needs full logits)
    pa = np.exp(np.array([da[i] for i in common]) - max(da[i] for i in common)); pa /= pa.sum()
    pb = np.exp(np.array([db[i] for i in common]) - max(db[i] for i in common)); pb /= pb.sum()
    kl = float((pa * np.log(pa / pb)).sum())
    print(f"A cuts {x['stages']} wire={x['wire']}   B cuts {y['stages']} wire={y['wire']}")
    print(f"top-1 equal: {ia[0] == ib[0]}   shared top-k ids: {len(common)}/{len(ia)}   max |dlogit|: {max(diffs):.5f}   KL(A||B) on shared top-k: {kl:.2e}")
    return 0 if ia[0] == ib[0] and kl < a.max_kl else 1


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
    s.add_argument("--prompt"); s.add_argument("--prompt-file"); s.add_argument("--ids"); s.add_argument("--chat", action="store_true"); s.add_argument("--system")
    s.add_argument("--max-new", type=int, default=128); s.add_argument("--temperature", type=float, default=0.0); s.add_argument("--top-p", type=float, default=1.0)
    s.add_argument("--seed", type=int); s.add_argument("--stop-id", action="append"); s.add_argument("--bench", action="store_true"); s.add_argument("--json")
    s.set_defaults(fn=cmd_generate)

    s = sp.add_parser("logits", help="save top-k logits for a prompt (cut-invariance check, step 1)"); pl(s)
    s.add_argument("--prompt"); s.add_argument("--ids"); s.add_argument("--k", type=int, default=64); s.add_argument("--save", required=True); s.set_defaults(fn=cmd_logits)

    s = sp.add_parser("compare", help="compare two saved logit files (cut-invariance check, step 2)")
    s.add_argument("a"); s.add_argument("b"); s.add_argument("--max-kl", type=float, default=1e-3); s.set_defaults(fn=cmd_compare)

    a = ap.parse_args(argv)
    if a.cmd == "generate" and not (a.prompt or a.prompt_file or a.ids):
        ap.error("generate needs --prompt, --prompt-file or --ids")
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
