#!/usr/bin/env python3
"""Serve a swp-exl3 Pipeline through Kyojin's OpenAI-compatible HTTP layer.

Kyojin's tools/glm/serve.py (MIT) already implements everything on the HTTP side: /v1/chat/completions, /v1/completions,
SSE streaming, chat-template rendering, tool-call and reasoning parsing, /v1/models, /health, /tokenize, /metrics. Its
`create_app(engine, model_id, template)` takes the engine as a plain object, so this file only supplies that object:
`PipelineEngine`, built on swp_exl3's public `Pipeline` API. Nothing in swp-exl3 is modified.

Run it on a machine that has the Kyojin checkout (the Strix):
    python swp_kyojin_server.py --pipeline pipeline.toml --model-dir /models/GLM-5.3-Flash-exl3 \
        --kyojin-root ~/engines/kyojin --ctx 8192 --port 8000
(needs `swp_exl3` and `exllamav3`/Kyojin importable, plus aiohttp and jinja2, which Kyojin's serve.py already uses.)

Engine contract it implements (from Kyojin's serve.py):
    async generate(prompt, *, max_tokens, temperature, top_p, stop, sampler_kw, cancel, images, stats, img_stats,
                   prompt_ids, ignore_eos) -> async iterator of incremental text
    count_tokens(text), prompt_ids(prompt), encode_ids(text, add_special_tokens), last_stats (dict), ctx, num_draft,
    supports_vision; optional: reset_prefix_cache().
Limits: one sequence at a time; sampling uses the head stage's top-64 (temperature, top_p, top_k, seed only);
no images; no min_p / penalties / logit_bias / grammar (a warning is logged once if a request asks for them).
"""
import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path


class _Cancelled(Exception):
    pass


SPEC_LOG_MAX_BYTES = 64 * 1024 * 1024      # rotate --spec-log at 64 MiB
SPEC_LOG_BACKUPS = 2                       # keep .1 and .2


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class _StopFilter:
    """Cuts the text stream at the first stop string; holds back a tail that could still become one."""

    def __init__(self, stops):
        self.stops = [s for s in (stops or []) if s]
        self.buf = ""
        self.hit = False

    def feed(self, text):
        if self.hit:
            return ""
        self.buf += text
        for s in self.stops:
            i = self.buf.find(s)
            if i >= 0:
                out, self.buf, self.hit = self.buf[:i], "", True
                return out
        hold = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(s[:k]):
                    hold = max(hold, k)
                    break
        out, self.buf = self.buf[:len(self.buf) - hold], self.buf[len(self.buf) - hold:]
        return out

    def flush(self):
        out, self.buf = ("" if self.hit else self.buf), ""
        return out


class _Detok:
    """Incremental detokenizer: emits text only when it cannot change (no dangling partial UTF-8)."""

    def __init__(self, decode):
        self.decode, self.ids, self.prefix, self.read = decode, [], 0, 0

    def feed(self, tok):
        self.ids.append(tok)
        before = self.decode(self.ids[self.prefix:self.read]) if self.read > self.prefix else ""
        now = self.decode(self.ids[self.prefix:])
        if len(now) > len(before) and not now.endswith("\ufffd"):
            delta = now[len(before):]
            self.prefix, self.read = self.read, len(self.ids)
            return delta
        return ""

    def flush(self):
        now = self.decode(self.ids[self.prefix:])
        before = self.decode(self.ids[self.prefix:self.read]) if self.read > self.prefix else ""
        self.prefix = self.read = len(self.ids)
        return now[len(before):] if len(now) > len(before) else ""


def _spec_mix(st):
    """Compact source mix: `n-gram 12r 44/44 | mtp 4r 9/12 | plain 1`."""
    parts = []
    if int(st.get("spec_ngram_rounds") or 0):
        parts.append(f"n-gram {int(st['spec_ngram_rounds'])}r "
                     f"{int(st.get('spec_ngram_accepted') or 0)}/{int(st.get('spec_ngram_drafted') or 0)}")
    if int(st.get("spec_mtp_rounds") or 0):
        parts.append(f"mtp {int(st['spec_mtp_rounds'])}r "
                     f"{int(st.get('spec_mtp_accepted') or 0)}/{int(st.get('spec_mtp_drafted') or 0)}")
    if int(st.get("spec_plain_rounds") or 0):
        parts.append(f"plain {int(st['spec_plain_rounds'])}")
    return " | ".join(parts)


def request_line(n, cache, pre_n, total, pre_s, out_n, gen_n, dec_s, accepted=0, rejected=0, rounds=0, mix=None):
    """One terminal line per request: cache verdict, prefill/decode rates, and (when drafting) the source mix."""
    eff_tps, new_tps = total / pre_s, pre_n / pre_s
    dec_n = max(gen_n - 1, 1)
    spec = ""
    mix_line = _spec_mix(mix or {})
    if mix_line:
        spec = f" | spec {mix_line}"
    elif accepted + rejected:
        pct = 100.0 * accepted / (accepted + rejected)
        spec = f" | spec {accepted}/{accepted + rejected} ({pct:.1f}%" + (f", {rounds} rounds)" if rounds else ")")
    return (f"[req {n}] {cache} | prefill {pre_n} new + {total - pre_n} cached in {pre_s * 1e3:.0f} ms "
            f"({new_tps:.1f} new tok/s, {eff_tps:.1f} effective) | decode {out_n} tok in "
            f"{dec_s * 1e3:.0f} ms ({dec_n / dec_s:.1f} tok/s){spec}")


class PipelineEngine:
    supports_prefix_reuse = True
    supports_vision = False
    row_budget = 0
    sessions = 1
    num_draft = 0

    def __init__(self, pipeline, tokenizer, ctx, eos_ids=(), seq_factory=None, log=print, spec=None, spec_log=None):
        self.pipeline, self.tok, self.ctx, self.log = pipeline, tokenizer, int(ctx), log
        self.eos_ids = {int(e) for e in eos_ids}
        self._seq = seq_factory or _default_seq
        self.spec = spec               # coordinator.SpecConfig (n-gram + MTP fallback); None = plain decode
        self.num_draft = max(int(getattr(spec, "k", 0) or 0), int(getattr(spec, "mtp_k", 0) or 0)) \
            if spec is not None else 0
        self.spec_log = spec_log       # optional JSONL path: one object per request
        self.last_stats = {}
        self._prev_ids = None           # exact ids of the previous prompt + reply (including the last sampled token)
        self._warned = set()
        self._requests = 0

    # ---------------------------------------------------------------- tokenizer surface (same shape as ResidentEngine)
    def _enc(self, text, add_bos=False):
        ids = self.tok.encode(text, add_bos=add_bos, encode_special_tokens=True)
        ids = ids[0] if isinstance(ids, tuple) else ids
        return [int(t) for t in ids.flatten().tolist()]

    def _dec(self, ids):
        out = self.tok.decode(self._seq(list(ids)), decode_special_tokens=True)
        return out[0] if isinstance(out, (list, tuple)) else out

    def count_tokens(self, text):
        return len(self._enc(text))

    def prompt_ids(self, prompt):
        ids = self.tok.encode(prompt, encode_special_tokens=True)
        ids = ids[0] if isinstance(ids, tuple) else ids
        return ids.reshape(-1)

    def encode_ids(self, text, add_special_tokens=False):
        return self._enc(text, add_bos=add_special_tokens)

    async def reset_prefix_cache(self):
        self._prev_ids = None
        await asyncio.get_running_loop().run_in_executor(None, self.pipeline.reset)

    # ---------------------------------------------------------------- prefix reuse
    def _ids_for(self, prompt):
        """Token ids for `prompt`, reusing the EXACT ids of the previous prompt+reply for the part of the text that is
        unchanged. Re-tokenising a model's own reply can differ from what it generated; on this model any mismatch inside
        the history costs a full replay (recurrent state cannot rewind), so keep the generated ids wherever possible."""
        prev = self._prev_ids
        if not prev:
            return self._enc(prompt)
        prev_text = self._dec(prev)
        if prompt.startswith(prev_text):       # append-only turn: no per-token binary search
            return prev + (self._enc(prompt[len(prev_text):]) if prompt[len(prev_text):] else [])
        common = common_prefix_len(prev_text, prompt)
        if common == 0:
            return self._enc(prompt)
        lo, hi = 0, len(prev)
        while lo < hi:                  # largest n whose decoded text stays inside the common text
            mid = (lo + hi + 1) // 2
            d = self._dec(prev[:mid])
            if len(d) <= common and prompt.startswith(d):
                lo = mid
            else:
                hi = mid - 1
        head = prev[:lo]
        consumed = self._dec(head) if head else ""
        if not prompt.startswith(consumed):
            return self._enc(prompt)
        return head + (self._enc(prompt[len(consumed):]) if prompt[len(consumed):] else [])

    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            self.log(f"[swp-kyojin-server] {msg}")

    def _append_spec_log(self, rec):
        """Append one JSONL record, rotating at SPEC_LOG_MAX_BYTES (keeps .1/.2 backups)."""
        path = self.spec_log
        try:
            if os.path.exists(path) and os.path.getsize(path) >= SPEC_LOG_MAX_BYTES:
                for i in range(SPEC_LOG_BACKUPS - 1, 0, -1):
                    src = f"{path}.{i}"
                    if os.path.exists(src):
                        os.replace(src, f"{path}.{i + 1}")
                os.replace(path, f"{path}.1")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except OSError as e:
            self._warn_once("spec_log", f"cannot write --spec-log {path}: {e}")

    # ---------------------------------------------------------------- generation
    async def generate(self, prompt, *, max_tokens, temperature, top_p, stop, sampler_kw=None, cancel=None,
                       images=None, stats=None, img_stats=None, prompt_ids=None, ignore_eos=False,
                       loop_guard=None):
        if images:
            raise RuntimeError("this engine does not support images")
        if loop_guard is not None:
            self._warn_once("loop_guard", "loop_guard is not implemented by this engine; ignoring it")
        loop = asyncio.get_running_loop()
        ids = await loop.run_in_executor(
            None,
            (lambda: [int(t) for t in prompt_ids]) if prompt_ids is not None
            else (lambda: self._ids_for(prompt)))
        # Admission in Kyojin's serve.py uses prompt_ids(); _ids_for can return more exact previous-reply ids,
        # so clamp now (and reject) against the ids actually fed, not only the re-tokenised length.
        room = self.ctx - len(ids) - 1 - self.num_draft
        if room < 1:
            from aiohttp import web
            raise web.HTTPBadRequest(
                reason=f"context length exceeded: prompt is {len(ids)} tokens, the server context is {self.ctx}")
        max_tokens = min(int(max_tokens), room)
        sk = dict(sampler_kw or {})
        top_k = max(1, min(int(sk.pop("top_k", 0) or 64), 64))
        seed = sk.pop("seed", None)
        ignored = sorted(k for k, v in sk.items() if v not in (None, 0, 0.0, 1.0, False, {}, []))
        if ignored:
            self._warn_once("sk:" + ",".join(ignored), f"ignoring unsupported sampler options {ignored} (head stage returns top-64 only)")

        eos = set() if ignore_eos else self.eos_ids
        history_before = list(self.pipeline.history)
        reused = common_prefix_len(history_before, ids)
        if reused >= len(ids):
            reused = len(ids) - 1
        q = asyncio.Queue()
        gen_ids, stop_flag, t0 = [], threading.Event(), time.perf_counter()
        marks, st_box = {}, {}
        spec = self.spec
        if spec is not None:
            spec = type(spec)(k=spec.k, match=spec.match, mtp_k=getattr(spec, "mtp_k", 2),
                              use_mtp=getattr(spec, "use_mtp", True),
                              accept=spec.accept, drafter=spec.drafter, seed=seed)

        def on_token(t):
            if stop_flag.is_set() or (cancel is not None and cancel.is_set()):
                raise _Cancelled()
            marks.setdefault("first", time.perf_counter())
            gen_ids.append(int(t))
            loop.call_soon_threadsafe(q.put_nowait, ("tok", int(t)))

        def work():
            try:
                _, st_box["st"] = self.pipeline.generate(
                    ids, max_new=max(1, int(max_tokens)), temperature=float(temperature or 0.0),
                    top_p=float(top_p if top_p is not None else 1.0), top_k=top_k,
                    stop_ids=tuple(eos), seed=seed, on_token=on_token, spec=spec)
                loop.call_soon_threadsafe(q.put_nowait, ("done", None))
            except _Cancelled:
                loop.call_soon_threadsafe(q.put_nowait, ("done", None))
            except BaseException as e:      # noqa: BLE001 - forwarded to the request
                loop.call_soon_threadsafe(q.put_nowait, ("err", e))

        fut = loop.run_in_executor(None, work)
        detok, filt = _Detok(self._dec), _StopFilter(stop)
        try:
            while True:
                kind, val = await q.get()
                if kind == "err":
                    raise val
                if kind == "done":
                    break
                if val in eos:                          # end-of-turn token: never shown
                    continue
                text = filt.feed(detok.feed(val))
                if text:
                    yield text
                if filt.hit:
                    stop_flag.set()
            if not filt.hit:
                tail = filt.feed(detok.flush()) + filt.flush()
                if tail:
                    yield tail
        finally:
            stop_flag.set()                             # client gone / error / finished: make the worker stop
            try:
                await fut
            except BaseException:                       # noqa: BLE001 - already reported through the queue
                pass
            self._prev_ids = ids + gen_ids
            end = time.perf_counter()
            first = marks.get("first", end)
            st = st_box.get("st") or {}
            replayed = int(st.get("replayed_tokens") or 0)
            # `reused` is the logical prefix; a clear+replay recomputed it, so nothing came from the cache.
            cached = 0 if replayed else reused
            accepted = int(st.get("spec_accepted") or 0)
            rejected = max(int(st.get("spec_drafted") or 0) - accepted, 0)
            self.last_stats = {"prompt_tokens": len(ids), "cached_tokens": cached, "replayed_tokens": replayed,
                               "rollback": st.get("rollback"),
                               "reused_prefix": reused, "new_tokens": len(gen_ids),
                               "time_prefill": first - t0, "time_generate": max(end - first, 0.0),
                               "accepted_draft_tokens": accepted, "rejected_draft_tokens": rejected,
                               "spec_match": int(getattr(self.spec, "match", 0) or 0)}
            for key in ("spec_rounds", "spec_drafted", "spec_accepted", "spec_mtp_drafts",
                        "spec_ngram_rounds", "spec_ngram_drafted", "spec_ngram_accepted",
                        "spec_mtp_rounds", "spec_mtp_drafted", "spec_mtp_accepted",
                        "spec_plain_rounds", "spec_lookup_hits", "spec_lookup_len_total"):
                self.last_stats[key] = int(st.get(key) or 0)
            for key in ("spec_ngram_verify_ms", "spec_mtp_verify_ms", "spec_mtp_roll_ms"):
                self.last_stats[key] = round(float(st.get(key) or 0.0), 3)
            self._requests += 1
            if self.spec_log:
                self._append_spec_log({"timestamp": time.time(), "request": self._requests, **self.last_stats})
            out_n = sum(1 for t in gen_ids if t not in eos)
            pre_s, dec_s = max(first - t0, 1e-6), max(end - first, 1e-6)
            if replayed:
                cache, pre_n = f"MISS (replayed {replayed})", len(ids)
            elif reused > 0:
                cache, pre_n = f"HIT {reused}/{len(ids)}", max(len(ids) - reused, 0)
            else:
                cache, pre_n = "MISS", len(ids)
            self.log(request_line(self._requests, cache, pre_n, len(ids), pre_s, out_n, len(gen_ids), dec_s,
                                  accepted, rejected, int(st.get("spec_rounds") or 0), mix=st))
            if stats is not None:
                stats.update(self.last_stats)


def _default_seq(ids):
    try:
        import torch
        return torch.tensor(ids, dtype=torch.long)
    except Exception:       # noqa: BLE001 - tests use list-based fake tokenizers
        return ids


# -------------------------------------------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(description="OpenAI-compatible server for a swp-exl3 pipeline, using Kyojin's HTTP layer")
    ap.add_argument("--pipeline", required=True, help="swp-exl3 pipeline.toml")
    ap.add_argument("--model-dir", required=True, help="model directory (tokenizer, config, chat_template.jinja)")
    ap.add_argument("--kyojin-root", default=os.environ.get("KYOJIN_ROOT", "~/engines/kyojin"))
    ap.add_argument("--model-id", default="glm-5.3-exl3")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ctx", type=int, default=8192, help="context window to advertise: the SMALLEST stage cache_tokens")
    ap.add_argument("--chat-template", default=None, help="jinja file (default: Kyojin's GLM-5.3 template, else <model>/chat_template.jinja)")
    ap.add_argument("--stop-id", action="append", default=[], help="extra stop token id (repeatable)")
    ap.add_argument("--spec", action="store_true", help="speculative decoding: prompt-lookup (n-gram) first, MTP fallback")
    ap.add_argument("--spec-k", type=int, default=3, help="k: drafts per round; the verify window is k+1 rows (measured Kyojin ceiling: 7)")
    ap.add_argument("--spec-mtp-k", type=int, default=2, help="MTP drafts per verify round when the n-gram lookup misses")
    ap.add_argument("--no-mtp", action="store_true", help="exact n-gram drafting only; never ask the head stage for MTP drafts")
    ap.add_argument("--spec-match", type=int, default=4,
                    help="n: minimum match length, strict. The last N tokens must already occur in the context (no 2-gram fallback)")
    ap.add_argument("--spec-log", default=None,
                    help="append one JSON object per request (stats + timestamp + id); rotates at 64 MiB, 2 backups")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    root = Path(args.kyojin_root).expanduser().resolve()
    sys.path.insert(0, str(root / "tools" / "glm"))
    sys.path.insert(0, str(root))
    from swp_exl3.config import load_pipeline, load_stage
    from swp_exl3.contract import contract_from_config
    from swp_exl3.coordinator import Pipeline, SpecConfig
    pc = load_pipeline(args.pipeline)
    # A local stage lives in this process: apply its env before importing the engine (or Kyojin's serve.py).
    for path in pc.stage_configs:
        if path:
            for k, v in load_stage(os.path.expanduser(path)).env.items():
                os.environ[k] = str(v)
    os.environ.setdefault("EXL3_PF_HC_FUSE", "0")
    os.environ.setdefault("EXL3_PF_GR_FUSE", "0")
    import serve as kserve                                   # Kyojin's HTTP layer
    from aiohttp import web
    from exllamav3 import Config, Tokenizer

    cfg = Config.from_directory(args.model_dir)
    tok = Tokenizer.from_config(cfg)
    pipe = Pipeline(pc.stages, secret=pc.secret(), wire_dtype=pc.wire_dtype, chunk=pc.chunk,
                    contract=contract_from_config(cfg), stage_configs=pc.stage_configs)
    pipe.connect()
    eos = set(getattr(cfg, "eos_token_id_list", None) or [tok.eos_token_id]) | {int(x) for x in args.stop_id}
    tpl = Path(args.chat_template).expanduser() if args.chat_template else Path(kserve.DEFAULT_TEMPLATE)
    if not tpl.exists():
        tpl = Path(args.model_dir) / "chat_template.jinja"
    spec = SpecConfig(k=max(0, args.spec_k), match=args.spec_match,
                      mtp_k=max(0, args.spec_mtp_k), use_mtp=not args.no_mtp) if args.spec else None
    engine = PipelineEngine(pipe, tok, args.ctx, eos, spec=spec, spec_log=args.spec_log,
                            log=lambda m: print(m, flush=True))
    app = kserve.create_app(engine, args.model_id, tpl.read_text(encoding="utf-8"))
    print(f"swp-kyojin-server: {len(pc.stages)} stages, ctx {args.ctx}, template {tpl}, http://{args.host}:{args.port}", flush=True)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
