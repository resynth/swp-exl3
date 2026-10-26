"""Pipeline coordinator (star topology): owns token history, positions, sampling; relays boundary tensors.
Stages are disposable caches: on a dropped connection or a stage reset the coordinator replays the history."""
import time
from dataclasses import dataclass
import numpy as np
from . import swp
from .bench import Trace


@dataclass
class StageSpec:
    addr: str
    layers: tuple


def ngram_drafts(tokens, k, n=2):
    """Prompt-lookup drafting: find the most recent earlier occurrence of the last `n` tokens and propose the
    k tokens that followed it. `tokens` is the committed history plus the last accepted token (not yet fed)."""
    if k <= 0 or n <= 0 or len(tokens) < n + 1:
        return []
    key = tuple(tokens[-n:])
    for i in range(len(tokens) - n - 1, -1, -1):
        if tuple(tokens[i:i + n]) == key:
            return [int(x) for x in tokens[i + n:i + n + k]]
    return []


@dataclass
class SpecConfig:
    """Drafting policy for `Pipeline.generate(spec=...)`. `drafter(context, k)` overrides the n-gram lookup."""
    k: int = 3
    match: int = 2
    accept: str = "argmax"                 # "argmax" (greedy) or "sample" (T>0: sample target, accept on equality)
    drafter: object = None
    seed: int = None

    def propose(self, context, k):
        if k <= 0:
            return []
        if self.drafter is not None:
            return [int(x) for x in (self.drafter(list(context), k) or [])][:k]
        for n in ((self.match, 2) if self.match != 2 else (2,)):
            d = ngram_drafts(context, k, n)
            if d:
                return d
        return []


def compare_topk(ids_a, vals_a, ids_b, vals_b):
    """Top-1 equality and KL over the shared top-k (renormalised). Same definition as `swp-exl3 compare`."""
    ia, va, ib, vb = (np.asarray(ids_a), np.asarray(vals_a, dtype=np.float64),
                      np.asarray(ids_b), np.asarray(vals_b, dtype=np.float64))
    common = [i for i in ia if i in set(ib.tolist())]
    da, db = dict(zip(ia.tolist(), va.tolist())), dict(zip(ib.tolist(), vb.tolist()))
    if not common:
        return {"top1_equal": bool(ia[0] == ib[0]), "shared": 0, "topk": len(ia), "max_abs": float("inf"), "kl": float("inf")}
    diffs = [abs(da[i] - db[i]) for i in common]
    pa = np.exp(np.array([da[i] for i in common]) - max(da[i] for i in common)); pa /= pa.sum()
    pb = np.exp(np.array([db[i] for i in common]) - max(db[i] for i in common)); pb /= pb.sum()
    kl = float((pa * np.log(pa / pb)).sum())
    return {"top1_equal": bool(ia[0] == ib[0]), "shared": len(common), "topk": len(ia),
            "max_abs": float(max(diffs)), "kl": kl}


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def sample_topk(ids, vals, temperature=0.0, top_p=1.0, top_k=0, rng=None):
    """ids/vals sorted by logit, descending (as returned by the head stage)."""
    if temperature <= 0:
        return int(ids[0])
    if top_k:
        ids, vals = ids[:top_k], vals[:top_k]
    z = vals.astype(np.float64) / temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    if top_p < 1.0:
        keep = np.cumsum(p) - p < top_p
        ids, p = ids[keep], p[keep] / p[keep].sum()
    return int((rng or np.random.default_rng()).choice(ids, p=p))


class Pipeline:
    def __init__(self, stages, secret=None, wire_dtype="f32", chunk=512, contract=None, session="s1",
                 node="coordinator", log=print, stage_configs=None):
        self.stages = [s if isinstance(s, StageSpec) else StageSpec(*s) for s in stages]
        self.secret, self.wire, self.chunk, self.contract = secret, wire_dtype, chunk, contract
        self.session, self.node, self.log = session, node, log
        self.peers, self.history, self.step, self.tail = [], [], 0, None
        self.replayed_tokens = 0        # tokens recomputed by clear+replay (prefix reuse lost)
        self.spec_k = 0                  # largest draft window every stage advertised (0 = no speculation)
        self.head_mtp = False            # head stage can propose next-round drafts after a verify/decode
        self.last_extra = {}             # head reply fields that are not the payload (drafts, basis, ...)
        self.stage_configs = (list(stage_configs or []) + [None] * len(self.stages))[:len(self.stages)]
        self._local = {}                 # stage index -> LocalPeer, kept across reconnects (no reload)
        self.trace = Trace([tuple(s.layers) for s in self.stages])

    # ------------------------------------------------------------------ connection
    def _make_local(self, i, path):
        import os
        from .config import load_stage
        from .exl3_backend import Exl3Backend
        from .server import StageServer, LocalPeer
        c = load_stage(os.path.expanduser(path))
        self.log(f"[local] stage {i}: loading layers {c.layers} from {c.model_dir}")
        be = Exl3Backend(c, log=self.log)
        be.load()
        if c.warmup:
            be.warmup(log=self.log)
        return LocalPeer(StageServer(be, c.node, "local", secret=None, wire_dtypes=c.wire_dtypes,
                                     max_chunk=c.max_chunk, log=self.log))

    def connect(self):
        self.close()
        n = len(self.stages)
        spec_ks, head_ready = [], {}
        for i, s in enumerate(self.stages):
            if self.stage_configs[i]:
                p = self._local.get(i)
                if p is None:
                    p = self._make_local(i, self.stage_configs[i])
                    self._local[i] = p
            else:
                p = swp.Peer(s.addr, node=self.node, secret=self.secret)
            ready, _ = p.rpc(swp.STAGE_CONFIGURE, {
                "session": self.session, "layers": list(s.layers), "has_embed": i == 0, "has_head": i == n - 1,
                "wire_dtype": self.wire, **({"contract": self.contract} if self.contract else {})}, want=swp.STAGE_READY)
            self.tail = ready.get("tail", self.tail)
            spec_ks.append(int(ready.get("spec_k", 0)))
            if i == n - 1:
                head_ready = ready
            self.peers.append(p)
        self.spec_k = min(spec_ks) if spec_ks else 0
        self.head_mtp = bool(head_ready.get("mtp")) and self.spec_k > 0
        self.history = []

    def close(self):
        from .server import LocalPeer
        for p in self.peers:
            if not isinstance(p, LocalPeer):
                p.close()
        self.peers = []

    def rtt_ms(self, n=50):
        out = []
        for p in self.peers:
            ts = []
            for _ in range(n):
                t = time.perf_counter()
                p.ping()
                ts.append((time.perf_counter() - t) * 1e3)
            out.append(float(np.median(ts)))
        return out

    # ------------------------------------------------------------------ forward / feed
    def _relay(self, start, toks, want, return_rows, k, phase, rewrite=False, extra=None):
        self.step += 1
        n, hidden = len(self.peers), b""
        res = None
        for i, p in enumerate(self.peers):
            last = i == n - 1
            h = {"session": self.session, "step": self.step, "mode": "prefill" if len(toks) > 1 else "decode",
                 "start_pos": start, "n_rows": len(toks), "token_ids": toks, "input": "tokens" if i == 0 else "hidden",
                 "dtype": self.wire, "want": want if last else "hidden", "return_rows": return_rows, "k": k, "rewrite": rewrite}
            if extra:
                h.update(extra)          # mode=verify / draft_ids / accept / sampling / next_drafts
            if i > 0:
                h["shape"] = [len(toks)] + list(self.tail)
            t = time.perf_counter()
            oh, payload = p.rpc(swp.FORWARD, h, hidden, want=swp.FORWARD_OUT)
            self.trace.add(phase, i, (time.perf_counter() - t) * 1e3, float(oh.get("stage_ms", 0.0)))
            if last:
                res = self._parse_head(oh, payload)
            else:
                hidden = payload
        return res

    def _parse_head(self, h, payload):
        kind = h["kind"]
        self.last_extra = {"drafts": [int(x) for x in (h.get("drafts") or [])],
                           "drafts_basis": h.get("drafts_basis"),
                           "timing": h.get("timing") or {}}
        if kind == "none":
            return ("none", None)
        if kind == "tokens":
            return ("tokens", np.frombuffer(payload, "<i4").copy())
        if kind == "spec":
            tokens = np.frombuffer(payload, "<i4").copy().tolist()
            return ("spec", {"accepted": int(h.get("accepted", max(len(tokens) - 1, 0))),
                             "tokens": [int(x) for x in tokens], "drafts": list(self.last_extra["drafts"]),
                             "drafts_basis": h.get("drafts_basis"), "timing": self.last_extra["timing"]})
        if kind == "topk":
            r, k = h["shape"]
            a = np.frombuffer(payload, "<i4", r * k).reshape(r, k).copy()
            v = np.frombuffer(payload, "<f4", r * k, offset=r * k * 4).reshape(r, k).copy()
            return ("topk", (a, v))
        raise swp.SwpError("bad_reply", f"unexpected head output kind {kind}")

    def _guarded(self, fn):
        try:
            return fn()
        except (ConnectionError, OSError, swp.SwpError) as e:
            if isinstance(e, swp.SwpError) and e.code not in ("no_session", "engine_error"):
                raise
            self.log(f"[pipeline] {type(e).__name__}: {e}; reconnecting and replaying {len(self.history)} tokens")
            self._recover()
            return fn()

    def _recover(self):
        hist = self.history[:]
        self.connect()
        self.replayed_tokens += len(hist)
        self._feed(hist, "none", "last", None, "replay")

    def _relay_recover(self, start, toks, want, return_rows, k, phase, rewrite=False, extra=None):
        """Forward, falling back to clear+replay if a stage cannot honour the lazy rewind."""
        try:
            return self._relay(start, toks, want, return_rows, k, phase, rewrite=rewrite, extra=extra)
        except swp.SwpError as e:
            if e.code != "unsupported":
                raise
            self.log(f"[pipeline] lazy rewind unsupported ({e.message}); clearing and replaying {len(self.history)} tokens")
            self._recover()
            return self._relay(len(self.history), toks, want, return_rows, k, phase, rewrite=False, extra=extra)

    def _feed(self, toks, want, return_rows, k, phase, rewrite=False):
        toks, out = list(toks), None
        for ci in range(0, len(toks), self.chunk):
            c = toks[ci:ci + self.chunk]
            last = ci + self.chunk >= len(toks)
            out = self._relay(len(self.history), c, want if last else "none", return_rows, k, phase, rewrite=rewrite)
            self.history += c
        return out

    def feed(self, toks, want="none", return_rows="last", k=None, phase="prefill", rewrite=False):
        """Append tokens at the current position (chunked). Only the final chunk asks the head for `want`."""
        start_len = len(self.history)
        toks = list(toks)

        def run():
            done = len(self.history) - start_len
            return self._feed(toks[done:], want, return_rows, k, phase, rewrite=rewrite)
        return self._guarded(run)

    # ------------------------------------------------------------------ cache control
    def reset(self):
        for p in self.peers:
            p.rpc(swp.CACHE_OP, {"session": self.session, "op": "clear"}, want=swp.CACHE_OK)
        self.history = []

    def _replay_to(self, pos):
        keep = self.history[:pos]
        self.connect()                      # fresh connections also reset every stage (unknown state after a drop)
        self.replayed_tokens += len(keep)
        self._guarded(lambda: self._feed(keep[len(self.history):], "none", "last", None, "replay"))

    def _truncate_all(self, pos):
        for p in self.peers:
            p.rpc(swp.CACHE_OP, {"session": self.session, "op": "truncate", "pos": pos}, want=swp.CACHE_OK)

    def rollback(self, pos):
        """Keep positions < pos. Stages that cannot rewind (recurrent state) answer `unsupported`; then we clear and replay."""
        if pos > len(self.history):
            return "noop"
        if pos == len(self.history):
            # A speculative round leaves the stage caches k-m rows past the committed history. Drop those rows;
            # with no overshoot this is a no-op on every stage.
            try:
                self._truncate_all(pos)
            except (ConnectionError, OSError):
                return "noop"           # the next forward reconnects and replays this same history
            except swp.SwpError as e:
                if e.code != "unsupported":
                    raise
                self._replay_to(pos)
                return "replayed"
            return "noop"
        if pos <= 0:
            self.reset()
            return "cleared"
        try:
            self._truncate_all(pos)
            del self.history[pos:]
            return "truncated"
        except (ConnectionError, OSError):
            self._replay_to(pos)
            return "replayed"
        except swp.SwpError as e:
            if e.code != "unsupported":
                raise
        self._replay_to(pos)
        return "replayed"

    # ------------------------------------------------------------------ generation
    def logits_topk(self, prompt_ids, k=64):
        """Reset, run the prompt, return head top-k (ids, logits) of the last row. Used for cut-invariance checks."""
        self.reset()
        out = self.feed(prompt_ids, want="topk", k=k)
        return out[1][0][0], out[1][1][0]

    def _spec_loop(self, tok, want, top_k, max_new, temperature, top_p, stop_ids, spec, rng, on_token, pick):
        """Speculative decode: propose k drafts, verify them in one k+1-row forward, emit the accepted prefix.
        Drafts come from the head stage's MTP drafter when it supplies them, otherwise from the n-gram lookup."""
        k = max(0, min(int(spec.k), int(self.spec_k)))
        gen, pending = [], []
        st = {"spec_rounds": 0, "spec_drafted": 0, "spec_accepted": 0, "spec_mtp_drafts": 0,
              "spec_verify_ms": 0.0, "spec_catchup_ms": 0.0, "spec_roll_ms": 0.0,
              "spec_depth": [0] * max(k, 1)}
        sampler = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "seed": int(spec.seed or 0)}
        while len(gen) < max_new:
            gen.append(tok)
            if on_token:
                on_token(tok)
            if tok in stop_ids or len(gen) >= max_new:
                break
            # Prefer a known continuation (prompt lookup) over the drafter: it is free and, on
            # repetitive output, exact. MTP is the fallback for novel text.
            lookup = spec.propose(self.history + [tok], k)
            if lookup:
                drafts, pending = lookup[:k], []
            else:
                drafts = pending[:k]
                pending = []
            want_mtp = bool(self.head_mtp and want == "token" and not lookup)
            if not drafts:
                extra = None
                if want_mtp:
                    extra = {"next_drafts": True, "draft_k": k,
                             "sampling": dict(sampler, round=st["spec_rounds"] + 1)}
                out = self._relay_recover(len(self.history), [tok], want, "last", top_k, "decode",
                                          rewrite=True, extra=extra)
                self.history.append(tok)
                tok = pick(out)
                nxt = self.last_extra.get("drafts") or []
                if nxt and self.last_extra.get("drafts_basis") == tok:
                    pending = nxt[:k]
                    st["spec_mtp_drafts"] += len(pending)
                continue
            extra = {"mode": "verify", "draft_ids": drafts,
                     "accept": "argmax" if want == "token" else "sample",
                     "sampling": dict(sampler, round=st["spec_rounds"] + 1),
                     "next_drafts": want_mtp}
            t_round = time.perf_counter()
            kind, data = self._relay_recover(len(self.history), [tok] + drafts, "spec", "all", top_k, "decode",
                                             rewrite=True, extra=extra)
            if kind != "spec":
                raise swp.SwpError("bad_reply", f"verify forward answered kind={kind}")
            st["spec_verify_ms"] += (time.perf_counter() - t_round) * 1e3
            m, tokens = int(data["accepted"]), data["tokens"]
            if len(tokens) != m + 1:
                raise swp.SwpError("bad_reply", f"spec reply has {len(tokens)} tokens for accepted={m}")
            tm = data.get("timing") or {}
            st["spec_catchup_ms"] += float(tm.get("catchup_ms", 0.0))
            st["spec_roll_ms"] += float(tm.get("mtp_roll_ms", 0.0))
            st["spec_rounds"] += 1
            st["spec_drafted"] += len(drafts)
            st["spec_accepted"] += m
            for j in range(len(drafts)):
                st["spec_depth"][j] += 1 if j < m else 0
            self.history += [tok] + drafts[:m]          # committed; rejected rows are rewound lazily next round
            for t in tokens[:-1]:
                if len(gen) >= max_new:
                    break
                gen.append(int(t))
                if on_token:
                    on_token(int(t))
                if t in stop_ids:
                    return gen, st
            tok = int(tokens[-1])                       # bonus: emitted at the top of the next round
            nxt = data.get("drafts") or []
            if nxt and data.get("drafts_basis") == tok:
                pending = nxt[:k]
                st["spec_mtp_drafts"] += len(pending)
            if len(gen) >= max_new:
                break
        return gen, st

    def generate(self, prompt_ids, max_new=256, temperature=0.0, top_p=1.0, top_k=64, stop_ids=(), seed=None,
                 on_token=None, spec=None):
        rng = np.random.default_rng(seed)
        prompt = list(prompt_ids)
        keep = common_prefix(self.history, prompt)
        if keep == len(prompt):
            keep -= 1                       # logits for the last prompt token are needed
        replayed0 = self.replayed_tokens
        rb = self.rollback(keep)
        want = "token" if temperature <= 0 else "topk"
        self.trace.hops.clear()
        t0 = time.perf_counter()
        out = self.feed(prompt[keep:], want=want, k=top_k, phase="prefill")
        pre_ms = (time.perf_counter() - t0) * 1e3
        pick = (lambda o: int(o[1][0])) if want == "token" else \
            (lambda o: sample_topk(o[1][0][0], o[1][1][0], temperature, top_p, top_k, rng))
        tok, t1 = pick(out), time.perf_counter()
        if spec is not None and self.spec_k > 0:
            gen, sst = self._spec_loop(tok, want, top_k, max_new, temperature, top_p, stop_ids, spec, rng, on_token, pick)
            if sst.get("spec_rounds"):
                sst["spec_accepted_per_round"] = sst["spec_accepted"] / sst["spec_rounds"]
                sst["spec_tokens_per_round"] = 1.0 + sst["spec_accepted_per_round"]
        else:
            if spec is not None:
                self.log("[pipeline] speculation requested but a stage advertises spec_k=0; plain decode")
            gen, sst = [], {}
            for i in range(max_new):
                gen.append(tok)
                if on_token:
                    on_token(tok)
                if tok in stop_ids or i == max_new - 1:
                    break
                tok = pick(self.feed([tok], want=want, k=top_k, phase="decode"))
        dec_ms = (time.perf_counter() - t1) * 1e3
        stats = {"prefill_tokens": len(prompt) - keep, "prefill_ms": pre_ms,
                 "decode_tokens": max(len(gen) - 1, 0), "decode_ms": dec_ms, "reused_prefix": keep,
                 "rollback": rb, "replayed_tokens": self.replayed_tokens - replayed0}
        stats.update(sst)
        return gen, stats

    def spec_check(self, prompt_ids, rows=20, k=3, klog=8):
        """Compare batched verify logits against sequential decode logits from the same state.
        `rows` verify rows (whole rounds of k+1), all drafts accepted."""
        k = max(1, min(int(k), int(self.spec_k)))
        if self.spec_k <= 0:
            return {"rows": 0, "top1": True, "max_kl": 0.0, "max_abs": 0.0, "k": 0, "skipped": True}
        self.reset()
        out = self.feed(prompt_ids, want="topk", k=klog)
        seq = [(out[1][0][0].copy(), out[1][1][0].copy())]          # seq[i] predicts token i
        seq_tokens = [int(out[1][0][0][0])]
        need = rows + k + 2
        while len(seq) < need:
            o = self.feed([seq_tokens[-1]], want="topk", k=klog)
            seq.append((o[1][0][0].copy(), o[1][1][0].copy()))
            seq_tokens.append(int(o[1][0][0][0]))
        self.reset()
        self.feed(prompt_ids, want="none")
        t, r, checked, bad_top1, max_kl, max_abs = seq_tokens[0], 0, 0, 0, 0.0, 0.0
        while checked < rows:
            kk = min(k, rows - checked - 1) if rows - checked > 1 else 0
            drafts = seq_tokens[r + 1:r + 1 + kk]
            if len(drafts) != kk:
                break
            kind, (ids, vals) = self._relay(len(self.history), [t] + drafts, "topk", "all", klog, "verify",
                                            rewrite=True, extra={"mode": "verify", "draft_ids": drafts})
            for j in range(ids.shape[0]):
                ref_i, ref_v = seq[r + 1 + j]
                c = compare_topk(ids[j], vals[j], ref_i, ref_v)
                checked += 1
                bad_top1 += 0 if c["top1_equal"] else 1
                max_kl, max_abs = max(max_kl, c["kl"]), max(max_abs, c["max_abs"])
            self.history += [t] + drafts
            r += kk + 1
            t = seq_tokens[r]
        return {"rows": checked, "top1": bad_top1 == 0, "bad_top1": bad_top1,
                "max_kl": max_kl, "max_abs": max_abs, "k": k}

    def report(self, stats, out=print):
        return self.trace.report(stats["prefill_tokens"], stats["prefill_ms"], max(stats["decode_tokens"], 1), stats["decode_ms"], out)
