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
                 node="coordinator", log=print):
        self.stages = [s if isinstance(s, StageSpec) else StageSpec(*s) for s in stages]
        self.secret, self.wire, self.chunk, self.contract = secret, wire_dtype, chunk, contract
        self.session, self.node, self.log = session, node, log
        self.peers, self.history, self.step, self.tail = [], [], 0, None
        self.trace = Trace([tuple(s.layers) for s in self.stages])

    # ------------------------------------------------------------------ connection
    def connect(self):
        self.close()
        n = len(self.stages)
        for i, s in enumerate(self.stages):
            p = swp.Peer(s.addr, node=self.node, secret=self.secret)
            ready, _ = p.rpc(swp.STAGE_CONFIGURE, {
                "session": self.session, "layers": list(s.layers), "has_embed": i == 0, "has_head": i == n - 1,
                "wire_dtype": self.wire, **({"contract": self.contract} if self.contract else {})}, want=swp.STAGE_READY)
            self.tail = ready.get("tail", self.tail)
            self.peers.append(p)
        self.history = []

    def close(self):
        for p in self.peers:
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
    def _relay(self, start, toks, want, return_rows, k, phase, rewrite=False):
        self.step += 1
        n, hidden = len(self.peers), b""
        res = None
        for i, p in enumerate(self.peers):
            last = i == n - 1
            h = {"session": self.session, "step": self.step, "mode": "prefill" if len(toks) > 1 else "decode",
                 "start_pos": start, "n_rows": len(toks), "token_ids": toks, "input": "tokens" if i == 0 else "hidden",
                 "dtype": self.wire, "want": want if last else "hidden", "return_rows": return_rows, "k": k, "rewrite": rewrite}
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

    @staticmethod
    def _parse_head(h, payload):
        kind = h["kind"]
        if kind == "none":
            return ("none", None)
        if kind == "tokens":
            return ("tokens", np.frombuffer(payload, "<i4").copy())
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
        self._feed(hist, "none", "last", None, "replay")

    def _feed(self, toks, want, return_rows, k, phase):
        toks, out = list(toks), None
        for ci in range(0, len(toks), self.chunk):
            c = toks[ci:ci + self.chunk]
            last = ci + self.chunk >= len(toks)
            out = self._relay(len(self.history), c, want if last else "none", return_rows, k, phase)
            self.history += c
        return out

    def feed(self, toks, want="none", return_rows="last", k=None, phase="prefill"):
        """Append tokens at the current position (chunked). Only the final chunk asks the head for `want`."""
        start_len = len(self.history)
        toks = list(toks)

        def run():
            done = len(self.history) - start_len
            return self._feed(toks[done:], want, return_rows, k, phase)
        return self._guarded(run)

    # ------------------------------------------------------------------ cache control
    def reset(self):
        for p in self.peers:
            p.rpc(swp.CACHE_OP, {"session": self.session, "op": "clear"}, want=swp.CACHE_OK)
        self.history = []

    def rollback(self, pos):
        """Keep positions < pos. Stages that cannot rewind (recurrent state) answer `unsupported`; then we clear and replay."""
        if pos >= len(self.history):
            return "noop"
        if pos <= 0:
            self.reset()
            return "cleared"
        try:
            for p in self.peers:
                p.rpc(swp.CACHE_OP, {"session": self.session, "op": "truncate", "pos": pos}, want=swp.CACHE_OK)
            del self.history[pos:]
            return "truncated"
        except swp.SwpError as e:
            if e.code != "unsupported":
                raise
        keep = self.history[:pos]
        self.reset()
        self._guarded(lambda: self._feed(keep[len(self.history):], "none", "last", None, "replay"))
        return "replayed"

    # ------------------------------------------------------------------ generation
    def logits_topk(self, prompt_ids, k=64):
        """Reset, run the prompt, return head top-k (ids, logits) of the last row. Used for cut-invariance checks."""
        self.reset()
        out = self.feed(prompt_ids, want="topk", k=k)
        return out[1][0][0], out[1][1][0]

    def generate(self, prompt_ids, max_new=256, temperature=0.0, top_p=1.0, top_k=64, stop_ids=(), seed=None, on_token=None):
        rng = np.random.default_rng(seed)
        prompt = list(prompt_ids)
        keep = common_prefix(self.history, prompt)
        if keep == len(prompt):
            keep -= 1                       # logits for the last prompt token are needed
        self.rollback(keep)
        want = "token" if temperature <= 0 else "topk"
        self.trace.hops.clear()
        t0 = time.perf_counter()
        out = self.feed(prompt[keep:], want=want, k=top_k, phase="prefill")
        pre_ms = (time.perf_counter() - t0) * 1e3
        pick = (lambda o: int(o[1][0])) if want == "token" else \
            (lambda o: sample_topk(o[1][0][0], o[1][1][0], temperature, top_p, top_k, rng))
        tok, gen, t1 = pick(out), [], time.perf_counter()
        for i in range(max_new):
            gen.append(tok)
            if on_token:
                on_token(tok)
            if tok in stop_ids or i == max_new - 1:
                break
            tok = pick(self.feed([tok], want=want, k=top_k, phase="decode"))
        dec_ms = (time.perf_counter() - t1) * 1e3
        return gen, {"prefill_tokens": len(prompt) - keep, "prefill_ms": pre_ms,
                     "decode_tokens": max(len(gen) - 1, 0), "decode_ms": dec_ms, "reused_prefix": keep}

    def report(self, stats, out=print):
        return self.trace.report(stats["prefill_tokens"], stats["prefill_ms"], max(stats["decode_tokens"], 1), stats["decode_ms"], out)
