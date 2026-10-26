"""Backend interface between the SWP stage server and an inference engine, plus a numpy FakeBackend used by tests.

A backend owns ONE loaded slice of layers and ONE active session (a single user, one sequence)."""
import struct
import numpy as np


def f32_to_bf16_bytes(a):
    """float32 array -> bf16 bytes (round to nearest even)."""
    u = np.ascontiguousarray(a, dtype="<f4").view("<u4")
    r = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + r) >> 16).astype("<u2").tobytes()


def bf16_bytes_to_f32(b):
    return (np.frombuffer(bytes(b), "<u2").astype("<u4") << 16).view("<f4")


class Unsupported(Exception):
    """Raised for operations the engine cannot do (e.g. partial rollback of recurrent state)."""
    code = "unsupported"


class Result:
    def __init__(self, kind, payload=b"", dtype=None, shape=None, timing=None, extra=None):
        self.kind, self.payload, self.dtype, self.shape = kind, payload, dtype, shape
        self.timing, self.extra = timing or {}, extra or {}


class Backend:
    contract: dict
    tail: tuple                    # boundary tail shape, e.g. (4, 4096)
    layers: tuple
    has_embed: bool
    has_head: bool
    recurrent: bool
    cache_tokens: int = 0
    supports_spec: bool = False
    supports_mtp: bool = False
    spec_k: int = 0                # max draft window this stage can roll back (0 = no speculation)

    WARMUP_ROWS = (1, 1, 1, 16, 64, 256, 512)         # 3 decode steps, then 4 prefill sizes

    def load(self) -> dict: ...                       # returns memory info
    def reset(self): ...                              # new session / clear (frees recurrent state)
    def cache_len(self) -> int: ...
    def forward(self, start_pos, token_ids, hidden, wire_dtype, want, return_rows, k, spec=None, draft=None) -> Result:
        """`spec`, when not None, is a verify forward:
        {"draft_ids": [...], "accept": "argmax"|"sample", "sampling": {...}, "next_drafts": bool}.
        `want == "spec"` asks the head for acceptance (kind="spec"); `want == "topk"` is a verify
        that returns logits for every row (used for the logit-equivalence check).
        `draft`, when not None, asks the head stage to attach next-round drafts (MTP) to a normal
        prefill/decode reply: {"next_drafts": bool, "k": int, "sampling": {...}}."""
    def truncate(self, pos): ...                      # may raise Unsupported
    def close(self): ...

    def warmup(self, rows=None, log=None) -> dict:
        """Run synthetic forwards so the first real prefill does not pay kernel/JIT warm-up, then reset.
        Rows are cumulative so every forward also warms a different cache offset; non-first stages get
        a zero boundary tensor (they have no embedding). Returns {"forwards", "ms", "per_forward_ms"}."""
        import time
        t0, start, done = time.perf_counter(), 0, []
        for n in (rows or self.WARMUP_ROWS):
            if self.cache_tokens and n > self.cache_tokens:
                continue
            if self.cache_tokens and start + n > self.cache_tokens:
                self.reset()
                start = 0
            hidden = None if self.has_embed else b"\0" * (n * int(np.prod(self.tail)) * 4)
            t = time.perf_counter()
            self.forward(start, [1] * n, hidden, "f32", "token" if self.has_head else "none", "last", None)
            start += n
            done.append(round((time.perf_counter() - t) * 1e3, 1))
        self.reset()
        info = {"forwards": len(done), "ms": round((time.perf_counter() - t0) * 1e3, 1), "per_forward_ms": done}
        if log:
            log(f"[warmup] {info['forwards']} synthetic forwards in {info['ms']:.0f} ms ({done})")
        return info


# ---------------------------------------------------------------------------------------------- numpy fake

def _emb(token, shape, seed=0):
    rng = np.random.default_rng((seed << 32) + int(token))
    return rng.uniform(-1, 1, size=shape).astype(np.float32)


def _sample_row(logits, temperature, top_p, top_k, rng):
    if temperature <= 0:
        return int(np.argmax(logits))
    z = logits.astype(np.float64) / temperature
    order = np.argsort(-z)
    if top_k:
        order = order[:top_k]
    z = z[order]
    p = np.exp(z - z.max())
    p /= p.sum()
    if top_p < 1.0:
        keep = np.cumsum(p) - p < top_p
        order, p = order[keep], p[keep] / p[keep].sum()
    return int(order[rng.choice(len(order), p=p)])


class FakeBackend(Backend):
    """Deterministic toy with the SAME boundary shape as GLM-5.3-Flash ([n, hc, hidden] fp32) and, optionally,
    recurrent state that cannot be rewound (like KDA layers), so rollback semantics are exercised realistically.

    Recurrent layers keep a per-position input history as well as the running sum, so a verify forward can be
    lazily rewound exactly like the engine's `GDNState.rewind`: only immediately after a verify forward, and
    only up to `max_history` rows (the toy analogue of the engine's staged-scan capacity)."""

    supports_spec = True

    def __init__(self, layers, n_layers=8, hc=4, hidden=8, vocab=64, has_embed=None, has_head=None, recurrent=True,
                 name="fake", delay_ms=0.0, fail_calls=(), spec_k=3, max_history=None, mtp_plan=None):
        self.layers = tuple(layers)
        self.n_layers, self.hc, self.hidden, self.vocab = n_layers, hc, hidden, vocab
        self.has_embed = layers[0] == 0 if has_embed is None else has_embed
        self.has_head = layers[1] == n_layers if has_head is None else has_head
        self.recurrent = recurrent
        self.spec_k, self.max_history = spec_k, spec_k if max_history is None else max_history
        self.mtp_plan = mtp_plan
        self.supports_mtp = mtp_plan is not None
        self.tail = (hc, hidden)
        self.contract = {"arch": "Fake", "n_layers": n_layers, "hidden_size": hidden, "vocab_size": vocab,
                         "hc_mult": hc, "boundary": {"tail": [hc, hidden], "canonical": "residual_pre_layer"}}
        self.name, self.delay_ms, self.fail_calls, self.calls = name, delay_ms, set(fail_calls), 0
        self.W = _emb(0, (hidden, vocab), seed=99)
        self.reset()

    def load(self):
        return {"fake": True}

    def reset(self):
        n = self.layers[1] - self.layers[0]
        self.pos = 0
        self.state = [np.zeros(self.tail, np.float32) for _ in range(n)]   # running sum (recurrent) per layer
        self.hist = [[] for _ in range(n)]                                 # per-position layer inputs
        self.last_was_verify = False
        self.last_history = 0

    def cache_len(self):
        return self.pos

    def truncate(self, pos):
        if pos >= self.pos:
            return
        if pos == 0:
            return self.reset()
        n = self.pos - pos
        if self.recurrent:
            if not self.last_was_verify or n > self.last_history or n > self.max_history:
                raise Unsupported(f"recurrent state can only be rewound right after a verify forward "
                                  f"(asked {n}; history {self.last_history}, capacity {self.max_history})")
        for li, h in enumerate(self.hist):
            del h[pos:]
            if self.recurrent:
                self.state[li] = np.sum(h, axis=0).astype(np.float32) if h else np.zeros(self.tail, np.float32)
        self.pos = pos
        self.last_was_verify = False
        self.last_history = 0

    def _layer(self, li, row):
        g = self.layers[0] + li
        self.hist[li].append(row.copy())
        if self.recurrent:
            self.state[li] = self.state[li] + row
            m = self.state[li] / np.float32(len(self.hist[li]))
        else:
            acc = np.zeros(self.tail, np.float32)
            for h in self.hist[li]:
                acc = acc + h
            m = acc / np.float32(len(self.hist[li]))
        return row + np.float32(0.05 + 0.01 * g) * (m / (np.float32(1) + np.abs(m)))

    def forward(self, start_pos, token_ids, hidden, wire_dtype, want, return_rows, k, spec=None, draft=None):
        import time
        self.calls += 1
        if self.calls in self.fail_calls:
            raise RuntimeError("injected engine failure")
        if self.delay_ms:
            time.sleep(self.delay_ms / 1e3)
        verify = spec is not None
        n = len(token_ids)
        if hidden is None:
            assert self.has_embed, "stage has no embedding"
            x = np.stack([np.broadcast_to(_emb(t, (self.hidden,)) + _emb(1_000_000 + start_pos + i, (self.hidden,)) * np.float32(.5),
                                          self.tail).copy() for i, t in enumerate(token_ids)])
        else:
            x = (bf16_bytes_to_f32(hidden) if wire_dtype == "bf16" else np.frombuffer(bytes(hidden), np.float32)).reshape((n,) + self.tail).copy()
        for li in range(self.layers[1] - self.layers[0]):
            for i in range(n):
                x[i] = self._layer(li, x[i])
        self.pos = start_pos + n
        self.last_was_verify = verify
        self.last_history = (n - 1) if verify else 0
        if not self.has_head:
            body = f32_to_bf16_bytes(x) if wire_dtype == "bf16" else x.astype(np.float32).tobytes()
            return Result("hidden", body, wire_dtype, [n] + list(self.tail))
        rows = list(range(n)) if (return_rows == "all" or want == "spec") else [n - 1]
        if want == "none":
            return Result("none")
        logits = x[rows].mean(axis=1) @ self.W                                  # [r, vocab]
        if want == "spec":
            drafts = [int(d) for d in (spec or {}).get("draft_ids", [])]
            if len(drafts) != n - 1:
                raise ValueError(f"verify forward has {n} rows for {len(drafts)} drafts")
            if (spec or {}).get("accept", "argmax") == "sample":
                s = (spec or {}).get("sampling") or {}
                rng = np.random.default_rng([int(s.get("seed") or 0), int(s.get("round") or 0)])
                pred = [_sample_row(logits[i], float(s.get("temperature", 1.0)), float(s.get("top_p", 1.0)),
                                    int(s.get("top_k") or 0), rng) for i in range(n)]
            else:
                pred = [int(v) for v in np.argmax(logits, -1)]
            m = 0
            while m < len(drafts) and pred[m] == drafts[m]:
                m += 1
            tokens = drafts[:m] + [pred[m]]
            nxt = []
            if self.supports_mtp and draft and draft.get("next_drafts"):
                nxt = [int(x) for x in (self.mtp_plan(start_pos + m + 2, int(draft.get("k") or 0)) or [])]
            extra = {"accepted": m, "tokens": tokens, "drafts": nxt, "drafts_basis": int(tokens[-1])}
            return Result("spec", np.asarray(tokens, "<i4").tobytes(), "i32", [len(tokens)], extra=extra)
        if want == "token":
            ids = np.argmax(logits, -1).astype("<i4")
            extra = {}
            if self.supports_mtp and draft and draft.get("next_drafts"):
                basis = int(ids[-1])
                extra = {"drafts": [int(x) for x in (self.mtp_plan(start_pos + n + 1, int(draft.get("k") or 0)) or [])],
                         "drafts_basis": basis}
            return Result("tokens", ids.tobytes(), "i32", [len(rows)], extra=extra)
        kk = k or 8
        idx = np.argsort(-logits, axis=-1, kind="stable")[:, :kk].astype("<i4")
        val = np.take_along_axis(logits, idx, -1).astype("<f4")
        return Result("topk", idx.tobytes() + val.tobytes(), "i32+f32", [len(rows), kk])

    def close(self):
        pass
