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
    def __init__(self, kind, payload=b"", dtype=None, shape=None, timing=None):
        self.kind, self.payload, self.dtype, self.shape, self.timing = kind, payload, dtype, shape, timing or {}


class Backend:
    contract: dict
    tail: tuple                    # boundary tail shape, e.g. (4, 4096)
    layers: tuple
    has_embed: bool
    has_head: bool
    recurrent: bool
    cache_tokens: int = 0

    WARMUP_ROWS = (1, 1, 1, 16, 64, 256, 512)         # 3 decode steps, then 4 prefill sizes

    def load(self) -> dict: ...                       # returns memory info
    def reset(self): ...                              # new session / clear (frees recurrent state)
    def cache_len(self) -> int: ...
    def forward(self, start_pos, token_ids, hidden, wire_dtype, want, return_rows, k) -> Result: ...
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


class FakeBackend(Backend):
    """Deterministic toy with the SAME boundary shape as GLM-5.3-Flash ([n, hc, hidden] fp32) and, optionally,
    recurrent state that cannot be rewound (like KDA layers), so rollback semantics are exercised realistically."""

    def __init__(self, layers, n_layers=8, hc=4, hidden=8, vocab=64, has_embed=None, has_head=None, recurrent=True, name="fake", delay_ms=0.0, fail_calls=()):
        self.layers = tuple(layers)
        self.n_layers, self.hc, self.hidden, self.vocab = n_layers, hc, hidden, vocab
        self.has_embed = layers[0] == 0 if has_embed is None else has_embed
        self.has_head = layers[1] == n_layers if has_head is None else has_head
        self.recurrent = recurrent
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
        self.hist = [[] for _ in range(n)]                                 # per-position inputs (non-recurrent)

    def cache_len(self):
        return self.pos

    def truncate(self, pos):
        if pos >= self.pos:
            return
        if pos == 0:
            return self.reset()
        if self.recurrent:
            raise Unsupported("recurrent state cannot be rewound; clear and replay")
        for h in self.hist:
            del h[pos:]
        self.pos = pos

    def _layer(self, li, row):
        g = self.layers[0] + li
        if self.recurrent:
            self.state[li] = self.state[li] + row
            m = self.state[li] / np.float32(self.pos_row + 1)
        else:
            self.hist[li].append(row.copy())
            acc = np.zeros(self.tail, np.float32)
            for h in self.hist[li]:
                acc = acc + h
            m = acc / np.float32(len(self.hist[li]))
        return row + np.float32(0.05 + 0.01 * g) * (m / (np.float32(1) + np.abs(m)))

    def forward(self, start_pos, token_ids, hidden, wire_dtype, want, return_rows, k):
        import time
        self.calls += 1
        if self.calls in self.fail_calls:
            raise RuntimeError("injected engine failure")
        if self.delay_ms:
            time.sleep(self.delay_ms / 1e3)
        n = len(token_ids)
        if hidden is None:
            assert self.has_embed, "stage has no embedding"
            x = np.stack([np.broadcast_to(_emb(t, (self.hidden,)) + _emb(1_000_000 + start_pos + i, (self.hidden,)) * np.float32(.5),
                                          self.tail).copy() for i, t in enumerate(token_ids)])
        else:
            x = (bf16_bytes_to_f32(hidden) if wire_dtype == "bf16" else np.frombuffer(bytes(hidden), np.float32)).reshape((n,) + self.tail).copy()
        for li in range(self.layers[1] - self.layers[0]):
            for i in range(n):
                self.pos_row = start_pos + i
                x[i] = self._layer(li, x[i])
        self.pos = start_pos + n
        if not self.has_head:
            body = f32_to_bf16_bytes(x) if wire_dtype == "bf16" else x.astype(np.float32).tobytes()
            return Result("hidden", body, wire_dtype, [n] + list(self.tail))
        rows = list(range(n)) if return_rows == "all" else [n - 1]
        if want == "none":
            return Result("none")
        logits = x[rows].mean(axis=1) @ self.W                                  # [r, vocab]
        if want == "token":
            return Result("tokens", np.argmax(logits, -1).astype("<i4").tobytes(), "i32", [len(rows)])
        kk = k or 8
        idx = np.argsort(-logits, axis=-1, kind="stable")[:, :kk].astype("<i4")
        val = np.take_along_axis(logits, idx, -1).astype("<f4")
        return Result("topk", idx.tobytes() + val.tobytes(), "i32+f32", [len(rows), kk])

    def close(self):
        pass
