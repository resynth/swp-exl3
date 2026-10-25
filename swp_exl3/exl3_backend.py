"""exllamav3 / Kyojin backend for the SWP stage server.

Every engine-facing assumption lives in this file. NOT exercised in the authoring environment (no GPU, no torch):
verified by reading exllamav3 1.6.4 (upstream) and Kyojin 1.5.10 sources, and by `swp-exl3 doctor` on your machines.

Call sequence mirrors exllamav3's own Model.forward():
    x = model.prepare_inputs(ids, params)      # attention + recurrent-state setup (GLM-5.3: prepare_for_attn + recurrence)
    prefetch layers (token-id keyed tables)
    y = model.forward_ls(x, params)            # runs this stage's modules; x is token ids (first stage) or the boundary tensor
    advance_recurrent_states(ids, params, model)
with the stage-specific parts: hidden-state input, hyper-connection flush, boundary/logit output.
"""
import importlib, inspect, os, time
from .backend import Backend, Result, Unsupported, bf16_bytes_to_f32, f32_to_bf16_bytes  # noqa: F401
from .contract import contract_from_config
from .slicing import slice_model


def _filter_kwargs(fn, kwargs):
    sig = inspect.signature(fn)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def flush_hyperconnections(params, y):
    """Apply a deferred hyper-connection residual update (params['hc_pending']) so the tensor leaving a stage is
    complete. Upstream 1.6.x: hyperconnections.hc_flush(params). Kyojin: HyperConnection.flush_pending(x, pending)."""
    pend = params.get("hc_pending")
    if pend is None:
        return y
    if pend[0].data_ptr() != y.data_ptr():
        raise RuntimeError("deferred hyper-connection update refers to a different tensor than the stage output")
    mod = importlib.import_module("exllamav3.modules.hyperconnections")
    if hasattr(mod, "hc_flush"):
        mod.hc_flush(params)
        return y
    hc = getattr(mod, "HyperConnection", None)
    if hc is not None and hasattr(hc, "flush_pending"):
        params.pop("hc_pending")
        return hc.flush_pending(y, pend)
    raise RuntimeError("hc_pending is set but this exllamav3 build has neither hc_flush nor HyperConnection.flush_pending")


def _sample_row(logits, temperature, top_p, top_k, rng):
    """Sample one token from a full-vocabulary logit row (numpy), mirroring the coordinator's top-k/top-p sampler."""
    import numpy as np
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


class Exl3Backend(Backend):
    def __init__(self, cfg, log=print):
        # Keep deferred/fused hyper-connection paths off where they are opt-in; the flush above covers the default path.
        for k, v in (cfg.env or {}).items():
            os.environ[k] = str(v)
        os.environ.setdefault("EXL3_PF_HC_FUSE", "0")
        os.environ.setdefault("EXL3_PF_GR_FUSE", "0")
        # On ROCm, torch's profiler registration leaves the HSA runtime event thread spinning on a full core for
        # the life of the process (from the first device op on). exllamav3 sets this before importing torch, but
        # this adapter imports torch first (just below), so the setdefault in exllamav3/__init__.py runs too late.
        # Mirror it here so the flag is in place before torch loads.
        os.environ.setdefault("ROCPROFILER_REGISTER_ENABLED", "0")
        import torch
        from exllamav3 import Config, Model, Cache
        self.torch, self.cfg, self.log = torch, cfg, log
        a, b = cfg.layers
        self.layers, self.has_embed, self.has_head = (a, b), cfg.embed, cfg.head
        self.cache_tokens = cfg.cache_tokens
        self.spec_k = int(getattr(cfg, "spec_k", 0) or 0)
        self.supports_spec = self.spec_k > 0
        # A verify forward of k+1 rows records k rows of history. The engine sizes its conv rings and staged
        # scan inputs for max_history+1 rows, so k+1 columns is the safe setting (max_history >= k).
        self.max_history = max(int(cfg.max_history), self.spec_k + 1 if self.spec_k else 0)
        if cfg.cache_tokens % 256 or cfg.cache_tokens < 256:
            raise ValueError("cache_tokens must be a positive multiple of 256 (page size)")
        config = Config.from_directory(cfg.model_dir)
        for k, v in (cfg.engine or {}).items():                  # e.g. moe_cpu_offload / moe_cpu_split on upstream 1.6.x
            setattr(config.infer_params, k, v)
            log(f"[engine] infer_params.{k} = {v}")
        self.config = config
        self.model = Model.from_config(config)
        self.contract = contract_from_config(config)
        self.tail = tuple(self.contract["boundary"]["tail"])
        self.slice = slice_model(self.model, a, b, cfg.embed, cfg.head)
        self.recurrent = self.slice.has_recurrent
        log(f"[slice] layers {a}..{b} blocks={self.slice.n_blocks} recurrent={self.recurrent} cache_layers={self.slice.n_cache_layers}")
        self._Cache, self._dtype_half = Cache, torch.half
        if "recurrent_dtype" not in inspect.signature(Cache.__init__).parameters and cfg.recurrent_dtype != "half":
            log(f"[warn] this exllamav3 build's Cache has no recurrent_dtype; ignoring recurrent_dtype={cfg.recurrent_dtype!r}")
        if self.max_history > int(cfg.max_history):
            log(f"[spec] max_history raised {cfg.max_history} -> {self.max_history} for k={self.spec_k} verify windows")
        self.cache = Cache(self.model, **_filter_kwargs(Cache.__init__, dict(
            max_num_tokens=cfg.cache_tokens, max_batch_size=1, max_history=self.max_history,
            recurrent_dtype=getattr(torch, {"half": "half", "float": "float", "bfloat16": "bfloat16"}[cfg.recurrent_dtype]))))
        self.rs, self.pos = None, 0
        self._last_was_verify, self._last_history = False, 0
        self._host_cache = {}                                    # (n, wire_dtype) -> pinned host buffer
        self.memory = {}

    # ------------------------------------------------------------------ lifecycle
    def load(self):
        if not self.cfg.device:
            raise ValueError("this adapter requires a single device (e.g. device = \"cuda:0\"): autosplit loading runs a "
                             "token-id dummy pass that a stage without an embedding cannot take")
        kw = dict(device=self.cfg.device, progressbar=True)
        kw.update(self.cfg.load or {})
        self.model.load(**_filter_kwargs(self.model.load_gen, kw))
        t = self.torch
        if t.cuda.is_available():
            free, total = t.cuda.mem_get_info(self.cfg.device)
            self.memory = {"device": str(self.cfg.device), "free_gb": round(free / 1e9, 2), "total_gb": round(total / 1e9, 2),
                           "allocated_gb": round(t.cuda.memory_allocated(self.cfg.device) / 1e9, 2)}
        return self.memory

    def _free_states(self):
        for st in self.rs or []:
            try:
                st.free()
            except Exception:
                pass
        self.rs = None

    def reset(self):
        self._free_states()
        self.pos = 0
        self._last_was_verify, self._last_history = False, 0

    def cache_len(self):
        return self.pos

    def close(self):
        self._free_states()

    def _settle_verify(self, pos):
        """Settle the previous verify pass at `pos` (the engine's `reject_remainder` equivalent).
        This is required even when every draft was accepted (`rewind(0)`): a history pass leaves the
        recurrent state and conv ring provisional until the rewind settles them. Legal only immediately
        after a verify forward and within the recorded window."""
        if not self._last_was_verify:
            if pos < self.pos:
                raise Unsupported("recurrent state can only be rewound right after a verify forward")
            return
        n = self.pos - pos
        if n < 0 or n > self._last_history or n > self.max_history:
            raise Unsupported(f"recurrent state can only be rewound right after a verify forward "
                              f"(asked {n}; history {self._last_history}, capacity {self.max_history})")
        for st in self.rs or []:
            st.rewind(n)
        self._sync()
        self.pos = pos
        self._last_was_verify, self._last_history = False, 0

    def truncate(self, pos):
        if pos >= self.pos:
            return
        if pos <= 0:
            return self.reset()
        if self.recurrent:
            self._settle_verify(pos)
        self.pos = pos          # attention reads only [0, past_len); later rows are overwritten

    # ------------------------------------------------------------------ forward
    def _sync(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def forward(self, start_pos, token_ids, hidden, wire_dtype, want, return_rows, k, spec=None):
        # The engine's own Model.forward() is @torch.inference_mode, but forward_ls() is not. The cache and the
        # recurrent state do in-place updates on inference tensors, which raises "Inplace update to inference
        # tensor outside InferenceMode" unless the whole pass runs inside it.
        with self.torch.inference_mode():
            return self._forward_impl(start_pos, token_ids, hidden, wire_dtype, want, return_rows, k, spec)

    def _forward_impl(self, start_pos, token_ids, hidden, wire_dtype, want, return_rows, k, spec=None):
        t = self.torch
        m, n = self.model, len(token_ids)
        verify = spec is not None
        if verify and want not in ("spec", "topk", "hidden", "none"):
            raise ValueError(f"verify forward cannot want {want!r}")
        if self.recurrent:
            self._settle_verify(start_pos)     # settle the previous verify pass (rewind(k-m), or rewind(0))
        t0 = time.perf_counter()
        ids = t.tensor([token_ids], dtype=t.long)
        params = {"attn_mode": "flash_attn", "cache": self.cache, "past_len": start_pos,
                  "batch_shape": (1, self.cfg.cache_tokens)}
        if self.recurrent:
            if self.rs is None:
                self.rs = [self.cache.get_new_state()]
            params["recurrent_states"] = self.rs
        if verify:
            # Record one row of recurrent history per draft. advance_recurrent_states sets
            # last_history = n - 1, which is what a later GDNState.rewind(num_rejected) expects.
            params["recurrent_history"] = True
        if self.has_head:
            params["last_tokens_only"] = n if (want not in ("none", "hidden") and return_rows == "all") else 1
        x = m.prepare_inputs(ids, params)
        t_prep = time.perf_counter()
        for pm in getattr(m, "_get_prefetch_layers", []):
            pm.prefetch(x, params)
        t_pf = time.perf_counter()
        if hidden is not None:
            if wire_dtype == "bf16":
                x = t.frombuffer(hidden, dtype=t.int16).view(t.bfloat16).float()
            else:
                x = t.frombuffer(hidden, dtype=t.float32)
            x = x.reshape((1, n) + self.tail)
        t1 = time.perf_counter()
        y = m.forward_ls(x, params)
        t_fwd = time.perf_counter()
        if not self.has_head:
            y = flush_hyperconnections(params, y)
        t_flush = time.perf_counter()
        from exllamav3.cache.recurrent_util import advance_recurrent_states
        advance_recurrent_states(ids, params, m)
        self._sync()
        t2 = time.perf_counter()
        self.pos = start_pos + n
        self._last_was_verify, self._last_history = verify, ((n - 1) if verify else 0)
        timing = {"h2d_ms": (t1 - t0) * 1e3, "compute_ms": (t2 - t1) * 1e3,
                  "prepare_ms": (t_prep - t0) * 1e3, "prefetch_ms": (t_pf - t_prep) * 1e3,
                  "forward_ls_ms": (t_fwd - t1) * 1e3, "flush_ms": (t_flush - t_fwd) * 1e3,
                  "advance_ms": (t2 - t_flush) * 1e3}
        if os.environ.get("SWP_DEBUG_TIMING"):
            self.log(f"[timing] n={n} " + " ".join(f"{kk}={vv:.1f}" for kk, vv in timing.items()))

        if not self.has_head:
            if tuple(y.shape) != (1, n) + self.tail:
                raise RuntimeError(f"stage output shape {tuple(y.shape)} != expected {(1, n) + self.tail}")
            dt = t.bfloat16 if wire_dtype == "bf16" else t.float32
            key = (n, wire_dtype)
            buf = self._host_cache.get(key)
            if buf is None:
                if len(self._host_cache) >= 8:
                    self._host_cache.clear()      # speculative decode alternates 1-row and k+1-row frames
                buf = t.empty((n,) + self.tail, dtype=dt, pin_memory=True)     # one pinned D2H per shape
                self._host_cache[key] = buf
            src = y[0]
            buf.copy_(src.to(dt) if src.dtype != dt else src)
            body = buf.view(t.uint8).reshape(-1).numpy()                       # byte view, no extra copy
            timing["d2h_ms"] = (time.perf_counter() - t2) * 1e3
            return Result("hidden", body, wire_dtype, [n] + list(self.tail), timing)

        if want == "none":
            return Result("none", timing=timing)
        logits = y.float()[0]                                   # [rows, vocab]
        if want == "spec":
            if not verify:
                raise ValueError("want=spec needs mode=verify")
            drafts = [int(d) for d in (spec or {}).get("draft_ids", [])]
            if logits.shape[0] != len(drafts) + 1:
                raise RuntimeError(f"verify head returned {logits.shape[0]} rows for {len(drafts)} drafts")
            import numpy as np
            if (spec or {}).get("accept", "argmax") == "sample":
                s = (spec or {}).get("sampling") or {}
                rng = np.random.default_rng([int(s.get("seed") or 0), int(s.get("round") or 0)])
                lr = logits.cpu().numpy()
                pred = [_sample_row(lr[i], float(s.get("temperature", 1.0)), float(s.get("top_p", 1.0)),
                                    int(s.get("top_k") or 0), rng) for i in range(lr.shape[0])]
            else:
                pred = logits.argmax(-1).to(t.int32).cpu().tolist()
            accepted = 0
            while accepted < len(drafts) and int(pred[accepted]) == drafts[accepted]:
                accepted += 1
            tokens = drafts[:accepted] + [int(pred[accepted])]
            body = np.asarray(tokens, "<i4").tobytes()
            res = Result("spec", body, "i32", [len(tokens)],
                         extra={"accepted": accepted, "tokens": tokens, "drafts": []})
        elif want == "token":
            body = logits.argmax(-1).to(t.int32).cpu().numpy().tobytes()
            res = Result("tokens", body, "i32", [logits.shape[0]])
        else:
            vals, idx = t.topk(logits, int(k or 64), dim=-1)    # sorted descending
            body = idx.to(t.int32).cpu().numpy().tobytes() + vals.cpu().numpy().tobytes()
            res = Result("topk", body, "i32+f32", [logits.shape[0], idx.shape[-1]])
        timing["d2h_ms"] = (time.perf_counter() - t2) * 1e3
        res.timing = timing
        return res
