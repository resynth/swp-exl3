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
import importlib, inspect, os, time, weakref
from .backend import Backend, Result, Unsupported, bf16_bytes_to_f32, f32_to_bf16_bytes  # noqa: F401
from .contract import contract_from_config
from .slicing import slice_model


class _MTPTargetShim:
    """Minimal target view for the MTP draft block: modules[0] is the embedding it must reuse and
    modules[logit_layer_idx] is the target's shared lm_head. `Glm5NextMTPModel` reaches both through
    `attached_model()`, which normally is the full model; the head stage only has the slice."""

    loaded_tp = False

    def __init__(self, embed, lm_head, config):
        self.modules = [embed, lm_head]
        self.logit_layer_idx = 1
        self.config = config


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
        self.cache_bits_requested = int(getattr(cfg, "cache_bits", 0) or 0)
        if self.cache_bits_requested not in (0, 8):
            raise ValueError(f"cache_bits must be 0 (fp16 KV) or 8 (packed int8 KV), got {self.cache_bits_requested!r}")
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
        if self.max_history > int(cfg.max_history):
            log(f"[spec] max_history raised {cfg.max_history} -> {self.max_history} for k={self.spec_k} verify windows")
        cache_kwargs = dict(max_num_tokens=cfg.cache_tokens, max_batch_size=1, max_history=self.max_history,
                            **self._quant_cache_kwargs(self.cache_bits_requested))
        self.cache_bits = 8 if self.cache_bits_requested else 0
        if "recurrent_dtype" in inspect.signature(Cache.__init__).parameters:
            cache_kwargs["recurrent_dtype"] = getattr(
                torch, {"half": "half", "float": "float", "bfloat16": "bfloat16"}[cfg.recurrent_dtype])
        elif cfg.recurrent_dtype != "half":
            log(f"[warn] this exllamav3 build's Cache has no recurrent_dtype; ignoring recurrent_dtype={cfg.recurrent_dtype!r}")
        self.cache = Cache(self.model, **_filter_kwargs(Cache.__init__, cache_kwargs))
        self.rs, self.pos = None, 0
        self._last_was_verify, self._last_history = False, 0
        self._host_cache = {}                                    # (n, wire_dtype) -> pinned host buffer
        self.memory = {}
        self.supports_mtp = bool(getattr(cfg, "mtp", False)) and cfg.head
        self.mtp_enabled = self.supports_mtp
        self.mtp_model = self.mtp_cache = self.embed = self._mtp_shim = None
        self._mtp_tap, self._mtp_pos, self._mtp_carry, self._mtp_committed = None, 0, None, 0
        if self.supports_mtp:
            self._init_mtp()

    # ------------------------------------------------------------------ MTP drafter
    def _init_mtp(self):
        """Build (but do not load) the MTP block, a read-only embedding copy and the drafter cache.
        The head stage's slice has no Embedding and no MTP block, so both come from the same pack."""
        from exllamav3 import Model
        from exllamav3.modules import Embedding
        d = self.cfg.mtp_model_dir or self.cfg.model_dir
        mtp_cfg = self.config if os.path.abspath(d) == os.path.abspath(self.cfg.model_dir) else self._config_from(d)
        self.mtp_config = mtp_cfg
        self.mtp_model = Model.from_config(mtp_cfg, component="mtp")
        ct = int(self.cfg.mtp_cache_tokens or self.cfg.cache_tokens)
        if ct % 256 or ct < 256:
            raise ValueError("mtp_cache_tokens must be a positive multiple of 256 (page size)")
        mtp_kwargs = dict(max_num_tokens=ct, max_batch_size=1, max_history=0)
        if "recurrent_dtype" in inspect.signature(self._Cache.__init__).parameters:
            mtp_kwargs["recurrent_dtype"] = self.torch.half
        self.mtp_cache = self._Cache(self.mtp_model, **_filter_kwargs(self._Cache.__init__, mtp_kwargs))
        keys = [k for k in mtp_cfg.stc.tensor_file_map if k.endswith(".embed_tokens.weight")]
        if len(keys) != 1:
            raise RuntimeError(f"MTP needs exactly one embed_tokens tensor in {d}; found {keys[:3]}")
        self.embed = Embedding(mtp_cfg, key=keys[0][:-len(".weight")], vocab_size=mtp_cfg.vocab_size,
                               hidden_size=mtp_cfg.hidden_size)
        self.log(f"[mtp] drafter built (cache {ct} tokens); embedding {keys[0]}")

    def _config_from(self, d):
        from exllamav3 import Config
        c = Config.from_directory(d)
        for k, v in (self.cfg.engine or {}).items():
            setattr(c.infer_params, k, v)
        return c

    def _quant_cache_kwargs(self, bits):
        """Build the quantized-KV cache kwargs, guarded by what this engine actually exposes.
        `Cache(layer_type=CacheLayer_quant, k_bits=..., v_bits=...)` is optional, but an explicit
        `cache_bits = 8` is not silently downgraded: a fallback would allocate the fp16 cache at the
        q8-sized `cache_tokens` and can OOM. Fail with a clear message instead; use `cache_bits = 0`
        on an engine without the packed cache."""
        if not bits:
            return {}
        try:
            from exllamav3 import Cache, CacheLayer_quant
        except Exception as e:                                          # no quantized cache in this build
            raise ValueError(f"cache_bits={bits} requested but this engine has no CacheLayer_quant; "
                             f"set cache_bits = 0 on this engine") from e
        cps = inspect.signature(Cache.__init__).parameters
        qps = inspect.signature(CacheLayer_quant.__init__).parameters
        has_var = any(p.kind == p.VAR_KEYWORD for p in cps.values())
        missing = [p for p in ("layer_type",) if p not in cps]
        missing += [p for p in ("k_bits", "v_bits") if p not in qps]
        if not has_var:
            missing += [p for p in ("k_bits", "v_bits") if p in qps and p not in cps]
        missing = sorted(set(missing))
        if missing:
            raise ValueError(f"cache_bits={bits} requested but this engine's Cache/CacheLayer_quant "
                             f"lacks {missing}; set cache_bits = 0 on this engine")
        self.log(f"[cache] quantized KV: CacheLayer_quant k_bits={bits} v_bits={bits}")
        return dict(layer_type=CacheLayer_quant, k_bits=bits, v_bits=bits)

    def _attach_mtp(self):
        """Attach the drafter to a shim that exposes the head slice's shared lm_head plus the embedding copy.
        `attach_to` itself asserts a full target model, hence this local equivalent."""
        from exllamav3.modules import Linear, RMSNorm
        norm = self.model.modules[self.model.logit_layer_idx - 1]
        if not isinstance(norm, RMSNorm):
            raise RuntimeError(f"MTP tap expected the final RMSNorm before lm_head, found {type(norm).__name__}")
        lm = self.model.modules[self.model.logit_layer_idx]
        if not isinstance(lm, Linear):
            raise RuntimeError(f"MTP expected lm_head as a Linear, found {type(lm).__name__}")
        tap = norm.key
        if os.getenv("EXL3_MTP_PRENORM_H", "0") == "1":
            from exllamav3.modules.hyperconnections import HyperHead
            for m in self.model.modules[self.model.logit_layer_idx::-1]:
                if isinstance(m, HyperHead) and m.mean:
                    tap = m.norm.key
                    break
            else:
                raise RuntimeError("EXL3_MTP_PRENORM_H=1 but no mean HyperHead before lm_head")
        self._mtp_shim = _MTPTargetShim(self.embed, lm, self.mtp_config)
        self.mtp_model.input_layer.attached_model = weakref.ref(self._mtp_shim)
        self.mtp_model.attached_model = weakref.ref(self._mtp_shim)
        self.mtp_model.target_embed = weakref.ref(self.embed)
        self.mtp_model.target_lm_head = weakref.ref(lm)
        self.mtp_model.draft_verifier_params = {"export_state_norm_keys": {tap}}
        self._mtp_tap = tap
        try:
            if self.mtp_model.load_eh_proj_sidecar():
                self.log(f"[mtp] eh_proj sidecar from {os.getenv('EXL3_MTP_EH_FP16')}")
        except Exception as e:                                          # optional accuracy overlay
            self.log(f"[mtp] eh_proj sidecar not applied: {type(e).__name__}: {e}")
        self.log(f"[mtp] attached: tap={tap} lm_head={lm.key}")

    def _mtp_params(self, pos, hidden):
        return {"attn_mode": "flash_attn", "cache": self.mtp_cache, "past_len": int(pos),
                "batch_shape": (1, int(self.mtp_cache.max_num_tokens)), "target_hidden": hidden}

    def _mtp_prefill(self, token_ids, hidden, pos):
        ids = self.torch.tensor([list(token_ids)], dtype=self.torch.long, device=hidden.device)
        self.mtp_model.prefill(ids, self._mtp_params(pos, hidden))

    def _mtp_roll(self, first_token, carry, k, pos):
        """Autoregressively propose k drafts. `carry` is the target hidden of the previous position;
        the first step writes the drafter row for `first_token` and predicts the token after it."""
        t = self.torch
        ids = t.tensor([[int(first_token)]], dtype=t.long, device=carry.device)
        out, h = [], carry
        for j in range(int(k)):
            p = self._mtp_params(pos + j, h)
            state = self.mtp_model.forward(ids, p)
            ids = self.mtp_model.sample_from_state(state, p)
            out.append(ids.reshape(1, 1))
            h = state
        if not out:
            return []
        return [int(x) for x in t.cat(out, dim=1)[0].cpu().tolist()]

    def _mtp_hidden(self, params, rows=None):
        st = params.get("export_states")
        if not st:
            raise RuntimeError("MTP is enabled but the target forward exported no hidden states "
                               "(export_state_norm_keys missing?)")
        h = st[0]
        return h if rows is None else h[:, :rows, :]

    def _mtp_after_verify(self, start_pos, token_ids, drafts, accepted, pred, params, next_drafts, k):
        """Settle the drafter to the accepted prefix, then optionally propose the next round.
        Row L holds the verify's first token paired with the previous target hidden; rows L+1.. are
        overwritten with the accepted tokens paired with their real target hiddens (catch-up)."""
        export = self._mtp_hidden(params)
        L = start_pos
        t0 = time.perf_counter()
        carry = self._mtp_carry
        if carry is None:
            carry = self.torch.zeros((1, 1, export.shape[-1]), dtype=export.dtype, device=export.device)
        # Always rewrite row L with the token actually being verified. Draft rows from the previous
        # round are provisional and may have been proposed for a different (sampled) bonus.
        self._mtp_prefill([token_ids[0]], carry, L)
        self._mtp_pos = L + 1
        m = int(accepted)
        if m:
            self._mtp_prefill(drafts[:m], export[:, :m, :], L + 1)
        self._mtp_pos = L + m + 1
        self._mtp_carry = export[:, m:m + 1, :].clone()
        self._mtp_committed = self._mtp_pos
        t1 = time.perf_counter()
        out = {"catchup_ms": (t1 - t0) * 1e3, "mtp_roll_ms": 0.0}
        next_list = []
        if next_drafts and int(k) > 0:
            bonus = int(pred[m])
            next_list = self._mtp_roll(bonus, self._mtp_carry, int(k), self._mtp_pos)
            self._mtp_pos += int(k)
            out["mtp_roll_ms"] = (time.perf_counter() - t1) * 1e3
        return next_list, out

    # ------------------------------------------------------------------ lifecycle
    def load(self):
        if not self.cfg.device:
            raise ValueError("this adapter requires a single device (e.g. device = \"cuda:0\"): autosplit loading runs a "
                             "token-id dummy pass that a stage without an embedding cannot take")
        kw = dict(device=self.cfg.device, progressbar=True)
        kw.update(self.cfg.load or {})
        self.model.load(**_filter_kwargs(self.model.load_gen, kw))
        if self.mtp_enabled:
            self.mtp_model.load(**_filter_kwargs(self.mtp_model.load_gen, kw))
            self.embed.load(self.torch.device(self.cfg.device))
            self._attach_mtp()
        t = self.torch
        if t.cuda.is_available():
            free, total = t.cuda.mem_get_info(self.cfg.device)
            try:
                cache_bytes = sum(l.storage_size() for l in self.cache.layers.values())
                recurrent_bytes = sum(l.storage_size() for l in self.cache.recurrent_layers.values())
            except Exception:
                cache_bytes = recurrent_bytes = 0
            self.memory = {"device": str(self.cfg.device), "free_gb": round(free / 1e9, 2), "total_gb": round(total / 1e9, 2),
                           "allocated_gb": round(t.cuda.memory_allocated(self.cfg.device) / 1e9, 2),
                           "cache_tokens": int(self.cfg.cache_tokens), "cache_bits": self.cache_bits,
                           "cache_bits_requested": self.cache_bits_requested,
                           "cache_gb": round(float(cache_bytes) / 1e9, 3), "recurrent_gb": round(float(recurrent_bytes) / 1e9, 3)}
        return self.memory

    def warmup(self, rows=None, log=None):
        info = super().warmup(rows, log)
        if self.mtp_enabled:
            t = self.torch
            t0 = time.perf_counter()
            h = t.zeros((1, 1, self.mtp_config.hidden_size), dtype=t.half, device=self.cfg.device)
            self._mtp_roll(1, h, 3, 0)
            self.reset()
            info["mtp_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
            info["mtp_forwards"] = 3
            if log:
                log(f"[warmup] MTP 3 draft steps in {info['mtp_ms']:.0f} ms")
        return info

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
        self._mtp_pos = self._mtp_committed = 0
        self._mtp_carry = None

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
        if self.mtp_enabled and pos < self._mtp_committed:
            # The drafter cache would need a target hidden for every row of the kept prefix; replay.
            raise Unsupported("MTP drafter cannot rewind into an earlier prefix; clear and replay")
        if self.recurrent:
            self._settle_verify(pos)
        self.pos = pos          # attention reads only [0, past_len); later rows are overwritten
        if self.mtp_enabled:
            # Drop provisional draft rows; the next verify re-syncs row L and catches up.
            self._mtp_pos = min(self._mtp_pos, pos)
            self._mtp_committed = min(self._mtp_committed, pos)

    # ------------------------------------------------------------------ forward
    def _sync(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def forward(self, start_pos, token_ids, hidden, wire_dtype, want, return_rows, k, spec=None, draft=None):
        # The engine's own Model.forward() is @torch.inference_mode, but forward_ls() is not. The cache and the
        # recurrent state do in-place updates on inference tensors, which raises "Inplace update to inference
        # tensor outside InferenceMode" unless the whole pass runs inside it.
        with self.torch.inference_mode():
            return self._forward_impl(start_pos, token_ids, hidden, wire_dtype, want, return_rows, k, spec, draft)

    def _forward_impl(self, start_pos, token_ids, hidden, wire_dtype, want, return_rows, k, spec=None, draft=None):
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
            if self.mtp_enabled:
                params["export_state_norm_keys"] = {self._mtp_tap}
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
        mtp_extra = {}
        mtp_t0 = time.perf_counter()
        if self.mtp_enabled and not verify:
            h = self._mtp_hidden(params)
            if self._mtp_pos != start_pos:
                raise Unsupported(f"MTP drafter cache at {self._mtp_pos}, target starts at {start_pos}; replay")
            if n == 1:
                carry = self._mtp_carry
                if carry is None:
                    carry = self.torch.zeros((1, 1, h.shape[-1]), dtype=h.dtype, device=h.device)
                self._mtp_prefill([token_ids[0]], carry, start_pos)
            else:
                carry = self._mtp_carry
                if carry is None:
                    carry = self.torch.zeros((1, 1, h.shape[-1]), dtype=h.dtype, device=h.device)
                self._mtp_prefill(token_ids, self.torch.cat((carry, h[:, :-1, :]), dim=1), start_pos)
            self._mtp_pos = start_pos + n
            self._mtp_carry = h[:, -1:, :].clone()
            self._mtp_committed = self._mtp_pos
            timing["mtp_sync_ms"] = (time.perf_counter() - mtp_t0) * 1e3
        elif self.mtp_enabled and verify and want == "topk":
            # A logit-equivalence verify does not run acceptance; leave the drafter unsynced and force a
            # replay if a later request tries to reuse this prefix.
            self._mtp_committed = -1
            timing["mtp_sync_ms"] = 0.0
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
            mtp_next, mtp_timing = [], {}
            if self.mtp_enabled:
                mtp_next, mtp_timing = self._mtp_after_verify(
                    start_pos, token_ids, drafts, accepted, pred, params,
                    bool((draft or {}).get("next_drafts")), (draft or {}).get("k") or 0)
                timing.update(mtp_timing)
                timing["mtp_ms"] = sum(v for kk, v in mtp_timing.items())
            body = np.asarray(tokens, "<i4").tobytes()
            res = Result("spec", body, "i32", [len(tokens)],
                         extra={"accepted": accepted, "tokens": tokens, "drafts": mtp_next,
                                "drafts_basis": int(tokens[-1])})
            mtp_extra = {"drafts": mtp_next, "drafts_basis": int(tokens[-1])}
        elif want == "token":
            body = logits.argmax(-1).to(t.int32).cpu().numpy().tobytes()
            res = Result("tokens", body, "i32", [logits.shape[0]])
            if self.mtp_enabled and bool((draft or {}).get("next_drafts")) and int((draft or {}).get("k") or 0) > 0:
                basis = int(logits[-1].argmax())
                mtp_t0 = time.perf_counter()
                mtp_extra = {"drafts": self._mtp_roll(basis, self._mtp_carry, int(draft["k"]), self._mtp_pos),
                             "drafts_basis": basis}
                self._mtp_pos += int(draft["k"])
                timing["mtp_ms"] = timing.get("mtp_ms", 0.0) + (time.perf_counter() - mtp_t0) * 1e3
        else:
            vals, idx = t.topk(logits, int(k or 64), dim=-1)    # sorted descending
            body = idx.to(t.int32).cpu().numpy().tobytes() + vals.cpu().numpy().tobytes()
            res = Result("topk", body, "i32+f32", [logits.shape[0], idx.shape[-1]])
        timing["d2h_ms"] = (time.perf_counter() - t2) * 1e3
        if mtp_extra:
            res.extra = dict(res.extra or {}, **mtp_extra)
        res.timing = timing
        return res
