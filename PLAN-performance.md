# swp-exl3 performance plan

For the coding assistant working on `swp-exl3` (and, where noted, `splitwire`). Goal: **faster decode and prefill** on the
Razer + Strix Halo pair, then bigger quants. Source of truth for current state: `BRINGUP.md`.

Everything marked *(hypothesis)* is unmeasured. Everything marked *(estimate)* is arithmetic on your own measurements.
Engine file/line pointers (Appendix A) come from a **Kyojin 1.5.10 checkout**; the Strix has **1.5.0** and the Razer has
upstream **1.6.4**. Re-verify each pointer against the installed source before relying on it.

---

## 1. Ground rules

1. **Measure before and after every change.** Append a row to `RESULTS.md` using the template in Appendix B. No row, no merge.
2. **Correctness gate for every performance change** (except MTP, see §5.8):
   - `swp-exl3 logits` + `compare` against the pre-change run: top-1 equal, KL <= 1e-4 at f32.
   - Greedy text identical on at least 5 prompts of different lengths (short, 300, 1000, 3000 tokens).
3. **Protocol changes go through the spec.** SWP is defined in the `splitwire` repo. Any new message field is added there
   (bump to `swp/1.1`, gate with a capability string), implemented in the Rust toy stage and `FakeBackend`, and covered by
   a test before the real backend uses it.
4. Keep the 15 existing tests green. New behaviour gets a fake-backend test first, real-engine test second.
5. Never make a change that silently degrades output (lossy wire, dropped layers, skipped flush). Lossy options stay opt-in.

---

## 2. What the measurements already tell us

| Observation (from BRINGUP) | Reading | Consequence |
|---|---|---|
| Decode 19.4 tok/s = 51 ms/token: Strix stage 44 ms (86% busy), Razer 4.8 ms (9% busy), wire+queue ~2.1 ms | Strix is the critical path; Razer idle 90% | Cut tuning cannot help much; **speculation** can (it amortises the serial chain over several tokens) |
| Moving 6 layers off the Strix (45 -> 39 layers) saved only about 1 ms *(compare to your ~22 tok/s whole-model figure; this may be a different benchmark)* | Strix decode may **not** scale with layer count: fixed per-token costs (graph launch, lm_head, hc_head, sampling, python) *(hypothesis)* | Phase 0 must settle this before any layer-balancing work |
| Local hop Razer stage -> coordinator costs ~0.9 ms wire+queue even over loopback | D2H copy + serialisation + thread wakeups, not network | ~2 ms/token (4%) is recoverable |
| Stage-0 prefill compute (~850-1150 ms) is close to stage-1 (~900-1390 ms) for the same prompt | Stages are balanced for prefill | **Prefill pipelining** can approach 1.6x on prompts long enough for several chunks *(estimate: 4 chunks, balanced -> 8/(2+3) = 1.6x)* |
| First prefill after process start: 2.5-44 s, then ~0.9 s | One-time kernel compile/autotune, cached to disk; per-process first-use cost remains | Warm the **stage** at startup (the kernels live in the stage servers, not the coordinator) |
| Fresh `generate` process = fresh coordinator = empty history = full re-prefill every call | Prefix reuse never happens in CLI use | A persistent coordinator gives **prefix reuse across turns** (the real UX win) |
| CPU offload costs ~7 ms per offloaded layer per token, nearly independent of how many experts are offloaded | Fixed per-layer cost (thread hand-off / affinity) dominates | Tune the CPU worker before scaling offload (Phase 3) |
| bf16 wire saves ~30 us of a ~51 ms token | Not a lever | Keep f32. Do not build int8 wire. |

---

## 3. Phase 0: measure (about a day, no behaviour changes)

Purpose: decide which later phases are worth doing. Add tiny tooling only.

- [ ] **0.1 Adapter-path baseline.** On the Strix run ONE stage covering `layers=[0,45]`, `embed=true`, `head=true`, with
      the coordinator on the Strix over localhost. Record decode ms/token and prefill tok/s for the same prompt and context
      size used on the pair. Then run Kyojin's own `tools/glm/serve.py` on the same prompt with MTP off and with MTP on.
      Decision: if the adapter path is much slower than `serve.py` MTP-off, **the adapter is losing an engine fast path**
      (likely candidates: `Model.forward` wrappers, block-graph capture, `last_tokens_only` handling). Fix that before
      anything else, because it speeds up every configuration.
- [ ] **0.2 Per-module decode timing on the Strix stage.** Set `[stage.env] EXL3_FWD_SPAN = "1"` and a low
      `EXL3_FWD_SPAN_MS` (see Appendix A, `forward_ls`; it times every module for <= 8 rows). Record where the 44 ms goes:
      KDA layers, MLA/DSA layers, MoE, hc ops, `hc_head`, norm, `lm_head`. Output: a table in `RESULTS.md`.
- [ ] **0.3 Cut sweep.** Decode ms/token and per-stage compute for cuts 3, 4, 5, 6 (and 7 if it fits in 16 GB). Fit
      `strix_ms ~ a + b * layers_on_strix`. If `b` is small, stop tuning the cut.
- [ ] **0.4 Rows curve.** Add a `swp-exl3 stagebench` command that calls the backend directly (no network) for
      `n_rows` in {1, 2, 3, 4, 6, 8} at a fixed context, decode-class, for each stage. This is the **verify cost curve** and
      is the key input to the MTP gain model in §5.2. It also shows how MoE time grows as more experts are touched.
- [ ] **0.5 MTP statistics for THIS checkpoint.** Using Kyojin's own tools on the Strix (`tools/glm/depth_mtp_table.py`,
      `depth_curve_mtp.py`, `acceptaudit_tf2.py` or equivalents), get per-depth acceptance probabilities and the drafter's
      per-forward cost for the 2.72 bpw model. Real prompts, not random ones.
- [ ] **0.6 Hop breakdown.** Split the 0.9 ms local "wire+queue" into D2H, serialise, socket, H2D, thread wake-up using
      `SWP_DEBUG_TIMING` plus `perf_counter` at the server edges. Output feeds task 1.4.

**Gate to Phase 2:** compute the MTP gain model (§5.2) with the numbers from 0.4 and 0.5. Proceed if predicted decode gain is
>= 1.3x over 19.4 tok/s. If it is below that, report back instead of building it.

---

## 4. Phase 1: quick wins (no protocol change)

- [ ] **1.1 Stage warm-up.** After `load()`, the stage server runs synthetic forwards (decode n=1 a few times; prefill rows
      16, 64, 256, 512; for non-first stages use zero boundary tensors), then `reset()`. Config key `warmup = true`
      (default true). Log the time taken. *Done when:* the first real prefill is within 20% of a warm one.
- [ ] **1.2 Persistent coordinator: `swp-exl3 serve`.** A long-running process that owns the `Pipeline` and exposes an
      OpenAI-compatible `/v1/chat/completions` (streaming, `stop`, `max_tokens`, `temperature`, `top_p`, `seed`) and
      `/v1/models`. It keeps the token history between requests and reuses the longest common prefix.
      - Reuse must survive tokenisation drift: keep the exact token ids of the previous assistant reply and splice them in
        when the new request's message prefix matches the previous conversation text.
      - A mismatch inside the history on this model is expensive (recurrent state cannot rewind: clear + replay). Log
        `reused_prefix`, `replayed_tokens` per request so the cost is visible.
      - Optional later: engine recurrent checkpoints (`GDNState.stash`/`unstash`) to make rollback cheap.
- [ ] **1.3 Prefill pipelining.** Coordinator sends chunk i+1 to stage 0 as soon as stage 0 returns chunk i, while chunk i
      goes to stage 1. One sender thread per stage plus ordered queues; stage 1 must still see chunks in order.
      Sweep chunk size in {256, 512, 1024, 2048} (a boundary row is 64 KiB at f32, so 2048 rows = 128 MiB; memory is fine).
      *Done when:* a >= 2k-token prompt prefills >= 1.4x faster and the correctness gate passes. Note: a prompt that fits in
      one chunk cannot benefit.
- [ ] **1.4 Hop overhead.** Targets, in order of expected value (confirm with 0.6 first):
      (a) run stage 0 **in the coordinator process** (a `LocalPeer` that calls the backend directly, no socket);
      (b) pinned host buffers and a single D2H into a preallocated buffer instead of `.cpu().numpy().tobytes()`;
      (c) `socket.sendmsg` with the header and payload together; avoid per-frame allocations;
      (d) optional `chain` topology (stage 0 sends directly to stage 1) if the star relay still shows in 0.6.
      *Done when:* decode wire+queue drops from ~2.1 ms/token to < 1 ms/token.
- [ ] **1.5 Engine environment flags.** Try, one at a time via `[stage.env]`, and keep only what measurably helps and passes
      the correctness gate: `EXL3_HC_FOLD` (hyper-connection fusion), `EXL3_BLOCK_GRAPH`, and whatever 0.2 points at.

---

## 5. Phase 2: MTP speculative decoding

### 5.1 Why this is the main decode lever
One serial chain per token cannot be overlapped, and the Strix is 86% busy. Speculation verifies several tokens per
traversal, so the fixed per-traversal costs (hop ~2 ms, the Razer's ~5 ms, launch overheads, expert weight reads that are
shared across rows) are spread over more than one token. The checkpoint ships an MTP layer (`model.language_model.layers.45`,
3.79 GB, already classified `extra` and pulled onto the last stage).

### 5.2 Gain model (fill in from Phase 0)
```
tokens_per_round E = 1 + sum_{i=1..k} prod_{j<=i} a_j        a_j = acceptance of draft j given previous accepted
round_ms R = razer_verify(k+1) + hop + strix_verify(k+1) + mtp_draft(k) + mtp_catchup + overhead
predicted decode tok/s = 1000 * E / R            baseline: 19.4 tok/s (51.4 ms)
```
`razer_verify`/`strix_verify` come from 0.4, `a_j` and `mtp_draft` from 0.5. Rough illustration *(estimate, not a
prediction)*: if k=3, E=2.3 and R=75 ms the result is ~31 tok/s; if E=1.5 it is ~20 tok/s (no gain). Acceptance is the
whole game, which is why 0.5 is a hard gate.

### 5.3 Design: one round trip per round
```
coordinator state: history (fed tokens), t (last accepted token, not yet fed), drafts d1..dk (from previous round)

round:
  FORWARD verify  rows = [t, d1..dk]  start_pos = len(history)  rewrite = true  -> stage 0 -> stage 1 (head)
  stage 1: forward all k+1 rows (recurrent_history = true, export post-norm hidden of every row)
           logits[0..k]; accept drafts while argmax(logits[i-1]) == d_i      (greedy; see 5.5 for T > 0)
           m = accepted count; bonus = argmax(logits[m])
           catch up the MTP cache with accepted rows (t, d1..dm) and their exported hiddens
           draft k new tokens from hidden[m] and embed(bonus)
           reply: kind "spec" {accepted: m, tokens: [d1..dm, bonus], drafts: [d1'..dk']}
  coordinator: history += [t, d1..dm]; t = bonus; drafts = drafts'
  next round's FORWARD uses start_pos = len(history) (< stage cache_len by k-m): that IS the rollback, done lazily.
```
Rollback is **lazy and rides on `rewrite`**: a stage that receives `start_pos < cache_len` rewinds by `n = cache_len - start_pos`:
- KV/attention state (MLA, DSA indexer): move the position back (`cache_len = start_pos`), as today.
- Recurrent (KDA) state: call `state.rewind(n)`; legal **only if** the previous forward on that stage was a verify forward
  that recorded history and `n <= state.last_history` (= rows - 1). Otherwise raise `Unsupported`, and the coordinator
  falls back to clear + replay (existing path).
This needs the stage `Cache` built with `max_history >= k` (config `max_history`; start with 4-6).

### 5.4 Protocol additions (spec change in `splitwire` first)
- Capability `spec`; `proto` stays `swp/1`, additive fields only.
- FORWARD header: `mode: "verify"`, `draft_ids: [..]` (head stage only), `accept: "greedy" | "match"`,
  `sampling: {temperature, top_k, seed}`, `next_drafts: k` (ask the head stage to draft after verifying).
  `mode: "verify"` implies `recurrent_history = true` on every stage.
- FORWARD_OUT from the head stage: `kind: "spec"`, `accepted: m`, `tokens: [...]`, `drafts: [...]`,
  `timing: {verify_ms, mtp_ms, catchup_ms}`.
- Non-head stages answer verify exactly as a normal hidden-state forward.
- Semantics of `rewrite` for recurrent state, as in 5.3. Document the "only immediately after a verify forward" rule.
- Extend `FakeBackend` (and the Rust toy stage if you want conformance parity) with per-position state history so
  rewind is testable. Add tests: verify/accept/rewind sequences equal a clean sequential run; rewind after a non-verify
  forward is `unsupported`; replay fallback still produces identical output.

### 5.5 Backend work (`exl3_backend.py`)
1. **Verify forward flags:** `params["recurrent_history"] = True` on every stage; on the head stage also
   `params["export_state_norm_keys"] = {<final norm key>}` so `params["export_states"]` returns per-row post-norm hidden.
   `advance_recurrent_states` then sets `last_history = rows - 1`.
2. **`truncate()` for recurrent models:** replace the blanket `Unsupported` with the rule in 5.3.
3. **Drafter on the head stage.** The engine's `attach_to` (Appendix A) requires the target's `Embedding` module in
   `target.modules`, `lm_head` as the last module, and an `RMSNorm` at `logit_layer_idx - 1`. A sliced last stage has the
   last two but **no Embedding**. Options (pick one, document why):
   (a) keep a loaded read-only copy of `embed_tokens` and call a local copy of `attach_to` that takes `embed_module=`
       (~30 lines; resilient to a missing engine feature, brittle to engine changes), or
   (b) a small upstreamable patch to `attach_to`.
   Provisioning: the last stage's pull needs `embed_tokens` (1.27 GB) in addition to the MTP layer
   (`swp-exl3 pull --last` currently only adds it for tied embeddings; add a `--with-embed` or make it automatic when MTP is on).
4. **Draft cache.** A separate `Cache` for the drafter with the same `max_num_tokens`, constructed **before** the drafter's
   `load()`. Mirror the engine's draft loop (`iterate_draftmodel_mtp_gen`): params `target_hidden`, `attn_mode`, cache,
   positions; greedy argmax per draft depth.
5. **Catch-up.** After each accepted round the drafter's cache must be extended with the accepted rows and their target
   hiddens (the engine calls this `mtp_pending` / `_flush_mtp_pending`). Start with the simple unfused version (explicit
   catch-up forward); fuse later if 0.5 shows it matters.
6. **Start minimal:** fixed k (3), greedy, no dynamic draft length, no confidence calibration, no spec gate. Add those only
   if measurements justify them (the engine has `dynamic_draft_tokens`, `DraftConfidenceCalibrator`, `SpecGate`).
7. **Sampling (T > 0):** use "sample then match": sample the target token at each verified row from the target
   distribution (top-k from the head stage); accept a draft only if it equals that sample. This is exact for the target
   distribution regardless of the drafter, at a lower acceptance rate than rejection sampling. Ratio-based acceptance
   (the engine's `spec_sample`) is a later optimisation.

### 5.6 Coordinator (`coordinator.py`)
`generate_spec`: the round loop of 5.3; handles stop tokens that appear inside an accepted run (truncate the run,
`rollback` to the stop position, which will replay on this model, so prefer to just stop and discard the cache tail);
keeps `history` exact; falls back to non-speculative decode when the drafter is unavailable or acceptance collapses
(simple moving-average gate). Report accepted tokens/round and MTP time in `--bench`.

### 5.7 Tests
- Fake backend with a scripted drafter (right/wrong patterns): speculative output **equals** non-speculative greedy output
  exactly, for several acceptance patterns including all-reject and all-accept, across 2- and 3-stage splits, with chunking.
- Rewind legality tests (5.4). Replay fallback test. Stop token inside an accepted run.
- Real engine: cannot require token identity (Kyojin documents that MTP-on and MTP-off greedy decode diverge at some
  point). Instead compare the verify forward's logits at each row against sequential single-token decode logits from the
  same state: KL <= 1e-3 and top-1 equal on >= 95% of rows.

### 5.8 Gates and exit criteria
- Gate A (Phase 0): predicted gain >= 1.3x, otherwise stop and report.
- Gate B (after 5.5 greedy works): measured decode >= 1.15x of 19.4 tok/s on at least 5 real prompts, with acceptance
  statistics recorded. If lower, tune k in {2, 3, 4, 6}; if still lower, report with the verify-cost curve.
- Output quality check: sample 20 prompts, compare greedy MTP vs non-MTP continuations for the first 50 tokens; record the
  divergence point distribution. Divergence is expected (see above); gross garbage is not.

### 5.9 Variants to evaluate only after 5.8 passes
- **Drafter on the Razer.** The Razer is idle ~90% and has 16 GB; the MTP layer is 3.79 GB. The Strix would send the
  accepted row's final hidden (4096 x 4 B = 16 KiB) back and receive k draft ids, trading ~2 hops for taking k drafter
  forwards off the critical path. Worth it only if the measured drafter cost on the Strix is comparable to a hop.
- **Draft-ahead.** Start the next round's drafting before verification returns, assuming all accepted. Complex; only if
  drafting shows up as a large share of R.
- **Dynamic draft length** and the engine's calibrator, if acceptance varies a lot between prompts.

---

## 6. Phase 3: bigger quants (CPU offload tuning)

The mechanism works (BRINGUP section 7); the cost does not scale. Each offloaded layer adds ~7 ms/token regardless of how many
experts move. Investigate in this order:
- [ ] **3.1** Sweep `moe_cpu_threads` (and the worker's CPU affinity; BRINGUP notes it pins to a single logical processor
      on a 16-thread laptop). Measure ms per offloaded layer vs threads. Check for contention with the coordinator and
      stage threads (set the stage process's own affinity away from the worker's cores).
- [ ] **3.2** Compare `moe_cpu_offload = N` vs `moe_cpu_split = N` at equal GPU-memory freed, on decode ms/token.
- [ ] **3.3** Find where the per-layer fixed cost goes (worker hand-off, H2D/D2H of activations per layer, synchronisation).
      If it is per-layer synchronisation, batching several offloaded layers' hand-offs is the lever; that is an engine-level
      change, so report it with data before touching engine code.
- [ ] **3.4** Capacity recipe check: pick a quant whose Strix share fits ~124 GB with headroom, Razer takes the rest with
      `moe_cpu_offload`, confirm end-to-end and record tok/s. Update BRINGUP section 11 with measured numbers.

---

## 7. Phase 4: correctness debts and housekeeping

- [ ] **hc_pending flush on real hardware.** `flush_ms` was 0.0 in every measured forward, but those were prefill chunks;
      the deferral applies to decode-class row counts (<= 32 rows per the engine's comments). Check the decode and verify
      forwards specifically. Force the path with `EXL3_HC_FOLD=1` in `[stage.env]`, then rerun the cut-invariance comparison
      at a cut right after a deferring site. If `hc_pending` is ever non-empty at a boundary, assert that the flush ran.
- [ ] **Reconnect/replay against real stages.** Kill the Strix stage mid-generation; the coordinator must reconnect (after
      you restart the stage) and replay. Record the replay time for a 1k-token history.
- [ ] **`doctor`:** replace the wrong `moe_cpu_host` probe with `modules/block_sparse_mlp_cpu` and report the real offload knobs.
- [ ] **README corrections:** Kyojin 1.5.0 (not 1.6.0); 15 tests (1 self-skips without `SWP_WEIGHTD`); Kyojin must be run
      with `PYTHONPATH` set because it is not pip-installed.
- [ ] **"Dependencies on this machine" note:** the exllamav3 `ext.py` `-std=c++20` patch for CUDA 13.3, the stale
      `torch_extensions/.../lock` hang, `ROCPROFILER_REGISTER_ENABLED=0`. Consider reporting the CUDA 13.3 build failure
      upstream with the minimal reproduction.
- [ ] **Repos:** `splitwire` and `swp-exl3` are separate repos. Pick one canonical `swp.py` (suggest `splitwire/python`,
      with `swp-exl3` depending on it), add a `.gitignore`, a licence for `swp-exl3`, commit `Cargo.lock`, tag `v0.1.0`/`v0.2.0`.
- [ ] **Planner (splitwire repo, low priority):** replace the bandwidth model with measured per-layer ms (from `--bench --json`
      and the 0.3 fit), with CPU-offloaded layers costed separately (~7x a GPU layer until Phase 3 improves it).

---

## 8. Explicitly not doing
- Wire quantisation (int8 or otherwise): the wire is ~5% of a token and ~2% of prefill capacity. bf16 stays opt-in.
- Cross-engine tensor parallelism, expert parallelism across machines.
- Tuning the cut for speed before 0.1-0.3 explain why the Strix stage barely changed.
- Dynamic draft-length machinery, calibration, and spec gates before a minimal fixed-k MTP shows a gain.

---

## Appendix A: engine pointers (Kyojin 1.5.10 checkout; re-verify on 1.5.0 and upstream 1.6.4)

| What | Where |
|---|---|
| Engine's MTP draft loop | `exllamav3/generator/generator.py`: `iterate_draftmodel_mtp_gen` (~line 875); drafter params `target_hidden`, `attn_mode`, `block_table`, `cache`, `cache_seqlens` |
| Verify + rollback of rejected drafts | same file, `reject_remainder` (~1395-1445): `batch_states[j].rewind(num_rejected)`, then cache positions return to the accepted position |
| MTP catch-up / pending rows | `generator.py`: `_flush_mtp_pending`, `mtp_pending`, `job.mtp_last_hidden` (job.py ~1564-1579; generator.py ~1041, ~1746) |
| Drafter attach requirements | `exllamav3/architecture/glm5_next_mtp.py`: `attach_to` (~185-230): needs target `Embedding`, last module `Linear`, `RMSNorm` at `logit_layer_idx - 1`; sets `draft_verifier_params = {"export_state_norm_keys": {...}}`; `EXL3_MTP_PRENORM_H` changes the tapped point |
| Recurrent history | `exllamav3/cache/recurrent_util.py`: `advance_recurrent_states` sets `last_history = rows - 1` when `params["recurrent_history"]` |
| Recurrent rewind | `exllamav3/modules/gated_delta_net.py`: `GDNState.rewind(num_tokens)` (~165); `rollback_capacity()` returns 0 (rewind only immediately after a history-recording forward); history buffers sized by `Cache(max_history=...)`; `save_history` (~1466) |
| Per-module timing diagnostic | `exllamav3/model/model_ls.py`: `_FWD_SPAN` / `EXL3_FWD_SPAN`, `EXL3_FWD_SPAN_MS` (decode and verify only, <= 8 rows) |
| Hyper-connection flush | upstream 1.6.4: `modules/hyperconnections.py::hc_flush(params)`; Kyojin 1.5.10: `HyperConnection.flush_pending(x, pending)` |
| Our code | `swp_exl3/exl3_backend.py` (`forward`, `truncate`), `server.py` (`_forward` rewrite handling), `coordinator.py` (`rollback`, `generate`), `backend.py` (`FakeBackend`) |

## Appendix B: `RESULTS.md` row template
```
| date | change (task id) | machine/config | prompt (tokens) | decode tok/s | decode ms/tok (stage0 / stage1 / wire) | prefill tok/s | correctness gate (KL, top-1, text) | notes |
```
Always record: cut, chunk size, wire dtype, `cache_tokens`, offload settings, engine versions, warm or cold process.
