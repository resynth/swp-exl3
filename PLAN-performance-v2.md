# swp-exl3 performance plan v2 (supersedes v1)

For the coding assistant. Goal: faster decode and prefill on Razer + Strix Halo, **with minimal test time**.
Current best: Model B (turboderp 3.09 bpw) at `[0,7)` / `[7,45)`, no offload: **14.9 tok/s decode, 217 tok/s warm prefill**.
Data: `MEASUREMENTS.md`. Engine pointers are from a Kyojin 1.5.10 checkout (Strix has 1.5.0, Razer has 1.6.4): re-verify before use.

## 0. Working rules (short)

1. **Time-box every task.** If it overruns its budget, report what you have and move on.
2. **Tests are small.** Unit tests on the fake backend (seconds, always run). On real engines use ONE smoke command
   (task 1), roughly 60-90 s. No multi-prompt sweeps, no long-context gates, no divergence studies.
3. **One benchmark line per change:** warm process, one ~200-token prompt, `--bench`. Append to `RESULTS.md`
   (change, decode tok/s, stage0/stage1/overhead ms per token, prefill tok/s).
4. Protocol additions go into the `splitwire` spec first (additive, capability-gated), then the fake backend, then the real one.

## 1. What your new measurements changed

| Finding | Consequence |
|---|---|
| Strix is the bottleneck: 54.8 of 67 ms/token (82% busy); Razer 7.6 ms (11%). Moving one layer from Strix to Razer saves ~1.9 ms on the Strix (cut 8 -> 7: 52.9 -> 54.8 ms). | Decode **is** roughly proportional to layers on the Strix (my v1 hypothesis that it was not is withdrawn). But the Razer cannot take more: cut 8 leaves 0.2 GB free. |
| Every offload variant lost: cut 8 + `split=32` 13.5, `split=48` 12.7, `[0,12)`+`offload=6` 9.5 tok/s vs 14.9 at cut 7 with no offload. | **Stop tuning cut and CPU offload.** Offload is for capacity only, and B fits. Phase 3 of v1 is dropped. |
| Per-token overhead = 67.0 ms wall - (7.6 + 54.8) = **~4.6 ms (7%)**, up from ~2.7 ms on Model A. | Worth a quick look (task 4): this is pure adapter cost. |
| Warm prefill, cut 7: Razer 202 ms vs Strix 1159 ms for 308 tokens. | **My v1 claim that prefill pipelining could give ~1.6x is withdrawn.** Stages are unbalanced 5.7:1, so the ceiling is (202+1159)/1159 = **1.17x**. The balanced figures from Model A (878 vs 880 ms) looked like leftover warm-up on the Razer. Pipelining is now optional. |
| Cold prefill 6 tok/s vs warm 217 tok/s; first-ever 45 s. | **Stage warm-up is the highest-value quick win.** |
| Remaining big lever is speculation (the Strix chain cannot be overlapped any other way). | Section 4. |

## 2. Order of work

### Task 1: smoke command + golden -- do this FIRST
`swp-exl3 smoke --pipeline p.toml [--update-golden]`. Runs, in under ~90 s: ping; two short prompts greedy for 48 tokens
(compare text to golden); one prompt's top-64 logits vs golden (top-1 equal and KL <= 1e-4); one `--bench` line printed.
Create the golden once from the current known-good Model B `[0,7)` build. Every later change must pass this and nothing more.
Also set `EXL3_HC_FOLD=1` once in `[stage.env]` and run smoke, to close the deferred hyper-connection question cheaply.

### Task 2: MTP ceiling check
Measure Model A in Kyojin's own `tools/glm/serve.py` on just the strix, with MTP on and MTP off.
**Gate:** if `g < 1.2`, skip section 4. Otherwise expect the pipeline to capture roughly 60-80% of the gain.
Then measure model A through swp-exl3 but still entirely on the strix, that way we can see if swp-exl3 causes any slowdown.
Then report and stop.

### Task 3: stage warm-up
After `load()`, each stage runs synthetic forwards (decode n=1 x3; prefill rows 16, 64, 256, 512; zero boundary tensors on
non-first stages), then `reset()`. Config `warmup = true` by default; log the duration.
Done when the first real prefill is within ~20% of a warm one (smoke prints it).

### Task 4: decode overhead, ~4.6 ms/token (1-2 h)
1. 3-minute look: run 20 decode tokens with `SWP_DEBUG_TIMING=1` on both stages and read `prepare / forward_ls / flush / advance`
   at n=1 (previous timings were prefill-sized only). Anything outside `forward_ls` is adapter overhead.
2. Fix in this order, stopping when overhead is < 2 ms/token:
   (a) run stage 0 **inside the coordinator process** (`LocalPeer` calling the backend directly, no socket);
   (b) preallocated pinned buffers and one D2H copy instead of `.cpu().numpy().tobytes()`;
   (c) send header + payload in one `sendmsg`; no per-frame allocations;
   (d) argmax/top-k on the GPU and return only ids (already done), avoid any extra `synchronize`.

### Task 5: engine env flags (15 min)
`EXL3_BLOCK_GRAPH`, `EXL3_HC_FOLD`, and at most one more that the Task 4 breakdown points to. One smoke + bench line each.
Keep only what helps and passes smoke.

### Task 6: persistent coordinator `swp-exl3 serve` (2-3 h)
Long-running process that owns the `Pipeline`; OpenAI-compatible `/v1/chat/completions` (stream, `stop`, `max_tokens`,
`temperature`, `top_p`, `seed`) and `/v1/models`. Keeps history; reuses the longest common prefix.
Tokenisation drift: keep the exact token ids of the previous assistant reply and splice them in when the new request's
message prefix matches. Log `reused_prefix` and `replayed_tokens`. (Any mismatch inside the history costs a full replay on this
model, because the recurrent state cannot rewind; speculation in section 4 changes that only for the last few tokens.)

### Task 7: prefill chunk size (10 min)
Two runs of a ~1.5k-token prompt at chunk 512 vs 1024 (maybe 2048). Keep the best. Prefill pipelining stays off the list
unless long prompts matter to you (ceiling 1.17x at this cut).

## 3. Housekeeping (batch it, ~1 h, no benchmarking)
- `doctor`: probe `modules/block_sparse_mlp_cpu` instead of `moe_cpu_host`; print the real offload knobs.
- README: Kyojin 1.5.0 (not 1.6.0), 15 tests (1 skips without `SWP_WEIGHTD`), `PYTHONPATH` needed for Kyojin.
- "Dependencies on this machine" note: exllamav3 `ext.py` `-std=c++20` patch (CUDA 13.3), stale `torch_extensions/.../lock`, `ROCPROFILER_REGISTER_ENABLED=0`.
- Repos: `splitwire` and `swp-exl3` stay separate; one canonical `swp.py`; `.gitignore`, licence, commit `Cargo.lock`, tag.
- Reconnect/replay on real stages: one manual check (kill the Strix stage mid-generation, restart, confirm recovery). Not a suite.

## 4. Speculation (the main decode lever), in three steps with early exits

### Mechanism (shared by all steps)
One round trip per round:
```
coordinator: history, t (last accepted token, not yet fed), drafts d1..dk
FORWARD verify rows [t, d1..dk], start_pos = len(history), rewrite = true  ->  stage 0  ->  stage 1 (head)
head stage: forward all k+1 rows; accept drafts while argmax(logits[i-1]) == d_i; m = accepted; bonus = argmax(logits[m])
            (for T>0: "sample then match": sample the target token at each row, accept a draft only if equal; exact, lower acceptance)
reply: kind "spec" {accepted: m, tokens: [d1..dm, bonus], drafts: next drafts (MTP step only)}
coordinator: history += [t, d1..dm]; t = bonus.   Next FORWARD's start_pos = len(history) < stage cache_len by (k - m).
```
**Rollback is lazy and rides on `rewrite`.** A stage receiving `start_pos < cache_len` rewinds `n = cache_len - start_pos`:
- attention / DSA caches: just move the position back (as today);
- recurrent (KDA) state: `state.rewind(n)`, legal only if the previous forward on that stage was a **verify** forward that
  recorded history and `n <= state.last_history` (= rows - 1). Otherwise `Unsupported`, and the coordinator falls back to
  clear + replay (existing path).
Required engine bits: verify forwards set `params["recurrent_history"] = True` on every stage (then
`advance_recurrent_states` sets `last_history = rows - 1`); stage `Cache(..., max_history >= k)` (start k = 3, max_history 4-6).

Protocol additions (splitwire spec, capability `spec`): FORWARD `mode:"verify"`, `draft_ids`, `accept`, `sampling`, `next_drafts`;
FORWARD_OUT `kind:"spec"` with `accepted`, `tokens`, `drafts`, `timing`. Document the "rewind only immediately after a verify
forward" rule. Extend `FakeBackend` with per-position state history so rewind is testable.

### Step S1: protocol + coordinator loop on the fake backend (2-3 h, no GPU)
Scripted drafter (right/wrong patterns). Tests: speculative output equals non-speculative greedy exactly for all-accept,
all-reject and mixed patterns, 2 and 3 stages; illegal rewind -> replay still correct. This is the only place exactness is
asserted. **Exit:** tests pass.

### Step S2: n-gram / prompt-lookup drafting on the real engines (2-3 h)
Drafts come from the coordinator (match the last 2-3 tokens against the history, propose the next k). No drafter model, no
embedding copy, no MTP cache. This exercises the hard part (verify batch across both stages, recurrent history, lazy rewind)
on real hardware cheaply, and speeds up repetitive output (code, quotes, summaries).
Smoke addition (one check): verify-forward logits at each row vs sequential decode logits from the same state, 20 tokens,
top-1 equal and KL <= 1e-3. **Exit:** smoke passes; bench on one repetitive prompt. If rewind misbehaves, fix here, not in S3.

### Step S3: MTP drafter on the head stage (0.5-1 day), only if Task 2 passed the gate
Head stage work (`exl3_backend.py`):
1. Drafter needs the target's `Embedding`, `lm_head` (last module) and an `RMSNorm` at `logit_layer_idx - 1`
   (engine `attach_to`, `glm5_next_mtp.py` ~185-230). A sliced last stage has the last two but **no Embedding**: keep a loaded
   read-only copy of `embed_tokens` (1.27 GB) and use a local copy of `attach_to` taking `embed_module=` (about 30 lines).
   Provisioning: `pull --last` must fetch `embed_tokens` (plus the MTP layer, already included) when MTP is on.
2. Drafter `Cache` with the same `max_num_tokens`, built before the drafter loads. Draft loop mirrors
   `iterate_draftmodel_mtp_gen` (generator.py ~875): params `target_hidden`, `attn_mode`, cache, positions; greedy argmax per depth.
3. Target hidden for the drafter: `params["export_state_norm_keys"] = {final norm key}` on the verify forward, take the row of
   the last accepted token from `export_states`.
4. Catch-up after each round: extend the drafter cache with the accepted rows and their hiddens (engine `mtp_pending` /
   `_flush_mtp_pending`). Start with an explicit catch-up forward; fuse later only if it shows up in the timings.
5. Minimal first: fixed k = 3, greedy. No dynamic draft length, calibrator, or gate.
**Exit (one smoke + one bench line):** decode >= 1.15x of 14.9 tok/s on the benchmark prompt. If lower, try k in {2, 4} (two more
runs), then report with the per-round timings (`verify_ms`, `mtp_ms`, `catchup_ms`, accepted tokens/round).
Do not chase token-identity with non-MTP decoding: Kyojin documents that MTP-on and MTP-off diverge. The S2 logit check is the guard.

### Later options (only if S3 works and drafting is a large share of the round)
Drafter on the Razer (idle 89% of decode; MTP layer is 3.8 GB): saves k drafter forwards on the Strix critical path, costs
about two extra hops of 16 KiB. Draft-ahead. Both need measured per-round timings to justify.

## 5. Explicitly not doing
Cut or offload tuning (measured: every variant lost), wire quantisation (wire is ~5% of a token), per-module profiling (unless
Task 4 points at something), cross-engine tensor/expert parallelism, prefill pipelining (<= 1.17x here), the planner rewrite,
large multi-prompt correctness suites.

## Appendix: engine pointers (verify on installed versions)
| What | Where |
|---|---|
| MTP draft loop, verify, rewind of rejected drafts | `exllamav3/generator/generator.py`: `iterate_draftmodel_mtp_gen` (~875); `reject_remainder` (~1395-1445) calls `batch_states[j].rewind(num_rejected)` |
| MTP pending rows / last hidden | `generator.py` `_flush_mtp_pending`, `job.mtp_last_hidden` (job.py ~1564-1579) |
| Drafter attach requirements | `exllamav3/architecture/glm5_next_mtp.py` `attach_to` (~185-230); `draft_verifier_params = {"export_state_norm_keys": {...}}` |
| `last_history` | `exllamav3/cache/recurrent_util.py` `advance_recurrent_states`: `last_history = rows - 1` if `params["recurrent_history"]` |
| Recurrent rewind | `exllamav3/modules/gated_delta_net.py` `GDNState.rewind(n)` (~165); `rollback_capacity()` is 0 (rewind only right after a history-recording forward) |
| Our code | `swp_exl3/exl3_backend.py` (`forward`, `truncate`), `server.py` (`_forward` rewrite handling), `coordinator.py` (`rollback`, `generate`), `backend.py` (`FakeBackend`) |
