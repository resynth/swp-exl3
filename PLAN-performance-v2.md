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

### Task 6: persistent coordinator `swp-exl3 serve` (2-3 h; independent of Section 4, do it after)
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

**Status (2026-10-25).** S1 and S2 are done (see `RESULTS.md`): the verify protocol, lazy rewind and n-gram drafting
are committed. S2 measured **1.66x** on repetitive output (100% acceptance) and **neutral** on the 149-token
non-repetitive prompt (15.02 vs 15.05 tok/s); top-1 is equal on all verify rows, with a batched-kernel KL floor of
~5e-3. **S3 is implemented and gated** (`RESULTS.md` §S3): MTP capacity/API pass, but the best k=3 result is
**15.5 tok/s vs 15.05 plain (1.03x)** on the 149-token prompt; no k reaches 1.15x, and the S3.4 stop rule
(<1.10x) fires. The n-gram path remains the default and `mtp = false`.

**Where it runs.** Everything here is implemented in the in-process `Pipeline` request loop that already owns
`history`, `start_pos` and the existing `rollback`/rewrite path (`Pipeline.generate` / `_spec_loop`), and is driven by the
existing `swp-exl3 generate --bench` and `swp-exl3 smoke` commands. The stages are the already long-lived
`swp-exl3 stage` processes. **Nothing here needs the persistent serving process of Task 6**: a fresh CLI process
gets the same verify forward and lazy rewind, and loses only cross-request prefix reuse. While iterating, run stage 0
as a remote stage (`addr`, not `stage_config`) so each run does not reload the Razer slice.

### Mechanism (shared by all steps)
The driver below is the existing `Pipeline` object; `history` is its token history, `t` the last accepted token
(not yet fed). One round trip per round:
```
driver: history, t (last accepted token, not yet fed), drafts d1..dk
FORWARD verify rows [t, d1..dk], start_pos = len(history), rewrite = true  ->  stage 0  ->  stage 1 (head)
head stage: forward all k+1 rows; accept drafts while argmax(logits[i-1]) == d_i; m = accepted; bonus = argmax(logits[m])
            (for T>0: "sample then match": sample the target token at each row, accept a draft only if equal; exact, lower acceptance)
reply: kind "spec" {accepted: m, tokens: [d1..dm, bonus], drafts: next drafts (MTP step only)}
driver: history += [t, d1..dm]; t = bonus.   Next FORWARD's start_pos = len(history) < stage cache_len by (k - m).
```
**Rollback is lazy and rides on `rewrite`.** A stage receiving `start_pos < cache_len` rewinds `n = cache_len - start_pos`:
- attention / DSA caches: just move the position back (as today);
- recurrent (KDA) state: `state.rewind(n)`, legal only if the previous forward on that stage was a **verify** forward that
  recorded history and `n <= state.last_history` (= rows - 1). Otherwise `Unsupported`, and the driver falls back to
  clear + replay (existing path).
Required engine bits: verify forwards set `params["recurrent_history"] = True` on every stage (then
`advance_recurrent_states` sets `last_history = rows - 1`); stage `Cache(..., max_history >= k)` (S2 uses k = 3, so
max_history = 4; S3 picks k from the S3.0 cost curve). Every verify must be settled before the next forward, even when
all drafts are accepted (`rewind(0)`); otherwise the recurrent state is left provisional. This was the one correctness bug
found in S2, not in the protocol.

Protocol additions (splitwire spec, capability `spec`): FORWARD `mode:"verify"`, `draft_ids`, `accept`, `sampling`, `next_drafts`;
FORWARD_OUT `kind:"spec"` with `accepted`, `tokens`, `drafts`, `timing`. Document the "rewind only immediately after a verify
forward" rule. Extend `FakeBackend` with per-position state history so rewind is testable.

### Step S1: protocol + speculative loop on the fake backend (2-3 h, no GPU)
Scripted drafter (right/wrong patterns). Tests: speculative output equals non-speculative greedy exactly for all-accept,
all-reject and mixed patterns, 2 and 3 stages; illegal rewind -> replay still correct. This is the only place exactness is
asserted. **Exit:** tests pass.

### Step S2: n-gram / prompt-lookup drafting on the real engines (2-3 h)
Drafts come from the driver side (match the last 2-3 tokens against the history, propose the next k). No drafter model, no
embedding copy, no MTP cache. This exercises the hard part (verify batch across both stages, recurrent history, lazy rewind)
on real hardware cheaply, and speeds up repetitive output (code, quotes, summaries).
Smoke addition (one check): verify-forward logits at each row vs sequential decode logits from the same state, 20 tokens.
Top-1 must be equal; KL <= 1e-2 on the shared top-8 (measured floor: 1-row verify is bit-identical, 2-row KL 3e-5,
3/4-row up to 5e-3 -- the engine's batched kernels, not SWP; a plain batched prefill shows the same delta, so 1e-3 is not
a correctness signal). **Exit:** smoke passes; bench on one repetitive prompt. If rewind misbehaves, fix here, not in S3.

### Step S3: MTP drafter on the head stage

Task 2 passed its gate, but on **Model A** (`g = 1.21`, 58.7% acceptance at k=2). Model B's MTP acceptance is unmeasured
(it does not fit alone), and S2's 1.66x batching result was at **100%** n-gram acceptance. The round economics are
therefore different: with a linear verify-cost curve, k=1 is ~1.18x, k=2 ~1.10-1.15x, and k=3 **below 1.0x** (the third
draft spends a full verify row for ~0.2 tokens). Do not start with a fixed k=3; measure and pick k first.

#### S3.0 Go/no-go measurements (no drafter code)

1. **Capacity.** Set the Strix `cache_tokens = 8192` (the Razer already caps context at 8192, so this is free).
   Load the MTP layer (3.76 GB) + a read-only `embed_tokens` copy (1.27 GB) + a drafter cache on the Strix and confirm
   no OOM (the S2 stage has 15.35 GB free; Task 2 saw Model A MTP-on OOM only at serve.py's 4608-token startup warm-up,
   which this adapter does not run). If it does not fit: reduce the drafter cache first, then report rather than evicting
   a target layer.
2. **Cost curve.** On the real pair, measure C(2), C(3), C(4) (wall per verify round) plus one MTP-layer forward and one
   explicit catch-up forward. Combine with Task 2's acceptance to pick k. If no k reaches ~1.15x, stop here and report;
   the n-gram S2 result is already banked.
3. **API check on Kyojin 1.5.0** (the plan's line numbers are 1.5.10): `attach_to` (it asserts an `Embedding`; pass one),
   `export_state_norm_keys` (tap key is the final norm; `EXL3_MTP_PRENORM_H` default off), `sample_from_state`,
   `mtp_pending`/`_flush_mtp_pending`, and the `eh_proj` sidecar. Add these to `check_api_against_source.py`.

#### S3.1 Drafter load and attach (head stage, `exl3_backend.py`)

4. The sliced last stage has the target's `lm_head` and final norm but no `Embedding` (the main model's `modules` list does
   not contain the MTP block; it is a separate `Glm5NextMTPModel`). Load the MTP model and a read-only embedding copy from
   the same full pack on the Strix (a pulled slice needs `pull --last` to include `embed_tokens`), and use a local copy of
   `attach_to` taking `embed_module=` (about 30 lines; `attach_to` also calls `load_eh_proj_sidecar`).
5. Build the drafter `Cache` (`max_num_tokens = 8192`, same as the target) **before** the drafter loads. The MTP block is
   an attention block, not recurrent; its cache position still has to be managed across rounds.

#### S3.2 Loop integration (backend + coordinator)

6. Set `next_drafts: true` on the verify FORWARD. The head stage returns the next round's `drafts` in the `kind:"spec"`
   reply; the coordinator must **consume `data["drafts"]`** for the next round (currently `_spec_loop` ignores them and
   re-runs `spec.propose`). Fall back to the n-gram lookup when the list is empty.
7. **Target hidden / carry.** Set `export_state_norm_keys` on the **prefill and verify** forwards. Unlike the native engine,
   this adapter already runs the final norm on every row (`forward_ls`, no `prefill_ls` shortcut), so the existing prefill
   can export all prompt hiddens -- do not add a second target prefill pass. Handle chunk carry for prompts > `chunk` and
   the shift (the drafter input at a position uses the target hidden of the previous position).
8. **Off-by-one after a round.** The next drafter input is the **bonus token**, paired with the hidden of the verify row
   that predicted it (row `m`), not simply "the last accepted token". The returned `drafts` are for the positions after
   the bonus. Get this right before tuning.
9. **Catch-up.** Extend the drafter cache with the accepted rows and their hiddens after each round (mirror
   `mtp_pending` / `_flush_mtp_pending`). Start with an explicit catch-up forward; fuse only if `catchup_ms` shows up.
   The drafter's KV position must be rewound/overwritten for rejected drafts even though its block is not recurrent.

#### S3.3 k and drafting policy

10. Start with **k = 2** (the native engine's measured choice), then run **k = 1 and 3**. Pick the best measured
    `(verify_ms + mtp_ms + catchup_ms) / accepted_per_round`, not the largest k.
11. Bring a **minimal confidence stop** forward: stop drafting at depth j when the drafter's top-1 probability (or
    top-1/top-2 margin) is below a threshold. `sample_from_state` already exports confidence. Record the accepted count
    per depth; this is the main lever if k=2 misses 1.15x. No full calibrator needed.

#### S3.4 Correctness, benchmark, exit

12. Keep the S2 verify-vs-sequential logit check with the drafter loaded (top-1 equal; KL <= 1e-2, the batched-kernel
    floor). A short MTP-on vs MTP-off token comparison is cheap and informative: in this design the target forward is
    unchanged except for a read-only norm export, so if they diverge, first check that the export is read-only and
    `last_tokens_only` is the same; do not spend days chasing identity -- the target logits are the guard.
13. Bench on the **149-token non-repetitive smoke prompt** (plain baseline 15.03 tok/s), not the repetitive code prompt.
    Report `verify_ms`, `mtp_ms`, `catchup_ms`, accepted/round and tokens/round. One repetitive prompt is optional, to
    check whether MTP and n-gram stack.
    **Exit:** decode >= 1.15x of 15.0 tok/s (>=17.3) on the 149-token prompt, smoke passes, no regression on the
    repetitive prompt. **Stop rule:** if the S3.0 cost math cannot reach 1.15x, or the implementation misses 1.10x after
    the k sweep, stop and report; do not grind. Keep the n-gram path either way.

### Later options (only if S3 works and `mtp_ms + catchup_ms` is a large share of the round)

Moving the drafter to the Razer is not a free win: the MTP layer is 3.8 GB and the Razer has 3.18 GB free, so a target
layer would have to move to the Strix; and drafting still depends on the target hidden, so the serial dependency remains.
Count the extra hops and the target-layer cost. Draft-ahead and fusing the catch-up only if the per-round timings show
they matter.

## 5. Explicitly not doing
Cut or offload tuning (measured: every variant lost), wire quantisation (wire is ~5% of a token), per-module profiling (unless
Task 4 points at something), cross-engine tensor/expert parallelism, prefill pipelining (<= 1.17x here), the planner rewrite,
large multi-prompt correctness suites.

## 6. Next: Plan security hardening.

## Appendix: engine pointers (verify on installed versions)
| What | Where |
|---|---|
| MTP draft loop, verify, rewind of rejected drafts | `exllamav3/generator/generator.py`: `iterate_draftmodel_mtp_gen` (~875); `reject_remainder` (~1395-1445) calls `batch_states[j].rewind(num_rejected)` |
| MTP pending rows / last hidden | `generator.py` `_flush_mtp_pending`, `job.mtp_last_hidden` (job.py ~1564-1579) |
| Drafter attach requirements | `exllamav3/architecture/glm5_next_mtp.py` `attach_to` (~185-230); `draft_verifier_params = {"export_state_norm_keys": {...}}` |
| `last_history` | `exllamav3/cache/recurrent_util.py` `advance_recurrent_states`: `last_history = rows - 1` if `params["recurrent_history"]` |
| Recurrent rewind | `exllamav3/modules/gated_delta_net.py` `GDNState.rewind(n)` (~165); `rollback_capacity()` is 0 (rewind only right after a history-recording forward) |
| Our code | `swp_exl3/exl3_backend.py` (`forward`, `truncate`, `_settle_verify`), `server.py` (`_forward` verify/rewrite handling), `coordinator.py` (`rollback`, `_spec_loop`, `generate`, `spec_check`), `backend.py` (`FakeBackend`) |
