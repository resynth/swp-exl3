# swp-exl3 performance plan v3 (supersedes v2)

Same hardware and model as `RESULTS.md`: Model B (3.09 bpw), Razer `[0,7)` + Strix `[7,45)`, f32 wire.
Current: **15.7 tok/s decode** (stage 0 in-process), prefill **473 tok/s** at chunk 2048, **24.9 tok/s** with n-gram speculation on
repetitive output, MTP implemented but gated off (1.03x measured, ~1.08x steady-state by your own arithmetic).
Estimates are labelled *(estimate)*; unmeasured claims *(hypothesis)*. Engine pointers are from a Kyojin 1.5.10 checkout: re-verify on 1.5.0.

## Task A: apply Kyojin's full `SPEED_ENV` on the Strix stage [DONE]
**Why.** `tools/glm/serve.py` sets 56 `EXL3_*` flags (`SPEED_ENV`) before importing the engine. The adapter sets none of them, and the
pair runs only toggled `EXL3_BLOCK_GRAPH(_MLA)` one at a time *(hypothesis: the Strix stage has been running without the rest)*. The
flags that matter for your measurements exist **only in Kyojin** (absent from upstream 1.6.4, so this is a Strix-only change):
`EXL3_VERIFY_FUSE`, `EXL3_VERIFY_ROW_SPLIT=2`, `EXL3_HIP_PREFILL_MIN_ROWS=2`, `EXL3_DEC_ROUTER_ROWS=2`, `EXL3_SPEC_SAMPLED`,
`EXL3_MOE_CFG=2`, `EXL3_PREFILL_BIG_CHUNK`, `EXL3_DEC_DSA_FAST`, plus the `EXL3_NC2_*` / `EXL3_PF_*` families.
Evidence it matters: Kyojin's native MTP round at k=2 costs ~87 ms vs 50 ms plain decode (1.55x); the pair's costs 144 ms vs 66.5 ms (2.0x).

**Do.**
1. Generate the block from the installed checkout and paste it into the Strix stage's `[stage.env]`:
```python
import ast
src = open("/home/m/engines/kyojin/tools/glm/serve.py").read()
i = src.index("SPEED_ENV = {"); j = src.index("\n}\n", i) + 3
d = ast.literal_eval(src[i+len("SPEED_ENV = "):j])
print("[stage.env]"); [print(f'{k} = "{v}"') for k, v in d.items()]
```
   (Optionally add a `swp-exl3 speed-env --kyojin-root DIR` command that prints this. I like this!) Leave the Razer unchanged.
2. Restart the stages, run `smoke`. **If it fails, bisect**: halve the flag list until the culprit is found, keep the rest.
   `EXL3_PF_HC_FUSE=1` is in the list and defers hyper-connection residual updates across blocks: the boundary flush in
   `flush_hyperconnections` has never been exercised on hardware, so a smoke failure there is a flush bug, not a flag problem.
   Do **not** add `EXL3_PF_GR_FUSE` (Kyojin's GLM server does not set it; only the Qwen server does). This replaces the old T4.
3. Four measurements, one run each, warm: decode (149-token prompt); prefill (1.5k-token prompt, chunk 2048); decode at 8k context
   (`EXL3_DEC_DSA_FAST` targets this); and the **round wall time at k=1..4** on the 100%-accept repetitive prompt (the verify curve).
**Exit.** Keep every flag that passes smoke. Record what moved. The verify curve decides Tasks B and D.

## Task B: longer n-gram draft window (1-2 h; after A)
Your assistant's T1, with two adjustments. The arithmetic holds: each extra row costs ~21 ms (117 / 136 / 160 / 181 ms for 2-5 rows) and
replaces a ~66 ms token. At 100% acceptance k=8 would be ~9 tokens per ~266 ms = ~34 tok/s *(estimate, extrapolating the slope past
5 rows)*; the assistant's 30-31 allows for imperfect acceptance.
1. `spec_k = 8` on both stages (`max_history` auto-raises to 9). If the Razer OOMs at load, use 6.
2. **Measure rows 6-9 first**: run k = 3, 5, 8 on the repetitive prompt and read the round wall time. If the per-row slope rises past
   5 rows (kernel path change), cap k where it does. After Task A the slope itself may be lower.
3. Adaptive length: draft length from the n-gram match length (e.g. k=8 when the match is >= 3 tokens, else 3). Also stop the draft at
   the end of the matched span. ~10 lines in `SpecConfig`/`_spec_loop`.
4. Sweep `--spec-match 2` vs `3`. Keep per-class settings if they differ.
5. Smoke's verify-vs-sequential check stays capped at `min(spec_k, 4)` rows (the batched-kernel KL floor grows with row count).
6. Honest caveat: speculative greedy runs on batched kernels, so at 9 rows a near-tie token may differ from plain greedy. Require
   identical text on the two smoke prompts; if a longer window ever diverges, report where, do not "fix" it by tightening the gate.
**Exit.** Repetitive prompt >= +25% over the current 24.9 tok/s with identical text; novel prompt within 2% of plain.

## Task C: warm-up shape coverage (1 h)
Your assistant's T3, extended. First-use costs seen: 149-token first prefill 1.68x warm; after chunk 2048, the first 1.5k-3k prefill pays a
shape autotune (6.5 -> 9.1 -> 19.1 s).
1. Find the shape buckets once: on a fresh stage time the first forward at rows 1, 2, 3, 4, 5, 17, 33, 65, 149, 257, 513, 1025, 2048
   (one script, one run). Warm up one row count per bucket that shows a first-call penalty, plus 2048 and a verify-shaped forward
   (`spec_k + 1` rows with `recurrent_history`).
2. Exit: first real prefill <= 1.2x warm at chunk 2048 and a first speculative round with no compile spike. Accept up to ~4 s extra
   startup on the Strix (a 2048-row forward alone is ~3 s there); startup is paid once per long-lived stage.

## Task D: MTP re-decision (30 min; after A and B)
The 1.03x came from a 48-token run whose one plain bootstrap round (~130 ms) dominates. By your own numbers k=3 steady state is
2.65 tokens / (150.6 ms target + 12.9 ms drafter) = **16.2 tok/s (1.08x)**. A 1.15x gate (17.3 tok/s) needs target verify <= ~140 ms
(-7%); -20% (120 ms) gives ~19.9 tok/s (1.32x) *(estimate)*.
1. After Task A, rerun the k = 2 and 3 sweep with **200 new tokens** on the novel prompt. Record accepted/round and round wall.
2. If >= 1.15x: enable MTP as the fallback when the n-gram lookup misses (already the design), try the depth-2 confidence stop
   (skip depth 2 when depth-1 confidence is low), and make it the default. If < 1.10x: leave `mtp = false`, record the verify curve,
   and stop. No further MTP work.

## Task E: server defaults and docs (30 min)
- `swp-kyojin-server`: `--spec` on by default (1.65x on repetitive output, neutral otherwise), with `spec_k` from Task B.
- README note for agent harnesses: echo `reasoning_content` back (or use a template that keeps it), otherwise reuse stops inside the
  reply and the whole prefix replays. Tell clients to stay inside the 8192-token cache.
- Report upstream/Kyojin: `Cache` forwards unknown kwargs into `CacheLayer_MLA_quant` (the `recurrent_dtype` crash); and CUDA 13.3 +
  torch 2.11 headers fail at `-std=c++17` (the `ext.py` patch).
- Record in the splitwire spec: "settle every verify, including a full accept (`rewind(0)`)" (it fixed 4/20 top-1 mismatches).

## Task F: stage-side sub-chunking so the Strix can take bigger prefill chunks (TEST DONE, didn't pass criteria)
Kyojin's own server uses 8192-row prefill chunks (`EXL3_PREFILL_BIG_CHUNK`); the pair is capped at 2048 by the Razer's 3.2 GB free.
**Measure before building:** on the Strix alone, prefill the 1.5k and 3k prompts at chunk 2048 vs 8192 (2 runs). If < +10%, drop this task.
Otherwise: let stage 0 accept up to 8192 rows and run them internally as 2048-row forwards, concatenating the boundary tensors
(8192 x 64 KiB = 512 MiB in host memory, stage 0 is in-process so no socket); stage 1 runs the full chunk. Pipeline `chunk = 8192`.
Exit: >= +10% prefill on a 3k prompt, smoke passes, no Razer OOM.

## Task G: checkpointed rollback (conditional, half a day)
Do this only if the server's `MISS (replayed n)` lines show up regularly in your real agent workload. Today any change inside the history
replays the whole prefix (~17 s at 8k tokens at 470 tok/s). The engine already has the pieces: `GDNState.stash/unstash`
(`modules/gated_delta_net.py` ~187/213) and DSA cache `stash/unstash` (`cache/dsa.py` ~257/273).
Design: each stage snapshots recurrent state (and the DSA indexer cache) at chunk boundaries and at prompt end, keeping the last few;
`CACHE_OP snapshot|restore` (reserved in the spec). `Pipeline.rollback(pos)`: pick the latest snapshot position `c <= pos` common to
all stages, restore it on every stage, then feed only `history[c:pos]` through the pipeline. Replay cost drops from `pos` to `pos - c`
(<= the snapshot interval). Fake-backend tests first (extend `FakeBackend` snapshots), then one smoke. Exit: rollback in the middle of a
4k history costs <= 1 s and output stays identical to a full replay.

## Dropped or defered
Strix C-state workaround (handled separately), `EXL3_PF_GR_FUSE` test (not in GLM's own flag set), cut/offload tuning, wire quantisation,
prefill pipelining (<= 1.17x at this cut)
