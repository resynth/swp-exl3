# RESULTS — performance plan v2

Environment for every row: Model B (`turboderp/GLM-5.3-Flash-exl3`, 3.09 bpw), Razer `[0,7)` +
Strix `[7,45)`, f32 wire, `pipeline-310-k7.toml`, chunk 512. Bench prompt = the smoke prompt
(149 tokens), 48 greedy new tokens. `overhead` = `1000/decode − (stage0 + stage1)` compute, i.e. the
adapter + wire + queue time per token.

| # | change | decode tok/s | stage0 ms/tok | stage1 ms/tok | overhead ms/tok | prefill tok/s | smoke |
|---|---|---|---|---|---|---|---|
| 1 | baseline, first forward after Razer stage load (golden creation) | 15.08 | 7.045 | 54.767 | 4.50 | 113.4 | golden written |
| 2 | baseline, warm | 15.31 | 6.834 | 54.313 | 4.17 | 174.9 | PASS |
| 3 | Razer `EXL3_HC_FOLD=1`, first forward | 15.02 | 7.259 | 54.838 | 4.48 | 140.8 | PASS |
| 4 | Razer `EXL3_HC_FOLD=1`, warm | 15.13 | 7.011 | 54.728 | 4.35 | 172.9 | PASS |
| 5 | Task 3: stage warm-up on, fresh stages | 15.08 | 6.855 | 54.965 | 4.49 | 174.2 (first real prefill 176.6) | PASS |
| 6 | Task 5 baseline, warm (in-session) | 15.05 | 7.181 | 54.794 | 4.47 | 172.1 | PASS |
| 7 | Task 5: Strix `EXL3_BLOCK_GRAPH=1` | 15.05 | 6.758 | 55.255 | 4.48 | 168.1 | PASS |
| 8 | Task 5: Strix `EXL3_BLOCK_GRAPH=1` + `EXL3_BLOCK_GRAPH_MLA=2` | 15.02 | 6.767 | 55.436 | 4.48 | 167.5 | PASS |

All smoke runs: both 48-token prompts token-identical to the golden; top-64 logits top-1 equal,
64/64 shared, **max |Δlogit| 0.00000, KL 0.00e+00**.

## Task 1 — smoke command + golden

* `swp-exl3 smoke --pipeline p.toml [--golden FILE] [--update-golden] [--max-kl F]` added
  (`swp_exl3/cli.py`). Runs ping, two fixed greedy prompts (48 tokens), one top-64 logit check on
  "The capital of France is", and one `--bench` report; exits non-zero on any mismatch. Warm run:
  **~13 s**.
* Golden created from the known-good build:
  `/home/b/models/pipeline-310-k7.golden.json` (default path = `<pipeline>.golden.json`).
* Fake-backend unit test covers pass, fail after perturbing the model, and `--update-golden`
  (`tests/test_cli.py`); full suite: **16 tests, OK** (1 skips without `SWP_WEIGHTD`).
* `EXL3_HC_FOLD=1` was put in the Razer `[stage.env]` and tested: output is bit-identical, but the
  flag is **performance-neutral** on this hardware (rows 2 vs 4: 15.31 vs 15.13 tok/s, stage0
  6.834 vs 7.011 ms/tok — within run-to-run noise). It does enable the folded/deferred
  hyper-connection path; whether `hc_pending` actually survived a stage boundary was not logged.

## Task 2 — MTP ceiling check

### Model B alone: impossible on this machine

Task 2 as written asks for Model B alone on the Strix. That does not fit:

* Model B weights ≈ 127.94 GB ≈ **119 GiB**; the Strix's GPU-visible pool is **118.00 GiB**.
* `tools/glm/serve.py --model <Model B> --no-vision -c 8192 --num-draft 0` OOMed **during weight
  loading**: `GPU 0 has a total capacity of 118.00 GiB ... 4.12 GiB is free. Of the allocated
  memory 113.70 GiB is allocated by PyTorch` and the next 128 MiB block failed.
* MTP-on adds the 3.756 GB MTP layer plus `num_draft` recurrent-history slots (~2.15 GiB each), so
  it cannot fit either.

### Model A proxy on the Strix (user-directed)

Model A (`NicSir`, 2.72 bpw, 113.94 GB) fits alone. Same `tools/glm/serve.py`, same MTP head, same
engine; raw `/v1/completions`, one fixed **186-token prompt**, greedy, `max_tokens=200`,
`--no-vision -c 8192`. No unquantized MTP `eh_proj` sidecar is present in this pack.

| run | decode tok/s | prefill tok/s | accepted / drafts | acceptance | rounds |
|---|---|---|---|---|---|
| MTP off (`--num-draft 0`), warm | **19.85** | 204.4 | — | — | — |
| MTP on (`--num-draft 2`), first run, no startup warm-up | 21.12 | 53.4 | 104 / 192 | 54.2% | 96 |
| MTP on (`--num-draft 2`), warm | **23.99** | 217.4 | 104 / 192 | 54.2% | 96 |

**`g` = 23.99 / 19.85 = 1.209** (or 1.204 against the swp-exl3 MTP-off baseline). This narrowly
passes the plan's `g >= 1.2` gate, with two caveats: it is Model A, not Model B, and the missing
`eh_proj` sidecar depresses acceptance (the Kyojin notes measure +0.077 acceptance and ~+4.8% decode
with the sidecar). The measured `g` should therefore be treated as a **lower bound**.

MTP-on at `-c 8192` could only start with `EXL3_SERVE_WARMUP=0`: the default 4608-token startup
warm-up OOMed (`113.12 GiB` allocated, `4.45 GiB` free, next 192 MiB failed). The warm row above is
after a first 200-token request; the first-run row is the un-warmed one.

### Model A through swp-exl3, entirely on the Strix

One stage holds `[0,45)` with `embed=true, head=true`; the coordinator runs on the same box over
loopback (`127.0.0.1:7803`, insecure, f32 wire, chunk 512). The stage was started with serve.py's
full `SPEED_ENV` set, except that the adapter's `EXL3_PF_GR_FUSE=0` default remains. Same prompt,
greedy, `--max-new 200`.

| run | decode tok/s | prefill tok/s | stage compute ms/tok | wire+q ms/tok | busy |
|---|---|---|---|---|---|
| swp-exl3, first run (cold kernels) | 19.77 | 77.6 | 50.384 | 0.187 | 100% |
| swp-exl3, warm | **19.93** | **217.7** | **49.988** | **0.171** | 100% |
| native serve.py, MTP off, warm | 19.85 | 204.4 | ~50.4 | n/a | n/a |

**swp-exl3 adds no measurable slowdown on loopback**: decode is +0.4% vs native MTP-off (within
noise), warm prefill is +6.5% (217.7 vs 204.4), and adapter/wire cost is **0.17 ms/token**. Greedy
text matches the native run. The ~4.2 ms/token overhead in the two-machine pipeline is therefore
the remote boundary transfer/queue, not the adapter itself.

## Task 3 — stage warm-up

Improvement for the **first prefill after a stage starts**, not for steady-state speed.

- First real prefill (149 tokens): **~1.29 s before → 844 ms now**, versus 855 ms warm. Previously first/warm was ~1.55x (560.5 vs 114.0 ms on Razer stage 0); now **0.99x**, with a second run at **1.06x** — inside the ≤1.20x target.
- Warm decode/prefill are unchanged: 15.08 tok/s decode, 174 tok/s prefill (baseline noise).
- Cost: **1.1 s on Razer / 3.4 s on Strix** added once at stage startup. Worth it for long-lived stages; smoke is now ~17 s instead of ~13 s.

* After `load()`, each stage runs 7 synthetic forwards: decode rows `n=1` ×3, then prefill rows
  `16, 64, 256, 512`, at cumulative positions (zero boundary tensors on non-first stages), then
  `reset()`. Config `[stage] warmup = true` by default; `warmup = false` disables it. The stage logs
  `[warmup] 7 synthetic forwards in N ms (per-forward list)`.
* Fresh pair, Model B cut 7 — warm-up duration: **Strix 3414 ms**
  (`[1208.6, 53.6, 52.4, 398.7, 405.4, 446.7, 841.1]`), **Razer 1075 ms**
  (`[569.8, 6.4, 5.4, 65.5, 148.0, 92.4, 187.5]`). The first synthetic forward dominates
  (lazy/JIT init); the rest are the prescribed sizes.
* **Acceptance:** first real prefill (149-token smoke prompt) **844 ms → 176.6 tok/s**, warm repeat
  **855 ms → 174.2 tok/s**, i.e. **first/warm = 0.99x** (target ≤ 1.20x). A second fresh-stage run
  measured 1.06x. Without warm-up the same stage-0 first prefill was 560.5 ms vs 114.0 ms warm
  (see "Other findings" below).
* Warm decode is unchanged: 15.08 tok/s; stage 0 6.855 ms/tok, stage 1 54.965 ms/tok, adapter+wire
  overhead 4.49 ms/tok. Smoke run time is now ~17 s (it also repeats the 149-token prompt to print
  the first/warm line; meaningful only when the stages were just started).
* Unit tests: **21 tests, OK** (1 skips without `SWP_WEIGHTD`); new `tests/test_warmup.py` covers the
  prescribed rows, zero boundary tensors, reset/clean state, the small-cache guard, and config default.

## Task 5 — engine env flags

Rows 6–8 above. All three runs bit-identical to the golden (top-1 equal, 64/64 shared,
max |Δlogit| 0.00000, KL 0.00e+00).

* **`EXL3_HC_FOLD` — already answered; nothing new to test.** The flag exists *only* in the upstream
  (Razer, CUDA) engine (`util/backend.py:56`); **Kyojin 1.5.0 has no `EXL3_HC_FOLD`** at all, so there
  is no second engine or stage to try it on. It was tested on the Razer in Task 1 (rows 3–4:
  15.13 vs 15.31 tok/s, stage0 7.011 vs 6.834 ms/tok) and is performance-neutral and bit-identical.
  Left at `1` in the Razer `[stage.env]`.
* **`EXL3_BLOCK_GRAPH=1`** (ROCm/Kyojin-only, `modules/block_graph.py`) is **neutral, marginally
  negative** on the Strix: decode 15.05 vs 15.05 tok/s, stage1 55.255 vs 54.794 ms/tok. It is not
  inert — a `EXL3_BLOCK_GRAPH_DEBUG=1` run shows **28 of the 38 layer-blocks capture and replay**
  (layers 8,9,10,12,…,42,44 — every GDN+MoE block); the 10 attention layers (7,11,15,19,23,27,31,35,39,43
  = `cache_layers=10`) decline. Capturing the majority of blocks changes decode by nothing, so **launch
  overhead is not the Strix bottleneck** — its 38 layers are genuinely compute-bound (~1.44 ms/layer
  against the Razer's ~1.09 ms/layer over 7 layers).
* **`EXL3_BLOCK_GRAPH_MLA=2`** (the documented extension, capturing the MLA/DSA post-segments) captures
  nothing extra here — the attention layers still decline — and is likewise neutral (15.02 tok/s).
* **Kept: nothing.** Block graph stays off; `EXL3_HC_FOLD` unchanged.

## Task 7 — prefill chunk size

1576-token prompt (multi-chunk), Model B cut 7, f32, **warm** stages; two warm runs each, plus a
3152-token robustness run at 2048.

| chunk | warm prefill tok/s | stage0 prefill ms | stage1 prefill ms | first-run (cold) prefill | decode tok/s |
|---|---|---|---|---|---|
| 512 | 312.6 / 312.6 | 662–669 | 4145–4149 | 6479 ms | 14.2–14.4 |
| 1024 | 397.8 / 398.7 | 577–581 | 3153–3161 | 4318 ms | 14.3–14.5 |
| **2048** | **472.3 / 472.7** | 535–537 | 2575–2580 | 9123 ms | 14.2–14.4 |

* Bigger chunks cut the **Strix's** prefill compute **38%** (4145 → 2575 ms); wall prefill goes
  **313 → 399 → 473 tok/s**. Decode is unchanged (n=1, so `chunk` does not touch it) at ~14.4 tok/s.
* Robustness: chunk 2048 handled a **3152-token** prompt (two full 2048-row chunks) with **no OOM** on
  the Razer (3.18 GB free); warm 455 tok/s.
* Caveat (first call only): the Task 3 stage warm-up covers rows ≤ 512, so the first real prefill at a
  larger chunk pays a one-time shape autotune on the stage (1.5k prompt: 6.5 s at 512 → 9.1 s at 2048;
  3.1k prompt at 2048: 19.1 s). Paid once per stage start; amortised for a long-lived stage.
* **Kept: `chunk = 2048`** in `~/models/pipeline-310-k7.toml` (was 512). Requires `max_chunk ≥ 2048`
  on both stages — set in `~/models/stage-310-razer-k7.toml` and `/home/m/swp/stage-310-strix-base.toml`.
  The Task 1 golden is unaffected (all smoke prompts are a single chunk).

## Task 4 — decode overhead (4.2–4.5 ms/token)

Measured with the real pair, 20 decode tokens per run, `SWP_DEBUG_TIMING=1`, and a raw RPC probe
that records each hop's timing dict (`/tmp/kilo/t4_probe.py`, not part of the repo).

### Where the time actually is

| stage | `prepare`+H2D | `forward_ls` (enqueue) | `advance` (incl. final sync) | D2H | RTT − compute |
|---|---|---|---|---|---|
| 0 (Razer, 7 layers) | 0.15–0.42 | 3.2–4.8 | 2.3–3.2 | 0.10–0.17 | **1.37** |
| 1 (Strix, 38 layers) | 0.62 | 12.2 | 42.7 | 0.09 | **3.08** |

* The plan's "anything outside `forward_ls` is adapter overhead" is **not** what the timings mean:
  `forward_ls` only *enqueues* kernels. The real GPU wait is in the `cuda.synchronize()` at the end
  of `advance`, so `advance_ms` (42.7 on the Strix) is the layer compute, not adapter overhead.
  The adapter cost is `prepare`/H2D + D2H + framing; everything else is the two stages' compute.
* Stage 0's 1.37 ms is loopback framing + prepare/D2H. Stage 1's 3.08 ms decomposes as
  prepare+H2D 0.62 + D2H 0.09 + **~2.3 ms of delivery/wake**. A 64 KiB `PING` round trip is 0.354 ms,
  so framing is not the cost.

### Fixes

* **(a) stage 0 in the coordinator process** — new `stage_config` key in `[[stage]]`; `LocalPeer`
  calls `StageServer.handle()` directly, no socket/framing. Stage 0 wire+adapter drops
  **1.35–1.60 → 0.36–0.73 ms**. `smoke` is bit-identical (top-1 equal, 64/64, Δlogit 0, KL 0).
* **(b) one pinned D2H** — preallocated pinned staging buffer, byte view returned without
  `.cpu().numpy().tobytes()` (one copy instead of three).
* **(c) one `sendmsg` per frame** — header + payload as iovecs, short-write loop only as fallback.
* (b) and (c) are neutral within run-to-run noise on this pair (the boundary is only 64 KiB), but
  they remove copies and stay.
* (d) was already done (argmax/top-k on the device; only ids cross the wire).

### Result

| # | change | decode tok/s | stage0 ms/tok | stage1 ms/tok | overhead ms/tok | smoke |
|---|---|---|---|---|---|---|
| 9 | Task 4 control: remote stage 0, one `sendmsg` + pinned D2H | ≈15.1 (20 tok) | 6.85 | 54.35 | 4.43 (1.35 + 3.18) | — |
| 10 | Task 4: stage 0 in-process (`stage_config`) | 15.67 | 6.928 | 54.607 | **2.14** (0.61 + 1.53) | PASS |
| 11 | Task 4: + Strix kept out of deep idle (external busy core) | 15.75 | 6.897 | 54.409 | **2.02** (0.57 + 1.44) | PASS |

Rows 9–11: 149-token prompt, 48 greedy tokens except row 9 (20 tokens; its decode rate is derived
from the stage sums because the DECODE line was not captured). Row 10 is the same
`pipeline-310-k7-local.toml` as the final smoke: prefill 175.8 tok/s, first/warm 0.97x.

**The remaining stage-1 cost is the Strix's idle power state, not the adapter.** With one core kept
busy on the Strix (or right after sustained traffic) stage 1's overhead is 1.40–1.53 ms; after the
package has been idle it is 2.8–3.1 ms — a reproducible ~1.5 ms wake penalty on every token. A busy
core on the Razer does not remove it (2.82 ms); a busy core on the Strix does (1.40 ms). The Strix
(AMD Ryzen AI Max+ 395) runs `acpi_idle` under the `powersave` governor: POLL/C1/C2/C3, C3 latency
**350 µs**, and C3 holds ~99.8% of idle residency. `/dev/cpu_dma_latency` is root-only there, and
there was no sudo in this session, so no platform setting was changed. For a latency-critical stage,
disable deep C-states (or keep one core busy) on the Strix; with a warm package, total decoder
overhead is ~2.0 ms/token. `STRIX-IDLE-WAKE.md` covers the options, costs and verification.

The final smoke (local stage 0, Strix back in its slow state) measured 15.47 tok/s with stage 0
0.36 ms and stage 1 3.04 ms, showing the same bimodality.

## Other findings

* **First-forward warm-up is real even with kernels cached.** A freshly loaded Razer stage spent
  560.5 ms of compute on the first 149-token prefill vs 114.0 ms on the next run (stage 0 only;
  stage 1 was 727.8 vs 717.3 ms). This is the Task 3 target.
* **Decode overhead is 4.2–4.5 ms/token** at this cut, matching the plan's ~4.6 ms; it is in the
  adapter/wire, not in either stage's compute. Task 4 splits it: ~0.7–0.8 ms is the remote stage-0
  socket (removed by `stage_config`), ~0.7 ms is prepare/H2D/D2H, and the rest is the Strix's
  idle-wake penalty (see Task 4; ~2.0 ms/token total with stage 0 in-process and the Strix awake).
* Running Model A fully on the Strix via swp-exl3 uses **108.99 GB (17.53 GB free)**; MTP on adds
  ~5 GB (drafter + history) and leaves no room for the default 4608-token startup warm-up.

## State after these measurements

All stage/serve processes are stopped; the Strix is idle (122 GB free), the Razer GPU is free.
The remote `/home/m/swp/swp-exl3` tree was synced to the current local tree for the single-stage run.
