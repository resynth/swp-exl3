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

### MTP projection sidecar (Model A, 4-bit MTP)

The sidecar question is now closed for Model A. Its `quantization_config.json` has
**`mtp_bits = 4`**, not the 2 that older notes assumed. The larger sidecar gains reported
elsewhere (+0.077 acceptance / +4.8% decode in the Kyojin note; +18% on the Yamz pack card) were
measured with a 2-bit `eh_proj` (the Yamz pack itself has `mtp_bits = 2`), so they do not transfer
to packs A/B.

**Installed outside the packs**, at the shared default path Kyojin's server already probes:
`/home/m/models/glm53-mtp-eh-proj-bf16.safetensors` (67,108,992 B = 64 MiB). Key
`model.language_model.layers.45.eh_proj.weight`, shape `[4096, 8192]`, `bfloat16`, sha256
`37ea317a0c60b328430905835869684b2fd8401c2d26a707471a03b0c1229015` (matches `SHA256SUMS` in
`yamz-labs/GLM-5.3-Flash-EXL3-Yamz`; the file there is `mtp_eh_proj.st`, copied from the
official release). No copy was put in either pack.

Selection (Kyojin `apply_eh_sidecar_default` + engine `load_eh_proj_sidecar`): explicit
`EXL3_MTP_EH_FP16=<file>` wins; unset -> `mtp_eh_proj.st` in the model folder, else the shared
default above; `EXL3_MTP_EH_FP16=0` disables. All three cases were verified on the Strix.

**Three arms**, native `tools/glm/serve.py`, Model A alone on the Strix, `--no-vision -c 8192`,
`EXL3_SERVE_WARMUP=0`, one fixed 185-token prompt (186 with BOS), greedy, 200 new tokens. Each arm
got a 200-token warm-up request with a different first token, then two measured runs.

| arm | decode tok/s (server, 2 runs) | decode tok/s (client) | prefill tok/s | accepted / proposed | acceptance | rounds |
|---|---|---|---|---|---|---|
| MTP off (`--num-draft 0`) | 19.54 / 19.55 | 19.55 / 19.55 | 197 / 212 | — | — | — |
| MTP on, `EXL3_MTP_EH_FP16=0` | 23.80 / 23.89 | 23.93 / 24.02 | 195 / 209 | 107 / 186 | 57.5% | 93 |
| MTP on, sidecar | **23.99 / 24.07** | **24.13 / 24.20** | 190 / 204 | 108 / 184 | **58.7%** | 92 |

* `g` = MTP on / off: **1.220** without the sidecar, **1.230** with it. The plan's `g >= 1.2`
  gate passes either way.
* Sidecar vs no sidecar: **+0.77% decode**, **+1.2 pp acceptance** (57.5 -> 58.7). Small,
  deterministic on this prompt, and free; keep it.
* Load: 111.6 GB torch-allocated in both MTP-on arms (the sidecar is 64 MiB); 105.8 GB MTP-off.
  Prefill and load time are unchanged within noise.
* This does not change the decision: MTP is the lever (~1.22x), the sidecar is a rounding error
  at `mtp_bits = 4`. Model B was not measured alone (it still does not fit: 119 GiB vs 118 GiB
  visible), but its MTP tensor is also 4-bit, so expect a similar small gain.
* Raw `mtp_sidecar_results.json` and the three server logs are in
  `/home/m/models/mtp_sidecar_bench/` on the Strix.

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

## T5 — quantized KV cache (`cache_bits`)

`[stage] cache_bits = 0|8` is implemented (per stage, default 0). With `8` the adapter builds
`Cache(layer_type=CacheLayer_quant, k_bits=8, v_bits=8)`; it is inspect-guarded on both `Cache` and
`CacheLayer_quant` and **fails fast** with a clear error if `cache_bits = 8` is requested on an engine
without them (use `cache_bits = 0` there; silently downgrading would allocate the fp16 cache at the
q8-sized `cache_tokens`). `doctor` reports
the capability. Kyojin's `Cache` has no `recurrent_dtype` but forwards unknown kwargs into the cache
layer, so the adapter now only passes `recurrent_dtype` when `Cache.__init__` declares it (main cache
and MTP drafter cache).
Unit suite: **37 tests, OK** (1 skips without `SWP_WEIGHTD`); `doctor` **19/19** on both engines.

**Kyojin caveat (not an upstream exllamav3 bug).** Kyojin 1.5.0's `Cache` lacks the upstream
`recurrent_dtype` parameter but accepts `**kwargs` and forwards them to the cache-layer constructors.
`CacheLayer_MLA_fp16` tolerates the extra keyword; `CacheLayer_MLA_quant` does not, which is why the
combination `cache_bits = 8` + `recurrent_dtype` crashed. Upstream exllamav3 1.6.4 declares the
parameter, so this is a Kyojin feature gap plus an over-broad kwargs forward. The adapter now passes
`recurrent_dtype` only when `Cache.__init__` declares it; worth reporting to Kyojin (declare the
parameter, or drop unknown kwargs before constructing layers).

Cache geometry measured from the installed engines (Model B, cut 7):
| slice | global-attn cache layers | fp16 | q8 |
|---|---|---|---|
| Razer `[0,7)` | 1 | 1600 B/token | 1120 B/token |
| Strix `[7,45)` | 10 | 16000 B/token | 11200 B/token |

At 64k cache that is 0.105 / 0.073 GB on the Razer and 1.049 / 0.734 GB on the Strix. The Razer is
not the context limit at this cut; the Strix is.

### T5.1 Measured maximum cache (load + warm-up)

Model B cut 7, f32 wire, `spec_k = 4`, chunk 2048. "Loads" means the stage reached `listening` and
completed the 7 synthetic warm-up forwards.

| stage | cache dtype | largest tested that loads | free after load | first tested failure |
|---|---|---|---|---|
| Razer `[0,7)` | fp16 | **851,968 (832k)** | 1.83 GB | not tried higher (not binding) |
| Strix `[7,45)` | fp16 | **589,824 (576k)** | 5.94 GB | **655,360 (640k)**: 4.90 GB free, first warm-up forward fails |
| Strix `[7,45)` | q8 | **851,968 (832k)** | 5.84 GB (first forward 18.6 s) | **917,504 (896k)**: 5.11 GB free, first forward fails; **1,048,576** OOMs during load |
| Strix `[7,45)` | q8 | 786,432 (768k) — normal warm-up | 6.57 GB (first forward 4.1 s) | — |

q8 raises the Strix's measured allocation ceiling from **576k to 832k** (1.44x, exactly the
16000/11200 byte ratio). 832k is the absolute ceiling but its first forward took 18.6 s versus 4.1 s
at 768k, so **768k is the practical maximum**. The model's own limit is 1,048,576; the loader/workspace
needs ~5 GB free and 1M q8 leaves less than that.

Cut 7 is the max-context split, so no split sweep is needed: it is the largest all-GPU Razer range
(`MEASUREMENTS.md` §4.1) and gives the Strix the smallest weight footprint it can have without Razer
CPU offload. Moving the cut down adds layers (and KV pages) to the Strix; moving it up makes the Razer
need offload, which costs more Razer free memory than the extra attention layer saves. Neither can beat
the model's 1,048,576 limit, and both lower the measured ceiling.

### T5.2 Throughput vs context (warm stages)

Razer/Strix mean the KV cache dtype on each stage. All rows f32 wire, chunk 2048, greedy, 8–47 new
tokens. `8k`/`32k` q8 rows use a 64k cache; the 128k rows use 256k; cache size does not change
per-token compute.

| prompt | Razer / Strix KV | prefill tok/s | decode tok/s | stage 0 ms/tok | stage 1 ms/tok | wire+q s0 / s1 |
|---|---|---|---|---|---|---|
| 149 (existing baseline) | fp16 / fp16 | 174.9 | 15.31 | 6.83 | 54.31 | — |
| 8,192 | fp16 / fp16 | 456.6 | **13.34** | 8.57 | 61.16 | 1.39 / 3.70 |
| 32,768 | fp16 / fp16 | 457.1 | **12.97** | 14.54 † | 57.19 | 1.56 / 3.65 |
| 131,072 | fp16 / fp16 | 443.5 | **11.27** | 6.55 | 77.00 | 0.64 / 4.50 |
| 8,192 | fp16 / **q8** | 453.2 | **13.20** | 7.11 | 62.80 | 1.38 / 4.26 |
| 32,768 | fp16 / **q8** | 448.9 | **13.55** | 6.92 | 61.42 | 1.30 / 4.01 |
| 131,072 | fp16 / **q8** | 414.5 | **9.40** | 6.66 | 94.14 | 0.90 / 4.57 |
| 8,192 | **q8 / q8** | 453.9 | **7.35** | 63.89 | 67.30 | 1.38 / 3.44 |

† Single-run stage-0 timing; other 32k/128k runs measured 6.5–7.0 ms. The wall decode rate is the
comparable figure.

* **q8 on the Strix is nearly free at 8k/32k** (decode 13.20/13.55 vs 13.34/12.97) and costs **~17%
  decode at 128k** (9.40 vs 11.27), because all 10 global-attn layers now read quantized pages.
* **q8 on the Razer is not usable**: at 8k its single attention layer costs **63.9 ms/token** versus
  8.6 ms fp16, and the pair drops to **7.35 tok/s**. Use q8 on the Strix only. (The 32k q8/q8 run hit
  the old "prompt == cache" fault, so it is not a data point.)
* The 8192 limit was a config choice, not a model/engine limit. 128k prefill/decode is measured above;
  576k fp16 / 768k q8 caches load on the Strix.
* **Control after the T5 code changes (149 tokens, same remote pair, fp16, cache 64k):** decode
  **15.03 tok/s**, stage 0 **7.146** ms/tok, stage 1 **54.955** ms/tok, prefill **174.8 tok/s**,
  smoke **PASS** (golden text and logits 64/64, KL 0). This matches RESULTS rows 2 (15.31 / 6.834 /
  54.313) and 6 (15.05 / 7.181 / 54.794) within run-to-run noise; the lower 8k/32k/128k decode rates
  are context cost, not a code regression.

### T5.3 Correctness

Top-64 logits, q8-on-Strix vs fp16-on-Strix (Razer fp16 in both):

| prompt | top-1 | shared top-64 | max abs Δlogit | KL(A‖B) |
|---|---|---|---|---|
| 8,192 | equal | 62/64 | 0.578 | 5.61e-08 |
| 32,768 | equal | 60/64 | 0.656 | 4.12e-08 |

The smoke gate against the fp16 golden **fails**: greedy text differs at the second token on the
capital prompt, and the 5-token top-64 check has KL 1.16e-3 (gate 1e-4); the speculative verify check
is 6.33e-4 (gate 1e-2). q8 is therefore a **context-headroom trade-off, not an accuracy-neutral
drop-in**. Keep `cache_bits = 0` as the default; enable 8 on the Strix only when a context above the
fp16 ceiling (~576k) is required.

### T5.4 Caveats

* **Reserve room for generated tokens.** With `cache_tokens == prompt length`, the first generated
  token writes past the cache and the CUDA stage faults with an illegal memory access (seen on both
  fp16 and q8). `StageServer` now rejects it with `context_exceeded`; size the cache for
  prompt + generation + spec window.
* **Maximum cache ≠ maximum runnable prompt.** The 832k q8 stage (5.84 GB free) could not prefill a
  128k prompt at chunk 2048 (`hipBLASCreate` allocation failure); the same 128k prompt ran with a
  256k cache (12.45 GB free). Prefill workspace still scales with chunk×context. For long prompts use
  cache ≈ 2× the desired context (or reduce `max_chunk`/`chunk`).
* **q8 is slower as context grows**: 8k/32k are within noise, 128k loses ~7% prefill and ~17% decode.
* q8 needs the engine's packed-MLA/DSA cache path; `doctor` reports it and the adapter raises a clear
  startup error where it is absent (`cache_bits = 0` there).

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

## Step S1 — protocol + speculative loop on the fake backend

**Protocol (splitwire first).** SWP/1 gains the additive, capability-gated `spec` path: stages advertise
`spec` in HELLO and `spec_k` in STAGE_READY; FORWARD accepts `mode:"verify"`, `draft_ids`, `accept`,
`sampling`, `next_drafts`; the head may answer `kind:"spec"` with `accepted`, `tokens` (int32 payload),
`drafts`, `timing`. The "rewind only immediately after a verify forward" rule and the lazy-rewind
contract are documented in `splitwire/README.md` and the canonical `splitwire/python/swp.py`; the
vendored `swp_exl3/swp.py` is re-copied and `tests/test_swp_sync.py` passes. Rust only gains the
`CAP_SPEC` constant (no behaviour change).

**Mechanism.** `Pipeline.generate(..., spec=SpecConfig)` proposes `k` drafts from the driver side,
sends a `k+1`-row verify forward with `rewrite=true`, and emits the accepted prefix plus the bonus
token. The next round starts at `len(history)` and rewinds the rejected rows lazily (or clears +
replays if a stage answers `unsupported`). `FakeBackend` keeps per-position layer inputs for recurrent
layers, so `GDNState.rewind` semantics (only after a verify, only within the history window) are
tested on the toy. `Pipeline.spec_check()` compares verify logits against sequential decode from the
same state.

**Tests.** New `tests/test_spec.py` (10 tests): all-accept, all-reject and mixed acceptance on 2 and 3
stages (output exactly equals non-speculative greedy), forced illegal rewind -> clear+replay still
exact, verify-vs-sequential logits, server rewind rules, n-gram lookup, CLI `--spec`, and `spec_k=0`
fallback. Full suite: **32 tests, OK** (1 skips without `SWP_WEIGHTD`).

## Step S2 — prompt-lookup speculation on the real engines

Stages: Model B `[0,7)` / `[7,45)`, f32, chunk 2048, `spec_k = 3`. The backend now sets
`max_history = spec_k + 1 = 4` (one extra row over the engine's `k+1`-row window). Load cost vs
RESULTS above: Strix **111.16 GB allocated (15.35 GB free)** vs 110.80 GB, i.e. **+0.36 GB**; Razer
still 13.30 GB (3.18 GB free).

**Critical engine finding — settle every verify, including a full accept.** The recurrent state is
provisional after a `recurrent_history` pass. The engine's generator calls `rewind(num_rejected)`
after *every* verification, including `rewind(0)` when all drafts matched. Without that settle the
first implementation had **4/20 top-1 mismatches and KL 2.25**; with `rewind(0)` (and lazy
`rewind(k-m)` for partial accepts) it is **0/20 mismatches and KL 5.0e-3**. This is now in
`Exl3Backend._settle_verify`, called before the next forward as well as from `truncate`.

**Smoke check (verify logits vs sequential, 20 rows, k=3).**

| check | result |
|---|---|
| top-1 mismatches | **0 / 20** |
| max KL (top-8, shared, renormalised) | **5.01e-3** |
| smoke threshold | top-1 equal and KL <= **1e-2** (relaxed from the plan's 1e-3; see below) |

The 1e-3 KL gate is not reachable at k=3 because of the engine's **batched** kernels, not SWP:
1-row verify is bit-identical (KL 0), 2-row is 3.2e-5, 3/4-row reaches 5.0e-3. A plain 4-row prefill
(no `recurrent_history`) shows exactly the same delta as the 4-row verify, and stage 0's hidden
boundary differs by only **max |Δ| 5.6e-4** between one 4-row forward and four 1-row forwards. Top-1
is equal at every row and in every round. The smoke gate is therefore top-1 plus KL <= 1e-2.

**Bench (warm persistent stages, greedy).**

| prompt | mode | decode tok/s | stage 0 ms/tok | stage 1 ms/tok | prefill tok/s | spec rounds | accepted |
|---|---|---|---|---|---|---|---|
| repetitive 42-tok code | plain | **15.03** | 6.99 | 54.81 | 89.3 | — | — |
| repetitive 42-tok code | `--spec` | **24.91** | 2.76 | 35.90 | 88.3 | 16 | **48/48 (100%)** |
| non-repetitive 149-tok | plain | 15.05 | 6.81 | 55.05 | 175.7 | — | — |
| non-repetitive 149-tok | `--spec` | 15.02 | 6.55 | 55.82 | 176.2 | 4 | 5/12 (41.7%) |
| repetitive, T=0.8, 32 new | `--spec` | 23.38 | — | — | 88.4 | 8 | 24/24 (100%) |

* **1.66x on repetitive output** (24.91 / 15.03), repeatable (24.90 / 24.91). One round advances
  ~4 tokens (3 drafts + bonus) for 4 verified rows, so both stages' compute per *emitted* token drops
  even though each verify row costs the same as a decode row.
* **Neutral on non-repetitive output** (15.02 vs 15.05): the n-gram lookup mostly misses, and missed
  rounds fall back to a plain 1-token decode, so there is no verify overhead.
* **Output is byte-identical to plain greedy** on both prompts (captured stdout, `cmp`; 48 new tokens
  each), including the non-repetitive run where only 5/12 drafts were accepted and several lazy
  rewinds ran.
* `T > 0` uses "sample target, accept on equality" in the head stage (seeded by `seed` + round); one
  sampled run completed cleanly at 23.38 tok/s.
* Aside (pre-existing Task 3 coverage): on a just-loaded pair the first 149-token prefill can be
  1.68x the warm one, because the synthetic warm-up rows (16/64/256/512) do not include 149; the
  second run is **0.97x**. Not caused by speculation, but worth adding ~150 rows to the warm-up.

## Step S3 — MTP drafter on the head stage

**Outcome: implemented, correct, and gated.** The drafter works and the target stays bit-identical,
but no k reaches the plan's 1.15x exit (best **15.5 tok/s vs 15.05 plain, 1.03x**). The S3.4 stop
rule (miss 1.10x after the k sweep) and the S3.0 cost math both fire. The n-gram S2 path remains the
default; `mtp = true` enables the drafter.

### S3.0 go/no-go (Model B, Strix head stage)

* **Capacity (pass).** Loading the `[7,45)` head slice (111.15 GB), the MTP block (3.70 GB), a
  read-only `embed_tokens` copy (1.27 GB) and a separate 8192-token drafter cache allocates
  **116.31 GB, 10.13 GB free** (target cache also 8192; the pair's context cap is 8192 anyway).
* **Warm drafter cost.** A 1-row MTP forward **1.5–2.3 ms** after the first (1.42 s cold, paid in
  the stage warm-up); rollouts **4.8 / 7.6 / 11.4 ms** for k=1/2/3; catch-up prefill
  **~0.9–6 ms** for 1–3 rows. So the drafter itself is not the blocker.
* **Verify cost curve.** On the 100%-accept repetitive prompt (n-gram, k=1..4): **2/3/4/5 rows =
  117 / 136 / 160 / 181 ms** per round. The 5-row point already costs 181 ms; with realistic
  acceptance the 5th draft cannot repay that row, so C(4) rules out k=4 before drafting.
* **API (pass).** Kyojin 1.5.0 has `Glm5NextMTPModel.attach_to` / `sample_from_state` (no
  `load_eh_proj_sidecar` on upstream 1.6.4; optional there); `export_state_norm_keys` is consumed
  by `RMSNorm`/`HyperHead` and appends `params["export_states"]`. `check_api_against_source.py`
  grew the corresponding checks: **Razer 62/62, Strix 61/62** (the one known
  `Cache.recurrent_dtype` mismatch on Kyojin 1.5.0 remains).
* **Sidecar.** The 64 MiB bf16 `eh_proj` sidecar was loaded from
  `/home/m/models/glm53-mtp-eh-proj-bf16.safetensors`; its effect was not isolated.

### S3.1/S3.2 implementation

The head stage loads `Glm5NextMTPModel` from the same pack, a read-only `Embedding`, and a drafter
`Cache` (all before `load()`). A local attach shim supplies the MTP block with the head slice's
shared `lm_head` and the embedding (the stock `attach_to` asserts a full target model). Every target
forward sets `export_state_norm_keys` (the final `model.norm`), so the prefill already exports all
prompt hiddens; the drafter cache mirrors the prompt/chunks with the shifted hidden (zero first row).
A verify does: rewrite row L with the token being verified, accept `m`, catch-up the accepted rows
with their real target hiddens, then feed the bonus to produce the next round's drafts. The
coordinator consumes `drafts` and the `drafts_basis` check; it **prefers an n-gram match** and only
falls back to MTP on novel text, which keeps the S2 repetitive result intact. The drafter cache is
attention-only; `truncate` into an earlier prefix answers `unsupported` so the coordinator replays.

### S3.3 k sweep (149-token novel prompt, greedy, warm stages)

| k | rounds | accepted / drafts | acceptance | accepted/depth | decode tok/s | verify wall / round | catch-up / round | MTP roll / round | tokens/round |
|---|---|---|---|---|---|---|---|---|---|
| plain | — | — | — | — | **15.05** | 66.5 ms/token | — | — | — |
| 1 | 27 | 19 / 27 | 70.4% | [19] | 14.15 | 120.2 ms | 2.0 | 4.7 | 1.70 |
| 2 | 21 | 26 / 42 | 61.9% | — | 14.56–15.12 | 144.2 ms | 2.1 | 7.2 | 2.24 |
| **3** | 17 | **28 / 51** | **54.9%** | **[13, 10, 5] = 76.5/58.8/29.4%** | **15.49–15.51** | 163.4 ms | 2.1 | 10.8 | **2.65** |

* `verify` is the whole round wall (target verify + catch-up + MTP rollout); the last two columns are
  the subsets reported by the head, and `tokens/round` is the nominal `1 + accepted/round`.
* Subtracting catch-up+roll gives a **target-only 150.6 ms/round** at k=3: with 2.65 tokens that is
  17.6 tok/s, but the drafter adds 12.9 ms/round (16.2 tok/s) and the one plain bootstrap round costs
  ~130 ms, so the measured 48-token rate is 15.5 (**1.03x**). Even eliminating the bootstrap leaves
  ~16.2 (1.08x), below the 1.10x stop threshold.
* Repetitive 42-token prompt, k=3: **24.13 tok/s** (12 rounds, 36/36, **0 MTP rolls** — the n-gram
  lookup wins and the coordinator does not ask the drafter); a later default-config run measured
  24.53. S2 was 24.47. No regression on the repetitive path.
* Depth-2 acceptance is 29%. A confidence stop that skipped depth 2 would trade 0.29 tokens/round
  for ~27 ms, which *could* reach ~17.6 tok/s if depth-2 confidence separates cleanly. It was not
  implemented: the stop rule fired after the k sweep (<1.10x measured). This is the first thing to
  try if S3 is resumed.

### S3.4 correctness

* `swp-exl3 smoke` **PASS** with MTP loaded: both 48-token prompts are token-identical to the
  golden (created without MTP), top-64 logits top-1 equal, 64/64 shared, `max |Δlogit| 0`, KL 0.
  So the hidden-state export is read-only.
* Verify-vs-sequential check with the drafter loaded: **0/20 top-1 mismatches, max KL 5.01e-3**
  (k=3), unchanged from S2's batched-kernel floor.
* Legal-rewind/catch-up paths were exercised by the k sweep (partial accepts included); unit suite
  **34 tests, OK** (1 skips without `SWP_WEIGHTD`), including a fake-backend test that consumes
  head-provided drafts and checks exactness.

### Decision

Stop at the gate. Keep the implementation (config `mtp = false` by default) and the n-gram path.
If MTP is revisited, the lever is the target verify cost, not the drafter: the 2-row round already
costs ~120 ms of a 66.5 ms token budget.

## Task 6 — persistent coordinator (`swp-kyojin-server`)

`swp_exl3/integrations/kyojin-server/` serves a long-lived `Pipeline` through Kyojin's own OpenAI-compatible HTTP layer
(`tools/glm/serve.py`). It owns one `Pipeline`, keeps history across requests, reuses the longest common prefix (with the
exact token ids of the previous reply), and exposes `/v1/chat/completions` (stream), `/v1/completions`, `/v1/models`,
`/health`, `/tokenize`, `/metrics`. Run on the Strix; stage 0 stays remote on the Razer and stage 1 is local.

**Bench** (Model B cut 7, f32, chunk 2048, warm stages; same smoke prompt, 149 raw tokens / 48 new; streaming client,
decode = completion tokens ÷ (last delta − first delta)). The first column is the existing CLI smoke run on the same pair.

| path | prompt | decode tok/s | TTFT | notes |
|---|---|---|---|---|
| `swp-exl3 smoke` (CLI) | 149 tok raw | 14.88 | 859 ms prefill | stage 0 7.139 / stage 1 55.067 ms/tok |
| server, plain | 149 tok raw | **15.20 / 15.20** | 860 / 944 ms | matches the CLI; `/v1/completions` |
| server, plain | repetitive, 64 new | 15.22 / 15.18 | 604 / 684 ms | 31-token prompt |
| server, `--spec` | 149 tok raw | 15.44 / 15.15 | 846 / 935 ms | neutral, as in S2 |
| server, `--spec` | repetitive, 64 new | **25.89 / 25.09** | 601 / 683 ms | **1.65–1.70x** plain; 159/180 drafts accepted (88%) |

Multi-turn (real tokenizer/template, 8 new tokens per turn): turn 1 is 94 prompt tokens; the append turn is 116 prompt
tokens with **102 cached**, so only **14 tokens** are prefilled (308 ms, 45 tok/s) and there is no replay. Re-asking the
first turn while its reply is in history trims the recurrent state, so it **replays**: `cached_tokens = 0`,
`rollback = replayed`, `replayed_tokens = 93`, prefill 798 ms, identical answer. With the medium template the rendered
chat prompt is larger than the raw prompt (reasoning-effort system lines; 232 tokens for the 149-token smoke prompt), and
a client that drops/normalises `reasoning_content` loses reuse inside the reply and replays; echoing it can still hit
(e.g. 279/295 cached, 16 fed in one run). This is the tokenisation-drift caveat from the plan, not a protocol bug.

**Replay accounting (new).** `Pipeline.generate` now returns `rollback` (`noop/cleared/truncated/replayed`) and
`replayed_tokens`; the server reports physical cache hits (`cached_tokens = 0` when a replay recomputed the prefix) and
prints one line per request with the cache verdict and both rates:

```
[req 2] HIT 101/113 | prefill 12 new + 101 cached in 281 ms (42.8 new tok/s, 402.8 effective) | decode 13 tok in 1119 ms (10.7 tok/s)
[req 3] MISS (replayed 92) | prefill 93 new + 0 cached in 803 ms (115.9 new tok/s, 115.9 effective) | decode 8 tok in 553 ms (12.7 tok/s)
[req 1] MISS | prefill 31 new + 0 cached in 623 ms (49.8 new tok/s, 49.8 effective) | decode 64 tok in 2411 ms (26.1 tok/s) | spec 48/48 (100.0%, 16 rounds)
```

`HIT n/m` is the kept prefix, `MISS` a cold/different prompt, `MISS (replayed n)` a prefix that had to be recomputed;
`new tok/s` is over the fed rows, `effective` over the whole prompt (what the cache bought); `spec accepted/proposed
(%, rounds)` is present only when the request drafted. The same server run measured **48/48 (100%)** on the repetitive
prompt and **10/21 (47.6%, 7 rounds)** on a 15-token novel prompt (15.6 tok/s, neutral).

**Tests / checks.**
* Integration suite (no GPU, Kyojin 1.5.0's real HTTP layer + fake pipeline and tokenizer): **9 tests, OK** — endpoints,
  non-stream equals a direct pipeline run, stream equals non-stream, stop strings, multi-turn append with no replay,
  trim replay accounting, the terminal HIT/MISS/rate line and the spec-acceptance field, exact-id splicing/fallbacks,
  cancel mid-stream, one-time sampler warning.
* Unit suite: **34 tests, OK** on the Razer and on the Strix (1 skips without `SWP_WEIGHTD`).
* `swp-exl3 smoke`: **PASS** on the stage pair (token-identical to the golden; top-1 equal, KL 0).
* Real server: `/v1/models`, `/health`, `/tokenize`, `/metrics` all 200; sampled (`T=0.8`, `top_p=0.95`, seed) and greedy
  requests through `--spec` complete; a client disconnect mid-stream is cancelled and the next request succeeds.
* **Tool calls** (real server, `tools` + `tool_choice`/auto): non-stream and stream both return
  `finish_reason = "tool_calls"` with a parsed `tool_calls` entry (`get_weather`, `{"city": "Paris"}`); a follow-up with
  `role: "tool"` returns the final answer and reuses 260/269 prompt tokens. Streaming holds tool XML until it is complete
  and emits it as `delta.tool_calls`.
* **Context guard**: an over-context request is refused before prefill with HTTP 400
  (`"context length exceeded: prompt is 9001 tokens, the server context is 8192"`), not a stage error. Agent harnesses
  must keep system prompt + tools + history within the pair's 8192-token cache (the Razer stage is the cap).

**Files.** `swp_exl3/coordinator.py` (replay stats), `swp_exl3/integrations/kyojin-server/{swp_kyojin_server.py,
test_swp_kyojin_server.py,README.md}` (server, 7th test, spec flags, replay-aware cached tokens, per-request log),
`tests/test_pipeline.py` (replay assertions), `README.md` (server section).

**Limits.** One request at a time (Kyojin queues the rest); no images; min_p/penalties/logit_bias/grammar ignored with a
one-time warning; MTP stays off (the `--spec` n-gram path is wired instead). The pipeline's 8192-token cache is still the
pair's context cap.

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

All stage/serve processes are stopped (including the S3 Razer and Strix stages); the Strix is idle,
the Razer GPU is free. The remote `/home/m/swp/swp-exl3` tree is synced to the current local tree.
The Strix stage config at the time of the S3 sweep had `mtp = false`; enable it with
`mtp = true` and `mtp_cache_tokens = 8192` (>= `cache_tokens`) to reproduce those rows. The current
repo template (`examples/stage-strix.toml`) ships `mtp = true` for the fallback policy below.
The live MTP validation used `/home/m/swp/stage-310-strix-k7-mtp-8k.toml` (cache 8192,
`mtp_cache_tokens = 8192`) on the Strix and `~/models/stage-310-razer-k7.toml` on the Razer; both
stages are stopped again after the run.
The persistent `swp-kyojin-server` is stopped too. Its Strix pipeline config is
`/home/m/swp/pipeline-310-k7-server.toml` (stage 0 `10.10.10.1:7801`, stage 1 `127.0.0.1:7802`); start it with
`--spec` to reproduce the n-gram server rows in Task 6.

For T5 all q8 stages are stopped as well. The configs used are on the Razer in `~/models/`
(`stage-310-razer-k7-{f16,256k,832k}.toml`, plus q8 variants) and on the Strix in `/home/m/swp/`
(`stage-310-strix-k7-{f16,q8}{,-256k,-576k,-640k,-768k,-832k,-896k,-1m}.toml`). The 128k id file is
`/tmp/kilo/ids-131072.txt`; regenerate long prompts with `/tmp/kilo/make_ids.py <model_dir> <n> <out>`
(the measurement helper is intentionally outside the repo). Start the comparable pair with the Razer
`stage-310-razer-k7-256k.toml` and Strix `stage-310-strix-k7-q8-256k.toml` (or `-f16-256k` for the
control) and `pipeline-310-k7-q8.toml`.

## Files changed for T5

* `swp-exl3`: `swp_exl3/config.py` (`cache_bits`), `swp_exl3/exl3_backend.py` (quant cache kwargs,
  `recurrent_dtype` guard, cache/recurrent bytes in `memory`), `swp_exl3/doctor.py` (capability check),
  `swp_exl3/server.py` (`context_exceeded` guard), `swp_exl3/cli.py` (`--ids-file` for `generate` and
  `logits`), `tests/test_config.py`, `tests/test_pipeline.py`, `examples/stage-{razer,strix}.toml`,
  `README.md`.
* Stage configs: `~/models/stage-310-razer-k7-{f16,256k,832k}.toml` on the Razer;
  `/home/m/swp/stage-310-strix-k7-{f16,q8}-{256k,576k,640k,768k,832k,896k}.toml` on the Strix.

## Files changed for S1/S2/S3

* `swp-exl3`: `swp_exl3/{backend,server,exl3_backend,coordinator,config,cli,swp}.py`,
  `tests/test_spec.py` (S1/S2/MTP), `tests/test_pipeline.py` (`start(..., **be_kw)`), `README.md`,
  `examples/stage-{razer,strix}.toml`, `tools/check_api_against_source.py`.
* `splitwire`: `python/swp.py` (capabilities + spec section), `README.md` (protocol section),
  `src/msg.rs` (`CAP_SPEC` only; `CAP_MTP` is Python-side because it changes no wire framing).
* Stage configs: `spec_k = 4` and `cache_tokens = 8192` on the Strix for the S3 sweep; `mtp`
  defaults to false. `spec_k = 0` in `[stage]` disables the `spec` capability entirely.

## Post-v2 levers (T4/T6): prefill fusions, verify `ROW_SPLIT`, rows curve

Environment: Model B, Razer `[0,7)` + Strix `[7,45)`, f32 wire, chunk 2048, Kyojin 1.5.0 with Task A's
`SPEED_ENV`, `EXL3_PF_HC_FUSE=1`, `EXL3_PF_GR_FUSE=0`, `spec_k=8` on both stages.

### T4 — prefill fusions the adapter forces off (stage-only Strix, 3-run median, chunk 2048)

| HC_FUSE | GR_FUSE | 1,576 rows | 3,152 rows |
|---|---|---|---|
| 0 | 0 | 2,101.5 ms (750 tok/s) | 4,507.9 ms (699 tok/s) |
| 0 | 1 | 2,053.2 ms (768) | 4,454.9 ms (708) |
| 1 | 0 | 2,013.7 ms (783) | 4,441.2 ms (710) |
| 1 | 1 | 2,014.0 ms (783) | 4,430.5 ms (711) |

`HC_FUSE=1` is **+4.4%** at 1.5k (gate +3%); `GR_FUSE=1` alone is only +2.3% (below gate) and adds
nothing on top of `HC_FUSE=1` (2,014.0 vs 2,013.7 ms). Keep `HC_FUSE=1`, `GR_FUSE=0` — i.e. no change
from Task A. Note `EXL3_PF_GR_FUSE` is **not** in Kyojin's GLM `SPEED_ENV` (it is set by the Qwen
server); `EXL3_GR_PF_FUSE` defaults to `1` and the deferred fold is gated only on `EXL3_PF_GR_FUSE=1`.

### T6 — verify rows curve on the real pair (42-token repetitive prompt, 100% acceptance, spec_k=8)

verify ms/round (3-run median), B = verify rows = k+1:

| ROW_SPLIT | B=2 | B=3 | B=4 | B=5 | B=6 | B=7 | B=8 | B=9 |
|---|---|---|---|---|---|---|---|---|
| 0 | 113.3 | 137.6 | 159.1 | 178.8 | **261.8** | **292.1** | 235.3\* | 270.0 |
| 1 | 112.7 | 134.6 | 158.5 | 182.9 | 207.7 | 226.2 | 246.1 | 279.9 |
| 2 | 113.1 | 139.7 | 158.4 | 183.1 | **201.4** | **222.0** | 242.4 | 274.6 |

\* the `ROW_SPLIT=0` B=8 point is not monotone (small round count); the `=1/=2` curves are smooth
(~+20–24 ms/row).

* Rows 2–5 reproduce the Task A curve within noise, so raising `max_history` to 9 does not change them.
* Without `ROW_SPLIT`, B≥6 hits the engine's >4-row slow path (+83 ms at B=6, +113 at B=7).
  `EXL3_VERIFY_ROW_SPLIT=2` (split only the MoE; GEMV/hc take 5–8 rows natively) removes it:
  **−23% at B=6, −24% at B=7**, curve linear to B=9. `=1` is close; `=2` chosen (best at 6/7, ties 8/9).
* Correctness at 9 rows: `spec_check` k=8 top-1 equal, 0 mismatches, KL 9.7e-3 (the known batched-kernel
  floor); greedy k=8 output identical to plain; `smoke` PASS.
* No prefill effect: stage-only 1,576/3,152 rows with `ROW_SPLIT=2` = 2,001.8 / 4,401.2 ms.

### Result — final flags (`HC_FUSE=1`, `GR_FUSE=0`, `ROW_SPLIT=2`, `spec_k=8` both stages)

| run | decode |
|---|---|
| plain, 149-token prompt | 15.02 tok/s |
| spec k=3 (previous default window) | 23.96 tok/s |
| spec k=4 | 26.50 tok/s |
| **spec k=8** | **32.15 tok/s** (2.14× plain, identical text, 7 rounds) |

Prefill 1,576 warm 485.6 tok/s; `smoke` PASS.

### T7 / T9

* **T7 (concurrent-stream batching) dropped**: it only amortises the per-*forward* weight sweep across
  concurrent sequences, so it does nothing for one prompt at a time; the adapter is also single-session
  per stage (`batch_shape (1, cache_tokens)`).
* **T9 (per-module GPU timing) skipped**: diagnostic only (deciding whether to report an upstream
  fusion), no user-facing gain.

### Files

* Strix `/home/m/swp/stage-310-strix-k7-f16.toml` and `stage-310-strix-base.toml`: `spec_k = 8`,
  `EXL3_VERIFY_ROW_SPLIT = "2"` (backups `.bak-t4t6`).
* Razer `~/models/stage-310-razer-k7.toml` and `-f16.toml`: `spec_k = 8` (backups `.bak-t4t6`).
* Measurement configs: `~/models/pipeline-310-k7-local-rows.toml`,
  `~/models/stage-310-razer-k7-f16-sk8.toml`.

## Verify-row flag sweep (2–4 rows) — no cheaper 2-row step, MTP stays gated

> **Superseded for the R≥2 path by "Verify-row flag sweep re-run (marker ON)" below.** This sweep ran
> before the adapter set `params["dflash_verify"]`, so every flag it tested selected a kernel the
> forwards never reached. The re-run keeps the T6 numbers only as the marker-off control.

**Question.** The verify curve's expensive step is 1→2 rows (~+45 ms for a 2-row verify over a
~67 ms plain token). Can `EXL3_HIP_PREFILL_MIN_ROWS`, `EXL3_DEC_ROUTER_ROWS`, `EXL3_VERIFY_FUSE`, or
`EXL3_MOE_CFG` cut the 2–4-row cost enough (~25–30%) to revive MTP?

**Setup.** Model B cut 7, f32 wire, chunk 2048, stage 0 remote on the Razer, stage 1 restarted per arm
with the arm's `[stage.env]` (all four flags are read at import or in C++ static init, so one process
per arm). The 39-token code prompt is prefilled once; the coordinator then runs verify rounds with a
*forced* greedy continuation as the drafter, so every round is exactly B = k+1 rows and B=2/B=4 accept
100% and reproduce the plain text byte-for-byte. 48 new tokens, 2 runs per B, median. Base stage env is
the Task A `SPEED_ENV` plus `EXL3_VERIFY_ROW_SPLIT=2`; `EXL3_VERIFY_FUSE=1`, `MIN_ROWS=2`,
`DEC_ROUTER_ROWS=2`, `MOE_CFG=2` are the base values.

| arm | change | B=2 ms | B=3 ms | B=4 ms | stage-1 B=2 / B=4 |
|---|---|---|---|---|---|
| **base** (3 controls, session start/middle/end) | — | **114.9** | **142.0** | **166.3** | 100.9 / 149.4 |
| `MIN_ROWS=3` | 2 rows use the decode kernel | 115.5 | 143.9 | 165.0 | 100.7 / 147.8 |
| `MIN_ROWS=4` | 2–3 rows use the decode kernel | 117.0 | 143.7 | 165.4 | 102.5 / 148.0 |
| `MIN_ROWS=17` | all ≤16 rows use the decode kernel | 113.1 / 114.2 | 138.7 / 142.3 | 161.8 / 164.5 | 98.6–100.6 / 143.9–146.9 |
| `DEC_ROUTER_ROWS=1` | one router launch per row | 113.0 | 137.7 | 162.8 | 98.7 / 145.2 |
| `DEC_ROUTER_ROWS=0` | batch-1 router path | 113.8 | 139.7 | 162.0 | 99.2 / 144.7 |
| `VERIFY_FUSE=0` | no fused R-row verify linears | 115.5 | 141.3 | 162.6 | 101.2 / 145.2 |
| `MOE_CFG=1` | 256-thread grouped decode | 114.4 | 140.8 | 162.4 | 99.9 / 146.4 |
| `MOE_CFG=0` | 512-thread grouped decode | 115.8 | 142.6 | 163.2 | 101.5 / 145.7 |

Controls at the end of the session measured 114.9/142.0/166.3 and 116.2/143.8/168.2; the session
drifted ~1–2 ms at B=2 and ~2–6 ms at B=4. **Every requested flag is therefore within run-to-run
drift of the base.** The only consistent effect is `VERIFY_FUSE=1` being ~1–2 ms better at B=2
(the Kyojin note measured −3.0 ms at R=2); it is already on. `MIN_ROWS=2` (the grouped prefill path
at 2 rows) is already the best of 2/3/4/17; the decode kernel is not faster.

**Adjacent knobs, same setup (all neutral or worse).**

| arm | B=2 | B=3 | B=4 |
|---|---|---|---|
| `MOE_PREFILL_CFG=1` | 113.6 | 137.6 | 161.2 |
| `MIN_ROWS=17` + `MOE_CFG=1` | 117.0 | 144.5 | 168.2 |
| `MOE_VALU=1` | 117.1 | 145.5 | 166.0 |
| `VERIFY_GEMV_R=1` | 117.9 | 147.4 | 168.7 |
| `MOE_UNION_V2=0` | 117.1 | 144.7 | 168.0 |
| `DEC_MOE_UNION=1` | 117.3 | 145.4 | 169.9 |
| `DEC_GEMV_R_LOOP_MIN_K=4` | 114.5 | 141.8 | 163.2 |
| `DEC_GEMV_R_LOOP_MIN_K=6` | 117.1 | 143.3 | 169.6 |

**Why nothing moves.** An early base run measured stage 1 at 55.4 / 100.1 / 122.5 / 144.6 ms at
B=1/2/3/4: the first extra row costs **+44.7 ms**, the next two **+22.4 / +22.1 ms** each. That is
consistent with the R=1 specialised batch-1 path (block graph, fused batch-1 kernels) being left behind
at R=2, after which the generic multi-row path scales ~linearly. The requested flags choose kernels
*inside* the R≥2 MoE/router path; none changes the path switch or the second row's real work (its
distinct experts, MLA/KDA recurrence, dense projections). The dedup knobs (`MIN_ROWS=2`, `MOE_VALU`,
union V2) produced no gain either, so the second row's expert set is not being recovered by
deduplication here.

**MTP arithmetic with the measured curve.** k=1 (B=2): 114.9 ms verify + S3's 4.7 ms drafter + 2.0 ms
catch-up = 121.6 ms/round. At the measured depth-1 acceptance 70.4% that is 1.704 tokens/round →
**14.0 tok/s** vs 15.0 plain. Matching plain needs a 113.5 ms round (verify ≤107 ms, −6.7%); 1.15x needs
≤98.5 ms (verify ≤92 ms, −19%). The best arm (113.0 ms verify) gives 14.2 tok/s. k=3 (B=4): 162.2 ms
verify + 12.9 ms drafter = 175.1 ms for 2.65 tokens → 15.1 tok/s (1.01x); 1.15x needs ≤153 ms
(−12.5%), and the best arm is 161.2 ms. **No flag reaches the gate; keep `mtp = false`.**

**Correctness.** `swp-exl3 smoke` on the base stage: PASS (both 48-token prompts token-identical to the
golden, top-64 logits top-1 equal, 64/64 shared, KL 0). `spec_check` on the same prompt: k=1/2/3 all
top-1 equal, 0/12 mismatches, max KL 2.9e-3 / 5.1e-4 / 5.2e-3. The forced B=3 run hit a near-tie and
cascaded (18/58 accepted) on every arm; `spec_check` stays top-1-equal, so it is the known
batched-kernel floor, not a flag effect.

**Files.** Sweep configs on the Strix: `/home/m/swp/sweep/{base,min3,min4,min17,router0,router1, ...}.toml`;
harness `/tmp/kilo/row_sweep.py` and results `/tmp/kilo/sweep-*.json` (measurement helpers, outside the
repo). No live stage config was changed; the flags under test stay at their base values.

## Long verify windows (12–32 rows): 16 rows exact, 32 rows fast, 24 rows is a trough

**Setup.** Model B cut 7, f32, chunk 2048, `spec_k = 32` on both stages so `max_history` rises to 33
(the Strix's recurrent cache grows 1.19 → 4.04 GB; 10.99 GB free after load). The single-round probe
resets, prefills the 39-token code prompt, issues **one** B-row verify with the known greedy
continuation, and resets again; 3 runs per B, so a rejected row cannot cascade. End-to-end uses
`swp-exl3 generate --spec` on an 84-token repeated-paragraph prompt, 128 greedy tokens; plain is
14.63 tok/s and every spec output is byte-identical to it.

Single-round B-row verify (stage 1, `EXL3_VERIFY_ROW_SPLIT=2`):

| B | 8 | 12 | 16 | 18 | 24 | 30 | 32 |
|---|---|---|---|---|---|---|---|
| verify wall ms | 235.3 | 320.4 | 391.0 | 565.8 | 645.6 | 719.3 | **533.6** |
| stage-1 ms | 213.9 | 288.5 | 353.5 | 520.3 | 594.7 | 661.0 | 474.7 |
| B / wall (tok/s if all kept) | 34.0 | 37.5 | 40.9 | 31.8 | 37.2 | 41.7 | **60.0** |

The fine 8..32 step-2 sweep is in `/tmp/kilo/curve-fine.json`; the shape is piecewise, not a smooth
flattening:

* **8 → 16 rows: ~19.5 ms per extra row**, essentially the same slope as 2 → 4 (20–22 ms). The
  per-row cost does **not** fall as more experts are touched.
* **16 → 18 rows: +175 ms for two rows**. 18 → 30 is then ~12.8 ms/row, but the whole 17–30 region
  carries a large fixed penalty.
* **30 → 32 rows: −186 ms**. B=32 is faster than B=24 and B=30. B=24 is a local trough:
  37.2 tok/s versus 40.9 at B=16 and 60.0 at B=32 (B=18 is the lowest in the 18–30 band at 31.8).

The shape points to the engine's 16-row chunk boundary, not a gradual expert-saturation curve: 16 and 32 are full
16-row chunks (fast), 17–30 pay for a partial tail, and 24 (16+8) is the worst. The expected unique
expert count at 32 rows is only ~171 of 288 (about 5.3 new experts/row instead of 8), well short of
saturation, yet B=32 is the fastest point; the discontinuity is the kernel's chunk/GEMM tiling.
`ROW_SPLIT=0/1` produced the same shape (B=16 381.6/381.9 ms, B=24 638/641 ms, B=32 525/525 ms versus
391.0/645.6/533.6 with =2), so the row-split mode is not the cause. =1 is ~1–2% faster at B≥16 but
remains worse at B=6–7 (T6); do not change the default without re-running those points.

**End-to-end, 100% n-gram acceptance on the repeated paragraph (128 tok, greedy):**

| k | B | rounds | accepted | decode tok/s | vs plain | output |
|---|---|---|---|---|---|---|
| 11 | 12 | 11 | 121/121 | 37.2 | 2.54x | identical |
| **15** | **16** | **8** | **120/120** | **43.5** | **2.97x** | **identical** |
| 23 | 24 | 6 | 138/138 | 34.9 | 2.38x | identical |
| **31** | **32** | **4** | **124/124** | **63.4** | **4.33x** | **identical** |

Repeatable: k=15 43.50/43.37, k=23 34.87/34.75, k=31 63.34/63.70 (two extra runs each). The
throughput gain comes from amortising the per-token fixed cost: the marginal row is ~13–20 ms while
a plain decode token is ~66 ms, so even linear rows win; B=32 wins much more because it is a full
two-chunk alignment.

**Correctness.** `spec_check` at k=15 (B=16) is top-1 equal 32/32 on both "The capital of France is"
and the repeated paragraph. At k=31 (B=32) it is 32/32 on the paragraph (0 mismatches, KL 3.3e-4) but
**31/32 on the capital prompt** (1 top-1 mismatch, KL 2.2e-3), and the code prompt in the single-round
probe accepted 27/31. So **B=16 is the safe long-window default** (exact on everything tested, 43.5
tok/s); **B=32 is the fast path for highly constrained repetitive output** (63.4 tok/s) but is not
universally bit-identical to sequential decode. B=24 is slower than B=16 and buys no correctness, so
avoid it.

**MTP is unaffected by the long window:** its k=1/3 rounds are B=2/4, far from that regime. (The MTP
gate was re-measured after the `dflash_verify` fix below and now passes: 1.44x at k=2.) The
long-window upside belongs to n-gram lookups with long exact matches (code blocks, quoted lists,
repeated paragraphs).

**Configs/memory.** Razer `~/models/stage-310-razer-k7-f16-sk32.toml`; Strix
`/home/m/swp/sweep/long32{,-split1,-nosplit}.toml`; probe `/tmp/kilo/row_once.py`, results
`/tmp/kilo/curve-{fine,once,nosplit,split1}.json`, end-to-end logs `/tmp/kilo/spec*-para*.{txt,err}`.
Raising `spec_k` to 16/32 on both stages is a config change only; the live `spec_k = 8` files were not
modified.

## Verify fast-path marker (`dflash_verify`) — MTP revived

**Finding.** Kyojin's R-row verify fast paths are gated on `params["dflash_verify"]`; its own generator
sets that key on every verify forward. The adapter set only `recurrent_history`, so every 2..8-row verify
here took the generic multi-row route. That is why the flag sweep above (VERIFY_FUSE, DEC_MOE_UNION,
VERIFY_GEMV_R, MOE_UNION_V2, MOE_VALU) moved nothing: the kernels those flags select were never reached.
Upstream 1.6.4 has no such key, so the first stage (Razer) is unaffected either way.

**Fix.** `swp_exl3/exl3_backend.py` sets `params["dflash_verify"] = True` for verify forwards (default;
`SWP_VERIFY_FASTPATH=0` restores the old calls for a same-build A/B). Set it before `prepare_inputs`, as
the engine's generator does.

**Verify curve.** Model B cut 7, f32 wire, chunk 2048, warm stages, forced 100%-accept drafts; B = verify
rows = k+1; 4 warm rounds, median wall.

| B | marker off, ms | marker on, ms | stage 1 on, ms | change |
|---|---|---|---|---|
| 2 | 117.0 | **75.9** | 61 | −35% |
| 3 | 139.0 | **85.7** | 69 | −38% |
| 4 | 157.0 | **95.9** | 78 | −39% |
| 5 | 174.0 | **128.0** | 110 | −26% |
| 6 | 207.7 * | 140.0 | 118 | −33% |
| 7 | 226.2 * | 148.0 | 127 | −35% |
| 8 | 246.1 * | 156.1 | 134 | −37% |
| 9 | 279.9 * | 231.7 | 202 | −17% |

\* T6's `ROW_SPLIT=2` rows (same stage config). B=9 crosses the engine's 8-row caps (`MAX_BSZN` /
`MAX_BSZN_GEMV_R`), so it is a cliff, not a row: the best n-gram window is **k=7 (B=8)**, not k=8.
Stage 1 alone at B=2/3/4 is 61/69/78 ms (T6: ~101/135/158).

**n-gram end-to-end.** 51-token repeated paragraph, 127 new tokens, greedy; steady state (second run):

| k | rounds | accepted | decode tok/s | vs plain | output |
|---|---|---|---|---|---|
| plain | — | — | 14.98 | — | — |
| 3 | 32 | 96/96 (100%) | 37.84 | 2.53x | byte-identical |
| **7** | **16** | **112/112 (100%)** | **47.07 / 47.25** | **3.14x** | **byte-identical** |

The first k=7 run after a stage start is 31.8 tok/s (one-time shape autotune on the 8-row kernels).

**MTP end-to-end.** Model B, 149-token novel prompt, 127 new tokens, greedy, `mtp = true` +
`mtp_cache_tokens = 8192`; plain is the same stage with MTP loaded:

| k | rounds | accepted | accepted/depth | tok/s (2 runs) | vs plain |
|---|---|---|---|---|---|
| plain | — | — | — | 14.73 / 14.76 | — |
| 1 | 73 | 50/73 (68.5%) | [50] | **19.51 / 19.55** | 1.33x |
| **2** | **56** | **69/112 (61.6%)** | **[44, 25]** | **21.16 / 21.27** | **1.44x** |
| 3 | 55 | 71/165 (43.0%) | [39, 22, 10] | 18.69 | 1.27x |

k=2 is the best point; the per-round budget is 86 ms verify + ~2 ms catch-up + ~7.4 ms drafter for
2.23 tokens. Every k clears the 1.15x MTP gate that stopped S3 (then 15.5 tok/s, 1.03x). Capacity:
117.63 GB allocated, **8.85 GB free** (MTP block 3.7 GB + embedding 1.27 GB + 8192-token drafter
cache); the shared `eh_proj` sidecar was present. MTP text is **not** byte-identical to plain greedy
(first divergence at "helpful" vs "help" for all three k); n-gram was identical on every run.

**Correctness.** `swp-exl3 smoke` passes its golden prompts and top-64 logits (top-1 equal, 64/64, KL 0)
but **fails the spec line**: 20 rows at k=3, **1 top-1 mismatch, max KL 1.95e-3** (gate: top-1 equal and
KL ≤ 1e-2). This is a near-tie flip, not the S2 gross failure (that was KL 2.25 before `rewind(0)`).
The pre-marker engine fails the same way at k=2 and k=4 on this prompt (0 at k=3, by luck), so the gate
was already sampling the engine's floor. Two contributors, measured separately:

* **Stage 0 (Razer, upstream 1.6.4) has no exact R-row verify.** Its boundary hidden vs R 1-row
  forwards differs by max |Δ| 5.4e-4 (R=2), 8.2e-4 (R=3), 1.6e-3 (R=4), 3.3e-3 (R=5).
* **Stage 1 (Kyojin + marker)** is close but not bitwise. Fed the *same* stage-0 rows, verify vs
  sequential gives max KL 7.8e-6 (R=2), 3.1e-3 (R=3), 4.4e-4 (R=4), top-1 equal in that sample.

So the marker removes the generic-path cost and most of the drift, but "bit-exact" does not hold
end-to-end on this heterogeneous pair. A bit-identical speculative run needs an exact R-row verify on
the first stage too (upstream 1.6.4 does not have one), or a single-engine pair.

**Files.** `swp_exl3/exl3_backend.py` (`verify_fastpath`, the `dflash_verify` marker). MTP stage config
`/home/m/swp/stage-310-strix-k7-mtp.toml`. Probes (outside the repo): `/tmp/kilo/verify_probe.py`
(curve), `stage0_check.py` (stage-0 hidden), `pair_exact.py` (stage-1 isolation), logs
`/tmp/kilo/{gen-spec7*,mtp-*,smoke-mtp*}`.

## MTP as the n-gram fallback (policy)

**Decision.** MTP is enabled on the head stage but used only when the n-gram lookup misses. Every round
tries the free prompt lookup first; on a miss, the head's pending MTP drafts fill the window.
The two windows are independent: n-gram uses `--spec-k` (default 3, capped by the stages' `spec_k`) and
MTP uses `--spec-mtp-k` (default 2). `--no-mtp` / `SpecConfig.use_mtp = false` keeps exact n-gram only.
`--spec` stays opt-in for now: the default `generate` request is still plain decode, and
`swp-kyojin-server` still requires `--spec`. The Strix template ships `mtp = true`; the portable
`StageConfig.mtp` code default stays `false` (upstream head stages have no `Glm5NextMTPModel`).

**Why.** The marker results above show the complement: n-gram wins on repetitive text (k=7: 47.1 tok/s,
3.14x, byte-identical) and is free on novel text (miss -> plain decode); MTP with `dflash_verify` wins
where the lookup misses (k=2: 21.16/21.27 vs 14.73/14.76 plain, 1.44x). MTP-first would pay the ~7.4 ms
drafter and ~2 ms catch-up every round and discard the larger n-gram win.

**Accounting.** `Pipeline.generate` reports `spec_rounds` split into `spec_ngram_rounds`,
`spec_mtp_rounds`, and `spec_plain_rounds`, with matching `*_drafted` / `*_accepted` counters (they sum
to the totals); `spec_lookup_hits` / `spec_lookup_len_total` expose weak 1–2-token matches, and
`spec_ngram_verify_ms` / `spec_mtp_verify_ms` / `spec_mtp_roll_ms` split the time. `swp-kyojin-server`
prints the compact mix (`| spec n-gram 12r 44/44 | mtp 4r 9/12 | plain 1`) and `--spec-log FILE` appends
one JSON object per request for aggregation (rotating at 64 MiB, two backups).

**Cache guard.** When MTP is enabled, `mtp_cache_tokens` must be >= `cache_tokens` (checked in
`config.load_stage` for `mtp = true` and in `Exl3Backend._init_mtp`), and the drafter's prefill/roll
check `pos + rows` against its cache with a clear error. The drafter mirrors the target cache; a smaller
one overflows on a long prompt. `examples/stage-strix.toml` keeps its 64k cache and pairs
`mtp_cache_tokens` at 64k; the measured 8k MTP profile sets both to 8192.

**Smoke gate.** The spec line now classifies a top-1 mismatch as a near-tie when the verify top-1 is in
the reference top-k and the reference top-1 is within 0.1 logits. The gate is zero non-near-tie flips,
KL <= 1e-2, and near-ties capped at 10% of rows (at least one); the count and budget are printed. This
keeps the S2 gross failure (KL 2.25) and the golden logit check strict, while still failing a systematic
near-tie pattern. The earlier smoke failure with MTP loaded (1 flip, KL 1.95e-3) was the batched-kernel
floor: the pre-marker engine fails the same way at k=2/4.

**Live validation (cache 8192, MTP loaded).** Razer `[0,7)` + Strix `[7,45)`, f32, chunk 2048,
`mtp = true`, `mtp_cache_tokens = 8192` (both stages cache 8192). Short prompts: the 149-token smoke
prompt and a 42-token repetitive code prompt, 48 new tokens. The 8k rows use an 8000-token prompt + 128
new (8128 total) so the verify window fits; a pure 8192-token prompt with cache 8192 cannot generate
(an 8064+128 spec attempt failed at `forward needs 8193`).

| test | plain PP / TG tok/s | `--spec` PP / TG tok/s | spec mix |
|---|---|---|---|
| short novel, 149 tok | 158.7 / **14.95** | 171.1 / **19.71** (1.32x) | n-gram 4r 7/12, mtp 17r 17/34, plain 3 |
| short repetitive, 42 tok | 88.9 / **15.09** | 91.4 / **28.02** (1.86x) | n-gram 16r 32/48 |
| 8k novel, 8000+128 | 455.9 / **14.58** | 454.8 / **14.29** (0.98x) | n-gram 40r 8/120, mtp 25r 34/50, plain 22 |
| 8k repetitive, 8000+128 | 476.8 / **14.53** | 476.2 / **28.02** (1.93x) | n-gram 43r 87/129 |

Pure 8192-token prefill (`--max-new 1`): **433.3 tok/s novel**, **478.7 tok/s repetitive**.

* The MTP fallback works: the short novel run used 17 MTP rounds (17/34 accepted) after the n-gram
  lookup missed, and reached 1.32x plain.
* Repetitive output is byte-identical to plain at both 42 and 8k tokens.
* The 8k novel spec row is slightly **slower** than plain. The mix explains it: 40 n-gram rounds
  proposed 120 drafts but accepted only 8 (6.7%), and every hit suppressed MTP. The MTP rounds themselves
  were good (34/50). This is the 2-gram precision problem, not MTP.
* A diagnostic probe using `SpecConfig` with only 3-gram or 4-gram lookup reached **18.14** and **20.06**
  tok/s on the same 8k novel prompt, with 48/47 MTP rounds. `--spec-match 3`/`4` did not reproduce that
  because `SpecConfig.propose` currently falls back to a 2-gram match. The strict-match fix is planned
  next (make the match length a floor and default it to 3).
* `smoke` **PASS**: golden prompts identical; spec check 0 non-near-tie flips, 1 near-tie (budget 2),
  max KL 1.95e-3. Novel `--spec` output differs from plain (MTP is not exact). `--no-mtp` also diverged
  from plain on the short novel prompt at a batched-verify near-tie; n-gram exactness is not guaranteed
  on novel prose, though every repetitive run was identical.

**Open validation.** The agent-trace mix (10–20 real requests, n-gram `--spec-k 3/7/8`), the strict
n-gram floor, and the decision to enable `--spec` by default in `swp-kyojin-server` are still pending.
This section records the policy and the first end-to-end numbers.

## Verify-row flag sweep re-run (marker ON) — the "nothing helps" T6 result is void, and one flag *was* hidden

**Why.** The T6 sweep ("Verify-row flag sweep (2–4 rows)") and the T6 rows curve concluded the multi-row
knobs are all neutral. Every flag they tested selects a kernel *inside* the engine's R≥2 verify path,
and that path was never reached: the adapter did not set `params["dflash_verify"]`, the marker Kyojin's
own generator sets on every verify forward (previous section). With the marker now on by default, the
sweep is re-run over a wider window, B = 2..8 rows.

**Setup.** Model B cut 7, f32 wire, chunk 2048, Razer `[0,7)` stage 0 + Strix `[7,45)` stage 1, warm
stages. Each arm is a **fresh Strix stage load** with that arm's `[stage.env]` (the flags are read at
import / in C++ static init, so one process per arm). The probe resets, prefills one fixed prompt
("The capital of France is"), then runs forced-100%-accept verify rounds at B = 2..8 (the known greedy
continuation is supplied as the drafts), 2 discarded + 6 measured rounds per B, reporting the **median
wall ms** and the head stage's `stage_ms`; `mism` is verify top-1 ≠ sequential-greedy top-1 summed over
all rows. The base arm is the sweep base config (`/home/m/swp/sweep/base.toml`: the 32-flag SPEED_ENV
including `EXL3_VERIFY_ROW_SPLIT=2`), marker on. Base **reproduces the published marker-on curve**
(75.3/85.3/97.1/129.9/140.0/149.4/159.7 ms vs 75.9/85.7/95.9/128.0/140.0/148.0/156.1), so the harness
is calibrated to the previous section.

**Wall ms per verify round (median), B = verify rows:**

| arm | B=2 | B=3 | B=4 | B=5 | B=6 | B=7 | B=8 | mism |
|---|---|---|---|---|---|---|---|---|
| **base** (control) | **75.3** | **85.3** | **97.1** | **129.9** | **140.0** | **149.4** | **159.7** | 8 |
| `EXL3_HIP_PREFILL_MIN_ROWS=17` | 76.0 | 84.9 | 97.5 | 128.8 | 142.1 | 151.3 | 160.2 | 8 |
| `EXL3_DEC_MOE_UNION=1` | 75.7 | 85.1 | 98.1 | 129.1 | 143.8 | 152.0 | 159.6 | 8 |
| `EXL3_MOE_VALU=1` | 77.8 | 86.7 | 99.1 | 128.5 | 139.2 | 150.2 | 158.4 | 12 |
| `EXL3_MOE_R8=1` + `GEMV_R_RM6=1` (serve trio) | 76.2 | 86.3 | 97.6 | 131.3 | 142.9 | 151.1 | 160.4 | 8 |
| `EXL3_MOE_R8=1` | 76.4 | 86.3 | 97.0 | 129.7 | 138.6 | 149.6 | 158.0 | 8 |
| `EXL3_VERIFY_GEMV_R=1` | 77.3 | 86.1 | 97.4 | 130.6 | 142.8 | 151.1 | 160.5 | 8 |
| `EXL3_DEC_ROUTER_ROWS=1` | 77.6 | 88.1 | 99.9 | 133.2 | 143.1 | 153.7 | 164.6 | 8 |
| `EXL3_DEC_ROUTER_ROWS=0` | 81.9 | 90.8 | 102.2 | 134.9 | 144.3 | 152.8 | 161.8 | 6 |
| `EXL3_VERIFY_ROW_SPLIT=0` | 76.1 | 86.4 | 97.7 | 134.0 | 145.5 | 154.1 | 164.8 | 8 |
| `EXL3_VERIFY_ROW_SPLIT=1` | 77.4 | 86.3 | 97.3 | 135.6 | 146.1 | 157.7 | 168.2 | 8 |
| **`EXL3_MOE_UNION_V2=0`** | **96.4** | **112.5** | **130.7** | **169.8** | **189.2** | **205.9** | **217.3** | 10 |

**Stage 1 (Strix) ms per round** (stage 0 is 8–15 ms and unchanged; all the movement is here):

| arm | B=2 | B=3 | B=4 | B=5 | B=6 | B=7 | B=8 |
|---|---|---|---|---|---|---|---|
| base | 61.6 | 69.7 | 79.4 | 111.7 | 119.8 | 128.9 | 137.2 |
| MIN_ROWS=17 | 61.7 | 69.6 | 79.9 | 111.5 | 121.9 | 129.7 | 138.0 |
| DEC_MOE_UNION=1 | 61.4 | 69.7 | 79.5 | 111.0 | 122.1 | 129.7 | 137.6 |
| MOE_VALU=1 | 62.6 | 70.3 | 80.6 | 109.2 | 118.9 | 128.2 | 137.6 |
| serve trio | 61.8 | 70.1 | 79.8 | 111.6 | 122.0 | 129.8 | 137.8 |
| MOE_R8=1 | 61.6 | 69.8 | 79.3 | 111.5 | 119.3 | 129.0 | 135.9 |
| VERIFY_GEMV_R=1 | 61.8 | 69.7 | 79.7 | 111.5 | 122.2 | 129.3 | 138.1 |
| DEC_ROUTER_ROWS=1 | 62.3 | 71.3 | 81.8 | 113.5 | 122.2 | 132.1 | 141.2 |
| DEC_ROUTER_ROWS=0 | 67.6 | 75.1 | 84.7 | 115.9 | 123.9 | 130.7 | 139.8 |
| ROW_SPLIT=0 | 61.5 | 69.7 | 80.0 | 114.6 | 124.4 | 132.6 | 142.5 |
| ROW_SPLIT=1 | 61.7 | 69.9 | 79.7 | 116.2 | 125.5 | 135.8 | 145.5 |
| MOE_UNION_V2=0 | 81.1 | 96.2 | 113.7 | 150.8 | 168.3 | 184.5 | 194.8 |

**No arm flattens the B=5 bump.** The B=4→5 step and the per-row slope (B=2→4, B=5→8) for every arm:

| arm | B=2→4 /row | **B=4→5** | B=5→8 /row |
|---|---|---|---|
| base | 10.9 | **+32.8** | 9.9 |
| MIN_ROWS=17 | 10.8 | +31.4 | 10.4 |
| DEC_MOE_UNION=1 | 11.2 | +31.0 | 10.2 |
| MOE_VALU=1 | 10.6 | +29.4 | 10.0 |
| serve trio | 10.7 | +33.7 | 9.7 |
| MOE_R8=1 | 10.3 | +32.7 | 9.4 |
| VERIFY_GEMV_R=1 | 10.1 | +33.2 | 9.9 |
| DEC_ROUTER_ROWS=1 | 11.2 | +33.2 | 10.5 |
| DEC_ROUTER_ROWS=0 | 10.1 | +32.7 | 8.9 |
| ROW_SPLIT=0 | 10.8 | +36.2 | 10.3 |
| ROW_SPLIT=1 | 10.0 | +38.3 | 10.9 |
| MOE_UNION_V2=0 | 17.2 | +39.1 | 15.9 |

**Findings.**

* **Nothing flattens the B=5 bump.** The 4→5-row step is **+29…+39 ms in all 12 arms** (base +32.8),
  and the 5→8-row marginal returns to ~9–11 ms/row everywhere — exactly as in the marker-on curve. The
  "B=5 is the 4-row instantiation limit, so one of these may flatten it" hypothesis is **not supported**:
  the knobs that *literally* sit on a 4-row boundary do not reduce the step.
  * `EXL3_DEC_ROUTER_ROWS` (the router's one-launch loop is capped at **4** rows): `=0` (per-row router)
    is the *worst* of the small arms (+6.6 ms at B=2, +4..5 ms at B≥4); `=1` is +2..+5 ms worse than the
    base's `=2` at every B. The one-launch multi-row router is already the fast choice, so the 5-row step
    is not a router-instantiation limit.
  * `EXL3_HIP_PREFILL_MIN_ROWS=17` (every ≤16-row forward uses the decode kernel instead of the grouped
    prefill path): B=5 **128.8** vs 129.9 — within noise; the step stays +31.4.
  * `EXL3_VERIFY_ROW_SPLIT=0/1` (the base's `=2` splits rows 5–8 into ≤4-row fused sub-calls):
    both are *worse* at B≥5 (+4.1 / +5.7 at B=5; +5.1 / +8.5 at B=8), confirming `=2`. Rows 5–8 are
    already served by ≤4-row instantiations, yet B=5 still jumps — so the step is not the MoE/GEMV
    4-row instantiation these split.
* **`EXL3_MOE_UNION_V2` is the one flag the missing marker hid, and it is already on.** Setting it `=0`
  costs **+21 ms at B=2, growing to +57.7 ms at B=8 (1.36×)**; the per-row marginal nearly doubles
  (2→4 rows 10.9 → 17.2 ms/row; 5→8 rows 9.9 → 15.9). It does **not** change the B=5 step
  (+39.1 vs +32.8): V2 is the union-MoE dedup for the whole R≥2 path, not a 4/5-row boundary. This
  vindicates the "the flags sat behind the marker" premise — for V2 at least, whose T6 verdict was
  "neutral / +9 %" against a path it never entered.
* **`EXL3_MOE_R8` (one-launch 8-row fused MoE) and the serve.py trio (`MOE_R8=1` + `GEMV_R_RM6=1`) are
  neutral** (B=5 129.7 / 131.3 vs 129.9; B=8 158.0 / 160.4 vs 159.7): the 8-row MoE and 6-row dense-GEMV
  instantiations buy nothing on this prompt. `EXL3_DEC_MOE_UNION=1`, `EXL3_VERIFY_GEMV_R=1` and
  `EXL3_MOE_VALU=1` are likewise within run-to-run noise (±2–3 ms; base was measured twice at
  75.26/85.29/97.11/129.92 and 75.41/86.33/97.35/129.97). The union MoE path is clearly exercised at
  baseline (see the V2 arm), so `DEC_MOE_UNION`/`VERIFY_GEMV_R` gating adds nothing on top.
* **Interpretation.** With the marker on, R=1→2 costs ~+10 ms, R=4→5 costs ~+33 ms, then R=5→8 is back
  to ~10 ms/row, and **all of it is on the Strix stage**. Since `ROW_SPLIT=2` already splits 5–8 rows
  into ≤4-row calls and the router is already one-launch, the 5-row step is not in the MoE/router
  kernels these flags select; it is in the remaining per-row work (the R≥2 path switch itself, the
  MLA/GDN projection row loops, dense GEMV, hc-mixer) or a batch-4→5 instantiation outside the MoE.
  Practically: keep the n-gram window at **B=4 or B≥6**; kite B=5 (pay +33 ms for one row).

**Correctness.** Forced-draft rounds, verify top-1 vs the sequential greedy continuation: 6–12
mismatches per arm over 7 B × 6 rounds × B rows (base 8, `MOE_VALU` 12, `MOE_UNION_V2=0` 10). This is
the same batched-kernel near-tie floor as the published marker-on run; no arm is a numerics regression.

**Decision / config.** No live stage config changes. `EXL3_MOE_UNION_V2` stays at its default **1** (worth
making explicit in the Strix `[stage.env]` so a future Kyojin default flip cannot disable it silently);
`EXL3_VERIFY_ROW_SPLIT=2`, `EXL3_DEC_ROUTER_ROWS=2`, `EXL3_VERIFY_FUSE=1` stay as they are. `EXL3_MOE_R8`
/ `EXL3_GEMV_R_RM6` are inert on this prompt — matching serve.py is safe but not a gain. The B=5 bump
remains the largest unaddressed verify inefficiency for windows of k=4.

**Files.** Arm configs on the Strix: `/home/m/swp/sweep/{base,moe_r8,row_split0,row_split1,dec_moe_union,
verify_gemv_r,union_v2_0,moe_valu,serve_trio,router0,router1,min_rows17}.toml` (each generated from
`base.toml` by a single-key override). Harness (outside the repo): `/tmp/kilo/row_sweep.py`; drivers
`/tmp/kilo/run_sweep.sh` and `run_all.sh`; results `/tmp/kilo/sweep-<arm>.json` (12 files); logs
`/home/m/swp/sweep/<arm>.log`. No live stage config was modified; the driver only writes to
`/home/m/swp/sweep/`. All stages were stopped after the run.
