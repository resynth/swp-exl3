# MEASUREMENTS — split-wire GLM-5.3-Flash EXL3 on Razer + Strix Halo

Complete measurement record for this session. Two checkpoints were brought up on the same two-machine
split; every number below was measured, not estimated. Companion narrative is in `BRINGUP.md`; this
file is the data.

Hardware: **Razer** (stage 0, RTX 3080 Ti Laptop 16 GB, upstream exllamav3 1.6.4 / CUDA) +
**Strix Halo "boss"** (stage 1, Radeon 8060S, Kyojin 1.5.0 / ROCm), over Thunderbolt `10.10.10.1` ↔ `10.10.10.2`.

**Conventions**

* *decode* = greedy/`--bench` decode rate over the generated tokens; *stage N compute* = ms/token of
  device-synced compute reported by that stage; *wire+q* = median round trip minus that stage's compute.
* *busy* = stage decode compute ÷ decode wall time (how much of the token the stage holds the critical path).
* *prefill compute* = total ms the stage spent on the prompt; *prefill* = prompt tokens ÷ wall.
* **cold vs warm**: these engines JIT-compile kernels and cache them on disk. The *first* prefill in
  a process (and the first ever, per machine) is far slower than later ones. Both numbers are given
  where they differ. Every `swp-exl3 generate` is a fresh process, so each run pays a small cold cost.
* GPU memory figures are the stage's own `allocated_gb` at load; "free" is the GPU's remaining VRAM.

---

## 1. Environment and link

### 1.1 Machines

| | Razer (stage 0) | Strix "boss" (stage 1) |
|---|---|---|
| OS / kernel | Ubuntu 24.04.5, 6.17.0-40 | Ubuntu 26.04.1 |
| RAM | 62 GB | 124 GB |
| GPU | RTX 3080 Ti Laptop, 16.76 GB | Radeon 8060S (gfx1151), unified |
| Engine | exllamav3 **1.6.4** (CUDA 13.3, gcc 13.3) | Kyojin **1.5.0** (ROCm 7.13) |
| torch | 2.11.0+cu128 | 2.12.0a0+rocm7.13.0a20260411 |
| Python | 3.12.0 | 3.14.4 |

### 1.2 Checks

| Check | Razer | Strix |
|---|---|---|
| `swp-exl3 doctor` | 17 passed, 0 failed | 17 passed, 0 failed (`Cache` missing `recurrent_dtype`, skipped by design) |
| `check_api_against_source.py` | 48 / 48 | 47 / 48 (only `Cache.__init__ has recurrent_dtype`) |
| unit tests (`unittest discover`) | 15 pass (with `SWP_WEIGHTD`) | — |
| `swp-drive --selftest` | 18 / 18 conformance | — |

### 1.3 Link (Razer → Strix), `swp-drive --probe --hidden 16384`

| Probe | value |
|---|---|
| empty round trip | min 85 µs, **median 130 µs**, p99 164 µs |
| 64 KiB message | 375 µs median → ~349 MB/s one-way |
| 1 MiB | 2148 µs → ~976 MB/s |
| 16 MiB | 29912 µs → ~1122 MB/s |
| 64 MiB | 119149 µs → **~1126 MB/s (~9.0 Gbit/s)** |
| decode hop, f32 64 KiB row | 189 µs (rtt is 69% of it) |
| decode hop, bf16 32 KiB row | 159 µs |
| decode hop, q8blk 17 KiB row | 146 µs |
| prefill ceiling, f32 | 17189 tok/s |
| prefill ceiling, bf16 | 34377 tok/s |
| prefill ceiling, q8blk | 64710 tok/s |

Measured `ping` RTT through the coordinator: **16–19 µs** local (stage 0), **128–134 µs** remote (stage 1).

### 1.4 ROCm idle-spin (Strix)

| | CPU over a 2 s idle window, all threads |
|---|---|
| `ROCPROFILER_REGISTER_ENABLED` unset | 210 jiffies (~1 core maxed) |
| `ROCPROFILER_REGISTER_ENABLED=0` (fix, §5.3 of BRINGUP) | 0 jiffies |

---

## 2. Model facts

| | **Model A** — `NicSir/GLM-5.3-Flash-EXL3-2.72bpw` | **Model B** — `turboderp/GLM-5.3-Flash-exl3` |
|---|---|---|
| exl3 bits | 2.72 (`head_bits 6`) | **3.09** (`head_bits 6`) |
| codebook / version | `mul1` / 1.4.6 | `mul1` / 1.6.1 |
| total size | **113.94 GB**, 14 shards | **127.94 GB**, 22 shards |
| layers / hidden / hc_mult | 45 / 4096 / 4 | 45 / 4096 / 4 |
| routed experts (active) | 288 (8) | 288 (8) |
| `first_k_dense_replace` / MTP | 3 / 1 | 3 / 1 |
| `tie_word_embeddings` | false | false |
| manifest tensors | 150,894 | — |
| manifest hash | `a79d9367…1242b2a` | — |
| boundary tensor | `[rows, 4, 4096]` fp32 = **64 KiB/row** | same |

### 2.1 Per-layer bytes

| group | Model A (2.72 bpw) | Model B (3.10 bpw) |
|---|---|---|
| L0–L2 (dense) | 0.134 GB each | 0.224 GB each |
| L3 | 2.836 GB | 3.775 GB |
| L4–L15 (A) / L4–L44 (B) | 2.82–2.84 GB | 2.861–2.869 GB |
| L16–L28 (A only) | 1.915–1.930 GB | (same as above) |
| L29–L44 | 2.82–2.84 GB | 2.861–2.869 GB |
| L45 (MTP) | 3.793 GB | 3.756 GB |
| embed | 1.269 GB | 1.269 GB |
| lm_head | 0.476 GB | 0.476 GB |
| other | 1.127 GB | 0.598 GB |

### 2.2 Stage loads actually used

| Model | Split | Razer GPU alloc (free) | Razer CPU | Strix alloc (free) |
|---|---|---|---|---|
| A | `[0,6)` / `[6,45)` | 9.10 GB (7.38) | 0 | 100.1 GB (26.4) |
| A | `[0,4)` / `[4,45)` | 3.47 GB (13.01) | 0 | 105.9 GB (20.7) |
| B | `[0,12)` / `[12,45)` offload=6 | 10.53 GB (5.94) | 6 layers (3–8) | 96.25 GB (30.26) |
| B | `[0,8)` / `[8,45)` split=48 | 13.82 GB (2.65) | 240/1440 experts | 107.99 GB (18.52) |
| B | `[0,8)` / `[8,45)` split=32 | 14.63 GB (1.84) | 160/1440 experts | 107.99 GB (18.52) |
| **B** | **`[0,7)` / `[7,45)` no offload** | **13.30 GB (3.18)** | **0** | **110.80 GB (15.71)** |

---

## 3. Model A (NicSir 2.72 bpw)

### 3.1 Performance by run

| Run | Split / config | prefill | decode | stage 0 compute (busy) | stage 1 compute (busy) | wire+q |
|---|---|---|---|---|---|---|
| A1 | `[0,6)`, cold | 11 tok, 6130 ms → 1.8 tok/s | 7 tok, 990 ms → **7.07 tok/s** | 2838 ms / 91.0 tok⁻¹ (64%) | 3232 ms / 46.4 tok⁻¹ (33%) | 21.7 / 38.0 ms |
| A2 | `[0,6)`, warm | 17 tok, 4598 ms → 3.7 tok/s | 63 tok, 3239 ms → **19.45 tok/s** | 4084 ms / 4.78 tok⁻¹ (9%) | 498 ms / 43.96 tok⁻¹ (86%) | 11.2 / 4.3 ms |
| A3 | `[0,6)`, 128 max-new | 13 tok, 4287 ms → 3.0 tok/s | 47 tok, 2424 ms → 19.39 tok/s | 1190 ms / 5.18 tok⁻¹ (10%) | 3047 ms / 44.31 tok⁻¹ (86%) | 13.2 / 35.8 ms |
| A4 | `[0,6)`, 291-tok prompt, **cold kernels** | 291 tok, 45844 ms → **6.3 tok/s** | 31 tok, 1617 ms → 19.17 tok/s | 44534 ms | 1270 ms | 16.1 / 23.8 ms |
| A5 | `[0,6)`, 291-tok prompt, **warm kernels** | 291 tok, 1809 ms → **160.8 tok/s** | 3 tok, 180 ms → 16.66 tok/s | 878 ms | 880 ms | 29.5 / 21.0 ms |
| A6 | `[0,6)` + `moe_cpu_split=144` | 13 tok, 682 ms → 19.1 tok/s | 47 tok, 3405 ms → **13.80 tok/s** | 393 ms / 22.39 tok⁻¹ (31%) | 281 ms / 45.25 tok⁻¹ (62%) | 2.1 / 6.0 ms |
| A7 | `[0,6)` + `moe_cpu_offload=3` | 13 tok, 782 ms → 16.6 tok/s | 47 tok, 3370 ms → **13.95 tok/s** | 494 ms / 22.54 tok⁻¹ (31%) | 280 ms / 45.40 tok⁻¹ (63%) | 2.4 / 6.3 ms |
| A8 | `[0,12)` + `moe_cpu_offload=6` | 13 tok, 3907 ms → 3.3 tok/s | 47 tok, 4519 ms → **10.40 tok/s** | 1694 ms / 51.90 tok⁻¹ (54%) | 2161 ms / 40.42 tok⁻¹ (42%) | 15.7 / 35.3 ms |

Load sizes (A): `[0,6)` → 10.15 GB slice, 10,609 tensors. `[0,12)` → 27.11 GB slice, 31,603 tensors.

### 3.2 Stage-0 prefill breakdown (A5, n=291) — `SWP_DEBUG_TIMING`

| phase | ms |
|---|---|
| `prepare` (attn + recurrence setup) | 11.3 |
| `prefetch` | 0.6 |
| **`forward_ls`** | **847.1** |
| `flush` (hyper-connection residual) | 0.0 |
| `advance` (recurrent state) | 16.0 |

`flush = 0` in every run: the deferred hyper-connection update never survived to a stage boundary here.

### 3.3 Levers at fixed split `[0,6)`

| Lever | Setting A → B | Razer GPU | decode | correctness |
|---|---|---|---|---|
| expert placement | GPU-only → `moe_cpu_split=144` | 9.10 → 5.07 GB | 19.45 → 13.80 tok/s | top-1 equal, 64/64 shared, max abs Δ 0.0625, **KL 6.7e-5** |
| expert placement | GPU-only → `moe_cpu_offload=3` | 9.10 → 0.85 GB | 19.45 → 13.95 tok/s | top-1 equal, 63/64 shared, max abs Δ 0.125, **KL 7.4e-5** |
| wire dtype | f32 → bf16 | — | no gain (wire is not the limit) | top-1 equal, 62/64 shared, max abs Δ 0.1875, **KL 8.7e-4** |
| cut point | `[0,6)` → `[0,4)` | — | — | top-1 equal, 63/64 shared, max abs Δ 0.094, **KL 5.3e-5** |
| cut point | `[0,6)` → `[0,12)` (+offload 6) | — | — | top-1 equal, 64/64 shared, max abs Δ 0.055, **KL 5.8e-5** |
| warmth | cold → warm kernels (A4 → A5) | — | prefill **6.3 → 160.8 tok/s** | — |

### 3.4 Capacity test (A, `[0,12)` = 27.11 GB)

| Config | Razer GPU | outcome |
|---|---|---|
| no offload | — | **`torch.OutOfMemoryError`** (15.56 GiB in use) |
| `moe_cpu_offload=6` | 9.93 GB | loads and runs end-to-end (A8) |
| `moe_cpu_offload=9` | **1.7 GB** | loads |

---

## 4. Model B (turboderp 3.10 bpw)

### 4.1 Razer GPU-capacity sweep (loaded from the `[0,12)` shard, no Strix reload)

| Razer range | config | GPU alloc | GPU free | CPU experts |
|---|---|---|---|---|
| `[0,12)`…`[0,9)` | no offload | — | — | OOM |
| `[0,8)` | no offload | 16.24 GB | 0.22 GB | 0 |
| `[0,8)` | `moe_cpu_offload=1` | 12.61 GB | 3.85 GB | 1 layer |
| `[0,8)` | `moe_cpu_offload=2` | 9.93 GB | 6.53 GB | 2 layers |
| `[0,8)` | `moe_cpu_split=24` | 15.03 GB | 1.42 GB | 120/1440 |
| `[0,8)` | `moe_cpu_split=32` | 14.63 GB | 1.84 GB | 160/1440 |
| `[0,8)` | `moe_cpu_split=48` | 13.82 GB | 2.65 GB | 240/1440 |
| `[0,8)` | `moe_cpu_split=64` | 13.02 GB | 3.45 GB | 320/1440 |
| `[0,8)` | `moe_cpu_split=96` | 11.40 GB | 5.07 GB | 480/1440 |
| **`[0,7)`** | **no offload** | **13.30 GB** | **3.18 GB** | **0** |

The Razer's 16 GB GPU tops out at `[0,8)` no-offload (16.24 GB, i.e. nothing left for a run); `[0,7)`
all-GPU is the largest range that fits with headroom.

### 4.2 Performance by run

| Run | Split / config | prefill | decode | stage 0 compute (busy) | stage 1 compute (busy) | wire+q |
|---|---|---|---|---|---|---|
| B1 | `[0,12)`, `offload=6` | 19 tok, 7198 ms → 2.6 tok/s | 63 tok, 8085 ms → **7.79 tok/s** | 4783 ms / 73.72 tok⁻¹ (57%) | 2360 ms / 50.35 tok⁻¹ (39%) | 18.2 / 36.8 ms |
| B2 | `[0,12)`, `offload=6`, 308-tok prompt | 308 tok, 5209 ms → **59.1 tok/s** | 23 tok, 2421 ms → 9.50 tok/s | 4115 ms / 51.77 tok⁻¹ (49%) | 1039 ms / 49.51 tok⁻¹ (47%) | 26.5 / 29.7 ms |
| B3 | `[0,12)`, `offload=6`, 31-tok prompt | 31 tok, 1683 ms → 18.4 tok/s | 79 tok, 8296 ms → 9.52 tok/s | 1170 ms / 51.70 tok⁻¹ (49%) | 502 ms / 49.41 tok⁻¹ (47%) | 3.2 / 7.5 ms |
| B4 | `[0,8)`, `split=48` | 19 tok, 3638 ms → 5.2 tok/s | 63 tok, 4944 ms → **12.74 tok/s** | 1256 ms / 19.61 tok⁻¹ (25%) | 2330 ms / 53.70 tok⁻¹ (68%) | 14.2 / 38.3 ms |
| B5 | `[0,8)`, `split=32` | 19 tok, 1661 ms → 11.4 tok/s | 63 tok, 4658 ms → **13.53 tok/s** | 1196 ms / 16.31 tok⁻¹ (22%) | 445 ms / 52.86 tok⁻¹ (71%) | 14.6 / 6.3 ms |
| B6 | **`[0,7)`, no offload** | 19 tok, 3180 ms → 6.0 tok/s | 63 tok, 4219 ms → **14.93 tok/s** | 730 ms / 7.59 tok⁻¹ (11%) | 2396 ms / 54.78 tok⁻¹ (82%) | 13.0 / 40.5 ms |
| B7 | **`[0,7)`, no offload, 308-tok prompt** | 308 tok, 1420 ms → **216.8 tok/s** | 23 tok, 1540 ms → **14.94 tok/s** | 202 ms / 7.33 tok⁻¹ (11%) | 1159 ms / 54.91 tok⁻¹ (82%) | 17.4 / 41.8 ms |

### 4.3 Progression on Model B (same model, only the split/offload changed)

| Config | Razer GPU | CPU | Strix | decode | prefill (warm) |
|---|---|---|---|---|---|
| `[0,12)` `offload=6` | 10.53 GB | 6 layers | 96.25 GB | 9.5 tok/s | 59.1 tok/s |
| `[0,8)` `split=48` | 13.82 GB | 240/1440 | 107.99 GB | 12.74 tok/s | — |
| `[0,8)` `split=32` | 14.63 GB | 160/1440 | 107.99 GB | 13.53 tok/s | — |
| **`[0,7)` no offload** | **13.30 GB** | **0** | **110.80 GB** | **14.93 tok/s** | **216.8 tok/s** |

Correctness is covered in §6; within a model, changing the cut or the offload only moves the boundary
(or the CPU/GPU split of the experts), which the cut-invariance checks show to be numerically neutral.

---

## 5. Levers → effects, at a glance

| # | Lever (where) | Change | Effect measured |
|---|---|---|---|
| 1 | **Cut point** (B) | `[0,12)` → `[0,8)` → `[0,7)` | decode **9.5 → 13.5 → 14.9 tok/s**; Razer GPU 10.5 → 14.6 → 13.3 GB; Strix 96.3 → 108.0 → 110.8 GB |
| 2 | **CPU expert placement** (A, fixed cut 6) | GPU-only → `split=144` → `offload=3` | Razer GPU **9.10 → 5.07 → 0.85 GB**; decode **19.4 → 13.8 → 13.9 tok/s** (CPU GEMM is ~3–4× the per-layer cost of the GPU) |
| 3 | **Whole-layer vs per-layer offload** (B, `[0,8)`) | `offload=1/2` vs `split=24/32/48/64/96` | whole-layer: 12.6 / 9.9 GB; split: 15.0 / 14.6 / 13.8 / 13.0 / 11.4 GB — split fills the GPU more for the same CPU budget |
| 4 | **GPU headroom** (B, `[0,8)`) | `offload=0` → any offload/split | 16.24 GB (0.22 free, unusable) → 12.6–15.0 GB (1.4–6.5 free); ~1.8 GB headroom is enough to run |
| 5 | **Cut / machine capacity** (A) | `[0,12)` no offload → `offload=6` | **OOM → runs**; GPU 9.93 GB |
| 6 | **Wire dtype** (A) | f32 → bf16 | no throughput change (wire ≈ 2.5 ms of a ~50 ms token); **KL 8.7e-4** — f32 kept |
| 7 | **Warmth** (A, B) | cold kernels → warm | prefill **6.3 → 160.8 tok/s** (A); **6.0 → 216.8 tok/s** (B). Fixes the "slow prefill" impression |
| 8 | **Which machine holds what** (A vs B) | Razer stage 0 always; Strix stage 1 | Strix is the bottleneck once it holds ≥33 layers (71–86% busy); Razer 11–25% busy at 7–8 layers |
| 9 | **ROCm idle spin** (Strix) | `ROCPROFILER_REGISTER_ENABLED` unset → 0 | 210 → 0 jiffies / 2 s (one core reclaimed) |

### 5.1 Bottleneck flip

| Config | stage 0 time/tok | stage 1 time/tok | bottleneck |
|---|---|---|---|
| A `[0,6)` GPU-only | 4.9 ms (9%) | 44.0 ms (86%) | Strix |
| A `[0,6)` offload | 22.5 ms (31%) | 45.4 ms (63%) | Strix (Razer close behind) |
| B `[0,12)` `offload=6` | 51.8 ms (49%) | 49.5 ms (47%) | balanced (~equal) |
| B `[0,8)` `split=32` | 16.3 ms (22%) | 52.9 ms (71%) | Strix |
| B `[0,7)` no offload | 7.6 ms (11%) | 54.8 ms (82%) | Strix, strongly |

---

## 6. Correctness (cut-invariance / offload equivalence)

All comparisons are top-k=64 logits for the prompt *"The capital of France is"*, f32 wire.

| A | B | top-1 | shared top-k | max abs Δlogit | KL(A‖B) |
|---|---|---|---|---|---|
| A `[0,6)` GPU-only | A `[0,4)` | equal | 63/64 | 0.094 | 5.30e-5 |
| A `[0,6)` GPU-only | A `[0,12)` + offload 6 | equal | 64/64 | 0.055 | 5.81e-5 |
| A `[0,6)` GPU-only | A `[0,6)` `split=144` | equal | 64/64 | 0.062 | 6.71e-5 |
| A `[0,6)` GPU-only | A `[0,6)` `offload=3` | equal | 63/64 | 0.125 | 7.44e-5 |
| A `[0,6)` f32 wire | A `[0,6)` **bf16** wire | equal | 62/64 | 0.188 | 8.68e-4 |

Residual KL at f32 is CUDA-vs-ROCm fp rounding, not a split bug; bf16 wire is the only lever with a
material effect, and it is one to two orders of magnitude larger than a cut change.

Text checks: greedy output is identical across cut points and offload modes (e.g. Model B B1/B6 both
produce the same "Paris is famous for many reasons: 1. **Landmarks** …" continuation; the 60 km/45 min
prompt returns the correct 80 km/h with steps).

---

## 7. Measurement caveats

* **Prefill in a fresh process always pays JIT warmup.** The *first* prefill ever on a machine is by
  far the worst (Model A: 44.5 s for 291 tok); after kernels are cached, the same work is ~1.8 s.
  Every CLI `generate` is a new process, so per-run prefill numbers are inflated by a fixed cost;
  only the *warm* numbers (A5, B2, B7) characterise steady-state prefill.
* **decode** numbers stabilise after the first token or two; B/C rows with few generated tokens are
  noisier.
* GPU figures are exllamav3's own `allocated_gb` at load and exclude KV/recurrent cache and
  activations, which are allocated lazily — always keep ~1.5–3 GB free on the Razer.
* `wire+q` is per stage and per token; the two stages are serialised (star topology), so token time
  ≈ Σ(stage compute) + Σ(wire+q).
* The Razer's GPU is a laptop part with an 80 W cap; absolute stage-0 throughput is power-limited.
* Model A and Model B differ in quant level (2.72 vs 3.10 bpw), so cross-model comparisons are
  indicative only; the lever tables isolate changes *within* a model.
