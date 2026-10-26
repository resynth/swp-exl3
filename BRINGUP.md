# BRINGUP — swp-exl3 two-machine bring-up (Razer + Strix Halo)

Status: **working**. The coordinator runs a real GLM-5.3-Flash EXL3 checkpoint split across the
Razer (layers 0..6) and the Strix Halo (layers 6..45) over Thunderbolt, generating coherent text.
This document is the record: environment, fixes, measurements, the CPU-expert-offload results, and
where to go next.

Author: bring-up session, 2026-10-24. Everything below was measured on this hardware, not inferred.

---

## 1. TL;DR

* Real two-machine inference works: **~19.4 tok/s decode**, prefill ~160–330 tok/s once warm, wire
  is ~2.5 ms/token (not the bottleneck).
* **Three bugs** had to be fixed first, one of which meant the Razer's engine had *never* built:
  1. Razer exllamav3 CUDA extension failed to compile under CUDA 13.3 (`-std=c++17` vs torch headers).
  2. `swp-exl3` called the engine outside `torch.inference_mode()`.
  3. On ROCm, a torch profiler/HSA thread spins a full core unless `ROCPROFILER_REGISTER_ENABLED=0`
     is set *before* torch is imported.
* **CPU expert offload works** on the Razer with the layer-sliced stage, in both modes
  (`moe_cpu_split`, `moe_cpu_offload`). A **27.1 GB** slice that OOMs the 16 GB GPU loads and runs
  once experts live in the Razer's RAM. This is the mechanism for bigger quants.
* Several statements in the pre-existing README / hand-off notes are wrong (Kyojin version, test
  count, the `moe_cpu_host` probe). Corrections in §4.

---

## 2. Hardware and software

| | Razer (coordinator + stage 0) | Strix Halo "boss" (stage 1) |
|---|---|---|
| IP (Thunderbolt) | `10.10.10.1` (`thunderbolt0`) | `10.10.10.2` |
| OS / kernel | Ubuntu 24.04.5, 6.17.0-40-generic | Ubuntu 26.04.1 LTS, ROCm 7.13 |
| CPU | laptop x86_64 | 32 threads |
| RAM | 62 GB | 124 GB |
| GPU | RTX 3080 Ti Laptop, 16 GB VRAM, driver 580 / CUDA 13.0 | Radeon 8060S (gfx1151), unified memory |
| Engine | upstream **exllamav3 1.6.4**, CUDA | **Kyojin 1.5.0**, ROCm |
| torch | `2.11.0+cu128` | `2.12.0a0+rocm7.13.0a20260411` (hip 7.13) |
| Python | 3.12.0 (venv at `~/ai/exllamav3/.venv`) | 3.14.4 (venv at `~/engines/kyojin/.venv`) |
| Toolchain | gcc 13.3, nvcc 13.3 | hipcc; Rust **not** installed (only `amdtop` in `~/.cargo/bin`) |
| Free disk | 168 GB on `/` | 635 GB on `/data` |

Rust toolchain was installed **on the Razer** (rustup, stable 1.99) to build `splitwire`; the
binaries were copied to the Strix (`/home/m/util/`). `uv` is present on both (the exllamav3 venv has
no `pip`).

### Model

`/data/hf/hub/models--NicSir--GLM-5.3-Flash-EXL3-2.72bpw/snapshots/db672fc5e61f1d61bdf6d641fb63538b80bb7613`
(HF-cache layout: real bytes in `blobs/`, snapshot holds symlinks; `du` on the snapshot lies).

* `Glm5NextForConditionalGeneration` / `glm5_next_text`; **45** layers, hidden **4096**, `hc_mult` **4**,
  vocab **154880**, **288** routed experts (8 active), `first_k_dense_replace` 3.
* exl3 quant: `bits 2.72`, `head_bits 6`, codebook **`mul1`**, `tie_word_embeddings false`.
* **14 shards, 113.94 GB.** Boundary tensor = `[rows, 4, 4096]` fp32 = **64 KiB/row**.

Per-layer bytes (from the safetensors index):

| range | size/layer |
|---|---|
| 0–2 (dense) | 0.134 GB |
| 3–15 | ~2.82–2.84 GB |
| 16–28 | ~1.92–1.93 GB |
| 29–44 | ~2.82–2.84 GB |
| 45 (MTP) | 3.79 GB |
| embed / lm_head / other | 1.269 / 0.476 / 1.127 GB |

Manifest (`swp-weightd index`): 150,894 tensors; dense 4.45, routed 102.82, embed 1.27, head 0.48,
MTP 3.79, other 1.127 GB; `manifest_hash a79d9367…1242b2a`.

Kyojin is launched by hand (`tools/glm/serve.py --model … --port 8000 -c 65536 --cache-bits 8`), and
is what we mirror for a stage.

---

## 3. What was done

### 3.1 Local test suite
`python -m unittest discover -s tests` → **15 tests, all pass** with
`SWP_WEIGHTD=…/target/release/swp-weightd`; 1 skipped without it (the provisioning test).
(The earlier hand-off said "15 tests pass from a clean extraction" but also "14 tests"; there are 15,
one of which self-skips without the Rust binary.)

### 3.2 Built the Rust tools
`~/.cargo/bin/cargo build --release` in `../splitwire`. `swp-drive --selftest` → **18/18 conformance
checks pass**. Binaries (`swp-weightd`, `swp-drive`, `swp-plan`, `swp-mockstage`) copied to the Strix.

### 3.3 Checks on both engines
* `tools/check_api_against_source.py`:
  * Razer upstream 1.6.4 → **48/48**.
  * Strix Kyojin 1.5.0 → **47/48**; the only failure is `Cache.__init__ has recurrent_dtype`.
* `swp-exl3 doctor`:
  * Razer → **17 passed, 0 failed**.
  * Strix → **17 passed, 0 failed** (`Cache(...)` reports `missing: ['recurrent_dtype']`; the adapter
    skips it with a warning, as designed).
* Dry-run slices against the *real* `Glm5NextModel` (no weights): `[0,6) first` and `[6,45) last` and
  `[0,12) first` all resolve cleanly (`boundary tail=[4,4096]`, last stage keeps `hc_head`/`norm`/`lm_head`).

### 3.4 Provisioning
`swp-weightd index` (manifest) → `swp-weightd serve` on the Strix :7700 → `swp-exl3 pull` on the
Razer. Pulled `[0,6)` (10.15 GB, 10,609 tensors) and later `[0,12)` (27.11 GB, 31,603 tensors). The
puller writes one normal safetensors shard plus `config.json`, `quantization_config.json` (48 MB!),
`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`,
`processor_config.json` — a loadable model dir.

### 3.5 Ran it
Split: Razer `[0,6)` first, Strix `[6,45)` last, coordinator on the Razer, `wire_dtype=f32`,
`chunk=512`. Razer GPU 9.10 GB, Strix 100.1 GB (26.4 GB headroom, vs the ~114 GB the Strix needs
alone). `ping` → 19 µs local / 130 µs remote.

```
$ swp-exl3 generate --pipeline pipeline.toml --prompt "..." --chat --max-new 64 --bench
Waves crash on the shore, / Salt spray kisses weathered rocks— / Ocean's endless song.
DECODE 29 tok in 1492 ms -> 19.43 tok/s
  stage layers   prefill: compute  wire+q   decode/tok: compute  wire+q  busy
      0  0..6              1157.1    12.9       5.344   0.887   10%
      1  6..45             1386.4     4.3      43.917   1.229   85%
```

---

## 4. Corrections to the previous notes / README

| Claim | Reality |
|---|---|
| Kyojin 1.6.0 on the Strix (and a 1.5.10 checkout for checks) | Strix is **Kyojin 1.5.0** |
| "15 tests" / "14 tests" | 15 tests, all pass with `SWP_WEIGHTD`; 1 self-skips without it |
| `doctor`'s `[info] no moe_cpu_host module` implies CPU offload is absent | Wrong probe name. CPU offload **exists** in 1.6.4; the implementation is `modules/block_sparse_mlp_cpu.py` (`moe_cpu_host` does not exist as a module). Cosmetic only. |
| Razer "upstream 1.6.4 with CPU expert offload" implied ready | The Razer's exllamav3 extension had **never built** — see §5. |
| `hc_flush` / deferred residual path | Present and wired, but `flush_ms=0.0` in every measured forward — the deferral never survived to a stage boundary here. Untested in practice (see §8). |

Also: **Kyojin is not pip-installed** in its venv. The user's `serve.py` works only because it is run
from the repo root. A `swp-exl3 stage` invocation must therefore run with
`PYTHONPATH=/home/m/engines/kyojin` (or `cd /home/m/engines/kyojin`).

---

## 5. Fixes

### 5.1 Razer: exllamav3 CUDA extension would not compile (blocker)
CUDA 13.3 `nvcc` cannot parse `torch/include/ATen/core/List_inl.h:202` at `-std=c++17`:

```
ATen/core/List_inl.h:202:52: error: need 'typename' before
  'decltype (...impl_->list)::difference_type' because ... is a dependent scope
```

Only translation units that include `<ATen/ATen.h>` hit it (`dsa_topk.cu`, `ple.cu`, `dflash2.cu`,
plus `quant/coop_autotune.cu`, `quant/exl3_moe_coop.cu`, `routing_gemm.cu`, `turing/fa75.cu`).
`-std=c++20` compiles it. torch only appends its own `-std=c++17` when the nvcc command line does not
already carry a `-std=` flag, so carrying one in `extra_cuda_cflags` suppresses the override.

**Change** — `~/ai/exllamav3/exllamav3/ext.py`, in the compiler-flags section (just before the
`CUDAHOSTCXX` block):

```python
# CUDA 13.x nvcc's device front end cannot parse ATen/core/List_inl.h at -std=c++17
# ("need 'typename' before 'decltype(...)::difference_type'"); -std=c++20 compiles it. torch only
# appends its own -std=c++17 to the nvcc command when no -std= is already present, so carrying one
# in extra_cuda_cflags suppresses the override.
if windows:
    extra_cflags += ["/std:c++20"]; extra_cuda_cflags += ["-std=c++20"]
else:
    extra_cflags += ["-std=c++20"]; extra_cuda_cflags += ["-std=c++20"]
```

Also had to delete a **stale build lock**:
`~/.cache/torch_extensions/py312_cu128/exllamav3_ext/lock`. This torch's `FileBaton` waits while the
lock file merely *exists* (`os.open(..., O_CREAT|O_EXCL)`), so a lock left by an interrupted build
blocks import forever. If `import exllamav3` hangs, check for this.

> This is a change to the dependency checkout, not to `swp-exl3`; it will be lost if exllamav3 is
> re-cloned. It is a machine-compat fix, not an upstream bug per se (torch headers × CUDA 13.3).

### 5.2 `swp-exl3`: forward ran outside `torch.inference_mode()`
`Model.forward()` is `@torch.inference_mode`, but `forward_ls()` is not. Calling the latter directly
raised `RuntimeError: Inplace update to inference tensor outside InferenceMode` on the first prefill.

**Change** — `swp_exl3/exl3_backend.py` (forward wrapper; body moved to `_forward_impl`):

```python
def forward(self, ...):
    with self.torch.inference_mode():
        return self._forward_impl(...)
```

### 5.3 `swp-exl3`: a ROCm core was spinning
A non-main thread burned ~100% of one core for the process lifetime. This is the HSA runtime event
thread that torch's ROCm build leaves spinning once the profiler is registered. exllamav3 guards it
with `os.environ.setdefault("ROCPROFILER_REGISTER_ENABLED", "0")` **before importing torch** — but
this adapter imports torch first, so the guard ran too late.

Measured (idle process, 2 s window):

| | CPU across all threads |
|---|---|
| flag unset | 210 jiffies / 2 s (~1 core) |
| flag = 0 | 0 jiffies / 2 s |

**Change** — `swp_exl3/exl3_backend.py`, in `Exl3Backend.__init__` before `import torch` (line 56):

```python
os.environ.setdefault("ROCPROFILER_REGISTER_ENABLED", "0")
```

Note `/proc/<pid>/environ` will *not* show it (Python `putenv` does not update the exec-time env
block); the CPU measurement is the proof.

### 5.4 Diagnostics added
`swp_exl3/exl3_backend.py`: when `SWP_DEBUG_TIMING=1`, the stage logs the per-forward split
(`prepare_ms`, `prefetch_ms`, `forward_ls_ms`, `flush_ms`, `advance_ms`). Off by default.

---

## 6. Benchmarks

### 6.1 Link (Razer → Strix), `swp-drive --probe --hidden 16384`
```
empty round trip : min 85 us   median 130 us   p99 164 us
64 MiB           : 119.1 ms median -> ~1126 MB/s one-way (~9.0 Gbit/s)
sustained model  1126 MB/s, rtt 130 us
  f32  64 KiB/row : decode hop 189 us   prefill ceiling 17189 tok/s
  bf16 32 KiB/row : decode hop 159 us   prefill ceiling 34377 tok/s
  q8blk17 KiB/row : decode hop 146 us   prefill ceiling 64710 tok/s
```
The wire is nowhere near the bottleneck at our rates.

### 6.2 Split at cut=6 (Razer `[0,6)`, Strix `[6,45)`), f32 wire
| metric | value |
|---|---|
| decode (63 tok) | **19.45 tok/s** |
| stage 0 decode compute | 4.78 ms/tok (9% busy) |
| stage 1 decode compute | 43.96 ms/tok (86% busy) |
| wire+queue | ~1.2–1.4 ms/tok per stage |
| prefill, warm, 291 tok | 1809 ms → **160.8 tok/s** |

Stage-0 prefill breakdown at n=291: `prepare 11.3`, `prefetch 0.6`, `forward_ls 847.1`,
`flush 0.0`, `advance 16.0` ms.

**Prefill "slowness" is one-time warmup.** The very first prefill after a fresh process can take
2.5–44 s (kernel compilation/autotuning, cached to disk afterwards); the same prompt then runs in
~0.9 s. Every `swp-exl3 generate` is a fresh process, so it pays a smaller version of this each time.
A long-lived coordinator/server would remove it.

### 6.3 bf16 wire (cut=6, same prompt)
top-1 equal, 62/64 shared top-k, max |Δlogit| 0.1875, **KL(f32‖bf16) 8.7e-4**. Halves bytes but the
wire is not the bottleneck, so **f32 stays the default**.

### 6.4 Cut-invariance
| A | B | top-1 | shared | max abs Δ | KL |
|---|---|---|---|---|---|
| cut 6 | cut 4 | equal | 63/64 | 0.094 | 5.30e-5 |
| cut 6 (GPU) | cut 12 (CPU offload) | equal | 64/64 | 0.055 | 5.81e-5 |

Residual KL is CUDA-vs-ROCm fp rounding, not a split bug.

---

## 7. CPU expert offload (the "bigger quants" mechanism)

Upstream 1.6.4 exposes (`infer_params`, set before load):

* `moe_cpu_offload = N` — the **whole routed-expert set of the first N block-sparse MoE layers** runs
  on the CPU worker (weights in system RAM).
* `moe_cpu_split = N` — the **tail N experts of every eligible MoE layer** run on the CPU worker,
  so CPU GEMMs overlap that layer's GPU compute.
* Both: layer-split mode only (we are), require the **`mul1`** codebook (this checkpoint is `mul1`),
  mutually exclusive. Worker is a separate process (here: avx2, 13 threads, host affinity LP 19).
  Tune with `moe_cpu_threads`.

`swp-exl3` passes `[stage.engine]` straight onto `config.infer_params` before `Model.from_config`.

### 7.1 Same slice (Razer `[0,6)`), three ways
| Razer config | Razer GPU | decode tok/s | vs GPU-only |
|---|---|---|---|
| GPU-only | 9.10 GB | 19.4 | — |
| `moe_cpu_split = 144` | 5.07 GB | 13.8 | KL 6.7e-5, 64/64 shared top-k |
| `moe_cpu_offload = 3` | **0.85 GB** | 13.95 | KL 7.4e-5, 63/64 |

Correctness is intact (differences are CPU-vs-GPU GEMM rounding). ~28% decode cost, because the two
stages still run sequentially per token — there is no cross-machine overlap.

### 7.2 Capacity proof — a slice that cannot fit GPU-only
Pulled `[0,12)` = **27.11 GB**, moved the Strix to `[12,45)`:

* **GPU-only: `torch.OutOfMemoryError`** (15.56 GiB in use).
* **`moe_cpu_offload = 6`** (layers 3–8 on CPU): loads at **9.93 GB** GPU and runs end-to-end:
  ```
  DECODE 47 tok in 4518 ms -> 10.40 tok/s
    0  0..12  decode compute 51.9 ms/tok  busy 54%
    1 12..45  decode compute 40.4 ms/tok  busy 42%
  ```
  Output identical to cut=6.
* **`moe_cpu_offload = 9`** (all MoE layers): loads into just **1.7 GB** of GPU.

So a 27 GB slice that OOMs a 16 GB GPU runs once experts live in Razer RAM. The binding limit then
becomes the Razer's **62 GB RAM** (~2.45 GB experts/layer → ~20 layers of experts could sit in RAM).

---

## 8. Known limits / not done

* **MTP / speculative decoding**: not implemented in the adapter (the model ships an MTP layer; the
  last stage could host it).
* **No cross-stage overlap**: stages serialize per chunk and per token; the star topology relays.
  `moe_cpu_split`'s overlap only helps within a stage.
* **Single device per stage** (`device = "cuda:0"`); autosplit loading is rejected on purpose.
* **Reconnect / replay** is unit-tested against the stub only, not exercised against the real
  engines (e.g. killing a stage mid-conversation).
* **`flush_hyperconnections`** measured `0 ms` in every run — the deferred hyper-connection update
  never survived to a stage boundary here, so that code path is effectively untested on real HW.
  Worth a targeted test (a boundary after a site that defers).
* **Prefill is not pipelined** and every CLI invocation pays per-process warmup.
* **Context**: the Razer stage uses `cache_tokens = 8192` in this bring-up — that is a **config
  choice**, not a hard cap. The checkpoint allows 1,048,576 positions. `RESULTS.md` T5 measures
  128k-token prefill/decode and loads 576k fp16 / 768k q8 caches (q8 on the Strix only); the 8192
  value is simply the small default used for the short-prompt benchmarks here.
* **`swp-exl3` changes are uncommitted** (see §9). The exllamav3 `ext.py` fix lives in the dependency
  checkout.
* **Network exposure.** SWP authenticates (shared-secret challenge) but does not encrypt, and
  `swp-kyojin-server` has no auth at all. Both hosts are on the internet, so the stage/weight
  servers bind their Thunderbolt IPs (`10.10.10.1` / `10.10.10.2`) and the OpenAI endpoint binds
  loopback. The `0.0.0.0` binds used during the original bring-up are not safe on an
  internet-connected host; use the firewall recipe in `README.md` §9 (which also keeps the Razer's
  BitTorrent TCP/UDP ports open).
* The `doctor` "kyojin-style / older" info line is wrong (bad module probe) — harmless.

---

## 9. Files changed / created

**Code changed**
* `~/ai/exllamav3/exllamav3/ext.py` — `-std=c++20` (Razer, CUDA 13.3). *Dependency, not in this repo.*
* `swp_exl3/exl3_backend.py` — `inference_mode()` wrapper (line ~134), `ROCPROFILER_REGISTER_ENABLED`
  (line 56), `SWP_DEBUG_TIMING` sub-timings (line ~179). Committed as `632776d`.

**Configs created**
* Razer `~/models/`: `pipeline.toml`, `pipeline-bf16.toml`, `pipeline-cut4.toml`, `pipeline-12.toml`,
  `stage-razer.toml` (default, GPU-only), `stage-razer-split.toml` (`moe_cpu_split=144`),
  `stage-razer-offload.toml` (`moe_cpu_offload=3`), `stage-razer-cut4.toml`,
  `stage-razer12-gpu.toml`, `stage-razer12-offload{6,9}.toml`.
* Strix `/home/m/swp/`: `stage-strix.toml` (`[6,45)`), `stage-strix-cut4.toml`, `stage-strix-12.toml`,
  `swp.env` (holds `SWP_SECRET`), plus the synced `swp-exl3/` tree.
* Slices on the Razer: `~/models/glm53-razer-slice` (10.15 GB, `[0,6)`),
  `~/models/glm53-razer-slice12` (27.11 GB, `[0,12)`).

**Secrets** — one random `SWP_SECRET`: Razer `/tmp/kilo/swp_secret`, Strix `/home/m/swp/swp.env`
(0600). Re-generate if you want; both ends must match.

---

## 10. Runbook

```bash
# --- Razer: build the Rust tools once ---
cd ~/ai/splitwire/splitwire && ~/.cargo/bin/cargo build --release
scp target/release/swp-{weightd,drive,plan,mockstage} m@10.10.10.2:/home/m/util/

# --- Strix: weight server (only needed to (re)pull slices) ---
ssh m@10.10.10.2 "set -a; . /home/m/swp/swp.env; set +a; \
  ~/util/swp-weightd serve --model /data/hf/hub/models--NicSir--GLM-5.3-Flash-EXL3-2.72bpw/snapshots/db672fc5e61f1d61bdf6d641fb63538b80bb7613 \
  --listen 10.10.10.2:7700 --secret-env SWP_SECRET"   # Thunderbolt IP, not 0.0.0.0

# --- Razer: pull this stage's slice ---
export SWP_SECRET=$(cat /tmp/kilo/swp_secret)
~/ai/exllamav3/.venv/bin/swp-exl3 pull --server 10.10.10.2:7700 \
  --out ~/models/glm53-razer-slice --layers 0 6 --first --secret-env SWP_SECRET

# --- Strix: stage 1 (note PYTHONPATH; Kyojin is not pip-installed) ---
ssh m@10.10.10.2 "set -a; . /home/m/swp/swp.env; set +a; cd /home/m/engines/kyojin && \
  PYTHONPATH=/home/m/engines/kyojin exec ./.venv/bin/swp-exl3 stage --config /home/m/swp/stage-strix.toml"

# --- Razer: stage 0 + coordinator ---
SWP_SECRET=$(cat /tmp/kilo/swp_secret) \
  ~/ai/exllamav3/.venv/bin/swp-exl3 stage --config ~/models/stage-razer.toml
SWP_SECRET=$(cat /tmp/kilo/swp_secret) \
  ~/ai/exllamav3/.venv/bin/swp-exl3 ping --pipeline ~/models/pipeline.toml
SWP_SECRET=$(cat /tmp/kilo/swp_secret) \
  ~/ai/exllamav3/.venv/bin/swp-exl3 generate --pipeline ~/models/pipeline.toml \
  --prompt "..." --chat --max-new 128 --bench

# --- Cut-invariance ---
swp-exl3 logits --pipeline p.toml --prompt "..." --save a.json
swp-exl3 compare a.json b.json
```

Note: the `swp-exl3` **stage servers currently run as session-scoped background processes** and stop
when the session ends — restart them with the commands above. The stage/weight listeners bind the
Thunderbolt IPs (`10.10.10.1` / `10.10.10.2`), so bring the Thunderbolt link up first and keep the
host firewalls as described in `README.md` §9.

---

## 11. Advice / how to go forward

**To run bigger quants (the stated goal).** The pipeline already supports it; the recipe is:
1. Pick a cut so the Strix holds what fits its ~124 GB with headroom; everything below the cut goes
   to the Razer.
2. Pull that slice over SWP (or `rsync` shards).
3. Set `[stage.engine] moe_cpu_offload = N` so at least `slice_bytes − ~14 GB` of experts lands in the
   Razer's RAM. Prefer `moe_cpu_split` if you want a softer speed/size trade (per-layer, overlaps).
4. Expect CPU-bound decode as the Razer's share grows. The Razer's 62 GB RAM is the hard limit.

**Quick wins.**
* **Persistent coordinator.** Today every `generate` is a fresh process and eats seconds of kernel
  warmup on first prefill. A long-running coordinator (or a `serve` mode) would make prefill feel
  right and is probably the single biggest UX improvement.
* **Cache sizing.** `cache_tokens = 8192` was only a small default. Raise it for long context:
  `RESULTS.md` T5 measures 128k prefill/decode, and the Strix loads 576k fp16 / 768k q8 caches at
  this cut (q8 only on the Strix; it is slow on the Razer's CUDA build). Size for prompt + generation
  + spec window and leave prefill-workspace headroom.
* **Keep f32 on the wire.** bf16 saves ~30 µs/token of a ~50 ms token; not worth the accuracy.

**Worth testing next.**
* The **hyper-connection flush** path — instrument a slice that ends right after a deferring site and
  confirm `hc_pending` is actually applied (it never triggered in these runs).
* **Reconnect/replay** against real stages (kill the Strix mid-generation and watch the coordinator
  re-feed history).
* **MTP / speculative decoding** on the last stage (the checkpoint has the layer; needs embedding +
  recurrent-history rollback for verify).
* **Razer CPU-offload tuning** — try `moe_cpu_threads`, and whether the CPU worker's single-LP
  affinity (LP 19) is a good default on a 16-thread laptop.
* **3-stage layouts** to place the Razer's RAM where it helps most (a `[0,a)` + Razer + Strix split
  means two stages on one host; only worth it once the middle is the bottleneck).

**Housekeeping.**
* The in-repo fixes (§5.2–5.4) are committed as `632776d "Fixes to get running"`; the exllamav3
  `ext.py` patch (§5.1) lives in the dependency checkout and is not covered by that commit — a
  re-clone of exllamav3 will silently re-break the Razer build. Consider a short "Dependencies on
  this machine" note.
* `doctor.py` looks for `exllamav3.modules.moe_cpu_host`; it should probe
  `modules/block_sparse_mlp_cpu` and report the real offload knobs.
* The README's Kyojin-version and test-count lines should be corrected.

---

## 12. Follow-up run — 3.10 bpw model (`turboderp/GLM-5.3-Flash-exl3`)

A second, larger quant was brought up on 2026-10-24.

**Model.** `…/models--turboderp--GLM-5.3-Flash-exl3/snapshots/b570edf6…` — exl3 **3.09 bpw**,
`head_bits 6`, **`mul1`**, version 1.6.1; **127.94 GB**, 22 shards, 45 layers, same
`Glm5NextForConditionalGeneration` shape (`hc_mult 4`). Per layer: 0–2 = 0.224 GB, 3 = 3.775 GB,
4–44 ≈ 2.861–2.869 GB, 45 (MTP) 3.756 GB, embed 1.269, lm_head 0.476.

**Split.** 128 GB does not fit the Strix's 124 GB, so: Razer `[0,12)` first (28.62 GB slice,
`moe_cpu_offload = 6` → layers 3–8 experts on the CPU worker) + Strix `[12,45)` last (96.25 GB).
Chosen to leave the Strix ~30 GB of headroom (it ran the 114 GB model "borderline" alone).

**Load result.** Strix allocated 96.25 GB (30.3 GB free); Razer 10.53 GB GPU (5.94 GB free) plus
6 layers' experts in system RAM (CPU worker: avx2, 13 threads). Both clean, no OOM.

**Coherence.** Yes, on two prompts:

```
What is the capital of France, and why is it famous?
→ Paris is famous for many reasons: 1. **Landmarks**: … Eiffel Tower, Notre-Dame … 2. **Art and Culture**: …
If a train travels 60 km in 45 minutes, what is its average speed in km/h? Show the steps.
→ Average speed = 60 km / 0.75 hours = 80 km/h   (steps shown, correct)
```

**Throughput.**

| metric | value |
|---|---|
| decode | **9.5 tok/s** (79 tok in 8296 ms; repeatable) |
| stage 0 (Razer, 12 layers, 6 offloaded) | 51.7 ms/tok, 49% busy |
| stage 1 (Strix, 33 layers) | 49.4 ms/tok, 47% busy |
| wire+queue | ~1.1 + ~2.7 ms/tok |
| prefill, warm, 308 tok | 5209 ms → **59.1 tok/s** |
| prefill, first call in a fresh process | ~2.6—18 tok/s (kernel warmup; see §6.2) |

**Reading it.** The two stages now cost about the same per token (≈50 ms each) and the pipeline
serialises them, so neither is idle. The Razer's six CPU-offloaded MoE layers cost as much as the
Strix's thirty-three layers. The obvious next lever is the split: the Strix is only at 96/124 GB, so
moving the cut down (e.g. `K = 9`–`10`) would take MoE work off the Razer's CPU and speed decode up;
`moe_cpu_offload = 5` (5.9 GB GPU still free) is a smaller, no-reload tweak. For reference, the same
setup on the 2.72 bpw model at cut 6 (all experts on GPU) gave 19.4 tok/s.

### 12.1 Tuned split — fill the Razer GPU, empty the CPU

The cut-12 layout above used only 10.5 of the Razer's 16.8 GB GPU and left 30 GB idle on the Strix.
Probing the Razer GPU capacity directly (loading different ranges from the `[0,12)` shard with no
offload, no Strix reload needed):

| Razer range | GPU allocated | free | CPU |
|---|---|---|---|
| `[0,9)`+ | OOM | — | — |
| `[0,8)` | 16.24 GB | 0.22 GB (too tight) | 0 |
| `[0,8)` + `moe_cpu_split=48` | 13.82 GB | 2.65 GB | 240/1440 experts |
| `[0,8)` + `moe_cpu_split=32` | 14.63 GB | 1.84 GB | 160/1440 experts |
| `[0,7)` | 13.30 GB | 3.18 GB | **0** |

**Final layout: Razer `[0,7)` all-GPU, Strix `[7,45)`** — the Razer GPU is 79% used, there is
literally no CPU offload, and the Strix carries 110.8 GB (15.7 GB free), matching the intended
"+15 GB on the Strix".

| metric | value |
|---|---|
| decode | **14.93 tok/s** (repeatable: 14.93 / 14.94) |
| stage 0 (Razer, 7 layers, no offload) | 7.3–7.6 ms/tok, 11% busy |
| stage 1 (Strix, 38 layers) | 54.8 ms/tok, 82% busy |
| wire+queue | ~1.4 + ~3.0 ms/tok |
| prefill, warm, 308 tok | 1420 ms → **216.8 tok/s** |

Progression on this model: **9.5 → 13.5 → 14.9 tok/s** (cut-12/heavy-offload → cut-8/split-32 →
cut-7/no-offload), and prefill 59 → 217 tok/s. The Strix is now clearly the bottleneck (82% busy vs
11%); going faster needs a shorter Strix stage, which memory does not allow, or a faster Strix stage
(Kyojin MTP/speculative decoding, cache-bit tuning).

Configs: `~/models/stage-310-razer-k7.toml`, `~/models/pipeline-310-k7.toml`,
`/home/m/swp/stage-310-strix-k7.toml`.


