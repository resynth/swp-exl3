# The Strix idle-wake penalty (~1.5 ms/token)

Measured 2026-10-25 on the two-machine pair. Companion to `RESULTS.md` Task 4. Everything below is
from that session unless marked *estimate* or *untested*.

---

## 0. TL;DR

The remote stage is **not** slow at framing. It is slow at **starting**: the Strix spends ~7 ms idle
while the Razer runs stage 0, and when the next 64 KiB request arrives after that gap the package has
to leave a deep idle state. That costs **~1.5 ms per token**, reproducibly, and disappears if one
Strix core is kept busy.

- Decode overhead: **4.2–4.5 ms/token** baseline; **~2.0 ms/token** after putting stage 0 in the
  coordinator and keeping the Strix awake.
- Stage 1's share: **2.8–3.1 ms** cold / **1.4–1.5 ms** awake.
- Worst case, the fix is worth **~2.3%** (`~0.36 tok/s` at 15.5 tok/s). It is not the main decode
  lever; it is a cheap, low-risk 2–3%.
- Preferred fixes, in order: a **bounded post-reply wake window** (no root, scoped to activity), or
  **PM QoS `/dev/cpu_dma_latency` with an idle timeout** (needs a one-time permission change). A
  permanent global C-state change costs real idle power and is not recommended for a box that idles.

---

## 1. Context

| | Razer (stage 0) | Strix "boss" (stage 1) |
|---|---|---|
| CPU | Intel laptop (16 threads) | **AMD Ryzen AI Max+ 395**, 32 threads |
| GPU | RTX 3080 Ti Laptop 16 GB | Radeon 8060S (gfx1151), unified |
| Engine | exllamav3 1.6.4 / CUDA | Kyojin 1.5.0 / ROCm 7.13 |
| Idle driver | `intel_idle` | **`acpi_idle`** |
| States exposed | POLL, C1E, C6, C8, C10 | POLL, C1, C2, C3 |
| Governor | `powersave` | `powersave` (`menu` idle governor) |
| Deep-state latency | not measured | **C3: 350 µs** (C2: 18 µs, C1: 1 µs) |
| Idle residency | not measured | **C3 ≈ 99.8%** of idle time |
| `/dev/cpu_dma_latency` | not checked | **root-only** (`crw------- root root`) |
| Tools | — | `cpupower`, `turbostat` present; no `powertop` |

Pipeline: Model B (`turboderp/GLM-5.3-Flash-exl3`, 3.09 bpw), Razer `[0,7)`, Strix `[7,45)`, f32 wire,
chunk 2048. Stage 0 can run in the coordinator process (`stage_config`, commit `f248632`).
Typical decode token: stage 0 **6.3–6.9 ms**, stage 1 **54–55 ms**, wall **63–67 ms**.

The star topology is sequential: stage 1 cannot start a token until stage 0 has produced the boundary
rows. During those ~7 ms the Strix has nothing to do.

---

## 2. The problem

### 2.1 Symptom

Decode overhead (`wall − stage compute`) is bimodal on stage 1:

| Strix state | stage 1 `rtt − compute` | Total overhead with local stage 0 |
|---|---|---|
| just after activity / one core busy | **1.40–1.53 ms** | **~2.0 ms/token** |
| after the package has been idle | **2.8–3.1 ms** | **~3.4–3.8 ms/token** |

The same run can show either: a final smoke run with local stage 0 measured 15.47 tok/s with
stage 1 at 3.04 ms; the 48-token runs just before it measured 15.67–15.75 tok/s with stage 1 at
1.44–1.53 ms.

### 2.2 Where the time is not

The obvious suspects were ruled out:

- **Framing/JSON is ~0.35 ms.** A 64 KiB `PING` round trip is **0.354 ms** (empty: 0.129 ms; 1 KiB:
  0.304 ms). A `FORWARD` request is 64 KiB.
- **Wire dtype does not matter.** f32 is already only a few percent of a token; bf16 saves ~30 µs.
- **`forward_ls` is not the adapter.** At n=1, stage 1 `forward_ls` is **12.2 ms** of kernel *enqueue*;
  the final `cuda.synchronize()` is inside `advance_ms` (**42.7 ms**). The plan's earlier reading
  ("anything outside `forward_ls` is adapter overhead") is wrong.
- **The adapter's own share is ~0.7 ms** on stage 1: `prepare` + H2D **0.62 ms**, D2H **0.09 ms**.
  Stage 0 (CUDA) is 0.15–0.42 + 0.10–0.17 ms.

Stage 1's honest breakdown at n=1:

```
prepare + H2D        0.62 ms
forward_ls (enqueue) 12.2  ms   <- mostly CPU enqueue; not the GPU wait
advance (incl. sync) 42.7  ms   <- real layer compute
D2H                  0.09 ms
delivery + wake      ~2.3  ms   <- the problem (when the package is cold)
```

### 2.3 Evidence for the idle-state hypothesis

Every hop is a `FORWARD` with a 64 KiB hidden payload; 20 decode tokens per run; same stages.

| Experiment | stage 1 `rtt − compute` | Reading |
|---|---|---|
| baseline, both stages remote | **3.0–3.2 ms** | cold package |
| busy core on the **Razer** | **2.82 ms** | Razer CPU state does not explain it |
| busy core on the **Strix** | **1.40 ms** | Strix idle state is the cause |
| coordinator heartbeat every 5 ms | 2.40 ms | partial wake |
| coordinator heartbeat every 1 ms | 1.85 ms | more wake |
| coordinator heartbeat, continuous | 0.92 ms | fully awake, but wasteful |
| stage 0 local, Strix cold | 3.04–3.09 ms | placement alone does not fix it |
| stage 0 local, Strix warm | 1.40–1.53 ms | same code, different power state |

The decisive pair is **busy Razer (2.82) vs busy Strix (1.40)**: the delay follows the *Strix* CPU's
idle state, not the link or the coordinator. A second pair shows the same for stage 0: a busy Razer
core reduces stage 0's remote hop from 1.35 to 0.65 ms; putting stage 0 in-process makes it 0.36–0.73.

### 2.4 Mechanism

1. Stage 1 sends its reply and blocks in `recv` (the handler thread is idle).
2. Stage 0 computes on the Razer for ~7 ms. During this window the stage's threads are blocked, so the
   `menu` governor picks **C3** (350 µs latency, 700 µs residency; residency confirms C3 dominates).
3. The 64 KiB request arrives. The core/package must exit C3 (and any package-level state), the
   handler thread is woken, the frame is read, and `prepare`/H2D runs.
4. That entry/exit path shows up as ~1.5 ms of `rtt − compute`, because the client's timer started
   when it sent and the server only begins `b.forward` after the wake.

The measured penalty (1.5 ms) is larger than the advertised 350 µs C3 latency. That is plausible if
the platform also enters a package-level state not exposed by per-core `acpi_idle`, if the ACPI
C3 entry/exit is slower than the advertised number, or if the wake is paid twice (request delivery
and reply delivery). **The exact state is inferred, not proven**; §7 is the confirming experiment.

### 2.5 What it is not, and what it is not worth

- Not the Thunderbolt link: a 64 KiB ping is 0.354 ms; a 5 ms heartbeat (which wakes both the
  link and the CPU) still leaves 2.40 ms, and a busy core fixes it without changing the link, so the
  local CPU state matters more.
- Not the adapter: `prepare`/H2D/D2H are ~0.7 ms and unchanged by the wake.
- Not a large win: 1.5 ms of a ~64 ms token is **~2.3%**. It is worth taking because it is cheap, but
  speculation is still the big decode lever.

### 2.6 Caveats

- One pair, one software stack (Python 3.14 / ROCm 7.13 / Kyojin 1.5.0). Re-measure after updates.
- The effect is bimodal; a "warm" run can hide it. Always test after a genuine idle period.
- The busy-core A/B used an external process. An in-process Python thread is **not** equivalent
  (GIL contention) — see §4.3.
- No wall-power meter was attached; §8 says exactly what to measure.
- `/dev/cpu_dma_latency` is root-only on this Strix, so the PM QoS option needs a one-time permission
  change (§4.2).

---

## 3. Performance impact

The measured configurations:

| Configuration | stage 0 wire+adapter | stage 1 wire+adapter | Total overhead | Decode |
|---|---|---|---|---|
| baseline (both remote, Strix cold) | 1.35–1.60 | 3.0–3.2 | **4.4–4.7 ms** | 15.05–15.31 tok/s |
| stage 0 in-process, Strix cold | 0.36–0.73 | 3.04–3.09 | 3.4–3.8 ms | 15.47 tok/s (smoke) |
| stage 0 in-process, Strix awake | 0.57–0.61 | 1.44–1.53 | **2.0–2.1 ms** | **15.67–15.75 tok/s** |

Expected change from the wake fix alone (assuming stage 0 remains in-process):

- Save **~1.5 ms** of a ~63.4 ms token → **~2.4%** → **+0.36–0.38 tok/s**.
- Combined with the in-process stage 0: **~2.4 ms** saved → **~3.7%** → 15.05 → **~15.6–15.8 tok/s**
  (measured 15.67–15.75).
- Prefill: only the first chunk after an idle period is affected; against an ~850 ms prefill it is
  noise.
- Per million generated tokens: ~2.4 ms × 1M = **~40 minutes** saved. (This is the honest way to size
  a 2–3% decode change.)

The same fix is *not* a substitute for speculation. Task 2 measured `g = 1.209` for MTP on Model A
(a lower bound); that is the larger lever (§4.7).

---

## 4. Fixes

Every option below is a way to handle the same 7 ms idle window. The real question is **where the
power/keep-alive lives**: in the request path, in the stage process, in the OS, or nowhere (global).

### 4.1 Bounded post-reply wake window (recommended, no privileges)

**What:** after sending a `FORWARD_OUT`, the stage polls its connection for `W` ms before falling
back to a blocking `recv`. The next request normally arrives in 7–9 ms, so a 12–20 ms window covers
the gap; after the last token the thread spends `W` ms awake and then sleeps normally.

**Sketch** (proposed; not implemented):

```python
# server.py, after send_frame for a FORWARD reply, before looping to recv_frame
if self.wake_window_ms > 0:
    deadline = time.perf_counter() + self.wake_window_ms / 1e3
    while time.perf_counter() < deadline:
        if select.select([c], [], [], 0.0002)[0]:   # 0.2 ms slice: C1/C2, not C3
            break
```

Config: `[stage] wake_window_ms = 15` (0 disables). Only for `FORWARD_OUT` on a configured session.

| | |
|---|---|
| **Pros** | No root; scoped to active decoding; no global state; self-restoring on crash; trivial to disable; no code outside the stage; works on any OS. |
| **Cons** | Burns CPU during generation; must exceed the real stage-0 gap (clock changes, queueing); the 0.2 ms-slice variant is **untested** (a pure spin is measured); one more tuning knob. |
| **Performance** | Measured equivalent (pure spin, external busy core): stage 1 **3.0 → 1.40 ms**; total **~2.0 ms**; **+0.36 tok/s**; decode **15.5 → 15.7–15.8** in isolation. |
| **Power during generation** | One core active for `W` ms per ~65 ms token. With `W = 15`: ~23% of one core at C0 (pure spin), or shallower with the 0.2 ms slice. The GPU is already at full tilt for 54 ms of each token. *Estimate:* a few watts marginal on a 32-thread SoC; **not measured**. |
| **Power at genuine idle** | **Zero** after `W` ms. This is the main advantage over every persistent option. |
| **Risk** | A pure `select(..., 0)` spin holds the GIL between calls; use a small timeout or a native helper if other Python threads matter. If the client disconnects, `select` reports readable and the normal `recv` path handles EOF. |

**Why this matches the "change the timings" idea:** it keeps the package out of C3 exactly for the
inter-token gap, and lets it go low-power as soon as the conversation stops.

### 4.2 PM QoS `/dev/cpu_dma_latency` (recommended if a permission change is acceptable)

**What:** a process opens `/dev/cpu_dma_latency`, writes a maximum acceptable exit latency, and keeps
the fd open. While held, cpuidle will not choose states above that latency. Closing the fd (normal
exit, crash, or kill) releases the constraint; no cleanup needed.

On this Strix: C1 = 1 µs, C2 = 18 µs, **C3 = 350 µs**. So:

| Value written | States allowed | Effect |
|---|---|---|
| `0` | POLL only | full spin, highest power |
| `20` | POLL, C1, C2 | blocks C3 (the deep one); likely enough |
| `400` | POLL, C1, C2, C3 | no useful constraint |

**Sketch** (proposed; not implemented):

```python
import os, struct
fd = os.open("/dev/cpu_dma_latency", os.O_WRONLY)
os.write(fd, struct.pack("i", 20))   # 20 us: C1/C2 allowed, C3 blocked
# keep fd open; os.close(fd) releases
```

Scope it to activity: acquire on the first `FORWARD`, refresh on each request, and a watchdog closes
the fd after `idle_release_s` (e.g. 60 s) with no requests; re-acquire on the next request. Then the
box behaves normally overnight, and only pays while a session is live.

**Prerequisite:** on the Strix the device is `crw------- root root` (verified); a normal user cannot
open it. One-time root change, e.g.:

```ini
# /etc/udev/rules.d/60-cpu-dma-latency.rules
KERNEL=="cpu_dma_latency", GROUP="<your-group>", MODE="0660"
```
then `udevadm control --reload && udevadm trigger` or reboot. (A systemd `chmod` oneshot works too.)

| | |
|---|---|
| **Pros** | Fine-grained (choose the latency), system-wide while held, no per-token CPU burn, auto-releases, no protocol/code changes in the request path. |
| **Cons** | Needs a one-time udev/permission change; affects every process on the host while held; must be tested on AMD `acpi_idle`; the right value (20 vs 50 µs) needs a sweep; the 1.5 ms penalty exceeding the 350 µs C3 latency means the exact state must be confirmed (§7). |
| **Performance** | Expected to recover the full **~1.5 ms** (stage 1 → ~1.4 ms) if C3/package state is the cause. **Untested on this hardware.** |
| **Power while held** | Idle cores sit in C1/C2; the package cannot deep-idle. More than normal idle, less than a C0 spin. **Estimate:** a few watts; measure. |
| **Power at genuine idle** | Normal after the idle timeout / process exit. |
| **Scoping** | Best of the persistent options: "on while swp-exl3 is working, off when it is genuinely idle". |

### 4.3 Dedicated keep-awake process (simplest, measured)

**What:** one `nice`d process pinned to a core, running a tight loop. This is exactly the A/B that
measured stage 1 at 1.40 ms.

```bash
nice -n 19 taskset -c 31 sh -c 'while :; do :; done' &
```

| | |
|---|---|
| **Pros** | No root, no code changes, measured, robust; can be started only for benchmarks. |
| **Cons** | Runs until killed; one core continuously; `nice` does not cap frequency; a Python **thread** in the stage is *not* a substitute (GIL contention). |
| **Performance** | Measured stage 1 **3.0 → 1.40 ms**; total ~2.0 ms. |
| **Power at genuine idle** | **The persistent option with the highest idle cost:** one core in C0 plus the package cannot deep-idle. *Estimate:* several watts continuously; measure. |
| **Scoping** | Start/stop with the stage via a wrapper or systemd; otherwise it is a 24/7 benchmark mode. |

### 4.4 Scoped system toggle (systemd; root)

**What:** disable only the deep state (C3) while the stage service runs, restore it on stop:

```ini
[Service]
ExecStartPre=/usr/bin/cpupower idle-set -d 3
ExecStart=/path/to/swp-exl3 stage --config /home/m/swp/stage-310-strix-base.toml
ExecStopPost=/usr/bin/cpupower idle-set -e 3
```

| | |
|---|---|
| **Pros** | No per-token CPU cost; no code; explicit restore; scoped to the stage's lifetime rather than the whole day. |
| **Cons** | Root; affects the whole host while the stage runs; **not** request-scoped, so genuine idle during a long-lived stage still pays; a manual `kill` that bypasses systemd can leave the state disabled (use the service). |
| **Performance** | Expected ~1.5 ms saved (same as the busy core). **Untested as a toggle.** |
| **Power** | Idle cores no longer enter C3 while the stage is up. Less than a C0 spin (C1/C2 still allowed); more than normal idle. |
| **Note** | Start with C3 only (`-d 3`). Disabling C2 too (`-d 2 -d 3`) is not obviously needed and costs more. |

### 4.5 Permanent global change (simplest; not recommended for an idle box)

**What:** disable deep C-states or cap C-state depth for the whole machine:

```bash
sudo cpupower idle-set -d 3              # only C3; minimal global change
sudo cpupower idle-set -d 2 -d 3         # also C2
# or kernel cmdline (AMD): processor.max_cstate=1
# extreme: idle=poll   (all CPUs spin; do not use except for a dedicated benchmark)
```

Also: `cpupower frequency-set -g performance` does **not** disable C-states; it only changes
frequency. It is orthogonal to this problem.

| | |
|---|---|
| **Pros** | Trivial, no code, applies to every workload, easy to verify. |
| **Cons** | Genuine-idle power rises permanently; affects everything on the box; not scoped; hides the real cause. |
| **Performance** | ~1.5 ms saved. |
| **Power** | **Highest genuine-idle cost:** all cores are limited whenever the machine is idle, not just while swp runs. On a 24/7 host this is the wrong trade for ~2%. |
| **Use** | Only on a dedicated benchmark box, or as the diagnostic that proves the C-state mechanism. |

### 4.6 Coordinator-side heartbeat (works, but wasteful)

**What:** keep a second connection from the coordinator and `PING` the Strix every N ms while a
generation is running.

Measured: 5 ms → 2.40 ms; 1 ms → 1.85 ms; continuous → 0.92 ms.

| | |
|---|---|
| **Pros** | No root, no stage code, can be toggled per run; confirms the wake mechanism. |
| **Cons** | Costs link traffic and CPU on **both** ends; continuous pinging competes with the coordinator's main thread for the GIL; less effective per unit of effort than a local keep-awake; N ms must be tuned. |
| **Power** | Constant tiny traffic plus CPU on both machines during generation; no idle cost only if the pinger stops with the generation. |
| **Verdict** | Diagnostic, not the production fix. |

### 4.7 Structural alternatives (the only way to remove the idle without burning power)

- **Speculation / MTP (plan §4, S3).** The Strix is idle ~7 ms per token while stage 0 computes. A
  drafter on the Strix could use that window, or the head stage could pipeline the next round. This
  is the "right" long-term answer: it turns idle time into work instead of burning a core. It is a
  0.5–1 day change, gated by Task 2 (`g = 1.209` on Model A, a lower bound), and is already the main
  decode lever. Not a quick fix.
- **Pipeline prefill.** The ceiling at this cut is 1.17x and it only helps prefill, not decode.
- **Change the split.** To make stage 0 shorter than the C3 residency threshold (~0.7 ms) is
  impossible with 7 layers; to make it longer makes the gap worse. Moving layers to the Razer is
  capacity-limited (3.18 GB free at cut 7). No win here.
- **Overlap stage 0 and stage 1 for decode.** Impossible for the same token: stage 1 needs stage 0's
  output. Only speculation creates work that can overlap.

### 4.8 Rejected or low-probability fixes

- **Wire dtype/quantisation:** the wire is a few percent of a token; bf16 costs accuracy for ~30 µs.
- **`SO_BUSY_POLL` / `net.core.busy_poll`:** needs a root sysctl, and it polls the NIC, not the CPU's
  package state; the idle window is ~7 ms, longer than a sane busy-poll budget.
- **TCP options:** `TCP_NODELAY` is already set; ping proves framing is ~0.35 ms; delayed ACK is not
  the issue.
- **Thread priority / `taskset`:** may shave scheduling latency but not the C-state exit; low expected
  effect, no power cost. Optional.
- **Thunderbolt/USB4/PCIe ASPM:** the busy-core A/B points at the CPU package, not the link; test
  only if §7 shows PM QoS does not help.

---

## 5. Comparison

| Option | Privileges | Scope | Stage-1 overhead | Decode gain | Idle power cost | Gen. power cost | Complexity |
|---|---|---|---|---|---|---|---|
| **4.1 wake window (15 ms)** | none | per request, auto-off | **~1.4 ms** (measured equivalent) | **+2.3%** | **none** | one core ~20% duty | low |
| **4.2 PM QoS (20 µs, 60 s idle)** | udev rule / root once | per session, auto-off | ~1.4 ms (*expected*) | +2.3% | none after timeout | shallow idle only | medium |
| **4.3 busy child** | none | process lifetime | **1.40 ms** (measured) | +2.3% | **high** (one core 24/7) | one core 24/7 | trivial |
| **4.4 systemd toggle (C3 off)** | root | stage lifetime | ~1.4 ms (*expected*) | +2.3% | high while stage runs | none extra | low |
| **4.5 global C-state off** | root | permanent | ~1.4 ms (*expected*) | +2.3% | **highest, permanent** | none extra | trivial |
| **4.6 heartbeat 1 ms** | none | while pinging | 1.85 ms (measured) | +1.2% | none | link + CPU both ends | low |
| **4.7 speculation** | none | — | eliminates the gap | ~20% (g = 1.209, Model A) | none | none | high |

Numbers in *italics* are the expected result, not yet measured with that specific mechanism.

---

## 6. Recommendation

1. **Implement 4.1 first** (`wake_window_ms`, default off, 15 ms). It is the closest match to "awake
   only between jobs", needs no privileges, and cannot affect genuine idle. Start with the 0.2 ms
   `select` slice to stay in C1/C2 rather than a C0 spin; if it does not recover the 1.5 ms, try a
   pure spin for the same window.
2. **In parallel, prepare 4.2**: add the udev rule for `/dev/cpu_dma_latency`, then test 20 µs with a
   60 s idle release. If it works, it is the lower-power steady-state answer and can replace 4.1.
3. **Do not** make 4.5 permanent on the Strix; it trades idle power for ~2% forever.
4. **Keep 4.3** as the benchmark/diagnostic tool; it is the simplest way to prove the mechanism and
   is not a deployment solution.
5. **Long term**, let speculation use the idle window (4.7); that is the only structural fix.

---

## 7. Verification plan (do this before trusting any fix)

Use the same bench prompt and 48 greedy tokens; run after 60 s of genuine idle and again immediately.

1. **Stage 1 overhead.** `generate --bench` with `pipeline-310-k7-local.toml`; look at the stage-1
   `wire+q` column. Acceptance: **≤ 1.6 ms** (from 2.8–3.1) and decode ≥ 15.6 tok/s. If it stays
   > 2.5 ms, the mechanism is not what this document says; stop and re-profile.
2. **Server-side wait.** With `SWP_DEBUG_TIMING=1`, the `[srv]` line for a FORWARD should show the
   same `handle` (~55 ms) and a shorter cold gap; the client's `rtt − stage_ms` is the metric that
   matters.
3. **Power, idle.** Attach a wall meter (or use `turbostat -i 1`; the Strix has it, no `powertop`).
   With the stage up but no requests for 60 s, record package power and C3 residency for: (a) normal,
   (b) each candidate fix active, (c) each fix after its idle release. A fix that still leaves C3
   residency near zero during that idle window is keeping the CPU awake when it should not, and is
   not acceptable for a shared host.
4. **Power, generation.** Run ~200 tokens and record average wall power for normal vs the fix. The
   GPU already dominates this window; expect a small delta for 4.1/4.2, a large one for 4.3/4.4.
5. **Confirm the state.** If possible, disable only C3 (4.4) and re-measure; if stage 1 drops to
   ~1.4 ms, C3 is confirmed. If not, record it and move the investigation to package states / NIC
   ASPM before writing more code.

---

## 8. Power accounting (what we know and do not know)

We did **not** measure wall power in this session. What is known:

- **Normal idle:** C3 holds ~99.8% of the Strix's idle residency (4.71e9 µs of C3 time vs 9.8e6 for
  C2), so the CPU is in its deepest exposed state almost all the time. Any fix that blocks C3 raises
  idle power; the question is by how much.
- **During generation:** the Strix GPU is busy ~54 of every ~65 ms. The CPU's marginal contribution
  is smaller there than at idle. The busy-core test did **not** change stage-1 compute
  (53.7 vs 54.4 ms, within noise), so there was no visible GPU-clock penalty, but a wall meter is
  needed to see the real board-power delta.
- **Unified memory:** the stage holds ~110 GB. Memory retention already sets a significant idle floor;
  the relative cost of one CPU core or shallow C-states may be smaller than on a normal laptop, but
  that is a hypothesis, not a measurement.
- **Order of magnitude (estimate, not measured):** a core held in C0 is a few watts; blocking C3 on
  idle cores is a few watts; a permanent global change is the largest. Use a wall meter; do not size
  a 24/7 decision from these ranges.

**Rule:** any option that keeps the machine shallow while `swp-exl3` is genuinely idle must justify
itself against ~2% decode. 4.1 and 4.2-with-timeout can; 4.3, 4.4, and 4.5 cannot for a shared host.

---

## 9. Implementation notes (this repo)

Proposed, minimal:

- `swp_exl3/config.py`: `StageConfig.wake_window_ms: int = 0` (or `pm_qos_us: int = 0` +
  `idle_release_s: int = 60`). Parse in `load_stage`.
- `swp_exl3/server.py`: in `_conn`, after `swp.send_frame(...)` for `swp.FORWARD_OUT` only, run the
  bounded poll before the next `recv_frame`. Add a `[stage]` log line at startup when enabled.
- `swp_exl3/cli.py` `cmd_stage`: for PM QoS, acquire/hold in the stage process (never in the
  coordinator); a watchdog thread with an idle timestamp. A failure to open the device is a warning,
  not a fatal error; the stage must still serve.
- Default **off** for both. The measured win is ~2%; the feature must not surprise a user with a
  power regression.
- PM QoS belongs to the stage process, not the coordinator. A coordinator-local stage
  (`stage_config`) would also benefit, but the host that runs the stage is the natural place.
- Tests: no GPU needed. Unit-test the wake-window loop against a fake socket pair (request arrives
  within/after the window; EOF; `wake_window_ms=0` is a no-op). PM QoS can be tested with a mocked
  `os.open`.

---

## 10. Appendix — raw session data

Environment: Razer `[0,7)` (Intel, RTX 3080 Ti, CUDA 13.3), Strix `[7,45)` (AMD Ryzen AI Max+ 395,
ROCm 7.13, Kyojin 1.5.0), f32 wire, chunk 2048, Model B 3.09 bpw.

Idle-state data (Strix):

```
driver acpi_idle, governor menu
states: POLL (0 us), C1 (1 us), C2 (18 us), C3 (350 us exit latency)
C3 min residency 700 us; C3 holds ~99.8% of idle time (4.71e9 us of state time)
/dev/cpu_dma_latency: crw------- root root  (user open fails: EACCES)
```

Bench/probe rows (20 decode tokens, mean):

| Config | stage 0 rtt−compute | stage 1 rtt−compute | Total |
|---|---|---|---|
| baseline, both remote | 1.37 | 3.08 | 4.45 |
| both remote + one `sendmsg`/pinned D2H | 1.38–1.60 | 3.00–3.08 | ~4.4–4.7 |
| stage 0 local, Strix cold | 0.36–0.73 | 3.04–3.09 | 3.4–3.8 |
| stage 0 local, Strix awake | 0.57–0.61 | 1.44–1.53 | 2.0–2.1 |
| remote Strix, one busy core | 1.42 | **1.40** | 2.82 |
| remote Strix, Razer busy core | 0.65 | 2.82 | 3.47 |
| continuous coordinator heartbeat | — | 0.92 | — |
| heartbeat every 1 ms | — | 1.85 | — |
| heartbeat every 5 ms | — | 2.40 | — |

48-token benchmark rows:

| Change | Decode | stage 0 | stage 1 | Overhead | Smoke |
|---|---|---|---|---|---|
| baseline (RESULTS row 6) | 15.05 | 7.181 | 54.794 | 4.47 | PASS |
| stage 0 in-process | 15.67 | 6.928 | 54.607 | 2.14 | PASS |
| + Strix busy core | 15.75 | 6.897 | 54.409 | 2.02 | PASS |
| final smoke, Strix cold | 15.47 | 6.350 | 54.839 | 3.40 | PASS |

Reproduction: `swp-exl3 generate --pipeline ~/models/pipeline-310-k7-local.toml --prompt <bench
prompt> --max-new 48 --bench`, after 60 s idle and again immediately; or the raw RPC probe used for
the per-hop timing dicts (`/tmp/kilo/t4_probe.py`, not in the repo).
