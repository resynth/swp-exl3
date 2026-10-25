# swp-exl3

Stage server + coordinator that splits one exllamav3 model across machines over SWP/1 (the protocol
from `splitwire`). Each machine loads a contiguous layer range; the coordinator (star topology) relays
the boundary tensor and owns sampling.

Target: GLM-5.3-Flash. Seen working on a Strix Halo (Kyojin, ROCm) + a Razer (upstream exllamav3,
CUDA); see `BRINGUP.md` and `MEASUREMENTS.md` for a full run record and benchmarks.

- `swp-exl3` is one console script with subcommands: `doctor`, `stage`, `pull`, `ping`, `generate`,
  `smoke`, `logits`, `compare`. It can also be run as `python -m swp_exl3 ...`.
- Only `stage` is machine-local work (it loads a slice and serves it). The rest are coordinator-side.
- Every network command needs a shared secret: put `secret_env = "SWP_SECRET"` in the config and
  `export SWP_SECRET=...` on each machine, or set `insecure = true` on a trusted/direct link.

---

## 1. Install (once per machine)

```bash
# python >= 3.11; exllamav3 (or Kyojin) must already be importable
cd swp-exl3
python -m pip install numpy blake3 && python -m pip install -e .
# with uv (what this repo used):  uv pip install --python <venv>/bin/python numpy blake3 -e .
```

Then verify, on **each** machine:

```bash
# static check of every engine API the adapter uses, against the installed source tree
python tools/check_api_against_source.py \
  "$(python -c 'import exllamav3,os;print(os.path.dirname(exllamav3.__file__))')"

# runtime API check (imports torch + exllamav3 and inspects it)
swp-exl3 doctor
```

`doctor` exits non-zero if anything fails. A build of exllamav3 that has never been imported will
JIT-compile its CUDA/HIP extension on first use (minutes); if `import exllamav3` hangs, look for a
stale lock at `~/.cache/torch_extensions/*/<ext>/lock` (this torch blocks while the file exists).

Verified on the reference pair: **exllamav3 1.6.4** (Razer, CUDA 13.3) and **Kyojin 1.5.0** (Strix,
ROCm 7.13). Local suite: `python -m unittest discover -s tests` → **37 tests**, all pass with
`SWP_WEIGHTD` set; 1 self-skips without it (the provisioning test). The Kyojin HTTP integration
(`swp_exl3/integrations/kyojin-server/`) has a separate **9-test** no-GPU suite.

> Kyojin is often *not* pip-installed — its `serve.py` works from the repo root. Run `swp-exl3`
> with the engine repo on `PYTHONPATH`, e.g.
> `PYTHONPATH=/path/to/kyojin /path/to/kyojin/.venv/bin/swp-exl3 stage --config ...`.

Rust tools (optional — only needed for weight provisioning and link probing):

```bash
cd ../splitwire && cargo build --release     # binaries in target/release/
./target/release/swp-drive --selftest        # 18 conformance checks, no GPU
```

---

## 2. Command reference

### `swp-exl3 doctor` — check the engine, optionally dry-run a slice

```
swp-exl3 doctor [--model-dir DIR] [--layers A B] [--first | --last]
```

| argument | meaning |
|---|---|
| `--model-dir DIR` | if given, also build the model structure (no weights) and dry-run `slice_model` |
| `--layers A B` | half-open layer range `[A, B)` to test slicing for |
| `--first` | this stage also holds the prolog (embedding, stream expansion) |
| `--last` | this stage also holds the epilog (`hc_head`, final norm, `lm_head`) |

```bash
swp-exl3 doctor                                                   # API checks only
swp-exl3 doctor --model-dir /models/GLM-5.3-Flash-exl3 --layers 0 6 --first    # validate a first-stage slice
swp-exl3 doctor --model-dir /models/GLM-5.3-Flash-exl3 --layers 6 45 --last    # validate a last-stage slice
```

A dry run prints the architecture, `hc_mult`, the 64 KiB/row boundary size, the modules kept and the
recurrent/cache layer counts. It loads **no weights**, so it is safe and fast.

When the engine ships `exllamav3.modules.block_sparse_mlp_cpu`, `doctor` also reports the CPU expert
offload knobs and their defaults (`moe_cpu_offload`, `moe_cpu_split`, `moe_cpu_threads`,
`moe_cpu_pinned_arena`, `draft_moe_cpu_*`). An engine without that module gets a warning, not a
failure, because not every build supports CPU offload.

### `swp-exl3 pull` — fetch one stage's tensors from a `swp-weightd`

```
swp-exl3 pull --server ADDR --out DIR --layers A B [--first | --last]
              [--tied-embed] [--vision] [--secret-env VAR | --insecure]
```

| argument | meaning |
|---|---|
| `--server ADDR` | `host:port` of a running `swp-weightd serve` |
| `--out DIR` | output model directory (shard + config/tokenizer files) |
| `--layers A B` | half-open range to pull |
| `--first` / `--last` | also pull the embedding (first) / `hc_head`+`norm`+`lm_head`+MTP (last) |
| `--tied-embed` | last stage also needs the embedding table when `lm_head` is tied |
| `--vision` | include the vision tower (first stage; skipped by default) |

```bash
export SWP_SECRET=...
swp-exl3 pull --server 10.10.10.2:7700 --out /models/glm53-razer-slice \
  --layers 0 7 --first --secret-env SWP_SECRET
```

Writes `swp-shard-L0-7-first.safetensors` plus `config.json`, `quantization_config.json`,
`tokenizer*.json`, `chat_template.jinja`, `generation_config.json`, `processor_config.json` — a normal
model directory the engine loads unchanged. Bytes are hash-verified against the manifest.

### `swp-exl3 stage` — load a slice and serve it (run on each machine)

```
swp-exl3 stage --config FILE
```

There are **no other flags**; everything comes from the config file (§3.1). Example configs are in
`examples/stage-razer.toml` and `examples/stage-strix.toml`.

```bash
# Razer (stage 0)
SWP_SECRET=... swp-exl3 stage --config examples/stage-razer.toml
# Strix (stage 1), Kyojin on PYTHONPATH
ssh strix "set -a; . /home/m/swp/swp.env; set +a; cd /path/to/kyojin && \
  PYTHONPATH=$PWD exec ./.venv/bin/swp-exl3 stage --config stage-strix.toml"
```

On start it prints `[slice] layers A..B blocks=… recurrent=… cache_layers=…`, then `memory: {...}`,
then `[warmup] …` (three decode-size and four prefill-size synthetic forwards, then a cache reset;
`warmup = false` in `[stage]` disables it), then `[node] layers (A, B) listening on :PORT
(authenticated|INSECURE)`. Useful env: `SWP_DEBUG_TIMING=1` logs a per-forward breakdown
(`prepare/prefetch/forward_ls/flush/advance`).

### `swp-exl3 ping` — TCP round trip to each stage

```
swp-exl3 ping --pipeline FILE [--no-contract]
```

```bash
SWP_SECRET=... swp-exl3 ping --pipeline examples/pipeline.toml
# 10.10.10.1:7801  layers 0..7   median RTT 16 us   (planner: "link_rtt_ms": 0.016)
# 10.10.10.2:7802  layers 7..45  median RTT 130 us
```

Also runs `STAGE_CONFIGURE`, so it validates the layer ranges, the wire dtype and the model contract.

### `swp-exl3 generate` — end-to-end generation

```
swp-exl3 generate --pipeline FILE [--no-contract]
                  (--prompt TEXT | --prompt-file FILE | --ids a,b,c)
                  [--chat] [--system S] [--max-new N] [--temperature F] [--top-p F]
                  [--seed N] [--stop-id ID ...] [--bench] [--json FILE]
                  [--spec] [--spec-k N] [--spec-match N]
```

| argument | default | meaning |
|---|---|---|
| `--prompt` / `--prompt-file` / `--ids` | — | one of these is required (`--ids` is a comma-separated token list, skipping the tokenizer) |
| `--ids-file` | — | file of comma/space-separated token ids (for prompts too long for the command line) |
| `--chat` | off | wrap the prompt in GLM chat format (`[gMASK]<sop>…<\|assistant\|>`) |
| `--system` | — | system prompt, used with `--chat` |
| `--max-new` | `128` | max new tokens |
| `--temperature` | `0.0` | 0 = greedy (`want="token"`); >0 samples (`want="topk"`) |
| `--top-p` | `1.0` | nucleus sampling |
| `--seed` | — | RNG seed for sampling |
| `--stop-id` | — | extra stop token id (repeatable); tokenizer EOS and `<\|user\|>`/`<\|endoftext\|>`/`<\|observation\|>` are always stops |
| `--bench` | off | print prefill/decode tok/s and per-stage compute vs wire |
| `--json` | — | write the bench report to a file (with `--bench`) |
| `--spec` | off | speculative decode with prompt-lookup (n-gram) drafting: one verify forward per round proposes `k` tokens and accepts the matching prefix |
| `--spec-k` | `3` | drafts per verify round, capped by the stages' `spec_k` |
| `--spec-match` | `2` | n-gram match length used to look up the draft in the existing history |

```bash
export SWP_SECRET=...
# greedy, chat-formatted
swp-exl3 generate --pipeline examples/pipeline.toml \
  --prompt "What is the capital of France, and why is it famous?" --chat --max-new 64 --bench

# sampled (top-k is the pipeline's `topk`)
swp-exl3 generate --pipeline examples/pipeline.toml --prompt "Write a haiku about the sea." \
  --chat --temperature 0.8 --top-p 0.95 --seed 7 --max-new 48

# long prompt from a file, JSON report
swp-exl3 generate --pipeline examples/pipeline.toml --prompt-file prompt.txt \
  --max-new 256 --bench --json bench.json

# speculative decode (prompt-lookup drafting; best on repetitive output such as code or lists)
swp-exl3 generate --pipeline examples/pipeline.toml --prompt "Continue: red green blue red green blue" \
  --max-new 64 --spec --spec-k 3 --bench
```

`--bench` output (per stage: device-synced compute vs wire+queue, and how much of the token the stage
holds):

```
PREFILL 308 tok in 1420.4 ms -> 216.8 tok/s
DECODE  23 tok in 1539.6 ms -> 14.94 tok/s
  stage layers   prefill: compute  wire+q   decode/tok: compute  wire+q   busy
      0  0..7               202.0    17.4                7.327   1.453    11%
      1  7..45             1159.0    41.8               54.911   3.064    82%
```

Note: each `generate` is a fresh process, so the first prefill pays a one-off kernel warmup. Steady
state is much faster — see `MEASUREMENTS.md` §7.

### `swp-exl3 smoke` — fast correctness gate

```
swp-exl3 smoke --pipeline FILE [--golden FILE] [--update-golden] [--max-kl F]
```

Pings both stages, runs the bench prompt once (the first real prefill, printed against a warm repeat
to show the stage warm-up), runs two fixed greedy prompts for 48 tokens, checks one top-64 logit row
(top-1 equal and KL ≤ `--max-kl`, default `1e-4`), **compares 20 batched verify rows against
sequential decode logits** (top-1 equal, KL ≤ 1e-2; skipped when a stage advertises `spec_k = 0`),
and prints one `--bench` report. Warm run is about 20 s. The golden defaults to
`<pipeline>.golden.json`; create it once on a known-good build with `--update-golden`, then every
change must pass the same check. Exits non-zero on mismatch.

```bash
SWP_SECRET=... swp-exl3 smoke --pipeline ~/models/pipeline-310-k7.toml --update-golden
SWP_SECRET=... swp-exl3 smoke --pipeline ~/models/pipeline-310-k7.toml
```

### `swp-kyojin-server` — persistent OpenAI-compatible coordinator

`swp_exl3/integrations/kyojin-server/` runs a long-lived `Pipeline` behind Kyojin's own HTTP layer
(`tools/glm/serve.py`): `/v1/chat/completions` (stream, templates, reasoning/tool parsing), `/v1/completions`,
`/v1/models`, `/health`, `/tokenize`, `/metrics`. Start the stage servers first, then on a machine that has the Kyojin
checkout (the Strix):

```bash
python swp_kyojin_server.py --pipeline pipeline.toml --model-dir /models/GLM-5.3-Flash-exl3 \
  --kyojin-root ~/engines/kyojin --ctx 8192 --host 0.0.0.0 --port 8000 [--spec]
```

`--ctx` is the smallest stage `cache_tokens`. The server keeps history and reuses the longest common prefix, including
exact token ids of the previous reply, so appended turns prefill only the new text; a trimmed/edited reply triggers the
normal clear+replay and is reported as `replayed_tokens` (with `cached_tokens = 0`). Each request prints a terminal
line with the cache verdict (`HIT n/m` / `MISS` / `MISS (replayed n)`), prefill tokens and tok/s (new and effective),
decode tokens and tok/s, and — when drafting — the speculative acceptance (`spec 48/48 (100.0%, 16 rounds)`).
`--spec` enables the n-gram speculative path (neutral on novel text, ~1.7x on repetitive). Tool calls work non-stream
and streamed (`finish_reason: "tool_calls"`, `role: "tool"` follow-ups); `--ctx` (the smallest stage
`cache_tokens`) is the context cap, and over-context prompts get a clean HTTP 400. The pair is not limited
to 8192: 128k tokens were measured, and 576k fp16 / 768k q8 caches load (see §5.1 and `RESULTS.md` T5).
No auth/TLS and one request at a time — bind loopback or a trusted link.
See its `README.md` for tests and limits.

### `swp-exl3 logits` / `compare` — cut-invariance check

```
swp-exl3 logits  --pipeline FILE [--no-contract] (--prompt TEXT | --ids a,b | --ids-file FILE) [--k N] --save FILE
swp-exl3 compare A.json B.json [--max-kl F]
```

```bash
# save top-k logits at cut A and at cut B (different stage configs), then compare
swp-exl3 logits  --pipeline cutA.toml --prompt "The capital of France is" --k 64 --save a.json
swp-exl3 logits  --pipeline cutB.toml --prompt "The capital of France is" --k 64 --save b.json
swp-exl3 compare a.json b.json
# A cuts [[0,6],[6,45]] wire=f32   B cuts [[0,4],[4,45]] wire=f32
# top-1 equal: True  shared top-k ids: 63/64  max |dlogit|: 0.09375  KL(A||B): 5.30e-05
```

`compare` exits non-zero if top-1 differs or KL ≥ `--max-kl` (default `1e-3`).

---

## 3. Configuration

### 3.1 `stage.toml` (one per stage)

```toml
[stage]
node        = "razer"            # name shown in logs
listen      = "0.0.0.0:7801"     # host:port to bind
secret_env  = "SWP_SECRET"       # or: insecure = true
model_dir   = "/models/glm53-razer-slice"
layers      = [0, 7]             # [first, end) — must match the pipeline file
embed       = true               # first stage: holds the embedding
head        = false              # last stage: holds hc_head + norm + lm_head
device      = "cuda:0"           # single device (HIP also presents as cuda:0)
cache_tokens = 8192              # KV/recurrent cache; multiple of 256
cache_bits  = 0                  # 0 = fp16 KV cache, 8 = packed int8 KV (CacheLayer_quant; longer context)
max_chunk   = 512                # max rows per FORWARD
wire_dtypes = ["f32", "bf16"]    # accepted wire dtypes
warmup      = true               # synthetic forwards after load() so the first real prefill is not cold
spec_k      = 3                  # speculative draft window this stage can roll back (0 disables `spec`)
mtp         = false              # head stage only: load the MTP drafter + embedding copy (see §6)
# mtp_model_dir = "..."          # pack holding the MTP/embed tensors (default: model_dir)
# mtp_cache_tokens = 8192        # drafter cache (default: cache_tokens)
# recurrent_dtype = "half"       # ignored where the engine's Cache lacks it (warned)
# max_history = 0                # extra attention history; auto-raised to spec_k for verify windows

# optional: set on config.infer_params before the model is built (engine passthrough)
[stage.engine]
moe_cpu_split = 32               # or moe_cpu_offload = N   (see §5)

# optional: extra kwargs for model.load()
[stage.load]

# optional: environment variables set before exllamav3 is imported
[stage.env]
```

### 3.2 `pipeline.toml` (coordinator)

```toml
[pipeline]
model_dir   = "/models/glm53-razer-slice"   # for the tokenizer + contract check; optional
secret_env  = "SWP_SECRET"                  # or: insecure = true
wire_dtype  = "f32"                         # "f32" or "bf16"
chunk       = 512                           # prefill rows per FORWARD
topk        = 64                            # top-k kept when sampling

[[stage]]
addr   = "10.10.10.1:7801"
layers = [0, 7]

# alternative: run this stage inside the coordinator process (no socket hop) -----------------
# [[stage]]
# stage_config = "/models/stage-razer.toml"    # a normal [stage] file; loaded by the coordinator
# layers       = [0, 7]

[[stage]]
addr   = "10.10.10.2:7802"
layers = [7, 45]
```

`[[stage]]` entries are in pipeline order; ranges must be **contiguous starting at 0**. Start the
stages before `ping`/`generate`.

An entry with `stage_config` instead of `addr` runs that stage **inside the coordinator process**:
the coordinator loads the slice and calls the backend directly (no socket, no SWP framing). Use it
for a stage on the coordinator's own machine, typically stage 0. Measured saving on this pair:
~0.7–0.8 ms/token. Any coordinator command that connects (`ping`, `generate`, `smoke`, `logits`)
loads such a stage, so it costs the same load time as a standalone stage.

### 3.3 Environment variables

| variable | where | effect |
|---|---|---|
| `SWP_SECRET` (name set by `secret_env`) | every machine | shared secret for the keyed-hash challenge |
| `SWP_WEIGHTD` | test host | path to the Rust `swp-weightd`, enables the provisioning test |
| `SWP_DEBUG_TIMING=1` | stage | log per-forward `prepare/prefetch/forward_ls/flush/advance` ms |
| `ROCPROFILER_REGISTER_ENABLED=0` | ROCm stage | set automatically by the adapter (before torch) to stop a core spinning |
| `PYTHONPATH` | Kyojin stages | point at the Kyojin checkout (it is usually not pip-installed) |
| `EXL3_MTP_EH_FP16` | MTP head stage | optional unquantized `eh_proj` sidecar (`0` disables); read by `load_eh_proj_sidecar` |
| `EXL3_MTP_PRENORM_H=1` | MTP head stage | tap the pre-norm hyper-head state (default off; matches the llama.cpp reference) |

---

## 4. Quick start (two machines)

```bash
# 0. both machines: install, doctor (§1)
# 1. on the machine that HAS the weights, serve them (Rust)
export SWP_SECRET=...
swp-weightd serve --model /data/hf/hub/models--turboderp--GLM-5.3-Flash-exl3/snapshots/<rev> \
  --listen 0.0.0.0:7700 --secret-env SWP_SECRET

# 2. on the other machine, pull this stage's slice
export SWP_SECRET=...
swp-exl3 pull --server <weights-host>:7700 --out /models/glm53-razer-slice \
  --layers 0 7 --first --secret-env SWP_SECRET

# 3. dry-run the slice against the real classes (no weights)
swp-exl3 doctor --model-dir /models/glm53-razer-slice --layers 0 7 --first

# 4. start both stages (each with its own stage.toml and this machine's model_dir),
#    then, from the coordinator machine:
swp-exl3 ping    --pipeline examples/pipeline.toml
swp-exl3 generate --pipeline examples/pipeline.toml --prompt "Hello" --chat --max-new 32 --bench
```

---

## 5. Sizing the split (levers that matter)

The cut and the expert placement are the two knobs; both are measured in `MEASUREMENTS.md`.

* **Cut point** `layers = [0, K)` on the first stage. Use `doctor --layers A B` to dry-run it. A stage
  holds only its range, so `model_dir` may point at a shard containing more layers than the stage uses.
* **Expert placement** — put MoE experts in the Razer's RAM/CPU to hold more than its 16 GB GPU:
  * `moe_cpu_split = N` — the tail `N` of every MoE layer's experts run on a CPU worker and overlap
    the GPU; fills the GPU more for the same CPU budget.
  * `moe_cpu_offload = N` — the whole routed-expert set of the first `N` MoE layers moves to CPU
    (biggest VRAM saving).
  * Both need **layer-split mode** (a single `device`) and the **`mul1`** codebook; they are mutually
    exclusive and are ignored in tensor-parallel mode.
* **Wire dtype** — `bf16` halves boundary bytes but the link is rarely the limit and it costs accuracy
  (KL ~9e-4 vs f32). Measured on this hardware: f32 kept.
* Probe the link first: `swp-drive --probe --stages <host>:7802 --hidden 16384` (the boundary row is
  `hc_mult*hidden*4 = 64 KiB` at f32).

### 5.1 Context and quantized KV

The 8192-token cap in earlier notes was a **config value** (`cache_tokens` on the Razer/Strix, and
`--ctx 8192` on the server), not a model or engine limit: the checkpoint declares
`max_position_embeddings = 1,048,576`. At cut 7 the Razer holds only one global-attention layer, so
the Strix's 10 attention layers are the binding KV budget. Measured on Model B, f32, warm stages:

| stage | KV | largest cache that loads + warms | free after load |
|---|---|---|---|
| Razer `[0,7)` | fp16 | 832k | 1.83 GB |
| Strix `[7,45)` | fp16 | **576k** | 5.94 GB |
| Strix `[7,45)` | q8 | **768k** (832k absolute ceiling, slow first forward) | 6.57 GB (5.84 GB) |

`cache_bits = 8` selects the packed int8 MLA/DSA pages (`CacheLayer_quant`) and raises the Strix
ceiling from 576k to 768–832k. It is **opt-in and not bit-identical**: top-1 logits stay equal at
8k/32k, but greedy text drifts from the fp16 golden, so keep `0` unless a context above ~576k is the
goal. On an engine without the packed cache it is a startup error (not a silent fp16 fallback);
`doctor` warns when the capability is absent. q8 is also **Strix-only** here: on the Razer's CUDA
build the one quantized MLA layer costs 63.9 ms/token at 8k vs 8.6 ms fp16.

Throughput at long context (Model B, cut 7, chunk 2048, fp16 on Razer):

| prompt | Strix KV | prefill tok/s | decode tok/s |
|---|---|---|---|
| 8,192 | fp16 | 456.6 | 13.34 |
| 8,192 | q8 | 453.2 | 13.20 |
| 32,768 | fp16 | 457.1 | 12.97 |
| 32,768 | q8 | 448.9 | 13.55 |
| 131,072 | fp16 | 443.5 | 11.27 |
| 131,072 | q8 | 414.5 | 9.40 |

For long prompts, size the cache for **prompt + generation + spec window** (a forward past
`cache_tokens` is now rejected as `context_exceeded`), and keep headroom: prefill workspace grows with
`chunk × context`, and the 832k q8 config could not prefill 128k tokens at chunk 2048 even though it
loaded. The 128k rows above use a 256k cache. Full tables and caveats are in `RESULTS.md` T5.

---

## 6. What GLM-5.3-Flash is (from `Glm5NextModel`)

Hybrid KDA linear attention (recurrent state) + MLA/DSA, MoE, **mHC hyper-connections**, MTP head.

* Boundary tensor = `[rows, hc_mult=4, hidden=4096]` fp32 = **64 KiB per row**. For
  `swp-drive --probe` use `--hidden 16384`.
* The hyper-connection residual update can be *deferred* (`hc_pending`); a stage must flush it before
  sending. Upstream 1.6.x: `hc_flush(params)`; Kyojin: `HyperConnection.flush_pending`.
  `flush_hyperconnections()` handles both.
* KDA layers have recurrent state that cannot rewind to an arbitrary position: `truncate(pos)` is
  exact for `pos >= cache_len` and `pos == 0`; anything else returns `unsupported` and the
  **coordinator clears all stages and replays its token history**. Appending turns needs no rollback.

---

## 7. Limits of this version

* Single device per stage (`device = "cuda:0"`); autosplit loading is rejected (its dummy pass assumes
  token ids as input). CPU expert offload is likewise layer-split-only.
* **Prompt-lookup (n-gram) speculative decoding is implemented** (SWP capability `spec`): the coordinator
  proposes the next `k` tokens from an earlier occurrence of the last 2–3 tokens, one verify forward
  checks `k+1` rows on both stages, and rejected rows are rewound lazily on the next round (or by
  clear+replay when a stage cannot). Sampling (`temperature > 0`) uses "sample target, accept on equality" in
  the head stage. Batched verify logits are not bit-identical to 1-row decode on this engine (measured:
  1-row verify exact, 2-row KL 3e-5, 3/4-row up to 5e-3; top-1 equal throughout, and a plain batched
  prefill shows the same delta — it is the engine's batched GDN/attention kernels, not SWP). No prefill
  pipelining across stages; star topology (coordinator relays).
* **MTP drafting on the head stage is implemented but off by default** (`mtp = true` enables it; needs
  Kyojin's `Glm5NextMTPModel`). The head loads the MTP block, a read-only embedding copy and a separate
  drafter cache, exports the target's final-norm hidden state on every forward, and returns the next
  round's drafts (`drafts` / `drafts_basis`). The coordinator prefers an n-gram match and falls back to
  MTP on novel text. Measured on Model B (3.09 bpw, 149-token novel prompt): acceptance 76.5 / 58.8 /
  29.4% at depths 1/2/3, best decode **15.5 tok/s vs 15.05 plain** (~1.03x) — below the plan's 1.15x
  gate, so the n-gram path remains the default. Capacity: +5.0 GB on the Strix (MTP 3.76 GB + embed
  1.27 GB); 116.3 GB allocated, 10.1 GB free at cache 8192. Details in `RESULTS.md` (S3).
* Prefill uses `forward_ls` on every stage (no early-exit `prefill_ls` shortcut), so the last stage
  also runs the head on the final row of each chunk.
* bf16 wire and `topk` output are implemented; the cut/offload equivalence checks in `MEASUREMENTS.md`
  cover them on real engines. Authentication is not encryption.
* `cache_bits = 8` (packed int8 MLA/DSA pages) is opt-in and **not bit-identical**: it raises the Strix
  cache ceiling from 576k to 768–832k tokens, but greedy text drifts from the fp16 golden. It is also
  slow on the Razer's CUDA engine, so use it on the Strix only. `cache_bits = 0` remains the default.
* Every `generate` is a fresh process and pays a per-process kernel warmup on the first prefill; a
  long-lived coordinator would remove that.
* Remote latency depends on the peer's CPU power state: after ~7 ms idle (the time the other stage
  computes), a stage on a host with deep C-states enabled pays ~1.5 ms extra to receive the next
  request. This is platform, not protocol; keep the host awake (performance governor / no deep idle)
  for latency-critical stages. See `RESULTS.md` Task 4.

---

## 8. Dependencies on this machine

Fixes applied to the *hosts*, not to this repo. Re-cloning or reinstalling the engine silently undoes
them; check these first when a previously working pair fails to start.

* **exllamav3 CUDA extension vs CUDA 13.x.** CUDA 13.3 `nvcc` cannot parse
  `torch/include/ATen/core/List_inl.h` at `-std=c++17` (`need 'typename' before … difference_type`).
  `-std=c++20` compiles it. Patch the compiler flags in `exllamav3/ext.py` (both `extra_cflags` and
  `extra_cuda_cflags`) — this lives in the dependency checkout, so a re-clone loses it. Torch only
  appends its own `-std=c++17` when the nvcc command has no `-std=` flag, so carrying `-std=c++20`
  suppresses the override.
* **Stale torch-extension build lock.** If `import exllamav3` hangs, delete
  `~/.cache/torch_extensions/py<XY>_cu<NNN>/exllamav3_ext/lock`. This torch's `FileBaton` waits while
  the lock file merely exists (`O_CREAT|O_EXCL`), so a lock left by an interrupted build blocks import
  forever.
* **ROCm idle spin (Strix).** `ROCPROFILER_REGISTER_ENABLED=0` must be set *before* torch is imported,
  or an HSA profiler thread burns a full core for the process lifetime (measured: 210 → 0 jiffies per
  2 s). `Exl3Backend.__init__` sets it before importing torch; anything else that imports torch first
  must export it itself.
* **Kyojin is not pip-installed.** Run it from the checkout with the repo on `PYTHONPATH` (see §1);
  otherwise `swp-exl3 stage` cannot import the engine.
