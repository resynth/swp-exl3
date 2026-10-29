# swp-kyojin-server

One file (`swp_kyojin_server.py`) that serves a `swp-exl3` pipeline through **Kyojin's own OpenAI-compatible HTTP layer**
(`tools/glm/serve.py`, MIT): chat templates, tool-call and reasoning parsing, SSE streaming, `/v1/completions`,
`/v1/models`, `/health`, `/tokenize`, `/metrics`. It supplies only the engine object that `create_app(engine, model_id,
template)` expects (`PipelineEngine`). It uses the public `Pipeline` / `load_pipeline` / `contract_from_config` API; the
only swp-exl3 change it needed was additive (`Pipeline.generate` now also reports `rollback` / `replayed_tokens`).

## Run (on the Strix, in Kyojin's venv, with swp_exl3 installed there)
    python swp_kyojin_server.py --pipeline pipeline.toml --model-dir /models/GLM-5.3-Flash-exl3 \
        --kyojin-root ~/engines/kyojin --ctx 8192 --host 127.0.0.1 --port 8000 \
        [--spec] [--spec-k 3] [--spec-mtp-k 2] [--spec-match 4] [--no-mtp] [--spec-log FILE]
`--ctx` must be the SMALLEST stage `cache_tokens` (8192 in this example). For the measured long-context
configs set both stages and `--ctx` to the same larger value; see `RESULTS.md` T5. Start the two stage servers first.
Template: `--chat-template FILE`, else Kyojin's `tools/lanes/assets/glm53-template-medium.jinja`, else `<model>/chat_template.jinja`.
`--spec` enables the pipeline's speculative decoding: prompt-lookup (n-gram) first, then the head stage's MTP
drafter as the fallback. `k` (`--spec-k`, default 3) is the number of drafts per round; the verify window is
`k+1` rows. `n` (`--spec-match`, default **4**) is a **strict minimum match length**: the last `n` tokens must
already occur earlier in the context, and there is no shorter fallback, so a weak 3-gram cannot suppress MTP.
`--spec-match 2` is the aggressive opt-in (old 2-gram lookup); a very large value (`--spec-match 99`) turns
the n-gram path off and leaves MTP/plain. `k` is not `n`: for repetitive n-gram work use `--spec-k 7`; `k = 8`
(B = 9) crosses Kyojin's 8-row caps (verify 232 ms vs 156 ms at B = 8). `--spec-mtp-k` is the independent MTP
window (default 2). `--no-mtp` keeps exact n-gram only. Repetitive text stays byte-identical and speeds up
~1.5-1.9x; novel prose can differ from plain on a batched-verify near-tie, and MTP output is not guaranteed exact.
`--spec-log FILE` appends one JSON object per request (`last_stats`, `timestamp`, `request` id),
rotating at 64 MiB and keeping two backups.

## What it does for multi-turn
Keeps the exact token ids of the previous prompt + reply. For the next prompt it keeps those ids for as long as the TEXT
is unchanged and re-tokenises only the rest, so appended turns do not re-prefill and never trigger a replay. If the client
edits or trims the reply (or the template strips reasoning), reuse stops at the first difference and the pipeline's
normal rollback applies (a full replay on this model, because recurrent state cannot rewind).

Each request prints one line to the server terminal:

    [req 2] HIT 101/113 | prefill 12 new + 101 cached in 281 ms (42.8 new tok/s, 402.8 effective) | decode 13 tok in 1119 ms (10.7 tok/s)
    [req 3] MISS (replayed 92) | prefill 93 new + 0 cached in 803 ms (115.9 new tok/s, 115.9 effective) | decode 8 tok in 553 ms (12.7 tok/s)
    [req 1] MISS | prefill 31 new + 0 cached in 623 ms (49.8 new tok/s, 49.8 effective) | decode 64 tok in 2411 ms (26.1 tok/s) | spec n-gram 12r 44/44 | mtp 4r 9/12 | plain 1

`HIT n/m` = the common prefix was kept; `MISS` = no shared prefix; `MISS (replayed n)` = the prefix existed but the
recurrent state could not rewind, so it was recomputed. `new tok/s` is the rate over the rows actually fed; `effective`
divides the whole prompt by the prefill time (what the cache bought). `decode tok/s` is inter-token. The spec mix
appears only when the request drafted: `n-gram`/`mtp` show rounds and accepted/drafted per source, and `plain` counts
1-token decodes with no draft. `usage.prompt_tokens_details.cached_tokens`
reports only tokens that were actually reused: a clear+replay sets it to 0 even when the logical prefix was longer.
`Pipeline.generate` also returns `rollback` and `replayed_tokens`; the server copies both into `last_stats`, along
with the per-source counters (`spec_ngram_*`, `spec_mtp_*`, `spec_plain_rounds`, `spec_lookup_*`) and the
active floor (`spec_match`, so a `--spec-log` records the policy).
With GLM's medium template, a client that wants a cache hit on the next turn should echo the previous
`reasoning_content` back in the assistant message; if it sends only `content`, the rendered prompt differs inside the
reply and the turn is replayed (expected).

## Tests (no GPU; Kyojin's real HTTP layer, fake pipeline and tokenizer)
    KYOJIN_ROOT=~/engines/kyojin SWP_EXL3_ROOT=~/swp-exl3 python -m unittest test_swp_kyojin_server -v
Covers: endpoints, non-stream output equals a direct pipeline run, stream equals non-stream, stop strings, multi-turn
append without replay, replay accounting on a trim (`cached_tokens = 0`, `replayed_tokens = kept`), exact-id splicing
and its fallbacks, cancel mid-stream, one-time warning for unsupported samplers, the terminal HIT/MISS/rate/spec line,
the source mix, `--spec-log` JSONL and rotation (including `spec_match`), and the `--spec-mtp-k` / `--spec-match` /
`--no-mtp` flags. 17 tests.
torch is stubbed only if it is not installed.

## Agent / chat-UI use
Works as a normal OpenAI endpoint (`http://<host>:8000/v1`, any/no API key): `/v1/models` lists `--model-id`,
chat/completions supports streaming, stop, `max_tokens`/`max_completion_tokens`, temperature/top_p/seed, tools and
`role: "tool"` follow-ups (verified non-stream and stream, `finish_reason: "tool_calls"`). Keep total prompt
(system prompt + tool schemas + history) under the served `--ctx` (8192 in the example; see `RESULTS.md`
T5 for the measured 128k/long-cache configs); over-context requests get a clean
HTTP 400. One request at a time; no authentication or TLS. Keep the default `--host 127.0.0.1`
and reach it remotely through an SSH tunnel (`ssh -L 8000:127.0.0.1:8000 m@host`); do **not**
use `--host 0.0.0.0` on a machine that is on the internet.

## Not covered / limits
* Verified against a Kyojin **1.5.0** checkout on the Strix (the 1.5.10 notes still apply). The module depends only on
  `serve.create_app` and `serve.DEFAULT_TEMPLATE`. Exercised end-to-end with the real exllamav3 `Tokenizer` on the
  Strix (Model B `[7,45)` head stage + Razer `[0,7)`), including multi-turn reuse, streaming, cancel and sampling.
* Sampling is whatever the head stage returns (top-64): temperature, top_p, top_k, seed. min_p, penalties, logit_bias,
  grammar are ignored (logged once). No images. One request at a time (Kyojin's handler queues the rest).
* MTP is wired as the `--spec` fallback (head stage `mtp = true`, Kyojin `Glm5NextMTPModel`); `--no-mtp` keeps
  exact n-gram only. The n-gram path is neutral on novel text and ~1.5-1.9x faster on repetitive output; the
  strict floor (`--spec-match 4`) avoids the weak 2-gram rounds that used to suppress MTP. MTP is ~1.4-1.7x on
  novel text but is **not** guaranteed byte-identical to plain greedy (see `RESULTS.md`).
* Prefill is silent until the first token; set Kyojin's `KYOJIN_SSE_PING_S` if a client times out on long prompts.
* Stage warm-up is a stage-server task (see the performance plan), not done here.
