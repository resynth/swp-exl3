# swp-kyojin-server

One file (`swp_kyojin_server.py`) that serves a `swp-exl3` pipeline through **Kyojin's own OpenAI-compatible HTTP layer**
(`tools/glm/serve.py`, MIT): chat templates, tool-call and reasoning parsing, SSE streaming, `/v1/completions`,
`/v1/models`, `/health`, `/tokenize`, `/metrics`. It supplies only the engine object that `create_app(engine, model_id,
template)` expects (`PipelineEngine`). It uses the public `Pipeline` / `load_pipeline` / `contract_from_config` API; the
only swp-exl3 change it needed was additive (`Pipeline.generate` now also reports `rollback` / `replayed_tokens`).

## Run (on the Strix, in Kyojin's venv, with swp_exl3 installed there)
    python swp_kyojin_server.py --pipeline pipeline.toml --model-dir /models/GLM-5.3-Flash-exl3 \
        --kyojin-root ~/engines/kyojin --ctx 8192 --host 127.0.0.1 --port 8000 [--spec]
`--ctx` must be the SMALLEST stage `cache_tokens` (8192 in this example). For the measured long-context
configs set both stages and `--ctx` to the same larger value; see `RESULTS.md` T5. Start the two stage servers first.
Template: `--chat-template FILE`, else Kyojin's `tools/lanes/assets/glm53-template-medium.jinja`, else `<model>/chat_template.jinja`.
`--spec` enables the pipeline's prompt-lookup (n-gram) speculative decoding (`--spec-k`, `--spec-match`); output stays
greedy-exact and repetitive text speeds up ~1.7x. MTP is not wired here.

## What it does for multi-turn
Keeps the exact token ids of the previous prompt + reply. For the next prompt it keeps those ids for as long as the TEXT
is unchanged and re-tokenises only the rest, so appended turns do not re-prefill and never trigger a replay. If the client
edits or trims the reply (or the template strips reasoning), reuse stops at the first difference and the pipeline's
normal rollback applies (a full replay on this model, because recurrent state cannot rewind).

Each request prints one line to the server terminal:

    [req 2] HIT 101/113 | prefill 12 new + 101 cached in 281 ms (42.8 new tok/s, 402.8 effective) | decode 13 tok in 1119 ms (10.7 tok/s)
    [req 3] MISS (replayed 92) | prefill 93 new + 0 cached in 803 ms (115.9 new tok/s, 115.9 effective) | decode 8 tok in 553 ms (12.7 tok/s)
    [req 1] MISS | prefill 31 new + 0 cached in 623 ms (49.8 new tok/s, 49.8 effective) | decode 64 tok in 2411 ms (26.1 tok/s) | spec 48/48 (100.0%, 16 rounds)

`HIT n/m` = the common prefix was kept; `MISS` = no shared prefix; `MISS (replayed n)` = the prefix existed but the
recurrent state could not rewind, so it was recomputed. `new tok/s` is the rate over the rows actually fed; `effective`
divides the whole prompt by the prefill time (what the cache bought). `decode tok/s` is inter-token. `spec accepted/proposed
(%, rounds)` appears only when the request drafted: it is the fraction of drafted tokens the target accepted. `usage.prompt_tokens_details.cached_tokens`
reports only tokens that were actually reused: a clear+replay sets it to 0 even when the logical prefix was longer.
`Pipeline.generate` also returns `rollback` and `replayed_tokens`; the server copies both into `last_stats`.
With GLM's medium template, a client that wants a cache hit on the next turn should echo the previous
`reasoning_content` back in the assistant message; if it sends only `content`, the rendered prompt differs inside the
reply and the turn is replayed (expected).

## Tests (no GPU; Kyojin's real HTTP layer, fake pipeline and tokenizer)
    KYOJIN_ROOT=~/engines/kyojin SWP_EXL3_ROOT=~/swp-exl3 python -m unittest test_swp_kyojin_server -v
Covers: endpoints, non-stream output equals a direct pipeline run, stream equals non-stream, stop strings, multi-turn
append without replay, replay accounting on a trim (`cached_tokens = 0`, `replayed_tokens = kept`), exact-id splicing
and its fallbacks, cancel mid-stream, one-time warning for unsupported samplers, and the terminal HIT/MISS/rate/spec line. 11 tests.
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
* MTP is not wired (this adapter keeps `mtp = false` on the head stage). The n-gram speculative path is available with
  `--spec`; it is neutral on novel text and ~1.7x faster on repetitive output in the measured pair (see `RESULTS.md`).
* Prefill is silent until the first token; set Kyojin's `KYOJIN_SSE_PING_S` if a client times out on long prompts.
* Stage warm-up is a stage-server task (see the performance plan), not done here.
