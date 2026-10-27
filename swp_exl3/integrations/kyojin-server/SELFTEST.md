# Self-test report — swp-kyojin-server (ouroboros run)

Date: 2026-10-27, ~12:30–12:55 UTC, on the Razer while being served *by* this stack
(coordinator on the Strix at `10.10.10.2:8000`, stages `[0,7)` here / `[7,45)` there).
Goal: exercise the integration's code paths from the inside — unit suite plus live
HTTP requests against the real server, including the tool-call loop.

## Unit suite (no GPU, real Kyojin HTTP layer, fake pipeline)

- Before changes: **10/10 pass** (`README.md` said 9 — doc drift, fixed in both READMEs).
- Added `test_over_context_gets_400`: the `ctx` clamp / HTTP 400 admission path in
  `PipelineEngine.generate` (`swp_kyojin_server.py:201-206`) had no coverage.
- After changes: **11/11 pass** in 0.8 s.

## Live tests against the running server (real model, real tokenizer, real stages)

| check | result |
|---|---|
| `GET /health`, `GET /v1/models` | OK |
| chat completion, non-stream | OK (greedy text, usage, timings) |
| tool call, non-stream | OK — `finish_reason: "tool_calls"`, `{"city": "Paris"}` |
| tool call, SSE stream | OK — role delta, `tool_calls` delta, `finish_reason`, `[DONE]` |
| `role: "tool"` follow-up | OK — model consumed the tool result and answered |
| `POST /tokenize` | OK (`count`, `max_model_len`, `tokens`) |
| `/v1/completions` with token-id prompt | OK (bypasses template and `_ids_for`) |
| multi-turn append with echoed reply (+ `reasoning_content`) | OK — **HIT 99/107**, `cached_tokens: 99` |

### The "cache always misses" investigation (not a bug)

Repeated identical requests reported `cached_tokens: 0`, which looked like broken
prefix reuse. Traced through `serve.py` (`admit`/`cache_salt`: no spurious resets)
and `coordinator.py` (`generate` → `rollback`): a repeated prompt *trims* the
previous reply out of the history, and KDA recurrent state cannot rewind to a
mid-history position, so the pipeline clears and replays — `cached_tokens = 0` is
the designed, correct verdict (`test_trim_reports_no_cache_hit_and_counts_the_replay`
encodes exactly this). Same for an id-prompt "append" that does not include the
previous reply's tokens. A genuine HIT requires the next prompt to extend the full
previous prompt+reply — confirmed live with the echoed-reply chat turn (99/107).

## Changes made

1. `swp_kyojin_server.py` — no code changes needed; behaviour matched its contract everywhere.
2. `test_swp_kyojin_server.py` — added the over-context 400 test (now 11 tests).
3. `README.md` (this dir) and repo `README.md` — test count 9 → 10 (→ 11 after the new test is noted here; repo README updated to 10).

## Still open / not covered live

- Server terminal lines (`[req n] HIT/MISS ...`) go to an interactive pts on the
  Strix and could not be read remotely; verdicts were inferred client-side from
  `usage.prompt_tokens_details.cached_tokens` and `timings`.
- Over-context HTTP 400 not exercised against the live server (would need a
  >256k-token prompt; covered by the new unit test instead).
- Cancel mid-stream and spec-decode acceptance not re-run live (covered by unit tests;
  live spec behaviour already measured in `RESULTS.md`).
- Kyojin on the Strix is a different checkout than `/home/b/ai/kyojin` here; the
  deployed `swp_kyojin_server.py` and `coordinator.py` were diffed against this repo —
  in sync (one cosmetic difference only).