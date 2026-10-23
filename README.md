# swp-exl3

Stage server + coordinator that split one exllamav3 model across machines over SWP/1 (the protocol from `splitwire`).
Target: GLM-5.3-Flash on a Strix Halo (Kyojin 1.6.0, ROCm) + a Razer (upstream exllamav3 1.6.4, CUDA).

## What was verified, and what was not
Verified here (no GPU, no torch available in the authoring environment):
* Protocol, server, coordinator, chunking, rollback/replay, reconnect recovery, auth, bf16 wire, bench accounting, CLI
  workflow: 14 tests against a numpy fake with the **same boundary shape as GLM-5.3** (`[rows, 4, 4096]`) and
  recurrent state that cannot be rewound (`python3 -m unittest discover -s tests`).
* Provisioning against the Rust `swp-weightd` (byte-identical shards, correct stage selection): `SWP_WEIGHTD=/path/swp-weightd`.
* Slicing logic against fakes mirroring `Glm5NextModel`'s module list.
* Every engine API the adapter touches exists, with the used signatures, in **upstream 1.6.4** (48/48) and in a
  **Kyojin 1.5.10** checkout (47/48; its `Cache` lacks `recurrent_dtype`, which the backend skips with a warning):
  `python tools/check_api_against_source.py <path to exllamav3 package dir>`.

**Not verified: any run against the real engine.** `exl3_backend.py` is written from reading the source. Run
`swp-exl3 doctor` and the tools/ checker on both of your machines first (your Kyojin is 1.6.0, which I have not seen).

## What GLM-5.3-Flash is (read from the source: `Glm5NextModel`)
Hybrid KDA linear attention (recurrent state) + MLA/DSA, MoE, **mHC hyper-connections**, MTP head. Consequences:
* Boundary tensor = `[rows, hc_mult=4, hidden=4096]` fp32 = **64 KiB per row** (not 16 KiB). For `swp-drive --probe` use `--hidden 16384`.
* The hyper-connection residual update can be *deferred* (`hc_pending`); a stage must flush it before sending its output.
  Upstream 1.6.x: `hc_flush(params)`; Kyojin 1.5.10: `HyperConnection.flush_pending`. `flush_hyperconnections()` handles both.
* KDA layers have recurrent state: it cannot rewind to an arbitrary position. `truncate(pos)` is exact for `pos >= cache_len`
  and `pos == 0`; anything else returns `unsupported` and the **coordinator clears all stages and replays its token history**.
  Appending turns/continuations (prefix reuse) needs no rollback.

## Install (each machine)
    pip install numpy blake3 && pip install -e .          # python >= 3.11; exllamav3 / kyojin must already be installed
    python tools/check_api_against_source.py "$(python -c 'import exllamav3,os;print(os.path.dirname(exllamav3.__file__))')"
    swp-exl3 doctor                                        # runtime API check
    swp-exl3 doctor --model-dir /models/GLM-5.3-Flash-exl3 --layers 0 20 --first     # dry-run slice, loads no weights

## Run order
1. `swp-exl3 doctor --model-dir ... --layers A B [--first|--last]` on each machine (validates slicing against the real classes).
2. Optional: provision slices over SWP instead of copying files. Strix: `swp-weightd serve --model DIR --secret-env SWP_SECRET`.
   Razer: `swp-exl3 pull --server strix-tb:7700 --out /models/glm53-razer-slice --layers 0 20 --first --secret-env SWP_SECRET`
   (the last stage also gets hc_head, norm, lm_head and the MTP layer; `--tied-embed` if lm_head is tied; vision tensors are skipped).
3. Start stages: `swp-exl3 stage --config examples/stage-razer.toml` and `... stage-strix.toml`.
4. `swp-exl3 ping --pipeline examples/pipeline.toml`, then
   `swp-exl3 generate --pipeline examples/pipeline.toml --prompt "..." --chat --max-new 128 --bench`
   (prefill and decode tok/s, per-stage compute vs wire time, `--json out.json`).
5. Cut-invariance: `swp-exl3 logits ... --save a.json` at one cut, restart stages at another cut, `... --save b.json`,
   `swp-exl3 compare a.json b.json` (top-1 equal and KL on shared top-k).
See `examples/loopback-test.md` to try everything on one machine with a smaller quant first.

## Limits of this version
* Single device per stage (`device = "cuda:0"`); autosplit loading is rejected (its dummy pass assumes token ids as input).
* No MTP / speculative decoding yet (needs the drafter on the last stage with a copy of the embedding, and recurrent-state
  history for verify rollback). No prefill pipelining across stages; star topology (coordinator relays).
* Prefill uses `forward_ls` on every stage (no early-exit `prefill_ls` shortcut), so the last stage also runs the head on the final row of each chunk.
* bf16 wire and `topk` output are implemented but untested on the real engine. Authentication is not encryption.
* CPU expert offload on the Razer is just `[stage.engine]` passthrough to `config.infer_params`; its interaction with a
  layer slice is unverified. Watch the first load.
