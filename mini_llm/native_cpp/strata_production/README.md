# Pinned Strata production engine

Full, unmodified upstream source: `../third_party/strata-712fec6b4297`, commit
`712fec6b4297c9cfd32ab43daaa2979a9aa01edf` from https://github.com/Niko1221/Strata.
Imported from the commit-addressed codeload archive with path/type/size checks
(6,777 archive members; 101,355,492 unpacked bytes). Archive SHA-256:
`26984c09b4d42476a8a3dee72d227d4583a07d0640572a1cb7e25b4f4d27b4f6`.
`source_provenance.json` records all 6,112 file hashes; `verify_source.py` checks
those and all nine files against `references/strata-712fec6b4297/manifest.json`.
No edits to that older reference snapshot or the external older checkout.
The vendor's `LICENSE`, all third-party notices, and
`third_party/strata-ggml/UPSTREAM.md` are preserved. The latter identifies its
ggml pin as `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

## Build

From `mini_llm/native_cpp/strata_production`:

```sh
cmake --preset rtx5080-release
timeout --signal=TERM --kill-after=5s 170s cmake --build --preset rtx5080-release
```

Resume the same command after a timeout; Ninja keeps completed objects. The
`mini_llm_strata_engine` ExternalProject target builds upstream `strata` and
`strata_kq`, not a replacement implementation or the upstream test matrix.
Artifacts are `build/rtx5080-release/upstream/strata` and its adjacent
`libstrata_kq.so`; retain them together. Existing experimental native code and
its CMake project are not changed or linked.

Release configuration: CUDA 13.4.92, GNU 13.3, `CMAKE_CUDA_ARCHITECTURES=120`
(SASS + PTX; upstream also selects 120a for its FP4-specific unit), portable
AVX2 CPU baseline, native CPU experts, Q6K, K-quant MMQ, trellis MMQ. Upstream
per-target flags are untouched, including `-O3 -DNDEBUG` and CUDA MMQ
`-use_fast_math`. CUDA prefill MMQ is built normally; `STRATA_PREFILL_MMQ` is
HIP-only and must not be enabled for CUDA. Token graphs, CPU arena,
resident/streamed GPU experts, MTP and DFlash2 remain in the upstream pipeline.
Runtime options and compatible model assets still determine which paths run.

Installed component `BUILD.json` declares version 0.1.59, source
`f-0213-rc 712fec6b4297`, portable=true, CUDA 13.0, architectures 75/86/89/120
and PTX. This build uses the pinned archive, vendored ggml, CUDA 13.4 and only
120. Installed metadata is a declaration, not independent proof of its full
source hash or byte-identical build. The Qwen target does not consume
`STRATA_BUILD_ID`; source identity is recorded outside the binary instead.

## Python-server handoff (binding not changed here)

Public upstream Python API: `serve/server.py:391` defines `Engine.generate(ids,
max_new, sampling, cancel) -> Iterator[int]`. Its concrete `StrataEngine`
constructor (`:1143`) takes `exe, args, cwd, log, env`; `generate` (`:2742`)
also supports heartbeat `None`, embeddings and prefill-only requests. Keep
one process resident and serialize solo requests. Reuse that adapter in an
isolated upstream module namespace/process, or implement this process boundary
in the parent server; do not mix the upstream `serve` package with this repo's
similarly named modules by accident. No Python layer/expert execution is needed.

The executable is upstream `src/program/generate.cpp` and speaks exactly:

- Launch with existing compatible pack/model paths via `--pack`, `--native`,
  `--max-context`, `--prefill auto`, `--serve`, plus the existing production
  launch's cache, pool and draft arguments. Do not invent cache sizing or turn
  speculation off to make startup easier. `--mtp`/drafter assets must already
  exist for the chosen speculative mode; no assets are downloaded by this build.
- Pin the child environment to
  `CUDA_VISIBLE_DEVICES=GPU-bea264b4-9e76-3be5-5883-7bde3fe9f82d`.
  Do not pass an overriding `--gpu` or any peer/multi-GPU options.
- Drain stderr independently; wait for stdout `READY <context> stop`, accepting
  startup/`INFO` lines before it. Never treat a successful help run as READY.
- Write and flush `GEN <max_new> temperature=F top_p=F top_k=N seed=N <id,id,...>\n`;
  optional sampling keys may be omitted. IDs are the complete templated prompt.
- Stream `T <id>`; handle `RESUME`, `PP`, `REUSED`, optional logprobs, `ERR`,
  and `DONE`. The first DONE fields are generated count, prompt count,
  prompt ms, decode ms, finish reason, drafts accepted, drafts offered,
  reused prompt count. Further positional/keyed telemetry follows; use the
  upstream parser instead of assuming a fixed field count.
- Cancel with `STOP\n`, drain to DONE before reuse; close with `QUIT\n` and
  bounded terminate/kill fallback. Reject `nonfinite` rather than report success.

## Actual validation and remaining limits

Source integrity and all nine snapshot comparisons passed, including after
compilation. The first build window timed out (170 s); the next incremental
window completed successfully, linking the full engine. The wrapper target
then passed with Ninja reporting no remaining work. A checked, bounded
`strata --help` passed with only the RTX 5080 UUID visible.
`build_validation.json` records final commands, artifact hashes and help result;
full local logs and compile commands are under `build/rtx5080-release/`.

No inference benchmark, model download, checkpoint/training change, or server
binding was performed. No measured claim of 2,500–3,000 prefill tokens/s,
~75 emitted decode tokens/s, or ~50 GB resident host arena is made. Those
remain end-to-end acceptance targets: the parent must bind the existing model,
pack, tokenizer and draft assets, reach READY, verify the actual arena/tier
telemetry, and measure fresh full prompts and emitted tokens on physical GPU 0.
