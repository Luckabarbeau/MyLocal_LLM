# Production prefill recovery: native Triton continuation and completed replay

## Status and verification boundary

The current prompt-expert prefill experiment is the native Triton policy,
selected with `--prefill-experts triton`. It lives entirely inside this
repository: `mini_llm/ops/triton_prefill_experts.py`, dispatched from
`_routed` in `mini_llm/external/qwen4exp_prefill.py`. **No Strata code,
kernels, or libraries link into MyLocal_LLM.** The former
`--prefill-experts strata` option, its narrow C ABI, and its build scripts
have been removed; no current build step, library path, or serving flag
references them.

A bounded two-request smoke replay **completed 2/2** at the chunk-4096,
3.5-GiB-cache configuration. Its measured numbers are recorded below and
come from artifacts under
`benchmarks/production_recovery_20261121/mylocal_triton_chunk4096_smoke_20261122/`.
The full six-request replay (1K and 4K payloads) **completed 6/6** under
`benchmarks/production_recovery_20261121/mylocal_triton_chunk4096_replay_20261122/`.
It was run by another agent; the completed results and newer continuation
measurements are recorded below, not attributed to this documentation turn.

Near-262,144-token prompt population in MyLocal_LLM remains unverified, and
nothing here claims it. Historical figures below belong to the deleted
Strata-kernel build or are explicitly marked as user reports.

## Prefill expert policies available today

| `--prefill-experts` | Behavior |
|---|---|
| `decode` | Default. Retained singleton expert dispatch (reference/fallback) |
| `grouped` | Multi-row experts preserving the frozen mixed CPU/GPU decode arithmetic |
| `stream` | Prompt experts use GPU Q8 row-dot arithmetic, streaming packed misses |
| `triton` | Experimental. Every prompt expert runs through the native Triton tensor-core policy (prefill only) |

Every non-`decode` policy requires `--prefill-chunk-size` greater than zero
(the server rejects the combination otherwise). The dispatch cascade is:
`triton` when selected and eligible, then `grouped`/`stream` when selected,
then the retained decode device plans, then the singleton reference. An
ineligible Triton call returns `None` before mutating any executor state.
Decode itself is unchanged: the policy affects chunked prompt prefill only;
generation still uses the retained singleton device-plan arithmetic.

## How the Triton policy runs a chunk

Selection gate. The fused path activates only under the CuPy backend on an
NVIDIA device, with the heterogeneous expert executor in `mixed` execution
mode, cache admission budget at zero (the default `warm-frozen` placement
sets this after warmup), no idle-eviction requests, no routing profiler, FP32
`[1, T, D]` activations at width 2,560, top-k between 1 and 16 with unique
valid experts per token, and **every active expert in the layer sharing one
ordered `(gate, up, down)` format tuple**: the three matrices within a trio
may be different supported formats (`IQ2_S`, `IQ3_S`, `IQ3_XXS`, `IQ4_NL`,
`IQ4_XS`, `Q2_0`), but a fourth format combination across experts in the
same layer returns `None`. Host bytes must be contiguous.

Streaming and waves. Routing is unchanged and every selected expert
executes; no route tail is skipped. The expert-ID space is partitioned into
ascending waves of at most `MINI_LLM_TRITON_WAVE` experts. For each wave,
packed GGUF expert bytes are copied into a pinned three-slot host ring with
GIL-free `ctypes.memmove` on `MINI_LLM_TRITON_THREADS` worker threads, then
uploaded asynchronously on a dedicated copy stream one wave ahead of use.
Native ring consumption events prevent device-slot overwrite until the compute
stream has consumed that slot. The coordinator publishes host-ring readiness
after worker fills complete, before scheduling the upload.

On-device dequantization. Each staged wave's weights dequantize on device
into FP16 tiles with the per-block scales baked in. Token activations are
cast to FP16. Inside a wave, token rows are bucketed per expert in ascending
order and multiplied in tiles of at least `MINI_LLM_TRITON_BM` rows. The
gate/up and down GEMMs are plain FP16 `tl.dot` products accumulated in FP32;
the intermediate `hbuf` is FP16.
A combine kernel adds each token's contributions in ascending expert order,
matching the grouped reference's add order.

Tunables and defaults. `MINI_LLM_TRITON_WAVE=32`, `MINI_LLM_TRITON_BM=64`
(floor 16), `MINI_LLM_TRITON_THREADS=6`.
`MINI_LLM_TRITON_FUSED_DEQUANT=0` and `MINI_LLM_TRITON_REUSE_WORKERS=0`
remain the defaults: fused packed dequantization and worker reuse are optional,
with inconclusive end-to-end benefit. A newer replay used reuse=1, but that
experimental run setting is not the final default.

The TILE128 materialized dequantization path was checked exact for all six
supported formats (`IQ2_S`, `IQ3_S`, `IQ3_XXS`, `IQ4_NL`, `IQ4_XS`, `Q2_0`).
Its reported **3.07–3.56x synthetic dequantization-kernel speedup is not an
HTTP speedup attribution**; it does not isolate whole-request performance.

## The arithmetic is a distinct policy, not decode parity

FP16 activations, FP16 dequantized weight tiles, and FP32 `tl.dot`
accumulation form a third prefill arithmetic policy, in the same sense as
`stream`. **No bitwise equivalence to the mixed CPU/GPU decode arithmetic is
performed or claimed**, because activation quantizers and accumulation paths
differ. The policy is opt-in and never changes decode numerics silently.

## Experts restream in full, every chunk

Nothing expert-related persists on the device across chunks or layers for
this policy. The ring holds at most three waves and the FP16 tile buffers
are sized per wave, so each MoE layer of each prompt chunk re-uploads and
re-dequantizes the full active expert set for that chunk. A single 1,024-
token chunk already streamed about 38.1 GB of packed expert bytes across 48
layers and 768 waves; a 4,096-token chunk approaches a full packed pass of
every layer.

Important correction about memory. Python references to the mmap-backed
`PackedGGUFWeight` source arrays (held by wave membership for each layer)
keep the objects alive but **do not pin pages in physical RAM**. Only the
ring's `alloc_pinned_memory` allocation is page-pinned. Expert file pages
are ordinary page cache: the kernel may evict them and the next chunk's
`memmove` re-reads from disk. Startup warmup touched about 6.73 GiB of
expert pages resident on the smoke host (`--host-moe-gib`,
`--ram-reserve-gib` control warmup and RAM headroom), and host swap was
reported at roughly 6.3 GiB before and 8 GiB after the smoke window; that
observation is **not** attributed to the policy without before/after run
telemetry.

Frozen placement is not an absent cache. `warm-frozen` sets the cache
admission budget to zero after warmup, which is what the Triton gate
requires, but the GPU expert cache built during warmup (3.5 GiB here) still
exists and serves decode device plans. The Triton prefill path simply
ignores that cache and streams host source bytes instead. Separately, the
verified prompt checkpoint lets an identical request skip prefill entirely;
the smoke repeat request reused all 1,024 tokens with zero Triton counter
movement (below).

## Measured 1K smoke (two-request replay, chunk 4096)

Server: `--context 262144 --qsa-state-storage host --cache-gib 3.5
--prefill-chunk-size 4096 --prefill-experts triton --no-thinking`, GPU 0
UUID, frozen placement after warmup. Source:
`.../mylocal_triton_chunk4096_smoke_20261122/` (`http/summary.json` shows
2 successes, 0 failures; per-request `after.json` metrics and counters).

| Metric | Fresh 1,024-token request | Exact repeat of same request |
|---|---:|---:|
| Prefilled / reused prompt tokens | 1,024 / 0 | 0 / 1,024 (prompt checkpoint restore) |
| Prefill seconds | 16.721 (61.24 tok/s) | 0 |
| Client TTFT | 16.8084 s | 0.1463 s (server first piece) |
| Decode, 256 tokens | 15.7742 model-forwards/s | 15.2945 model-forwards/s |
| Total request seconds | 33.001 | 16.850 |

Counter deltas after the fresh prefill (repeat moved none):
`prefill_triton_layers` 48, `prefill_triton_experts` 18,601 executed expert
activations, `prefill_triton_waves` 768, `prefill_triton_streamed_bytes`
38,102,630,400, and `prefill_routed_rows` 491,520 (= 1,024 tokens x 10
experts x 48 layers).

**Historical counter caveat (now fixed):** this smoke's archived
`prefill_triton_rows` value of 7,864,320 was 16x the routed total because the
old executor added the whole chunk's row count once per wave. Current code
sums only the routed rows in each wave; it no longer has that overcount.
Do not reinterpret the archived smoke counter as corrected data;
`prefill_routed_rows` was the correct total in that artifact.

## Completed replay and continuation measurements

Continuation figures below are supplied handoff results, not new HTTP runs by
this documentation agent. Artifact paths are relative to
`benchmarks/production_recovery_20261121/`.

| Run | Fresh 1,024-token prefill | Fresh 4,095-token prefill | Decode forwards/s, 1K / 4K |
|---|---:|---:|---:|
| Other-agent completed six-request `mylocal_triton_chunk4096_replay_20261122` | 16.824 s | 157.959 s | 12.51 / 7.42 |
| Newer `native_triton_worker_reuse_realcode` (reuse=1) | 9.517 s | 121.926 s | 25.64 / 13.79 |
| Archived external `strata_realcode` reference | 1.3542 s | 2.2515 s | Not compared here |

Runtime variation confounds the two native runs: neither the prefill nor the
decode difference establishes a causal worker-reuse improvement. The newer
reuse=1 run does not promote that option to a default. HTTP completion is not
coding-quality acceptance or competitive production recovery.

Fresh matched 1K A/B medians from the continuation were **9.323 s** for
materialized dequantization with reuse=1, **9.176 s** for fused packed
dequantization with reuse=1, and **8.995 s** for materialized dequantization
with reuse=0. These were measured in sequential order, not an interleaved or
randomized comparison; they are not causal proof for either option.

## Reproduce the measured configuration

```sh
.venv/bin/python scripts/run_qwen4exp_serving_benchmark.py \
  --output benchmarks/production_recovery_20261121/mylocal_triton_chunk4096_replay_20261122 \
  --context 262144 --chunk-size 4096 --prefill-experts triton \
  --cache-gib 3.5 --no-thinking \
  --replay-from benchmarks/production_recovery_20261121/strata_realcode
```

The `--replay-from` argument takes the **directory**
`benchmarks/production_recovery_20261121/strata_realcode`, whose per-request
`*/request.json` payloads replay in sorted order with only the model name
replaced; saved assistant history is never swapped for new output. Do not
pass `results.jsonl` here, and do not edit the payloads: their hashes are
what made the archived comparison matched.

The runner pins GPU 0
(`CUDA_VISIBLE_DEVICES=GPU-bea264b4-9e76-3be5-5883-7bde3fe9f82d`), starts
and cleans up only its own server process group, and records `launch.json`,
`ready.json`, and `server.log` in the new output directory (which must not
already exist). The full six-request replay at that exact artifact path
has completed; use a fresh output directory for any new run. Concurrency: the user explicitly accepts a live Strata
server on GPU 1 sharing host RAM. Protect that foreign process: never signal
or restart it, keep benchmark state on GPU 0 and in a fresh directory, and
report observed host RAM/swap contention in results rather than treating
co-execution as forbidden. Default model and CPU GGML library paths remain
the workstation's existing paths; see `--help` for overrides.

## Historical measurements from the deleted Strata-kernel build

These numbers come from the archived 2026-11-21 comparison, which linked
only `moe_fused.cu` and `moe_fused_iq.cu` from a local Strata checkout
through a narrow C ABI (build provenance recorded source revision
`c77572806996349b49995e4f98d41b5294d36a21`). That build path, its flag, and
its library argument no longer exist in the repository. **The table is kept
as history and must not be read as current production behavior or as a
baseline the Triton policy has matched.**

Matched HTTP suite: real repository source, non-thinking, temperature 0,
streaming, 256 requested output tokens, context capacity 262,144, all six
saved request payload hashes matching the Strata originals exactly.

| Metric | Strata (archived) | Prior MyLocal chunk64 | Strata-kernel MyLocal (deleted build) |
|---|---:|---:|---:|
| 1,024-token fresh prefill | 1.354 s | 30.834 s | 10.394 s |
| 1K prefill throughput | 756.2 tok/s | 33.2 tok/s | 98.5 tok/s |
| 4,095-token fresh prefill | 2.252 s | 193.420 s | 116.874 s |
| 4K prefill throughput | 1,818.8 tok/s | 21.2 tok/s | 35.0 tok/s |
| 1K cold client TTFT | 1.506 s | 30.902 s | 10.466 s |
| 4K cold client TTFT | 2.345 s | 193.583 s | 117.014 s |
| 1K exact-repeat client TTFT | 0.165 s | 0.078 s | 0.078 s |
| 4K exact-repeat client TTFT | 0.189 s | 0.074 s | 0.083 s |
| 1K exact-repeat processed/reused | 7 / 1,017 | 0 / 1,024 | 0 / 1,024 |
| 4K exact-repeat processed/reused | 7 / 4,088 | 0 / 4,095 | 0 / 4,095 |
| 1K warm generated tokens/s | 74.5 | 31.0 | 29.9 |
| 4K warm generated tokens/s | 76.3 | 14.5 | 14.1 |
| 1K warm full-model forwards/s | Unavailable | 30.8 | 29.8 |
| 4K warm full-model forwards/s | Unavailable | 14.5 | 14.1 |

The Strata decode figures (74.5 and 76.3 tokens/s) are uncapped archived
measurements. Prefill then improved 2.97x at 1K and 1.65x at 4K over the
prior measured configuration while remaining roughly 7.7x and 51.9x slower
than the archived Strata figures. Artifacts:
`benchmarks/production_recovery_20261121/` including `INITIAL_COMPARISON.md`
and `mylocal_strata_prefill_realcode/http/results.jsonl`; the exact-replay
payload reference remains the whole `strata_realcode` directory. Retained
failed experiments sit in that tree too. Archived resource samples for the
final deleted-build run: selected-GPU total 10.78 to 13.38 GiB including
other consumers, process RSS 39.84 to 52.08 GiB including reclaimable
mapped expert pages, process CPU about 83 to 1,096 percent, whole-GPU
utilization samples 4 to 100 percent, and about 123.2 GB of packed bytes
streamed cumulatively (a transfer counter, not resident memory). All
answers reached the output cap; HTTP success and numerical parity do **not**
establish completed-answer coding quality, and no coding-quality pass was
ever claimed.

## Live Strata reference (user-reported)

The user reports a live Strata session **capped at 32 GiB of host RAM**
(the cap is memory, not an output-token cap) sustaining roughly 1,000
prefill tokens/s and 55 to 60 decode tokens/s with dflash 2. This was not
measured during this session and is recorded as a report for orientation
only, not a benchmark.

## Verification status

This documentation turn ran the bounded CPU regression (120-second timeout):

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_qwen4exp_prefill_policy.py tests/test_qwen4exp_chunked_prefill.py tests/test_qwen4exp_serving_state.py tests/test_benchmark_qwen4exp_serving.py
```

Result: **98 passed, 19 skipped, 7 subtests passed in 12.60 s**.

The supplied continuation handoff reports a fresh GPU-0 run of
`tests/test_triton_prefill_worker_reuse.py`: **23 passed this turn**. That GPU
result was supplied, not executed again by this documentation agent; it is
not the older archived 389-pass GPU count.

Prior real-checkpoint same-policy parity passed bitwise for a **32-token**
populated prefix, recorded in
`benchmarks/production_recovery_20261121/native_triton_checkpoint_parity32.json`
(inspected here, not rerun). It compares singleton explicit GPU prefill with
chunk sizes 16 and 64, with fused dequantization disabled; it is not parity
against mixed-CPU decode arithmetic. A 64-row setting on 32 tokens does not
exercise 64 populated rows. Completed 4K HTTP coverage does not establish
4K numerical parity, answer quality, or near-262K populated-context support.

## Next work

1. Use controlled, interleaved matched runs with RAM/swap telemetry to
   distinguish runtime variation from optional dequantization/worker effects.
2. Extend numerical and coding-quality validation beyond the short checkpoint
   parity case before claiming production parity.
3. Validate larger populated contexts explicitly. Near-262K population,
   competitive prefill, and production recovery remain unverified.
