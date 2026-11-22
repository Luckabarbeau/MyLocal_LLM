# Production prefill recovery — measured progress, not completion

## Status

Strata was launched directly on **GPU 0 (RTX 5080) only**, configured for
262,144 tokens, and benchmarked through HTTP. Its standard suite completed
21/21 required requests and 12/12 sanity checks. The coding diagnostic suite
also completed 24 requests, including 260,992 populated prompt tokens and
follow-ups. GPU 1 was excluded. Engines were not run concurrently.

MyLocal_LLM initially failed 262K admission (`need 27929515008`,
`budget 15032385536` device bytes). Host-backed FP32 QSA now permits actual
262K-configured serving with populated 4K prompts and generation. **Real
near-262K population in MyLocal_LLM remains unverified.** Synthetic boundary
retrieval tests and empty-capacity allocation are not substitutes.

Artifacts and the initial scaling comparison are under
`benchmarks/production_recovery_20261121/`, including `INITIAL_COMPARISON.md`.
The directory contains retained failed experiments and large state snapshots;
not every subdirectory represents the final implementation.

## What “use Strata's setup” now means

The explicit `--prefill-experts strata` path calls **Strata's actual fused int8
CUDA tensor-core expert kernels**, not a separate Strata engine/server. A narrow
C ABI links only `moe_fused.cu` and `moe_fused_iq.cu` from the existing local
MIT-licensed checkout. Build provenance records source revision
`c77572806996349b49995e4f98d41b5294d36a21`, source hashes and nvcc arguments.
This checkout is not claimed to be the exact source revision of installed
Strata engine 0.1.59.

The production chunk path now includes:

- Batched packed dense projections and exact native-GEMV coefficient batches.
- Whole-chunk HC kernels retaining singleton reduction order.
- Gated-delta updates across a chunk inside a CUDA kernel, rather than Python
  dispatch per token. Convolution and normalization parallelize across tokens.
- GPU grouping of token/expert assignments for the Strata fused backend.
- Resident packed expert reuse and an eight-slot pinned-host/device streaming
  ring for misses. Every selected expert executes; no route-tail skipping.
- Batched causal QSA, pooling and bulk history commits for the all-cell prefix.
  A chunk crossing that boundary uses the fast prefix and reference sparse tail.
- A verified prompt checkpoint, allowing an identical request to reuse prompt
  state/logits without re-prefill or copying the full QSA history.

**Still not equivalent to the entire Strata pipeline:** sparse QSA beyond 2,051
cells remains per-query, PLE retains its causal reference operator, the expert
staging ring is smaller than Strata's long-prompt ring, and QSA history remains
FP32 rather than Strata's int8 KV. Default Strata serving also offered/accepted
drafts and skipped some missed tail experts; MyLocal_LLM does neither.

### Explicit arithmetic policies

| `--prefill-experts` | Behavior |
|---|---|
| `decode` | Retained singleton expert dispatch reference/fallback |
| `grouped` | Multirow experts preserving frozen mixed CPU/GPU decode arithmetic |
| `stream` | All prompt experts use GPU Q8 row-dot arithmetic, streaming misses |
| `strata` | Actual upstream fused int8 tensor-core prompt-expert arithmetic |

The latter two are **not claims of bitwise equivalence to mixed CPU/GPU decode
arithmetic**. GPU and CPU activation quantizers and accumulation paths differ.
The Strata policy is opt-in, not a silent default precision change. All policies
retain full routing and all selected experts.

## Matched production HTTP comparison

Real repository source, non-thinking requests, temperature 0, streaming, 256
requested output tokens; context capacity remains 262,144. All six saved request
payload hashes match the Strata originals exactly. Follow-ups replay **Strata's
assistant history**, so MyLocal_LLM correctly prefills different assistant
trajectories rather than falsely claiming they are already committed.

These are unprofiled measurements. “Prior” is the earlier native chunk64 run,
not an assertion that only one variable changed: final configuration uses the
Strata fused policy, chunk4096 and a 3.5 GiB cache; prior used the mixed policy,
chunk64 and 6 GiB cache.

| Metric | Strata | Prior MyLocal | Strata-kernel MyLocal |
|---|---:|---:|---:|
| 1,024-token fresh prefill | 1.354 s | 30.834 s | **10.394 s** |
| 1K prefill throughput | 756.2 tok/s | 33.2 tok/s | **98.5 tok/s** |
| 4,095-token fresh prefill | 2.252 s | 193.420 s | **116.874 s** |
| 4K prefill throughput | 1,818.8 tok/s | 21.2 tok/s | **35.0 tok/s** |
| 1K cold client TTFT | 1.506 s | 30.902 s | **10.466 s** |
| 4K cold client TTFT | 2.345 s | 193.583 s | **117.014 s** |
| 1K exact-repeat client TTFT | 0.165 s | 0.078 s | **0.078 s** |
| 4K exact-repeat client TTFT | 0.189 s | 0.074 s | **0.083 s** |
| 1K exact-repeat processed/reused | 7 / 1,017 | 0 / 1,024 | **0 / 1,024** |
| 4K exact-repeat processed/reused | 7 / 4,088 | 0 / 4,095 | **0 / 4,095** |
| 1K warm generated tokens/s | 74.5 | 31.0 | **29.9** |
| 4K warm generated tokens/s | 76.3 | 14.5 | **14.1** |
| 1K warm full-model forwards/s | Unavailable | 30.8 | **29.8** |
| 4K warm full-model forwards/s | Unavailable | 14.5 | **14.1** |

Prefill improves **2.97× at 1K** and **1.65× at 4K** relative to the prior
measured configuration, but remains approximately **7.7× / 51.9× slower than
Strata** on these two fresh prompts. This is not competitive production prefill.

Final artifact: `mylocal_strata_prefill_realcode/http/results.jsonl`.
Reference: `strata_realcode/results.jsonl`.

The final run's counters report 192 fused expert layers and 2,726,880 routed
rows, matching all ten experts for each of the four newly prefilled prompts.
It streamed approximately 123.2 GB of packed bytes cumulatively. This is a
transfer counter, not resident memory.

Resource samples for the final run:

- Selected-GPU total memory: 10.78–13.38 GiB, including other GPU consumers.
- Selected process RSS: 39.84–52.08 GiB, including reclaimable mapped expert pages;
  not a measurement of anonymous state alone.
- Process CPU: approximately 83–1,096%, where one logical CPU is 100%.
- Whole-GPU utilization samples: 4–100%.

All answers in this paired suite reached the output cap. Successful HTTP and
numerical parity do **not** establish completed-answer coding quality. Some
sample critiques were not justified by the source; no coding-quality pass is
claimed.

## Correctness and tests

- Exact mixed policy: real checkpoint, 1,024 prompt tokens, chunks 64 and 1,024,
  state/logits/routing/subsequent decode **bitwise identical** to sequential.
  Artifact: `chunk_grouped_parity1024.json`.
- Strata fused policy: real checkpoint, 128 prompt tokens, chunks 16/64/256/1,024,
  **bitwise identical** to singleton execution of that same fused policy.
  Artifact: `strata_fused_prefill_parity128.json`. Larger chunk settings in this
  test do not imply that more than 128 rows executed.
- CPU regression run: **266 passed, 30 skipped, 1 deselected**, plus seven
  unittest subtests. Final isolated GPU-0 regression run: **389 passed, 34 skipped**,
  including fused expert integration, ring wraparound, chunk recurrence, QSA
  boundaries and the existing graph/decode regressions. Artifact:
  `final_gpu_regressions/`.
- Earlier attempted GEMM/native-recurrence substitutions failed real checkpoint
  parity. Their reports remain saved; they were corrected, not accepted by
  widening tolerance. Mixed-policy final results have maximum error zero.
- Completed asynchronous scratch is retired before replacement-state admission.
  Four-GiB-cache / chunk4096 fused experiments still hit conservative physical
  admission after CUDA module loading; the measured 3.5-GiB setup completed.

## Reproduce

Build the expert kernels once (CUDA SDK and matching local Strata/GGML headers
required; no download, global installation or whole-engine link):

```sh
.venv/bin/python scripts/build_strata_prefill.py \
  --output benchmarks/my_strata_prefill_build
```

Launch the opt-in measured configuration:

```sh
CUDA_VISIBLE_DEVICES=GPU-bea264b4-9e76-3be5-5883-7bde3fe9f82d \
.venv/bin/python scripts/serve_qwen4exp.py \
  --context 262144 --qsa-state-storage host \
  --cache-gib 3.5 --prefill-chunk-size 4096 --prefill-experts strata \
  --strata-prefill-library benchmarks/my_strata_prefill_build/libmylocal_strata_prefill.so
```

Use `--no-thinking` for the measured non-thinking policy. The default model and
CPU GGML library paths remain the workstation's existing paths. See `--help`
for overrides. `--expert-placement warm-frozen` is the default: it enables native
decode device plans after warmup, with all CPU misses still executed. An
`adaptive` escape hatch retains the previous placement behavior.

The bounded runner starts and cleans up only its own process group. For exact
request replay, use `scripts/run_qwen4exp_serving_benchmark.py --replay-from`;
for an already-running server, use `scripts/benchmark_qwen4exp_serving.py`.
Do not benchmark it concurrently with Strata.

## Next production-critical work

1. Whole-chunk **sparse** QSA beyond the all-cell prefix: batched causal index
   scoring, exact selection, bounded selected-history staging and attention.
   The current 4K profile attributes most remaining time to `Qwen4ExpAttention.step`.
2. Amortize expert source packing and enlarge/overlap staging within an explicit
   budget; retaining the tensor-core kernels alone does not reproduce Strata's
   packed-host arena and large streaming ring.
3. Measure larger populated contexts and complete coding answers. Do not claim
   near-262K model support, 40–50 production forwards/s, or competitive serving
   until those actual pipeline measurements pass.
