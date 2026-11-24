# WP3 complete native QSA — validation and bounded layer benchmark

Date: 2026-11-23. **Requested validation and benchmark gates PASS.** Parent owns
WP3 acceptance and status. `IMPLEMENTATION_STATUS.md`, the reference manifest,
runtime sources, and the production Python reference were not edited by this task.
The incomplete-verification account in `WP3_COMPLETE_QSA.md` is historical; the
executions below provide the newer evidence for this checkout/library.

## Executed gates

| Gate | Actual result |
| --- | --- |
| Expanded actual-checkpoint test | **24 passed, 19 deselected in 34.08 s** |
| Broad native suite, including expanded checkpoint test | **396 passed, 3 skipped in 52.22 s** |
| Full-step benchmark, actual layer 3, both policies × seven lengths | **14 cases passed**, each with one warmup and three measured fresh-state trials |

The actual-checkpoint test is parametrized over layers
`3,7,11,15,19,23,27,31,35,39,43,47` and policies `ggml_cpu,strata`.
**All 12 layers and both policies were actually executed**, not merely collected.
Each case performs nine consecutive real steps from an empty state, crossing pool
completions at n=4 and n=8. Every step checks output, exact selected IDs, scores,
and complete committed raw/K/V/pooled state. Maximum checkpoint output absolute
error: **3.5762786865234375e-7**. Maximum per-case native accounted peak:
**64,354,872 bytes**. Cases run serially, holding only one layer's weights.

The three broad-suite skips are the existing opt-in attention CUDA timing test,
standalone hidden-frontend checkpoint test, and standalone projection checkpoint
test. The complete-layer checkpoint cases are **not skipped**. No remaining known
failure in the executed gates. The prior 372-pass/4-skip report is not substituted
for current evidence: enabling the old single checkpoint test and expanding it to
24 explains the current 396-pass/3-skip count.

## Reproduction

Run from repository root, using the existing built library:

```sh
# First execution: all twelve checkpoint layers, both policies.
MINI_LLM_BACKEND=numpy \
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MINI_LLM_QSA_LAYER_CHECKPOINT=1 \
.venv/bin/python -m pytest -q -s tests/test_native_engine_qsa_layer.py \
  -k actual_checkpoint -x

# Second execution: benchmark, matched GPU reference.
MINI_LLM_BACKEND=cupy \
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 \
NVIDIA_TF32_OVERRIDE=0 \
.venv/bin/python -m benchmarks.benchmark_native_qsa_layer \
  --device 0 --layer 3 --samples 3 --warmups 1 \
  --output docs/native_engine/WP3_COMPLETE_QSA_BENCHMARK.json

# Third execution: broad native regression, expanded checkpoint gate enabled.
MINI_LLM_BACKEND=numpy \
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MINI_LLM_QSA_LAYER_CHECKPOINT=1 \
.venv/bin/python -m pytest -q tests/test_native_engine*.py -x
```

The benchmark defaults to all seven lengths and both policies. `--samples` is
bounded to 1..10; `--warmups` to 0..3. `--layer` selects one of the twelve actual
attention layers; this run benchmarks **only layer 3**. With
`MINI_LLM_BACKEND=numpy` the script also supports the validated tiled CPU projection
oracle, but that path was **not benchmarked here** and is explicitly labeled as
unmatched CPU/GPU hardware. CuPy was available and used instead.

## Benchmark contract

- Device 0: **NVIDIA GeForce RTX 5080**, driver **580.178.04**. Python/CuPy/NumPy
  versions, full sample arrays, environment, capacity and policy are in the JSON.
- `n` means the post-step committed length; timed position is `n-1`. Capacities
  are 4096 for n≤4096 and 16384 otherwise. Each trial resets native state and
  imports the identical pre-step raw/K/V/pool prefix used by the Python reference.
  A new reference state is allocated/populated for every trial. No stale output,
  advancing-history timing, or invalid post-capacity step is used.
- Actual weights/gamma are read from the checkpoint identified by
  `WP2_GGUF_INVENTORY.json`. Only the selected layer's ten tensors are uploaded.
  The same stored bytes feed both paths. CPU decoding uses bounded 32-row tiles;
  CuPy holds only this layer's six FP32 projection matrices. Native keeps packed
  weights and performs its own full matvecs. No full-model construction/loading.
- Reference imports and validation helpers are reused from
  `tests/test_native_engine_qsa_layer.py`: deterministic `hidden(t)`, real
  reference-operator `seed_reference`, `import_state`, readback and state checks.
  Hidden trajectory is `0.2*sin(arange(2560)*0.03125+(t%11)*0.125)` in FP32.
  Setup-only memoization of the 11 distinct key/value/raw projections avoids
  redundant projection work while seeding; all position-dependent reference
  norm/RoPE and chronological left-to-right FP32 pool additions still execute.
  Memoization is removed before any timed step. Seed history is generated once
  with the unchanged reference operators and reused identically for both policies.
- Native timing is `perf_counter_ns` around the **full public `layer.step`**,
  including scalar status/commit synchronization. Reference timing is around the
  **full unchanged `Qwen4ExpAttention.step`** followed by stream synchronization.
  Both include all six projections, frontend, scoring, policy selection,
  attention, gate, output and state commit. Both exclude import, reset, allocation,
  weight decode/upload, hidden upload, diagnostics and correctness readback.
- FP32 arithmetic on both paths; TF32 disabled for this execution. Gamma,
  epsilon, scale, RoPE, state geometry and explicit tie policy match. The native
  packed matvec and CuPy decoded FP32 GEMV have different reduction orders.
  Fast decode is off, and native-selection override is off for the Python oracle.
- Each warmup and measured trial checks output and entire committed state.
  A separate reference replay, outside timing, checks **exact IDs** and scores
  without charging diagnostics only to the reference latency.
- Existing predeclared tolerances are unchanged: output `rtol=atol=2e-4`;
  scores `rtol=5e-5, atol=2e-4`; state base bound `5e-5*(1+abs(reference))`,
  with the existing coordinate-wise long-position RoPE rounding allowance.
  Maximum benchmark output absolute error: **1.3783574104309082e-7**.

### Measured medians (milliseconds, complete attention-layer step)

| n | ggml_cpu native | ggml_cpu CuPy reference | strata native | strata CuPy reference |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.447945 | 4.035692 | 0.446920 | 4.032484 |
| 2051 | 2.746523 | 20.783384 | 2.743362 | 20.359401 |
| 2052 | 2.752936 | 21.057163 | 2.749126 | 20.622676 |
| 2053 | 2.756786 | 20.776316 | 2.750218 | 20.626340 |
| 4096 | 4.003604 | 39.992757 | 3.992357 | 37.670677 |
| 8192 | 6.030836 | 76.267253 | 6.438462 | 71.936996 |
| 16384 | 10.066784 | 149.220433 | 9.491425 | 140.250682 |

These are wall-latency observations, **not a model speedup claim**. The unchanged
CuPy reference still has a per-block Python scoring loop and CPU-based selection;
it is not a fully GPU-resident optimized competitor. The matrix decoding/storage
strategy also differs: decoded resident reference FP32 matrices versus native
packed-weight matvecs. Three samples provide bounded observations, not a robust
performance distribution. No model throughput, end-to-end latency, or generation
performance is inferred.

## Memory and provenance

Native engine admission is **192 MiB**; CuPy memory-pool limit is **320 MiB**.
Their sum is 512 MiB, and sampled combined reserved payload is asserted strictly
below 512 MiB. Import uploads are closed before allocating the reference trial
state. Unused CuPy pool blocks are released between trials/cases.

- Native accounted peak: **188,532,280 bytes** (including native pinned accounting).
- Maximum observed native device + CuPy reserved payload: **445,239,848 bytes**
  (about 424.61 MiB).
- This is a bound/account of tensor/cache/scratch payload, **not total GPU process
  memory**: CUDA context, loaded modules, driver and library-internal allocations
  are not included in the payload counters.

`WP3_COMPLETE_QSA_BENCHMARK.json` records actual samples/configuration,
SHA-256 of each consumed tensor payload, native library SHA-256, source hashes,
and checkpoint inventory hash/path. It does not hash/read all 83 GB of model
payload. `WP3_COMPLETE_QSA_VALIDATION.json` preserves all 24 checkpoint case
records, raw test summaries, benchmark artifact hash and additional imported
helper hashes. Benchmark source/library hashes were checked for stability during
the run; captured source hashes were checked again after regression and unchanged.

## Remaining scope / blockers

No known blocker remains for **the requested tests and bounded benchmark**.
Parent must make the acceptance/status decision; status/manifest files remain
untouched. This evidence does **not** establish full-model native inference,
HC/residual/MLP integration, all-layer long-context actual-checkpoint parity,
CUDA fault injection, sanitizers, concurrent workloads, or end-to-end model
performance. Long-context actual-checkpoint timing/parity here is layer 3 only;
the all-12-layer actual-checkpoint gate is nine short repeated steps per policy.
The three optional standalone tests listed above remain unexecuted in this run.
