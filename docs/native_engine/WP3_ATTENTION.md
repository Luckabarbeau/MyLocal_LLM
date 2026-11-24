# WP3 — selected FP32 GQA attention and isolated history-validation optimization

Date: 2026-11-23. **WP3 remains in progress.** Parent-owned implementation status
and reference manifest are not updated here. No production Python Qwen changes.
Existing uncommitted WP2/selection/attention work is preserved.

## Existing implementation contract

Authority: `mini_llm/external/qwen4exp_attention.py`, `step` lines 363–386.
The tiled online softmax implementation reuses the computational pattern in
`mini_llm/ops/qsa_decode_gpu.py`, extended to consume selected device IDs.

- `QSASelection.attention()` / `ml_qsa_attention_create` retains native selection
  and engine ownership. No selected-ID CPU readback in the computational path.
- Exact stored-byte F32 uploads: normalized/rotated q `[24,256]`, separate gate
  `[24,256]` (not combined qg), current K/V `[2,256]`, previous history K/V
  `[capacity,2,256]`. Native GGML shapes are reversed. All inputs are same-engine.
- Device selection supplies valid n and ascending IDs, width `min(n,2051)`.
  Query head h uses KV head `h//12`. ID `n-1` uses supplied current K/V rather
  than the uncommitted history slot. Other selected IDs use history directly.
- Scaled FP32 logits, tiled stable softmax, ungated contexts and exact sigmoid
  gating. Default scale is `1/16`. No output projection or state/cache mutation.
  FMA contraction is disabled; no fast math. Selection/arithmetic/tolerances are
  not part of this optimization.
- q/gate/current K/V and **all** history K/V rows `[:n-1]`, including unselected
  rows, must be finite. Current history slot and unused capacity are ignored.
  Invalid IDs and nonfinite logits/numerators/denominator/outputs are rejected.
- Run invalidates its previous output-valid flag, retains uploaded inputs until
  stream completion (including error paths), synchronizes and checks scalar
  device status, then marks outputs valid only on success. Failed runs cannot
  expose results. Successful snapshots survive later selection updates/close.
- Fixed additional payload: **49156 device + 8 pinned bytes**. No per-run payload
  allocation; repeated runs preserve live/peak counters. This excludes CUDA
  context/stream/event/allocator overhead and host/caller storage, not total VRAM.
- Attention-only reuses valid device selection. Optional selection_inputs reruns
  the unchanged selector before attention on each combined timed iteration.

Sources: `mini_llm/native_cpp/qsa_attention.{h,cu}`, `runtime.{h,cpp}`,
`mini_llm/native/__init__.py`, `mini_llm/native_cpp/CMakeLists.txt`.
Tests/benchmark: `tests/test_native_engine_qsa_attention.py`.

## Persisted baseline — before the optimization

These are **previous-turn observations**, recorded before editing the kernel in
this task, not a fresh rerun. Raw numeric samples were available in the preceding
printed tool output and are transcribed without inventing samples into
`WP3_ATTENTION_BENCHMARK_BEFORE.json` (with provenance and environment).

- Release build: `/tmp/mini-llm-wp1-build`, CUDA architecture 120, succeeded.
- Attention tests: **77 passed, 1 optional benchmark skipped in 3.75s**.
- Exact full native suite: **247 passed, 1 skipped in 4.85s** (includes previous
  170 tests). Optional benchmark separately: **1 passed in 5.39s**.
- `git diff --check` passed. Existing gguf_loader.cpp misleading-indentation
  compiler warnings were not changed.
- GPU: NVIDIA GeForce RTX 5080, device 0, driver 580.178.04.

Tests cover n=1,2,3,4,5,127,1024,2048,2051,2052,2053,4096,8192,16384;
flat ties and controlled scores under both policies; exact selected membership;
NumPy-reference ungated/gated outputs; current-token inclusion/exclusion; large
positive logits; ignored NaN current/future history slots; source and arithmetic
nonfinite rejection; dynamic n/reset, snapshots and ownership, budgets, invalid
and stale handles, cross-engine inputs, immutable uploaded bytes, unchanged
allocation counters, and explicit readback capacities/buffer checks.

Numerical tolerance is unchanged: `rtol=4e-5, atol=3e-6`. BLAS dot/GEMV reductions
and online tile rescaling differ from the NumPy reference graph. Absolute
allowance protects near-zero contexts, not selector membership. No baseline
numerical failure led to production edits or tolerance relaxation.

### Before timings (ms/iteration, medians)

| n | Attention ggml_cpu | Combined ggml_cpu | Attention strata | Combined strata |
|---:|---:|---:|---:|---:|
| 1 | 0.006502 | 0.034966 | 0.006534 | 0.035021 |
| 2051 | 1.913504 | 3.759088 | 1.893526 | 3.605200 |
| 2053 | 1.853203 | 3.589507 | 1.802592 | 3.594512 |
| 4096 | 3.068342 | 6.102144 | 3.067462 | 6.116253 |
| 8192 | 5.561978 | 10.045488 | 5.523379 | 10.086493 |
| 16384 | 15.338458 | 26.591709 | 15.405248 | 26.763025 |

## Evidence and fixed implementation plan

Inspected `qsa_attention.cu` before edits: lines 23–28 scan `(n-1)*256`
elements for KV head `kh=h/12` in **each of 24 head CTAs**. Thus each existing
K/V element is checked twelve times with only 24 CTAs total. This is direct
source evidence of redundant history validation; timings alone do not attribute
all latency to that region.

One isolated change: move exactly that history scan to one parallel grid-stride
validation kernel, preceding the unchanged 24-head attention kernel on the same
engine stream. Flatten history as `(n-1)*2*256` elements per K/V array. Each
flattened offset belongs to exactly one grid-stride thread, checking both K and V.
For n=0 retain failure without underflow; n=1 has no historical elements.
Do not clear status in the new kernel. Keep all q/gate/current/ID/arithmetic
checks in attention and the runtime's final synchronized status/validity gate.
No new workspace, host n/ID readback, API change or selection change.

## Reproduction

Build:

```sh
cmake -S mini_llm/native_cpp -B /tmp/mini-llm-wp1-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build /tmp/mini-llm-wp1-build
```

Full unchanged 247-test suite (benchmark skipped by default):

```sh
MINI_LLM_BACKEND=numpy \
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m pytest -q tests/test_native_engine_*.py
```

Optional benchmark, run separately with no concurrent test run:

```sh
MINI_LLM_BACKEND=numpy \
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MINI_LLM_QSA_ATTENTION_BENCH=1 \
.venv/bin/python -m pytest -q -s \
  tests/test_native_engine_qsa_attention.py::test_cuda_timings
```

Benchmark: capacity 16384, seed 363386 for attention arrays, controlled selection
fixture seed 20261123; one combined warmup per n/policy, three samples of ten
repeats per case. CUDA events cover status clear and attention (including source
validation), plus selection in combined mode. They exclude input/n uploads,
final status D2H/synchronization, and diagnostic readback. Event intervals can
include host enqueue idle gaps. Input+workspace payload accounting:
69392936 device + 20 pinned bytes. No exclusive-GPU, locked-clock, model latency,
or matched production baseline claim. Policies label selection membership;
attention-only does not rerun the policy.

## Implemented optimization and verified after-state

Only computational source changed in this follow-up:
`mini_llm/native_cpp/qsa_attention.cu`. The history loop was removed from the
head CTA, and `validate_history<<<256,256>>>` was added before the unchanged
`attention<<<24,128>>>` launch. Both use the same stream and status pointer.
Each offset in `[0,(n-1)*512)` has exactly one grid-stride owner. No host n is
needed for launch sizing; no historical element is omitted based on selection.
The validation launch error is checked before enqueueing attention. Neither
kernel clears status; the runtime still clears once per run and only publishes
outputs after synchronized scalar status checks. Arithmetic may still execute
on invalid sources, as before, but those results cannot become trusted outputs.

All other source checks, ID checks, arithmetic and numerical tolerances are
unchanged. No selection, runtime, ABI, Python binding, test, workspace/budget or
production Qwen edits were made in this follow-up. Attention now uses two kernel
launches per iteration (parallel history validation + all parallel heads), not
one. Repeated benchmark iterations each retain their full validation.

Observed after the single implementation, **no corrective edit needed**:

- `cmake --build /tmp/mini-llm-wp1-build`: succeeded.
- The exact unchanged full suite command above: **247 passed, 1 skipped in
  4.27s**. This establishes the verified state for the edited kernel with the
  unchanged runtime, selector, binding and native-engine tests.
- Optional benchmark, separately: **1 passed in 2.08s**. Both phases account
  for exactly 69392936 device + 20 pinned bytes.
- Additional isolated Python experiment (no test-file changes): **8 cases
  passed**, covering K/values × both KV heads × NaN/Inf at `[4094,kh,255]` for
  n=capacity=4096 and flat-score `strata` selection. Diagnostic IDs confirmed
  row 4094 was unselected. Every case first ran valid attention successfully,
  then supplied the poisoned history upload: run rejected nonfinite input,
  readback rejected the formerly valid snapshot as unavailable, and allocation
  counters were unchanged during the failed run. This specifically checks late
  unselected-source coverage and the output-validity gate, in addition to the
  unchanged suite's current/future-slot exclusion tests.
- `git diff --check`: passed.
- The after benchmark GPU/driver was checked: RTX 5080 / 580.178.04 on device 0.

### After timings (ms/iteration, medians)

Raw samples are captured directly from this task's benchmark output in
`WP3_ATTENTION_BENCHMARK_AFTER.json`. The baseline remains separate in
`WP3_ATTENTION_BENCHMARK_BEFORE.json`. Each file has 24 cases / 72 actual sample
values; all stored medians were checked against their three samples. No raw
samples were reconstructed from medians.

| n | Attention ggml_cpu | Combined ggml_cpu | Attention strata | Combined strata |
|---:|---:|---:|---:|---:|
| 1 | 0.008560 | 0.037053 | 0.008560 | 0.037018 |
| 2051 | 0.609910 | 2.497379 | 0.609731 | 2.498490 |
| 2053 | 0.609882 | 2.449376 | 0.587875 | 2.391770 |
| 4096 | 0.593549 | 3.629782 | 0.589808 | 3.654595 |
| 8192 | 0.600032 | 5.123840 | 0.600346 | 5.169232 |
| 16384 | 0.626294 | 6.942007 | 0.627302 | 7.013370 |

At n=16384, ggml_cpu attention-only changed from 15.338458 to 0.626294 ms,
and combined from 26.591709 to 6.942007 ms. At n=1, attention-only regressed
from 0.006502 to 0.008560 ms and combined from 0.034966 to 0.037053 ms:
there is an extra validation launch even when history is empty. No special-case
optimization is included. These are observed component-event timings from
separate runs, not interleaved paired measurements, isolated validation-kernel
profiles, or complete-model speedups. GPU load, clocks, caching and enqueue gaps
can affect them. The single source change removes known redundant scans; the
measurements do not establish attribution of every microsecond or general
performance beyond these fixtures. The slow selector remains in combined time.

## Limitations and next computational boundary

Fixed 24/2 heads and D256 FP32; no combined-qg or strided API. Full history finite
validation remains O(n), even with selection capped at 2051. Source validation
and attention are not graph-captured; calls remain synchronous and globally
serialized. The existing reference selector is single-threaded and unoptimized.
No output projection, RMSNorm/RoPE, pooling/commit, session integration, checkpoint
parity or complete-model speedup claim. No multi-GPU attention, sanitizer/race
checker or extensive fuzzing validation is established. No safety check is
intentionally removed by this optimization; no new first-output trust boundary.

Next computational step after this isolated work: add the native output
projection consumer of gated `[24,256]` output using the existing exact-upload
weight contract, without prematurely adding cache mutation or sequence commit.
Reinspect the actual projection weight format/shape and reusable native matvec
before choosing that implementation; this note is not a claim it is implemented.
