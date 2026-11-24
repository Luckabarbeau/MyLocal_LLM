# WP4 — isolated reference_fp32 hyperconnection primitives

## Scope and handoff boundary

New implementation only:

- `mini_llm/native_cpp/hyperconnection.h`
- `mini_llm/native_cpp/hyperconnection.cu`
- `tests/native_hc_primitives_harness.py` (standalone test, temporary test-only shim)
- this document

No runtime, runtime header, CMake, Python production binding, production dispatch,
shared status, or shared documentation changes are part of this work. Parent WP3
work is preserved. These are independently tested primitives, **not completed
WP4/runtime integration**, checkpoint parity, or end-to-end decoder validation.

## Source mathematics and licensing

Direct implementation authorities inspected for this work:

1. `mini_llm/external/qwen4exp_hyperconnection.py`: `_mix`,
   `_read_uncaptured`, `write`, `collapse`, and converted-gamma/epsilon validation.
2. `mini_llm/ops/hyperconnection_decode_gpu.py`: separate divide then gamma
   multiply, stable sigmoid, FP32 write, `--fmad=false` convention.
3. `mini_llm/ops/hyperconnection_read_gpu.py`: 256-thread grouped normalization
   and down/inject reductions, eight feature warps per up/mix block, four-stream
   sum order, single-token output-first layouts.
4. `mini_llm/native_cpp/qsa_frontend.h/.cu`: caller-owned stream/buffers,
   sticky device status and immediate CUDA launch-error convention. HC intentionally
   retains its own divide-then-gamma graph, not the QSA inverse-RMS graph.
5. `docs/qwen4exp/0205_hyperconnections.md`: existing mathematical/provenance
   record. It cites llama.cpp `171e8846b4af9766c354064cb776cb34a50f053f`
   and Radiance `a555ef392a48fbdac3df4a0eeb1197ec3f6e6100`; this work did not
   re-inspect or revalidate those external source trees or execute their engines.

The new CUDA is adapted from the in-repository Python/CUDA equations and launch
layout above, under the repository's `LICENCE.TXT` (GNU LGPL version 3 text).
No third-party source files, Strata kernels, or model weights were copied into
these files; no new MIT/Apache relicensing claim is made. The historical upstream
references are provenance of the existing graph, not a claim of independent
upstream execution parity or a new audit of upstream licensing.

For one token, `h[4,D]`, converted multiplicative `gamma[4,D]`, and caller epsilon:

```text
n[c,j] = (h[c,j] / sqrt(sum_j(h[c,j]^2)/D + epsilon)) * gamma[c,j]
a[r]   = (sum_k down[r,k] * flatten(n)[k]) * 0.25
low[r] = a[r] * sigmoid(a[r])
g[c,j] = sigmoid(sum_r up[c*D+j,r] * low[r])
mix[j] = (((0 + n[0,j]*g[0,j]) + n[1,j]*g[1,j])
          + n[2,j]*g[2,j] + n[3,j]*g[3,j]) * 0.25
gain[c] = 2 * sigmoid((sum_k inject[c,k] * flatten(n)[k]) * 0.25)
new_h[c,j] = ORIGINAL_h[c,j] + gain[c] * block_output[j]
```

Normalization is **four independent groups**, never a single width-4D RMS.
There is no additional `+1` on gamma. Division by four precedes SiLU; read gates
are per-feature, write gains per-stream. Write uses the original residual, not
normalized streams or the read mixture. Final collapse is exactly the read mix,
without injection or a conventional output normalization afterward. Caller owns
embedding replication, block computation, paired read/write ordering and history.

All weights, buffers and intermediates are FP32. There is no BF16 storage or
rounding mode and no implicit quantization. Stable sigmoid uses `exp(-abs(x))`;
this does not make RMS or projections overflow-safe. Arithmetic uses explicit
round-to-nearest multiply/add/divide/sqrt intrinsics to preserve graph boundaries.
FP32 reductions can differ from Python GEMM; bitwise equality to arbitrary GEMM
implementations is not the contract. Compile without fast math or FTZ.

## Exact integration ABI requirements

The header is the authoritative **C++ internal** launch ABI (no `extern "C"`
production export). The three entry points return `cudaError_t`:

```cpp
ml_hc_read_launch(stream, d, rank, epsilon,
    h, gamma, down, up, inject, normalized, low, mixed, gains, status);
ml_hc_collapse_launch(stream, d, rank, epsilon,
    h, gamma, down, up, normalized, low, mixed, status);
ml_hc_write_launch(stream, d, h, block_output, gains, new_h, status);
```

All tensor pointers are contiguous FP32 device buffers on the caller's current
device; `status` is a device `int*`. Shapes/counts:

| Buffer | Elements/layout |
|---|---|
| `h`, `new_h`, `gamma`, `normalized` | `4*D`, stream-major `[4,D]` |
| `down` | `R*4*D`, row-major `[R,4*D]` |
| `up` | `4*D*R`, row-major `[4*D,R]` |
| `inject` | `4*4*D`, row-major `[4,4*D]` |
| `low` | `R` |
| `mixed`, `block_output` | `D` |
| `gains` | `4` |
| `status` | one device int, caller zero-initialized |

`D>0`, `R>0`, `D<=INT_MAX/4`, `R<=INT_MAX-4`, epsilon finite positive FP32,
including positive FP32 subnormals with FTZ disabled. All pointers are required
except the pair `inject/gains`: both null selects read-only; exactly one null is
invalid. Collapse always selects this read-only path. No batch dimension or
implicit striding is supported; caller dispatches separate tokens if needed.

Caller allocates scratch `normalized[4D]` and `low[R]`: `(4D+R)*sizeof(float)`
bytes, 42,240 bytes at recorded WP2 dimensions `D=2560, R=320`. Scratch, outputs
and status must be disjoint from all inputs and each other; inputs may share
read-only storage. No in-place write is promised. Caller checks actual sizes,
contiguity, pointer validity/device ownership and nonoverlap. Host validation
checks nulls/scalars/index limits, not allocation metadata or gain provenance.

Read enqueues normalization, down/injection, and up/mix on **exactly the supplied
stream**, checking each submission. Write enqueues one kernel. Collapse uses the
same three-launch read graph with injection absent. Nothing allocates, copies,
synchronizes, caches activations, switches device, or uses the default stream
implicitly. Buffers remain alive and stable through the caller's completion gate;
concurrent calls need separate scratch/outputs and appropriate input ordering.

Caller zeroes status on the stream before the transaction. Any nonfinite consumed
input, arithmetic intermediate or output atomically sets it to 1, even if sigmoid
would mask the nonfinite logit. Status is never cleared by these launches, so it
may be shared across ordered stages. On status!=0, discard outputs; no sanitizing
fallback is provided. Invalid host arguments return `cudaErrorInvalidValue`
without launches or status changes. Return success means submission only: parent
must also check asynchronous CUDA errors and status at its existing completion
boundary, without adding per-kernel synchronization. A submission error can leave
earlier stages enqueued; preserve buffer lifetime in that case too.

Parent follow-up, deliberately not performed here:

1. Add `hyperconnection.cu` to its build with the project's CUDA architecture,
   FP32/no-fast-math settings and this header.
2. Upload/lease converted FP32 gamma and output-first dense down/up/inject weights;
   any packed storage conversion belongs outside this ABI.
3. Allocate caller scratch and two residual buffers. Retain each read's gains
   until its paired block output writes to that read's original residual.
4. Call collapse once at the final HC with its own gamma/down/up; do not add an
   injection or extra conventional output norm.
5. Integrate existing stream/error/status ownership and then perform checkpoint,
   layer and decoder validation. No new binding or public runtime ABI is supplied.

## Standalone validation (2026-11-23)

The harness builds **only** `hyperconnection.cu` and a temporary C-linkage test
shim. The shim/library are deleted on exit, and neither is a production binding.
`nvcc` subprocess timeout is 120 seconds; outer runs below were bounded to 180
seconds. Exact compile options used:

```sh
/usr/local/cuda/bin/nvcc -std=c++17 -O2 --fmad=false -ftz=false \
  -prec-div=true -prec-sqrt=true -arch=sm_120 --shared -Xcompiler -fPIC \
  -I mini_llm/native_cpp mini_llm/native_cpp/hyperconnection.cu \
  /tmp/<harness-generated-directory>/shim.cu -o /tmp/<harness-generated-directory>/hc.so
```

Run the reproducible harness (it supplies real temporary paths automatically):

```sh
MINI_LLM_BACKEND=cupy .venv/bin/python tests/native_hc_primitives_harness.py
MINI_LLM_BACKEND=cupy /usr/local/cuda/bin/compute-sanitizer \
  --tool memcheck --error-exitcode 99 \
  .venv/bin/python tests/native_hc_primitives_harness.py
```

Verified on CUDA 13.4 `nvcc`, RTX 5080 (device 0) and RTX 5070 Ti (device 1):

- Standalone compilation and numerical/error tests passed on both devices.
- Latest run: **memcheck ERROR SUMMARY: 0 errors**, exit 0.
- Latest maximum absolute difference across all recorded comparisons:
  **2.38418579e-7**, requested `rtol=atol=2e-6` (FP32 kernels, no finite differences).
- Geometries include `(D,R)=(1,1),(3,2),(7,5),(33,35),(257,257),
  (2560,32),(2560,320)`, covering odd widths, warp/rank tails and production
  dimensions. Eight fixtures per device including zero and saturated cases.
- Compared reads/gains against existing non-fast Python FP32 HC and an
  independent FP64 graph; two nonlinear synthetic paired writes per fixture;
  final collapse and null-injection equivalence; exact hand fixture for converted
  gamma/grouped RMS, exact write arithmetic, input/output/gain non-mutation.
- Explicit nonblocking streams; invalid dimensions/pointer pairs/epsilon;
  positive subnormal epsilon; NaN/+Inf/-Inf in each read source; finite overflow
  in normalization/projections; write nonfinites/overflow; sticky status retention.

The first harness run compiled successfully but stopped before kernel invocation:
its ctypes declaration incorrectly counted eleven read pointer arguments instead
of ten. A localized **test-only** declaration correction resolved it. Kernel
implementation was unchanged. After adding the production-rank fixture, the
latest full sanitizer run above passed. These results establish only the isolated
files and harness state, not the concurrently evolving parent runtime or full
repository. No benchmark, external-engine bitwise parity, real checkpoint, graph
capture, batch/prefill, or end-to-end completion claim is made.
