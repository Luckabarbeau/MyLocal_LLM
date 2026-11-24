# WP5: native caller for the existing GGML CPU row pool

## Scope and arithmetic

This is the independent **CPU caller gate**, not complete WP5 MoE integration.
C++ owns library resolution, source registration, shared mmap lifetime, registered
submission, job buffers, completion, and retirement. Python does not execute a
per-expert dispatch loop. No new scheduler or expert arithmetic was introduced.

The explicit arithmetic name is **`ggml_q8`**, NOT `reference_fp32`:

1. Quantize the shared FP32 input using the required GGML Q8_K/Q8_0 quantizer.
2. Existing packed gate and up integer-dot kernels produce FP32 values.
3. Compute `(gate / (1 + exp(-gate))) * up` in the existing bridge.
4. Quantize each FP32 hidden row for the down projection's activation type.
5. Existing packed down dot kernels produce **unweighted FP32 `[count,D]` rows**.

No routing multiplication, shared expert, GPU activation staging, model dispatch,
GPU Q8_1 approximation, placement policy, or whole-model allocation is added.
Host pool/job scratch is not charged as device/pinned payload in engine stats.
The existing Engine is still a CUDA engine (creation needs a device/stream), but
CPU job execution makes no CUDA calls and allocates no device/pinned payload.

## API and ownership

- `mini_llm/ops/ggml_packed_cpu.h`: declaration-only existing ABI (`Quantize`,
  `Dot`, `Projection`, `PoolExpert`, pool functions). The original `.cpp` only
  replaces those declarations with an include; worker/arithmetic code is unchanged.
- `mini_llm/ops/ggml_packed_cpu.py`: cache digest now includes this shared header,
  preventing stale bridge reuse after header edits.
- `mini_llm/native_cpp/cpu_expert_runtime.inc`: private native caller/registries.
- `mini_llm/native_cpp/runtime.h`: `ml_cpu_experts_create`, `_register`, `_submit`,
  `_destroy`; `ml_cpu_job_wait`, `_readback_for_test`, `_destroy`.
- `mini_llm/native/__init__.py`: thin `Engine.cpu_experts`, `CPUExperts`,
  `CPUExpertJob` wrappers. Diagnostic output is an explicit detached NumPy copy.
- CMake links `${CMAKE_DL_LIBS}`. The CPU bridge is loaded, not linked into a
  second scheduler implementation.

`Engine.cpu_experts(d_model=..., d_ff=..., threads=..., capacity=10,
bridge_library=..., ggml_library=..., arithmetic='ggml_q8')` requires explicit
library paths; there is no deployment path default, search, or silent fallback.
Callers may obtain the bridge path once at setup from the existing
`mini_llm.ops.ggml_packed_cpu._load_bridge()._name`; that helper may compile the
existing bridge into its cache. It is not used by per-expert submission.

`pool.register(gate_lease, up_lease, down_lease, offsets=(g,u,d))` returns a native
registration index. Offsets are tensor-relative **bytes**, not expert IDs or
file offsets. Each source must be a supported packed rank-2 matrix or rank-3
expert bank with the exact inner shape. Offsets must select a whole matrix:
`expert_id * rows * row_bytes`. Shape, activation-block alignment, type, and
subtensor bounds are checked before registration. Dimensions are configured once;
registration can mix formats across experts. Duplicate registration is allowed.
Source files must remain immutable throughout ownership.

`pool.submit(x, indices)` snapshots one contiguous FP32 `[D]` or `[1,D]` input
into native-owned storage and returns a job. Indices are registration indices,
not model expert IDs. Output preserves submission order. One outstanding job is
allowed per pool until `job.wait()`, readback, job close, or pool close drains it.
Busy registration/submission, invalid/stale/wrong-kind handles, invalid indices,
nonfinite input, bad offsets/shapes, and short diagnostic buffers are rejected.
C create/submit output handles are zeroed on failure; registration index is -1.

Native source retention is independent of Python objects: closing the public
store, source leases, or engine does not invalidate registered sources/jobs.
Pool close explicitly drains its pending batch and invalidates the public pool
handle. An existing job can still wait/read back after pool close. A job retains
its pool (and therefore mappings, engine and both library handles) until job
close; completed jobs also retain that owner. Job close drains before release.
Results already copied out remain detached and valid across future epochs.
All public ABI operations follow the existing runtime's serialized boundary lock.

## Dynamic loading and ABI validation

The native caller uses `dlopen(..., RTLD_NOW | RTLD_LOCAL)` for both explicit
paths and resolves symbols once. It retains both handles until after pool worker
destruction, and uses RAII for partial-setup failures. Resolution errors are
reported, never converted into a fallback backend.

Bridge symbols resolved: `mylocal_ggml_pool_create`, `_register`,
`_submit_registered`, `_wait`, `_destroy`.

GGML setup resolves/calls `ggml_cpu_init`, resolves `ggml_get_type_traits_cpu`,
and applies the same traits layout/checks as `GGMLPackedCPUExecutor`: a non-null
traits pointer and vec-dot pointer, and the expected vec-dot activation type.
It resolves `quantize_row_q8_K`, `quantize_row_q8_0`, plus:

| Weight format | GGML type | Activation type | Dot suffix after `ggml_vec_dot_` |
|---|---:|---:|---|
| IQ2_S | 22 | 15 / Q8_K | iq2_s_q8_K |
| IQ3_S | 21 | 15 / Q8_K | iq3_s_q8_K |
| IQ3_XXS | 18 | 15 / Q8_K | iq3_xxs_q8_K |
| IQ4_XS | 23 | 15 / Q8_K | iq4_xs_q8_K |
| IQ4_NL | 20 | 8 / Q8_0 | iq4_nl_q8_0 |
| Q2_0 | 42 | 8 / Q8_0 | q2_0_q8_0 |

`Projection.qtype` is the **activation** type, not the weight's GGML type.
Traits checks do not establish compatibility with arbitrary GGML forks. ABI
reference: `171e8846b4af9766c354064cb776cb34a50f053f`,
`ggml.h`, `ggml-cpu.h`, `ggml-common.h`, `ggml-cpu/quants.h`.
No claim is made that the loaded library's source revision has been verified.

## Provenance and licensing

The declaration extraction and private caller remain original project
LGPL-3.0-or-later code. The existing bridge retains its own SIMD/header notices;
GGML is an external MIT dependency. No external arithmetic was copied into the
native caller, and no Strata engine API was introduced.

Observed local test inputs (identities, **not defaults**):

- GGML selected path:
  `/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so`
- Resolved GGML filename: `libggml-cpu.so.0.25.3` in that directory.
- GGML SHA256:
  `924ac13f3e949cb1b62eddff20221600a6efcb1a2b99eb4abd390e75c23df839`.
- Bridge after declaration extraction:
  `/home/lucka/.cache/mylocal_llm/native/ggml_packed_cpu_e231d10a715274f9e9e6.so`.
- Bridge SHA256:
  `9be971e4091c0eda183e635d41d2bdb8fc24e3ac17b86b4ed90060bf7f7b6f5b`.

The bridge cache filename changes with source/header/compiler inputs; obtain it
from the setup helper instead of embedding this measured filename in deployment.

## Verified state (2026-11-23)

Build completed:

```sh
cmake -S mini_llm/native_cpp -B /tmp/mini-llm-wp1-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build /tmp/mini-llm-wp1-build -j 4
```

Build emitted existing `gguf_loader.cpp` misleading-indentation warnings; no
unrelated edits were made to suppress them.

New caller tests plus requested existing CPU regressions:

```sh
PYTHONDONTWRITEBYTECODE=1 MINI_LLM_BACKEND=numpy NVIDIA_TF32_OVERRIDE=0 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so \
MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so \
.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_native_engine_cpu_experts.py \
  tests/test_0256_ggml_packed_cpu.py \
  tests/test_0259_ggml_row_pool.py \
  tests/test_0271_registered_cpu_descriptors.py
```

**60 passed, 3 skipped in 2.04 s.** The skips are optional existing CPU tests.
The new tests use synthetic `D=256,F=320` packed banks, not checkpoints, covering
all seven actual gate/up/down triples:

- IQ3_XXS / IQ3_XXS / IQ4_NL
- IQ2_S / IQ2_S / Q2_0
- IQ2_S / IQ2_S / IQ4_NL
- IQ3_XXS / IQ3_XXS / Q2_0
- IQ3_S / IQ3_S / IQ4_NL
- IQ3_S / IQ3_S / Q2_0
- IQ4_XS / IQ4_XS / IQ4_NL

Native rows are exactly equal to the existing Python GGML executor's oracle
output, including count 10. Tests forbid subsequent Python `expert()` dispatch,
exercise native input ownership, busy/invalid/stale errors, draining pending
close, source mapping survival after public close, and final unmapping.

Full native suite with the established QSA and PLE flags:

```sh
PYTHONDONTWRITEBYTECODE=1 MINI_LLM_BACKEND=numpy NVIDIA_TF32_OVERRIDE=0 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MINI_LLM_TEST_PLE_CUDA=1 MINI_LLM_QSA_LAYER_CHECKPOINT=1 \
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so \
MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so \
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_native_engine*.py
```

**505 passed, 3 skipped in 71.07 s**, extending the parent 495/3 state by ten
caller tests. Both test commands completed within their 120-second tool bounds.
No status/manifest updates were made; parent owns status exclusively.

## True next tasks

1. Integrate this caller's native job ownership with bounded native GPU
   input/output staging and the existing handoff/service completion protocol.
   Preserve unweighted CPU rows, explicit ggml_q8 policy, and source lifetimes.
2. Integrate separately implemented MoE CUDA kernels under C++ launch ownership:
   routed IQ2_S/IQ3_S/IQ3_XXS/IQ4_XS GU, IQ4_NL/Q2_0 down, full-softmax top-10
   normalization (floor 2^-14), ascending-ID ordered weighted combination.
3. Add shared expert projections (Q4_K/Q5_K/Q6_K/IQ4_XS GU; IQ4_NL down and
   layer-47 Q8_0 down) and scalar shared sigmoid gate.
4. Validate mixed CPU/GPU routing and complete FFN HC read/compute/write ownership
   before making native MoE, complete decoder, or performance claims.
