# R9700 CuPy BF16 GEMM repair

## Cause and scope

CuPy 14.2.0's `cupy_backends/hip/cupy_hipblas.h` omitted CUDA datatype
14 (BF16) from `convert_hipblasDatatype_t`. BF16 GEMM consequently threw an
uncaught C++ `unrecognized type` exception. The fix maps it to `HIPBLAS_R_16B`.
ROCm's BF16 compute support was already present.

`PATCH_0107D_CUPY_HIP_BF16.patch` applies to the **CuPy v14.2.0 source tree**,
not this repository. Only the `cupy_backends.cuda.libs.cublas` extension was
rebuilt. No model equations, NVIDIA packages, or checkpoint files were changed.

## Installed local state

- Python environment: `.venv-rocm`, Python 3.12, CuPy 14.2.0.
- ROCm: `/opt/rocm-7.2.4`; Triton 3.8.0 ROCm build; target `hip/gfx1201`.
- MIOpen: matching `miopen-hip` 3.5.1.70204-93~24.04 package extracted under
  `.venv-rocm/rocm-deps`, not installed system-wide.
- Patched source: `.venv-rocm/cupy-14.2.0`.
- Build output: `.venv-rocm/cupy-bf16-build/lib/cupy_backends/cuda/libs/`.
- Original installed binding retained beside the replacement as
  `cublas.cpython-312-x86_64-linux-gnu.so.before-bf16-fix` under
  `.venv-rocm/lib/python3.12/site-packages/cupy_backends/cuda/libs/`.

These virtual-environment artifacts are local. Reinstalling CuPy can overwrite
the fix, and a different CuPy/Python/ROCm version needs a matching rebuild.
Do not copy this binary into the NVIDIA environment.

## Reproduce the build

Download and extract the CuPy v14.2.0 source archive into `.venv-rocm`, apply
the patch there (`patch -p1 < ../../PATCH_0107D_CUPY_HIP_BF16.patch`), then run
from the repository root (replace the absolute project path if relocated):

```sh
CUPY_SOURCE=/home/lucka/Documents/my_local_LLM/MyLocal_LLM/.venv-rocm/cupy-14.2.0 \
  .venv-rocm/bin/python scripts/build_rocm_cupy_blas.py build_ext --build-lib lib
```

The helper uses an isolated build directory and does **not** install the result.
Back up the installed `cublas` extension before copying the built extension over
it; use a fresh Python process for validation. The helper requires Cython,
setuptools, a C++ compiler, Python headers, and ROCm development headers.

## Verified commands and results

```sh
sh scripts/run_r9700_preflight.sh
sh scripts/run_r9700_preflight.sh scripts/probe_0107d_model_b_rocm.py --dtype bfloat16
sh scripts/run_r9700_preflight.sh scripts/probe_0107d_model_b_rocm.py --dtype float32
```

- Both preflight scripts passed, including CuPy BF16 validation and Triton FP32
  and BF16 tests (Triton errors zero).
- A standalone 16x16 BF16 ones GEMM passed with zero error.
- Twelve random BF16 GEMMs passed for `(M,K,N)` of `(16,16,16)`, `(17,33,19)`,
  `(64,128,48)` and all four contiguous/transposed input combinations.
  Reference: quantized BF16 inputs multiplied in NumPy FP64, output rounded to
  BF16 then converted to FP32. `rtol=0.01`, `atol=0.01`; largest absolute
  discrepancy was 0.0078125. This validates these cases, not arbitrary GEMMs.
- Full registered Model B, 348,261,376 parameters, BF16: forward, loss, backward,
  and all finite-gradient checks passed; loss 10.315400, cold process 21.14 s.
- Same Model B FP32 regression passed; loss 10.317940, cold process 17.52 s.

The model diagnostic uses random weights, 256 synthetic input tokens and eight
scored predictions. It retains the registered architecture/context capacity,
uses activation checkpointing, and does not save or load checkpoints. Without
`--optimizer-step` it tests only forward/backward. Neither mode is a long-context
benchmark, warmed throughput comparison, or verification of the complete pinned
BF16-mixed production profile.
Timings include initialization and variable kernel-cache effects; do not infer
a BF16/FP32 speed ratio from these two runs.

## Hybrid Muon/AdamW optimizer-step gate

Verified command:

```sh
sh scripts/run_r9700_preflight.sh -u scripts/probe_0107d_model_b_rocm.py --dtype bfloat16 --optimizer-step
```

This opt-in gate uses BF16 parameters, FP32 gradients and master weights, full
optimizer offload (default), 16 MiB AdamW staging, LR 1.5e-4, weight decay 0.1,
AdamW betas 0.9/0.95, Muon momentum 0.95, five Newton–Schulz iterations with
BF16 GEMMs, and global gradient clipping at 1.0. It uses the same short-context
synthetic workload, not the sourced production kernel profile or 512-way
microbatch accumulation.

Observed successful run:

- 348,261,376 parameters; 220 Muon tensors and 82 AdamW tensors.
- Forward/loss/backward passed in 16.97 s including initialization.
- Pre-clipping gradient norm: 272.105377; clipping scale: 0.003675.
- Both optimizer groups advanced to step 1 with finite parameters, masters,
  momentum, and AdamW second moments.
- Bounded 4,096-element-per-tensor samples changed in 220 Muon masters / 217
  Muon model tensors and 82 AdamW masters / 41 AdamW model tensors. Zero or
  unchanged BF16 samples in some tensors are permitted; each group must show
  both master and model updates.
- Refreshed model compute buffers and recomputed loss on the same batch:
  **10.315400 -> 5.774983** (finite). A decrease is observed, not a test requirement.
- Optimizer update: **2.46 s**; complete diagnostic including state construction,
  checks, and second forward: **46.35 s**.

This verifies one real update only. It does not establish multi-step stability,
optimizer numerical parity, full-profile Triton correctness, or production
throughput. No checkpoint files were read or written.
