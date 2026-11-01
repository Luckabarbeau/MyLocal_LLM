# PATCH 0103A2 — Memory-aware GPU parity regression

## Why this patch exists

The `gpu/matrix_pool_triton_parity` gate failed at the Triton `_dq_kernel` launch with `Triton Error [CUDA]: out of memory` on a 513-token, production-width BF16 test. This is **not** a demonstrated numerical mismatch. Available GPU memory may be constrained by the simultaneously running 348M training job; the traceback alone does not establish the exact allocation responsible.

## Changes

- The `test_production_width_triton_mixed_window_and_global_parity` test copies reference forward results and gradients to **CPU** arrays before constructing the Triton operator. It then drops unreachable GPU objects, runs Python collection, and releases the CuPy memory pools' unused cached allocations. The Triton run likewise returns CPU arrays. The production model width, sequence length (513), forward/backward operations and tolerances are unchanged.
- The GPU runner now reports **actual free/total VRAM before each suite** in an isolated process and includes those values in its JSON report.
- A suite is marked **blocked, not passed**, when less than 2 GiB is free at suite startup. This is a conservative diagnostic guard, **not a proof** that 2 GiB suffices for all tests. CUDA OOM can still occur after startup (for example, if training changes its allocations).
- CPU runner tests validate both the VRAM query and the blocked/not-passed behavior.

## What this does not do

- It does **not** change a numerical kernel, model configuration, optimizer, training checkpoint, or GPU training path.
- It does **not** prove Triton/CuPy parity. That requires executing the GPU gate after applying the patch.
- It does **not** guarantee that a parity test can fit alongside a running training job.
- It does **not** downgrade a failed numerical assertion to a warning or skip.

## Recommended execution

Apply after 0103-0, 0103A, and 0103A1:

```bash
git apply --check PATCH_0103A2_GPU_MEMORY_AWARE_PARITY.patch
git apply PATCH_0103A2_GPU_MEMORY_AWARE_PARITY.patch
python scripts/run_0103a_regressions.py --mode cpu \
  --report /tmp/model_b_cpu_regression.json
```

Before the GPU run, check whether training or serving occupies VRAM:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

Ideally run the GPU correctness gate **when the 348M training process is not occupying the GPU**:

```bash
python scripts/run_0103a_regressions.py --mode gpu \
  --report /tmp/model_b_gpu_regression.json
```

If the test still reports CUDA OOM with ample GPU memory available, investigate Triton's memory requirements and the device/pool state; do not assume model correctness or blindly suppress the error. The next diagnostic would compare memory before/after the reference and Triton runs, and inspect the kernel launch's scratch requirement.

## Validation in the patch-authoring environment

- New/updated CPU runner unit tests: pass.
- Full CPU gate after changes: 64 passed, 4 GPU-only skipped.
- The environment has no accessible GPU; GPU parity remains **unverified**.
