#!/usr/bin/env python3
"""Single-AMD-GPU ROCm/CuPy/Triton preflight for MyLocal_LLM (no training).

Run from the MyLocal_LLM root with its dedicated .venv-rocm Python:
    MINI_LLM_BACKEND=cupy .venv-rocm/bin/python -u scripts/probe_0107a_r9700.py

Does not alter GPU configuration, checkpoints, or the NVIDIA environment.
"""
from __future__ import annotations

import os
import sys
import subprocess
import traceback

os.environ.setdefault("TRITON_DEFAULT_BACKEND", "amd")


def _triton_vector_test(cp):
    """Compile and run a real Triton kernel on CuPy HIP buffers."""
    global tl  # Triton JIT resolves tl via the compiled function's globals.
    import triton
    import triton.language as tl

    target = triton.runtime.driver.active.get_current_target()
    print("Triton target:", target, flush=True)
    if target.backend != "hip" or not str(target.arch).startswith("gfx1201"):
        raise RuntimeError(f"Refusing to launch on non-R9700 target: {target}")

    class CuPyPointer:
        def __init__(self, array, dtype):
            self.array = array  # Keep backing allocation alive during launch.
            self.dtype = dtype

        def data_ptr(self):
            return int(self.array.data.ptr)

    @triton.jit
    def vector_add(x, y, out, n: tl.constexpr, block: tl.constexpr):
        idx = tl.program_id(0) * block + tl.arange(0, block)
        a = tl.load(x + idx, idx < n, other=0.0)
        b = tl.load(y + idx, idx < n, other=0.0)
        tl.store(out + idx, a + b, idx < n)

    x = cp.arange(256, dtype=cp.float32)
    y = cp.ones(256, dtype=cp.float32)
    z = cp.empty_like(x)
    args = (CuPyPointer(x, tl.float32), CuPyPointer(y, tl.float32),
            CuPyPointer(z, tl.float32), 256, 256)
    cp.cuda.get_current_stream().synchronize()
    # MyLocal_LLM uses explicit CuPy-stream launch for Triton 3.8; match it.
    compiled = vector_add.warmup(*args, grid=(1,), num_warps=4)
    compiled[(1, 1, 1)](*args, stream=int(cp.cuda.get_current_stream().ptr))
    cp.cuda.get_current_stream().synchronize()
    err = float(cp.max(cp.abs(z - (x + y))).get())
    if err != 0.0:
        raise AssertionError(f"Triton vector add max error: {err}")
    print(f"PASS: Triton {triton.__version__}, CuPy HIP pointer + kernel execution", flush=True)


def main() -> int:
    print("=== MyLocal_LLM R9700 ROCm preflight (fixed script) ===", flush=True)
    print("Python:", sys.executable, flush=True)
    print("MINI_LLM_BACKEND:", os.environ.get("MINI_LLM_BACKEND", "<unset>"), flush=True)
    print("ROCM_HOME:", os.environ.get("ROCM_HOME", "<unset>"), flush=True)

    try:
        import cupy as cp
        print("CuPy:", cp.__version__, "HIP:", cp.cuda.runtime.is_hip, flush=True)
        if not cp.cuda.runtime.is_hip:
            print("STOP: CUDA CuPy detected; no GPU tests performed.", flush=True)
            return 2
        n = cp.cuda.runtime.getDeviceCount()
        print("HIP GPU count:", n, flush=True)
        if n < 1:
            print("FAIL: No HIP GPU detected", flush=True)
            return 2
        cp.cuda.Device(0).use()
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get("name", "unknown")
        print("HIP GPU 0:", name.decode(errors="replace") if isinstance(name, bytes) else name,
              flush=True)
    except Exception as exc:
        print("FAIL: CuPy HIP device initialization:", repr(exc), flush=True)
        traceback.print_exc(limit=5)
        return 2

    failures = []
    try:
        a = cp.arange(128 * 128, dtype=cp.float32).reshape(128, 128) / 16384.0
        result = a @ cp.eye(128, dtype=cp.float32)
        cp.cuda.get_current_stream().synchronize()
        error = float(cp.max(cp.abs(result - a)).get())
        if error > 1e-6:
            raise AssertionError(f"FP32 GEMM max error: {error}")
        print("PASS: CuPy FP32 rocBLAS GEMM", flush=True)
    except Exception as exc:
        failures.append("FP32 GEMM")
        print("FAIL: FP32 GEMM:", repr(exc), flush=True)
        traceback.print_exc(limit=5)

    try:
        # A native CuPy BF16 abort must not prevent the Triton check below.
        probe = os.path.join(os.path.dirname(__file__), "probe_0107b_amd_bf16.py")
        result = subprocess.run(
            [sys.executable, "-u", probe, "--stage", "bf16-model"],
            check=False, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"BF16 validation subprocess returned {result.returncode}")
        print("PASS: MyLocal_LLM backend BF16 GEMM", flush=True)
    except Exception as exc:
        failures.append("MyLocal_LLM BF16 GEMM")
        print("FAIL: MyLocal_LLM BF16 GEMM:", repr(exc), flush=True)
        traceback.print_exc(limit=5)

    try:
        _triton_vector_test(cp)
    except Exception as exc:
        failures.append("Triton HIP kernel")
        print("FAIL: Triton HIP kernel:", repr(exc), flush=True)
        traceback.print_exc(limit=5)

    if failures:
        print("RESULT: NOT READY —", ", ".join(failures), flush=True)
        return 1
    print("RESULT: PASS — ready to attempt Model B kernel parity tests (not full training)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
