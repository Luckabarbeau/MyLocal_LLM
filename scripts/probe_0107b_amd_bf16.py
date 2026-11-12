#!/usr/bin/env python3
"""Isolate BF16 CuPy/ROCm aborts without killing other MyLocal_LLM probes.

Run from the project root:
    MINI_LLM_BACKEND=cupy .venv-rocm/bin/python -u scripts/probe_0107b_amd_bf16.py

Each case executes in a new Python interpreter so a native SIGABRT is recorded
without hiding subsequent (notably Triton) results. Read-only: no checkpoints,
source modifications, or GPU power/clock changes.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import traceback

os.environ.setdefault("TRITON_DEFAULT_BACKEND", "amd")

STAGES = (
    ("hip-fp32", "CuPy HIP + FP32 GEMM baseline"),
    ("bf16-dtype", "NumPy/ml_dtypes and CuPy BF16 dtype recognition"),
    ("bf16-allocate", "CuPy BF16 array allocation/fill"),
    ("bf16-cast", "CuPy float32 -> BF16 conversion"),
    ("bf16-model", "MyLocal_LLM validate_bfloat16_backend"),
    ("triton-fp32", "Triton FP32 vector arithmetic via CuPy HIP pointers"),
    ("triton-bf16", "Triton BF16 tensor-core dot via FP32 CuPy buffers"),
)


def _ptr(array, triton_dtype):
    class Pointer:
        def __init__(self, a, d):
            self.array = a   # Maintain lifetime of the underlying allocation.
            self.dtype = d

        def data_ptr(self):
            return int(self.array.data.ptr)

    return Pointer(array, triton_dtype)


def _stage(name: str):
    import cupy as cp
    if not cp.cuda.runtime.is_hip:
        raise RuntimeError("CUDA CuPy found instead of HIP CuPy")
    print("CuPy", cp.__version__, "HIP", cp.cuda.runtime.is_hip, flush=True)

    if name == "hip-fp32":
        x = cp.ones((64, 64), dtype=cp.float32)
        z = x @ x
        cp.cuda.get_current_stream().synchronize()
        print("FP32 [0,0] =", float(z[0, 0].get()), flush=True)
        assert abs(float(z[0, 0].get()) - 64.0) < 1e-5
        return

    if name in ("bf16-dtype", "bf16-allocate", "bf16-cast", "bf16-model"):
        import numpy as np
        import ml_dtypes
        print("NumPy", np.__version__, "ml_dtypes", ml_dtypes.__version__, flush=True)
        print("NumPy BF16 dtype:", np.dtype(ml_dtypes.bfloat16), flush=True)
        if name == "bf16-dtype":
            print("About to construct CuPy BF16 dtype", flush=True)
            print("CuPy BF16 dtype:", cp.dtype(ml_dtypes.bfloat16), flush=True)
            return
        if name == "bf16-allocate":
            print("About to run cp.ones(..., dtype=ml_dtypes.bfloat16)", flush=True)
            x = cp.ones((16, 16), dtype=ml_dtypes.bfloat16)
            cp.cuda.get_current_stream().synchronize()
            print("BF16 array allocated:", x.dtype, x.shape, flush=True)
            return
        if name == "bf16-cast":
            x = cp.ones((16, 16), dtype=cp.float32)
            print("About to run float32.astype(ml_dtypes.bfloat16)", flush=True)
            x = x.astype(ml_dtypes.bfloat16)
            cp.cuda.get_current_stream().synchronize()
            print("BF16 cast completed:", x.dtype, flush=True)
            return
        if name == "bf16-model":
            from mini_llm.backend import BACKEND_NAME, validate_bfloat16_backend
            print("MyLocal_LLM backend:", BACKEND_NAME, flush=True)
            if BACKEND_NAME != "cupy":
                raise RuntimeError(f"Expected cupy backend, got {BACKEND_NAME!r}")
            print("About to run validate_bfloat16_backend()", flush=True)
            print("Returned dtype:", validate_bfloat16_backend(), flush=True)
            return

    import triton
    import triton.language as tl
    # Triton captures `tl` from the function's globals during JIT compilation.
    globals()["tl"] = tl
    print("Triton", triton.__version__, flush=True)
    target = triton.runtime.driver.active.get_current_target()
    print("Triton target:", target, flush=True)
    if target.backend != "hip" or not str(target.arch).startswith("gfx1201"):
        raise RuntimeError(
            "Refusing to launch: Triton selected " + str(target) +
            ". Install the AMD ROCm Triton wheel and verify HIP/gfx1201."
        )

    if name == "triton-fp32":
        @triton.jit
        def add_kernel(A, B, C, N: tl.constexpr, BLOCK: tl.constexpr):
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(A + i, i < N, other=0)
            y = tl.load(B + i, i < N, other=0)
            tl.store(C + i, x + y, i < N)

        a = cp.arange(256, dtype=cp.float32)
        b = cp.ones((256,), dtype=cp.float32)
        c = cp.empty_like(a)
        # Triton 3.8 removed the explicit stream= launch argument.
        # Device-wide barriers keep this smoke test correct even when
        # CuPy and Triton use different default streams.
        cp.cuda.runtime.deviceSynchronize()
        add_kernel[(1,)](_ptr(a, tl.float32), _ptr(b, tl.float32),
                         _ptr(c, tl.float32), 256, 256)
        cp.cuda.runtime.deviceSynchronize()
        error = float(cp.max(cp.abs(c - (a + b))).get())
        print("Triton FP32 max error:", error, flush=True)
        assert error < 1e-5
        return

    if name == "triton-bf16":
        @triton.jit
        def dot_kernel(A, B, C):
            r = tl.arange(0, 32)
            k = tl.arange(0, 32)
            a = tl.load(A + r[:, None] * 32 + k[None, :]).to(tl.bfloat16)
            b = tl.load(B + k[:, None] * 32 + r[None, :]).to(tl.bfloat16)
            c = tl.dot(a, b)
            tl.store(C + r[:, None] * 32 + r[None, :], c)

        # BF16 is *inside Triton only*: no CuPy BF16 arrays are created here.
        a = cp.ones((32, 32), dtype=cp.float32)
        b = cp.ones((32, 32), dtype=cp.float32)
        c = cp.empty((32, 32), dtype=cp.float32)
        print("Compiling Triton tl.dot(BF16, BF16) -> FP32", flush=True)
        cp.cuda.runtime.deviceSynchronize()
        dot_kernel[(1,)](_ptr(a, tl.float32), _ptr(b, tl.float32),
                         _ptr(c, tl.float32))
        cp.cuda.runtime.deviceSynchronize()
        error = float(cp.max(cp.abs(c - 32.0)).get())
        print("Triton BF16 GEMM max error:", error, flush=True)
        assert error < 1e-3
        return

    raise ValueError(f"Unknown stage: {name}")


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--stage":
        try:
            _stage(sys.argv[2])
        except Exception:
            traceback.print_exc()
            return 1
        return 0

    print("=== MyLocal_LLM ROCm crash-isolation diagnostic (0107B) ===", flush=True)
    print("Python:", sys.executable, flush=True)
    print("ROCM_HOME:", os.environ.get("ROCM_HOME", "<unset>"), flush=True)
    print("MINI_LLM_BACKEND:", os.environ.get("MINI_LLM_BACKEND", "<unset>"), flush=True)
    results = []
    for name, description in STAGES:
        print(f"\n=== {name}: {description} ===", flush=True)
        result = subprocess.run([sys.executable, "-u", os.path.abspath(__file__),
                                 "--stage", name], check=False, timeout=60)
        if result.returncode < 0:
            signum = -result.returncode
            status = f"NATIVE CRASH: {signal.Signals(signum).name} ({signum})"
        elif result.returncode:
            status = f"FAIL (exit {result.returncode})"
        else:
            status = "PASS"
        print(f"=> {name}: {status}", flush=True)
        results.append((name, status))

    print("\n=== SUMMARY ===", flush=True)
    for name, status in results:
        print(f"{name:18s} {status}", flush=True)
    return 0 if all(status == "PASS" for _, status in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
