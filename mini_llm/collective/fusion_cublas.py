"""Allocation-free FP32 cuBLAS contractions for probability fusion."""

import numpy as np
import cupy as cp
from cupy.cuda import cublas, device


_CUBLAS_OP_N = 0
_CUBLAS_POINTER_MODE_HOST = 0
_ALPHA_ONE = np.asarray(1.0, dtype=np.float32)
_BETA_ZERO = np.asarray(0.0, dtype=np.float32)
_BETA_ONE = np.asarray(1.0, dtype=np.float32)


def fp32_matmul_into(a, b, out, *, accumulate=False):
    """Submit row-major ``out = a @ b + beta*out`` on the current stream."""
    for name, array in (("a", a), ("b", b), ("out", out)):
        if (not isinstance(array, cp.ndarray) or array.ndim != 2
                or array.dtype != cp.float32 or not array.flags.c_contiguous):
            raise ValueError(f"{name} must be a C-contiguous FP32 CuPy matrix")
    rows, inner = map(int, a.shape)
    b_inner, columns = map(int, b.shape)
    if b_inner != inner or out.shape != (rows, columns) or min(rows, inner, columns) < 1:
        raise ValueError(f"incompatible matrix shapes: {a.shape} @ {b.shape} -> {out.shape}")
    if max(rows, inner, columns) > 2**31 - 1:
        raise ValueError("matrix dimensions exceed cuBLAS int32 limits")
    out_start = int(out.data.ptr)
    out_end = out_start + int(out.nbytes)
    for array in (a, b):
        start = int(array.data.ptr)
        if start < out_end and out_start < start + int(array.nbytes):
            raise ValueError("output must not overlap either matrix input")

    handle = int(device.get_cublas_handle())
    cublas.setStream(handle, int(cp.cuda.get_current_stream().ptr))
    cublas.setPointerMode(handle, _CUBLAS_POINTER_MODE_HOST)
    beta = _BETA_ONE if accumulate else _BETA_ZERO
    # cuBLAS is column-major. Row-major C=A@B is submitted as C^T=B^T@A^T.
    cublas.sgemm(
        handle, _CUBLAS_OP_N, _CUBLAS_OP_N,
        columns, rows, inner,
        int(_ALPHA_ONE.ctypes.data), int(b.data.ptr), columns,
        int(a.data.ptr), inner,
        int(beta.ctypes.data), int(out.data.ptr), columns,
    )
