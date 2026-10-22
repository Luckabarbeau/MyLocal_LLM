"""Graph-capturable direct cuBLAS FP16 matrix multiplication.

CuPy 14 deliberately rejects its high-level cuBLAS wrappers during CUDA stream
capture.  This module performs the same row-major ``out = a @ b`` operation
through a pre-bound libcublas handle and writes into caller-owned storage.
All loading, handle setup, and allocation must happen before capture.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass

from mini_llm.backend import BACKEND_NAME, xp, is_bfloat16_dtype


_CUBLAS_STATUS_NAMES = {
    0: "CUBLAS_STATUS_SUCCESS",
    1: "CUBLAS_STATUS_NOT_INITIALIZED",
    3: "CUBLAS_STATUS_ALLOC_FAILED",
    7: "CUBLAS_STATUS_INVALID_VALUE",
    8: "CUBLAS_STATUS_ARCH_MISMATCH",
    11: "CUBLAS_STATUS_MAPPING_ERROR",
    13: "CUBLAS_STATUS_EXECUTION_FAILED",
    14: "CUBLAS_STATUS_INTERNAL_ERROR",
    15: "CUBLAS_STATUS_NOT_SUPPORTED",
    16: "CUBLAS_STATUS_LICENSE_ERROR",
}

_CUBLAS_OP_N = 0
_CUDA_R_16F = 2
_CUDA_R_16BF = 14
_CUBLAS_COMPUTE_32F = 68
_CUBLAS_GEMM_DEFAULT = -1
_CUBLAS_POINTER_MODE_HOST = 0
_ALPHA_ONE = ctypes.c_float(1.0)
_BETA_ZERO = ctypes.c_float(0.0)
_LIB = None
_GEMM_EX = None


@dataclass(frozen=True)
class FP16GemmContext:
    function: object
    handle: int
    stream_ptr: int
    device_id: int


def _load_gemm_ex():
    global _LIB, _GEMM_EX
    if _GEMM_EX is not None:
        return _GEMM_EX
    if BACKEND_NAME != "cupy":
        raise RuntimeError("direct FP16 cuBLAS GEMM requires the CuPy backend")
    candidates = []
    found = ctypes.util.find_library("cublas")
    if found:
        candidates.append(found)
    candidates.extend(("libcublas.so.13", "libcublas.so.12", "libcublas.so"))
    last_error = None
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            function = lib.cublasGemmEx
            function.argtypes = (
                ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
            )
            function.restype = ctypes.c_int
            _LIB = lib
            _GEMM_EX = function
            return function
        except Exception as exc:  # pragma: no cover - installation dependent
            last_error = exc
    raise RuntimeError("unable to load cublasGemmEx") from last_error


def prepare_fp16_gemm_context(stream) -> FP16GemmContext:
    """Load and bind a borrowed CuPy cuBLAS handle before stream capture."""
    if BACKEND_NAME != "cupy":
        raise RuntimeError("direct FP16 cuBLAS GEMM requires the CuPy backend")
    from cupy.cuda import device
    from cupy_backends.cuda.libs import cublas

    function = _load_gemm_ex()
    handle = int(device.get_cublas_handle())
    stream_ptr = int(stream.ptr)
    cublas.setStream(handle, stream_ptr)
    cublas.setPointerMode(handle, _CUBLAS_POINTER_MODE_HOST)
    return FP16GemmContext(
        function=function,
        handle=handle,
        stream_ptr=stream_ptr,
        device_id=int(xp.cuda.runtime.getDevice()),
    )


def _validate_array(name, array, context, *, allow_fortran=False):
    if not isinstance(array, xp.ndarray):
        raise TypeError(f"fp16_matmul_into: {name} must be a CuPy array")
    layout_ok = array.flags.c_contiguous or (
        allow_fortran and array.flags.f_contiguous
    )
    if array.ndim != 2 or array.dtype != xp.float16 or not layout_ok:
        expected = "C- or F-contiguous" if allow_fortran else "C-contiguous"
        raise ValueError(
            f"fp16_matmul_into: {name} must be {expected} FP16 rank-2; "
            f"got shape={array.shape}, dtype={array.dtype}, "
            f"c_contiguous={array.flags.c_contiguous}, "
            f"f_contiguous={array.flags.f_contiguous}"
        )
    if int(array.device.id) != context.device_id:
        raise ValueError(f"fp16_matmul_into: {name} is on the wrong CUDA device")


def _ranges_overlap(left, right):
    left_start = int(left.data.ptr)
    right_start = int(right.data.ptr)
    return (
        left_start < right_start + int(right.nbytes)
        and right_start < left_start + int(left.nbytes)
    )


def fp16_matmul_into(a, b, out, *, context: FP16GemmContext) -> None:
    """Submit row-major FP16 ``out = a @ b`` without allocating or synchronizing."""
    _validate_array("a", a, context)
    _validate_array("b", b, context, allow_fortran=True)
    _validate_array("out", out, context)
    m, k = (int(value) for value in a.shape)
    bk, n = (int(value) for value in b.shape)
    if min(m, n, k) <= 0 or bk != k or tuple(out.shape) != (m, n):
        raise ValueError(
            "fp16_matmul_into: incompatible geometry: "
            f"a={a.shape}, b={b.shape}, out={out.shape}"
        )
    if max(m, n, k) > 2**31 - 1:
        raise ValueError("fp16_matmul_into: dimensions exceed cuBLAS int32 limits")
    if _ranges_overlap(out, a) or _ranges_overlap(out, b):
        raise ValueError("fp16_matmul_into: output must not overlap either input")
    current_stream = int(xp.cuda.get_current_stream().ptr)
    if current_stream != context.stream_ptr:
        raise RuntimeError(
            "fp16_matmul_into: current stream differs from prepared cuBLAS stream"
        )

    b_is_fortran = bool(b.flags.f_contiguous and not b.flags.c_contiguous)
    trans_b_storage = 1 if b_is_fortran else _CUBLAS_OP_N
    b_leading_dimension = k if b_is_fortran else n
    status = context.function(
        ctypes.c_void_p(context.handle),
        ctypes.c_int(trans_b_storage), ctypes.c_int(_CUBLAS_OP_N),
        ctypes.c_int(n), ctypes.c_int(m), ctypes.c_int(k),
        ctypes.byref(_ALPHA_ONE),
        ctypes.c_void_p(int(b.data.ptr)), ctypes.c_int(_CUDA_R_16F),
        ctypes.c_int(b_leading_dimension),
        ctypes.c_void_p(int(a.data.ptr)), ctypes.c_int(_CUDA_R_16F), ctypes.c_int(k),
        ctypes.byref(_BETA_ZERO),
        ctypes.c_void_p(int(out.data.ptr)), ctypes.c_int(_CUDA_R_16F), ctypes.c_int(n),
        ctypes.c_int(_CUBLAS_COMPUTE_32F), ctypes.c_int(_CUBLAS_GEMM_DEFAULT),
    )
    if status != 0:
        name = _CUBLAS_STATUS_NAMES.get(int(status), "CUBLAS_STATUS_UNKNOWN")
        raise RuntimeError(
            f"cublasGemmEx failed: {name} ({int(status)}); "
            f"row-major a={a.shape}, b={b.shape}, out={out.shape}; "
            f"column-major m={n}, n={m}, k={k}, "
            f"transa={trans_b_storage}, lda={b_leading_dimension}, "
            f"ldb={k}, ldc={n}"
        )


def _validate_bf16_array(name, array, context, *, allow_fortran=False):
    if not isinstance(array, xp.ndarray):
        raise TypeError(f"bf16_matmul_into: {name} must be a CuPy array")
    layout_ok = array.flags.c_contiguous or (
        allow_fortran and array.flags.f_contiguous
    )
    if array.ndim != 2 or not is_bfloat16_dtype(array.dtype) or not layout_ok:
        expected = "C- or F-contiguous" if allow_fortran else "C-contiguous"
        raise ValueError(
            f"bf16_matmul_into: {name} must be {expected} BF16 rank-2; "
            f"got shape={array.shape}, dtype={array.dtype}, "
            f"c_contiguous={array.flags.c_contiguous}, "
            f"f_contiguous={array.flags.f_contiguous}"
        )
    if int(array.device.id) != context.device_id:
        raise ValueError(f"bf16_matmul_into: {name} is on the wrong CUDA device")


def bf16_matmul_into(a, b, out, *, context: FP16GemmContext) -> None:
    """Submit row-major BF16 ``out = a @ b`` with FP32 accumulation."""
    _validate_bf16_array("a", a, context)
    _validate_bf16_array("b", b, context, allow_fortran=True)
    _validate_bf16_array("out", out, context)
    m, k = (int(value) for value in a.shape)
    bk, n = (int(value) for value in b.shape)
    if min(m, n, k) <= 0 or bk != k or tuple(out.shape) != (m, n):
        raise ValueError(
            "bf16_matmul_into: incompatible geometry: "
            f"a={a.shape}, b={b.shape}, out={out.shape}"
        )
    if max(m, n, k) > 2**31 - 1:
        raise ValueError("bf16_matmul_into: dimensions exceed cuBLAS int32 limits")
    if _ranges_overlap(out, a) or _ranges_overlap(out, b):
        raise ValueError("bf16_matmul_into: output must not overlap either input")
    if int(xp.cuda.get_current_stream().ptr) != context.stream_ptr:
        raise RuntimeError(
            "bf16_matmul_into: current stream differs from prepared cuBLAS stream"
        )

    b_is_fortran = bool(b.flags.f_contiguous and not b.flags.c_contiguous)
    trans_b_storage = 1 if b_is_fortran else _CUBLAS_OP_N
    b_leading_dimension = k if b_is_fortran else n
    status = context.function(
        ctypes.c_void_p(context.handle),
        ctypes.c_int(trans_b_storage), ctypes.c_int(_CUBLAS_OP_N),
        ctypes.c_int(n), ctypes.c_int(m), ctypes.c_int(k),
        ctypes.byref(_ALPHA_ONE),
        ctypes.c_void_p(int(b.data.ptr)), ctypes.c_int(_CUDA_R_16BF),
        ctypes.c_int(b_leading_dimension),
        ctypes.c_void_p(int(a.data.ptr)), ctypes.c_int(_CUDA_R_16BF), ctypes.c_int(k),
        ctypes.byref(_BETA_ZERO),
        ctypes.c_void_p(int(out.data.ptr)), ctypes.c_int(_CUDA_R_16BF), ctypes.c_int(n),
        ctypes.c_int(_CUBLAS_COMPUTE_32F), ctypes.c_int(_CUBLAS_GEMM_DEFAULT),
    )
    if status != 0:
        name = _CUBLAS_STATUS_NAMES.get(int(status), "CUBLAS_STATUS_UNKNOWN")
        raise RuntimeError(
            f"cublasGemmEx failed: {name} ({int(status)}); "
            f"BF16 row-major a={a.shape}, b={b.shape}, out={out.shape}; "
            f"column-major m={n}, n={m}, k={k}, "
            f"transa={trans_b_storage}, lda={b_leading_dimension}, "
            f"ldb={k}, ldc={n}"
        )
