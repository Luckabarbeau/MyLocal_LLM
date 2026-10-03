"""Thin opt-in cuBLAS grouped-GEMM bridge for BF16 row-major matrices.

CuPy's high-level ``@`` path is excellent for individual GEMMs, but local
attention executes many related matrices with different chunk sizes. CUDA
12.5+ exposes ``cublasGemmGroupedBatchedEx`` for exactly this case.  This
module intentionally keeps the bridge tiny and lazy so CPU/reference runs do
not import or load CUDA libraries.

The public helper accepts ordinary row-major matrix descriptions.  cuBLAS is
column-major, so each problem is mapped through ``C^T = B^T @ A^T`` without
moving matrix data.  A/B/C may therefore be direct pointers into strided
Q/K/V tensors as long as elements within each row are contiguous.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..backend import BACKEND_NAME, xp


@dataclass(frozen=True)
class RowMajorGemmProblem:
    """One ``C = op(A) @ op(B)`` row-major GEMM.

    Pointers are integer device addresses. ``lda/ldb/ldc`` are physical row
    strides in elements of the underlying, *untransposed* row-major arrays.
    """

    a_ptr: int
    b_ptr: int
    c_ptr: int
    m: int
    n: int
    k: int
    lda: int
    ldb: int
    ldc: int
    trans_a: bool = False
    trans_b: bool = False


@dataclass(frozen=True)
class RowMajorGemmGroup:
    """Uniform-size GEMMs that share one grouped-cuBLAS descriptor."""

    problems: Sequence[RowMajorGemmProblem]


_LIB = None
_GROUPED = None
_BATCHED = None
_DISABLED = False
_LOAD_ERROR = None


def _enabled():
    raw = __import__("os").environ.get(
        "MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM", "0"
    ).strip().lower()
    return BACKEND_NAME == "cupy" and raw not in {"0", "false", "off", "no"}




def local_cublas_mode():
    """Return the local-attention cuBLAS dispatcher selected for 0053E.

    ``grouped`` preserves the validated heterogeneous GroupedBatchedEx path.
    ``batched`` executes each uniform chunk/gradient group with
    GemmBatchedEx while keeping the exact same direct-pointer descriptors.
    """
    raw = __import__("os").environ.get(
        "MINI_LLM_LOCAL_CUBLAS_MODE", "grouped"
    ).strip().lower()
    if raw in {"grouped", "group", "grouped_batched"}:
        return "grouped"
    if raw in {"batched", "batch", "gemm_batched"}:
        return "batched"
    raise ValueError(
        "MINI_LLM_LOCAL_CUBLAS_MODE must be 'grouped' or 'batched'"
    )


def _autotune_requested():
    raw = __import__("os").environ.get(
        "MINI_LLM_CUBLAS_AUTOTUNE", "1"
    ).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def strict_enabled():
    raw = __import__("os").environ.get(
        "MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT", "0"
    ).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def is_enabled():
    return _enabled()


def _load_grouped_symbol():
    global _LIB, _GROUPED, _DISABLED, _LOAD_ERROR
    if _DISABLED:
        return None
    if _GROUPED is not None:
        return _GROUPED
    if BACKEND_NAME != "cupy":
        return None

    candidates = []
    found = ctypes.util.find_library("cublas")
    if found:
        candidates.append(found)
    candidates.extend(("libcublas.so.13", "libcublas.so.12", "libcublas.so"))

    last = None
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            fn = lib.cublasGemmGroupedBatchedEx
            # cublasStatus_t cublasGemmGroupedBatchedEx(
            #   handle, transa[], transb[], m[], n[], k[], alpha[], Aarray,
            #   Atype, lda[], Barray, Btype, ldb[], beta[], Carray, Ctype,
            #   ldc[], group_count, group_size[], computeType)
            c_int_p = ctypes.POINTER(ctypes.c_int)
            fn.argtypes = (
                ctypes.c_void_p,
                c_int_p, c_int_p,
                c_int_p, c_int_p, c_int_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, c_int_p,
                ctypes.c_void_p, ctypes.c_int, c_int_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, c_int_p,
                ctypes.c_int, c_int_p, ctypes.c_int,
            )
            fn.restype = ctypes.c_int
            _LIB = lib
            _GROUPED = fn
            return fn
        except Exception as exc:  # pragma: no cover - depends on CUDA install
            last = exc

    _LOAD_ERROR = last
    _DISABLED = True
    if strict_enabled():
        raise RuntimeError(
            "cublasGemmGroupedBatchedEx is unavailable on this CUDA runtime"
        ) from last
    return None



def _load_batched_symbol():
    """Load cublasGemmBatchedEx lazily from the same cuBLAS library."""
    global _LIB, _BATCHED, _DISABLED, _LOAD_ERROR
    if _DISABLED:
        return None
    if _BATCHED is not None:
        return _BATCHED
    if BACKEND_NAME != "cupy":
        return None

    # Ensure a libcublas handle is loaded first. GroupedBatchedEx is available
    # on the CUDA versions that support this optimization stack.
    _load_grouped_symbol()
    if _LIB is None:
        return None
    try:
        fn = _LIB.cublasGemmBatchedEx
        fn.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        )
        fn.restype = ctypes.c_int
        _BATCHED = fn
        return fn
    except Exception as exc:  # pragma: no cover - depends on CUDA install
        _LOAD_ERROR = exc
        if strict_enabled():
            raise RuntimeError(
                "cublasGemmBatchedEx is unavailable on this CUDA runtime"
            ) from exc
        return None


def _resolve_batched_algo(cublas):
    """Choose CUDA-13 autotune when exposed, otherwise normal heuristics.

    CUDA 13 added CUBLAS_GEMM_AUTOTUNE to GemmBatchedEx. CuPy versions built
    against older headers may not export the enum even when the runtime is
    newer, so falling back to CUBLAS_GEMM_DEFAULT is intentional and safe.
    """
    default = int(getattr(cublas, "CUBLAS_GEMM_DEFAULT", -1))
    if not _autotune_requested():
        return default
    value = getattr(cublas, "CUBLAS_GEMM_AUTOTUNE", None)
    if value is not None:
        return int(value)
    # Some CuPy wheels were built against pre-CUDA-13 headers even when used
    # with a CUDA-13 runtime, so the new enum may be missing from the Python
    # wrapper. The CUDA-13 cublasGemmAlgo_t value is 999. Only use that numeric
    # fallback when the active runtime is definitely CUDA 13 or newer.
    try:
        from cupy_backends.cuda.api import runtime
        if int(runtime.runtimeGetVersion()) >= 13000:
            return 999
    except Exception:  # pragma: no cover - runtime-specific
        pass
    return default

def available():
    if not _enabled():
        return False
    return _load_grouped_symbol() is not None


def _i32(values):
    return np.asarray(values, dtype=np.int32)


def batched_bf16_gemm(
    groups: Sequence[RowMajorGemmGroup], alpha=1.0, beta=0.0, *, output_dtype="bf16"
):
    """Execute uniform groups with cublasGemmBatchedEx.

    0053E deliberately keeps the same RowMajorGemmGroup descriptors used by
    GroupedBatchedEx. Each group becomes one tuned homogeneous batched GEMM;
    all pointer arrays are uploaded together once, then addressed by offset.
    """
    if not _enabled():
        return False
    fn = _load_batched_symbol()
    if fn is None:
        return False
    groups = tuple(g for g in groups if g.problems)
    if not groups:
        return True
    if output_dtype != "bf16":
        raise ValueError(
            "batched BF16 GEMM requires BF16 C output on this local path"
        )

    # Validate groups and flatten the three pointer tables into one upload.
    signatures = []
    a_ptrs = []
    b_ptrs = []
    c_ptrs = []
    offsets = []
    offset = 0
    for group in groups:
        p0 = group.problems[0]
        sig = (
            p0.m, p0.n, p0.k, p0.lda, p0.ldb, p0.ldc,
            p0.trans_a, p0.trans_b,
        )
        for p in group.problems:
            other = (
                p.m, p.n, p.k, p.lda, p.ldb, p.ldc,
                p.trans_a, p.trans_b,
            )
            if other != sig:
                raise ValueError(
                    "all GEMMs in a batched descriptor must be uniform"
                )
        signatures.append(sig)
        offsets.append(offset)
        offset += len(group.problems)
        # Row-major C=A@B maps to column-major C^T=B^T@A^T.
        for p in group.problems:
            a_ptrs.append(int(p.b_ptr))
            b_ptrs.append(int(p.a_ptr))
            c_ptrs.append(int(p.c_ptr))

    a_dev = xp.asarray(np.asarray(a_ptrs, dtype=np.uintp))
    b_dev = xp.asarray(np.asarray(b_ptrs, dtype=np.uintp))
    c_dev = xp.asarray(np.asarray(c_ptrs, dtype=np.uintp))

    from cupy.cuda import device
    from cupy_backends.cuda.libs import cublas
    try:
        from cupy_backends.cuda.api import runtime
        cuda_r_16bf = int(getattr(runtime, "CUDA_R_16BF", 14))
    except Exception:  # pragma: no cover
        cuda_r_16bf = 14

    handle = int(device.get_cublas_handle())
    cublas.setStream(handle, int(xp.cuda.get_current_stream().ptr))
    compute_32f = int(getattr(cublas, "CUBLAS_COMPUTE_32F", 68))
    algo = _resolve_batched_algo(cublas)
    alpha_host = np.asarray(np.float32(alpha))
    beta_host = np.asarray(np.float32(beta))
    ptr_bytes = np.dtype(np.uintp).itemsize

    for group, sig, off in zip(groups, signatures, offsets):
        m, n, k, lda, ldb, ldc, trans_a, trans_b = sig
        # Swapped operands/dimensions implement row-major GEMM through cuBLAS.
        transa = 1 if trans_b else 0
        transb = 1 if trans_a else 0
        cm_m, cm_n, cm_k = int(n), int(m), int(k)
        cm_lda, cm_ldb, cm_ldc = int(ldb), int(lda), int(ldc)
        a_array = int(a_dev.data.ptr) + off * ptr_bytes
        b_array = int(b_dev.data.ptr) + off * ptr_bytes
        c_array = int(c_dev.data.ptr) + off * ptr_bytes
        status = fn(
            ctypes.c_void_p(handle),
            ctypes.c_int(transa), ctypes.c_int(transb),
            ctypes.c_int(cm_m), ctypes.c_int(cm_n), ctypes.c_int(cm_k),
            ctypes.c_void_p(alpha_host.ctypes.data),
            ctypes.c_void_p(a_array), ctypes.c_int(cuda_r_16bf), ctypes.c_int(cm_lda),
            ctypes.c_void_p(b_array), ctypes.c_int(cuda_r_16bf), ctypes.c_int(cm_ldb),
            ctypes.c_void_p(beta_host.ctypes.data),
            ctypes.c_void_p(c_array), ctypes.c_int(cuda_r_16bf), ctypes.c_int(cm_ldc),
            ctypes.c_int(len(group.problems)),
            ctypes.c_int(compute_32f), ctypes.c_int(algo),
        )
        if status != 0:
            message = (
                f"cublasGemmBatchedEx failed with status {status} "
                f"for row-major geometry m={m}, n={n}, k={k}"
            )
            if strict_enabled():
                raise RuntimeError(message)
            return False
    return True


def grouped_bf16_gemm(
    groups: Sequence[RowMajorGemmGroup], alpha=1.0, beta=0.0, *, output_dtype="bf16"
):
    """Dispatch local BF16 GEMMs through grouped or homogeneous batched cuBLAS.

    ``MINI_LLM_LOCAL_CUBLAS_MODE=batched`` is the 0053E experimental path;
    the default ``grouped`` mode preserves the validated 0053D behavior.

    Inputs are BF16 and Tensor-Core accumulation is FP32
    (``CUBLAS_COMPUTE_32F``). This grouped mixed-precision path uses BF16
    outputs; final gradient reductions/scatters promote those BF16 temporaries
    into FP32 accumulation buffers. ``output_dtype`` is retained only to fail
    early with a clear message if FP32 C output is requested.
    """
    if local_cublas_mode() == "batched":
        return batched_bf16_gemm(
            groups, alpha=alpha, beta=beta, output_dtype=output_dtype
        )
    if not _enabled():
        return False
    fn = _load_grouped_symbol()
    if fn is None:
        return False
    groups = tuple(g for g in groups if g.problems)
    if not groups:
        return True

    # Validate uniformity inside each group and build row-major -> cuBLAS
    # column-major descriptors.  For Cr=A@B, Cr^T=B^T@A^T, hence the swapped
    # operand pointers and dimensions below.
    transa = []
    transb = []
    ms = []
    ns = []
    ks = []
    ldas = []
    ldbs = []
    ldcs = []
    group_sizes = []
    a_ptrs = []
    b_ptrs = []
    c_ptrs = []

    for group in groups:
        p0 = group.problems[0]
        signature = (
            p0.m, p0.n, p0.k, p0.lda, p0.ldb, p0.ldc,
            p0.trans_a, p0.trans_b,
        )
        for p in group.problems:
            sig = (p.m, p.n, p.k, p.lda, p.ldb, p.ldc, p.trans_a, p.trans_b)
            if sig != signature:
                raise ValueError("all GEMMs in a grouped descriptor must be uniform")

        # cuBLAS operand A is the row-major B storage; operation flags are
        # swapped but otherwise unchanged by the transpose identity.
        transa.append(1 if p0.trans_b else 0)  # CUBLAS_OP_T / CUBLAS_OP_N
        transb.append(1 if p0.trans_a else 0)
        ms.append(int(p0.n))
        ns.append(int(p0.m))
        ks.append(int(p0.k))
        ldas.append(int(p0.ldb))
        ldbs.append(int(p0.lda))
        ldcs.append(int(p0.ldc))
        group_sizes.append(len(group.problems))
        for p in group.problems:
            a_ptrs.append(int(p.b_ptr))
            b_ptrs.append(int(p.a_ptr))
            c_ptrs.append(int(p.c_ptr))

    transa = _i32(transa); transb = _i32(transb)
    ms = _i32(ms); ns = _i32(ns); ks = _i32(ks)
    ldas = _i32(ldas); ldbs = _i32(ldbs); ldcs = _i32(ldcs)
    group_sizes = _i32(group_sizes)
    alpha_host = np.full(len(groups), np.float32(alpha), dtype=np.float32)
    beta_host = np.full(len(groups), np.float32(beta), dtype=np.float32)

    # Aarray/Barray/Carray are *device arrays of pointers* for the grouped API.
    a_dev = xp.asarray(np.asarray(a_ptrs, dtype=np.uintp))
    b_dev = xp.asarray(np.asarray(b_ptrs, dtype=np.uintp))
    c_dev = xp.asarray(np.asarray(c_ptrs, dtype=np.uintp))

    from cupy.cuda import device
    from cupy_backends.cuda.libs import cublas

    handle = int(device.get_cublas_handle())
    # Keep the borrowed CuPy handle on the same stream as surrounding kernels.
    cublas.setStream(handle, int(xp.cuda.get_current_stream().ptr))

    # Stable CUDA/cuBLAS enum values. Prefer CuPy's wrappers when exported by
    # the installed version, but keep CUDA-12/13-compatible fallbacks.
    try:
        from cupy_backends.cuda.api import runtime
        cuda_r_16bf = int(getattr(runtime, "CUDA_R_16BF", 14))
    except Exception:  # pragma: no cover
        cuda_r_16bf = 14
    compute_32f = int(getattr(cublas, "CUBLAS_COMPUTE_32F", 68))
    if output_dtype == "bf16":
        c_type = cuda_r_16bf
    elif output_dtype == "float32":
        raise ValueError(
            "grouped BF16 GEMM requires BF16 C output on this cuBLAS path; "
            "promote during the final reduction/scatter instead"
        )
    else:
        raise ValueError("output_dtype must be 'bf16' or 'float32'")

    c_int_p = ctypes.POINTER(ctypes.c_int)
    status = fn(
        ctypes.c_void_p(handle),
        transa.ctypes.data_as(c_int_p), transb.ctypes.data_as(c_int_p),
        ms.ctypes.data_as(c_int_p), ns.ctypes.data_as(c_int_p), ks.ctypes.data_as(c_int_p),
        ctypes.c_void_p(alpha_host.ctypes.data), ctypes.c_void_p(int(a_dev.data.ptr)),
        cuda_r_16bf, ldas.ctypes.data_as(c_int_p),
        ctypes.c_void_p(int(b_dev.data.ptr)), cuda_r_16bf, ldbs.ctypes.data_as(c_int_p),
        ctypes.c_void_p(beta_host.ctypes.data), ctypes.c_void_p(int(c_dev.data.ptr)),
        c_type, ldcs.ctypes.data_as(c_int_p),
        len(groups), group_sizes.ctypes.data_as(c_int_p), compute_32f,
    )
    if status != 0:
        message = f"cublasGemmGroupedBatchedEx failed with status {status}"
        if strict_enabled():
            raise RuntimeError(message)
        global _DISABLED
        _DISABLED = True
        return False
    return True
