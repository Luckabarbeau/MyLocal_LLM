"""0102L: fused RMSNorm for the dual-resolution *block compressor* only.

Unlike model-wide RMSNorm this runs on the two normalization sites inside
128-token compression blocks. It retains FP32 inverse RMS, computes the
normalization and input-gradient in Triton, and sums gamma-gradient tiles on
GPU without FP32 [num_tokens, width] intermediates. The CPU/CuPy reference
remains available with MINI_LLM_DUAL_RESOLUTION_FUSED_RMSNORM=0.
"""
from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


class _Pointer:
    """CuPy allocation adapter for Triton's typed-pointer runtime."""

    def __init__(self, array, dtype):
        self.array = array
        self.dtype = dtype

    def data_ptr(self):
        return self.array.data.ptr


def _ptr(array):
    mapping = {
        "float32": tl.float32,
        "float16": tl.float16,
        "bfloat16": tl.bfloat16,
    }
    dtype = mapping.get(str(array.dtype))
    if dtype is None:
        raise TypeError(f"unsupported compressor RMSNorm dtype: {array.dtype}")
    return _Pointer(array, dtype)


if triton is not None:
    @triton.jit
    def _norm_forward(X, GAMMA, Y, INV,
                      N: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr,
                      R: tl.constexpr, C: tl.constexpr):
        row = tl.program_id(0) * R + tl.arange(0, R)
        col = tl.arange(0, C)
        valid = (row[:, None] < N) & (col[None, :] < D)
        x = tl.load(X + row[:, None] * D + col[None, :],
                    mask=valid, other=0).to(tl.float32)
        gamma = tl.load(GAMMA + col, mask=col < D, other=0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
        y = x * inv[:, None] * gamma[None, :]
        tl.store(Y + row[:, None] * D + col[None, :], y, mask=valid)
        tl.store(INV + row, inv, mask=row < N)

    @triton.jit
    def _norm_backward(DY, X, GAMMA, INV, DX, DG_PARTIAL,
                       N: tl.constexpr, D: tl.constexpr,
                       R: tl.constexpr, C: tl.constexpr,
                       WRITE_DG: tl.constexpr):
        group = tl.program_id(0)
        row = group * R + tl.arange(0, R)
        col = tl.arange(0, C)
        valid = (row[:, None] < N) & (col[None, :] < D)
        x = tl.load(X + row[:, None] * D + col[None, :],
                    mask=valid, other=0).to(tl.float32)
        dy = tl.load(DY + row[:, None] * D + col[None, :],
                     mask=valid, other=0).to(tl.float32)
        gamma = tl.load(GAMMA + col, mask=col < D, other=0).to(tl.float32)
        inv = tl.load(INV + row, mask=row < N, other=0).to(tl.float32)
        dn = dy * gamma[None, :]
        projection = tl.sum(dn * x, axis=1) / D
        dx = (dn - x * (inv[:, None] * inv[:, None]) * projection[:, None]) * inv[:, None]
        tl.store(DX + row[:, None] * D + col[None, :], dx, mask=valid)
        if WRITE_DG:
            partial = tl.sum(dy * x * inv[:, None], axis=0)
            tl.store(DG_PARTIAL + group * D + col, partial, mask=col < D)


def _flatten(value):
    import cupy as cp
    if value.ndim < 2 or int(value.shape[-1]) < 1:
        raise ValueError("compressor RMSNorm expects [*,D] with at least one row")
    width = int(value.shape[-1])
    # Inputs from the packed projection/residual path are generally contiguous.
    # Make the rare noncontiguous case explicit; kernels intentionally use
    # contiguous [rows, D] indexing to avoid strided broadcast intermediates.
    flat = cp.ascontiguousarray(value.reshape(-1, width))
    return flat, int(flat.shape[0]), width


def _check_backend():
    if triton is None:
        raise RuntimeError("fused compressor RMSNorm requires Triton")
    import cupy as cp
    return cp


def forward(value, gamma, eps=1e-6):
    """Return (value_dtype output, FP32 per-token inverse RMS)."""
    cp = _check_backend()
    data, n, d = _flatten(value)
    if tuple(gamma.shape) != (d,):
        raise ValueError("compressor RMSNorm gamma dimension mismatch")
    if not (eps > 0):
        raise ValueError("compressor RMSNorm epsilon must be positive")
    if n == 0:
        raise ValueError("empty compressor RMSNorm sequence")
    out = cp.empty(data.shape, dtype=value.dtype)
    inv = cp.empty((n,), dtype=cp.float32)
    cols = triton.next_power_of_2(d)
    rows = max(1, min(8, 4096 // cols))
    _norm_forward[(triton.cdiv(n, rows),)](
        *map(_ptr, (data, gamma, out, inv)),
        n, d, float(eps), rows, cols,
        num_warps=4,
    )
    return out.reshape(value.shape), inv


def backward(gradient, value, gamma, inverse_rms, *, accumulate_gamma=True):
    """Return FP32 dX and optional FP32 dgamma (sum over all rows).

    Gamma reductions are tiled so memory is proportional to (N/R)*D,
    rather than allocating one FP32 dgamma contribution per token.
    """
    cp = _check_backend()
    data, n, d = _flatten(value)
    grad, gn, gd = _flatten(gradient)
    if gn != n or gd != d or tuple(gamma.shape) != (d,) or tuple(inverse_rms.shape) != (n,):
        raise ValueError("incompatible compressor RMSNorm backward tensors")
    if n == 0:
        raise ValueError("empty compressor RMSNorm sequence")
    out = cp.empty((n, d), dtype=cp.float32)
    cols = triton.next_power_of_2(d)
    rows = max(1, min(8, 4096 // cols))
    groups = triton.cdiv(n, rows)
    partial = cp.empty((groups, d), dtype=cp.float32) if accumulate_gamma else cp.empty((1,), dtype=cp.float32)
    _norm_backward[(groups,)](
        *map(_ptr, (grad, data, gamma, inverse_rms, out, partial)),
        n, d, rows, cols, bool(accumulate_gamma), num_warps=4,
    )
    dgamma = cp.sum(partial, axis=0) if accumulate_gamma else None
    return out.reshape(value.shape), dgamma
