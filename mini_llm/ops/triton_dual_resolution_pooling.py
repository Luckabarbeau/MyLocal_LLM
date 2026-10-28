"""Fused learned per-block attention pooling (experimental 0102J CUDA path).

Each program handles one (complete-block, compression-head) pair.  Scores,
softmax, pooled value, and their adjoints are fused without materializing
[N,H,T,D] intermediate products.  This operator does *not* implement
compressor self-attention; its query is the learned per-head pool query.

The caller retains the original CuPy path for A/B tests and CPU use. This
module imports Triton only when explicitly enabled on a CUDA machine.
"""
from __future__ import annotations


def _require_gpu():
    import cupy as cp
    import triton
    import triton.language as tl
    return cp, triton, tl


# Import Triton below deliberately: importing this module is itself allowed on
# CPU, but using it requires Triton/CUDA.
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _pool_forward(K, V, Q, OUT, PROBS,
                      K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr,
                      V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr, V3: tl.constexpr,
                      Q0: tl.constexpr, Q1: tl.constexpr,
                      O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr,
                      P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                      T: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
                      BT: tl.constexpr, BD: tl.constexpr):
        n = tl.program_id(0)
        h = tl.program_id(1)
        t = tl.arange(0, BT)
        d = tl.arange(0, BD)
        mask = (t[:, None] < T) & (d[None, :] < D)
        k = tl.load(K + n * K0 + h * K1 + t[:, None] * K2 + d[None, :] * K3,
                    mask=mask, other=0).to(tl.float32)
        v = tl.load(V + n * V0 + h * V1 + t[:, None] * V2 + d[None, :] * V3,
                    mask=mask, other=0).to(tl.float32)
        q = tl.load(Q + h * Q0 + d * Q1, mask=d < D, other=0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * SCALE
        scores = tl.where(t < T, scores, float('-inf'))
        probs = tl.exp(scores - tl.max(scores, axis=0))
        probs = probs / tl.sum(probs, axis=0)
        context = tl.sum(probs[:, None] * v, axis=0)
        tl.store(OUT + n * O0 + h * O1 + d * O2, context, mask=d < D)
        tl.store(PROBS + n * P0 + h * P1 + t * P2, probs, mask=t < T)


    @triton.jit
    def _pool_backward(K, V, Q, PROBS, DC, DK, DV, DQ,
                       K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr,
                       V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr, V3: tl.constexpr,
                       Q0: tl.constexpr, Q1: tl.constexpr,
                       P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                       C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
                       A0: tl.constexpr, A1: tl.constexpr, A2: tl.constexpr, A3: tl.constexpr,
                       B0: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr, B3: tl.constexpr,
                       R0: tl.constexpr, R1: tl.constexpr, R2: tl.constexpr,
                       T: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
                       BT: tl.constexpr, BD: tl.constexpr):
        n = tl.program_id(0)
        h = tl.program_id(1)
        t = tl.arange(0, BT)
        d = tl.arange(0, BD)
        mask = (t[:, None] < T) & (d[None, :] < D)
        k = tl.load(K + n * K0 + h * K1 + t[:, None] * K2 + d[None, :] * K3,
                    mask=mask, other=0).to(tl.float32)
        v = tl.load(V + n * V0 + h * V1 + t[:, None] * V2 + d[None, :] * V3,
                    mask=mask, other=0).to(tl.float32)
        q = tl.load(Q + h * Q0 + d * Q1, mask=d < D, other=0).to(tl.float32)
        p = tl.load(PROBS + n * P0 + h * P1 + t * P2, mask=t < T, other=0).to(tl.float32)
        dc = tl.load(DC + n * C0 + h * C1 + d * C2, mask=d < D, other=0).to(tl.float32)
        dp = tl.sum(v * dc[None, :], axis=1)
        ds = p * (dp - tl.sum(p * dp, axis=0))
        dk = ds[:, None] * q[None, :] * SCALE
        dv = p[:, None] * dc[None, :]
        dq = tl.sum(ds[:, None] * k, axis=0) * SCALE
        tl.store(DK + n * A0 + h * A1 + t[:, None] * A2 + d[None, :] * A3, dk, mask=mask)
        tl.store(DV + n * B0 + h * B1 + t[:, None] * B2 + d[None, :] * B3, dv, mask=mask)
        tl.store(DQ + n * R0 + h * R1 + d * R2, dq, mask=d < D)


class _Pointer:
    """Expose CuPy device memory through Triton's typed pointer protocol.

    CuPy ndarrays cannot be bound directly by the Triton version used by the
    project.  Keep the allocation alive while the kernel is launched, as in
    the existing local/global Triton attention adapters.
    """

    def __init__(self, array, dtype):
        self.array = array
        self.dtype = dtype

    def data_ptr(self):
        return self.array.data.ptr


def _ptr(array):
    name = str(array.dtype)
    dtype = {
        'float32': tl.float32,
        'float16': tl.float16,
        'bfloat16': tl.bfloat16,
    }.get(name)
    if dtype is None:
        raise TypeError(f'unsupported Triton pooling dtype: {name}')
    return _Pointer(array, dtype)


def _strides(arr):
    return tuple(int(s // arr.itemsize) for s in arr.strides)


def forward(pool_k, pool_v, pool_query, scale):
    """Return FP32 pooled context [blocks,heads,dim], probs [blocks,heads,T]."""
    cp, tri, _ = _require_gpu()
    if pool_k.shape != pool_v.shape or pool_k.ndim != 4:
        raise ValueError('pool K/V must have the same [N,H,T,D] shape')
    n, h, t, d = map(int, pool_k.shape)
    if pool_query.shape != (h, d) or not (n > 0 and t > 0 and d > 0):
        raise ValueError('invalid learned pooling query or empty inputs')
    out = cp.empty((n, h, d), dtype=cp.float32)
    probs = cp.empty((n, h, t), dtype=cp.float32)
    _pool_forward[(n, h)](
        *map(_ptr, (pool_k, pool_v, pool_query, out, probs)),
        *_strides(pool_k), *_strides(pool_v), *_strides(pool_query),
        *_strides(out), *_strides(probs),
        t, d, float(scale), tri.next_power_of_2(t), tri.next_power_of_2(d),
        num_warps=4 if t * d <= 4096 else 8,
    )
    return out, probs


def backward(d_context, pool_k, pool_v, pool_query, probabilities, scale):
    """Return K/V gradients and query gradient, all FP32.

    The per-block query gradients are reduced across blocks *after* the
    kernel, so no floating-point atomics or inter-program races are needed.
    """
    cp, tri, _ = _require_gpu()
    n, h, t, d = map(int, pool_k.shape)
    if (pool_v.shape != pool_k.shape or d_context.shape != (n, h, d)
            or probabilities.shape != (n, h, t)
            or pool_query.shape != (h, d)):
        raise ValueError('incompatible learned pooling backward shapes')
    dk = cp.empty((n, h, t, d), dtype=cp.float32)
    dv = cp.empty_like(dk)
    dq_partial = cp.empty((n, h, d), dtype=cp.float32)
    _pool_backward[(n, h)](
        *map(_ptr, (pool_k, pool_v, pool_query, probabilities, d_context,
                    dk, dv, dq_partial)),
        *_strides(pool_k), *_strides(pool_v), *_strides(pool_query),
        *_strides(probabilities), *_strides(d_context),
        *_strides(dk), *_strides(dv), *_strides(dq_partial),
        t, d, float(scale), tri.next_power_of_2(t), tri.next_power_of_2(d),
        num_warps=4 if t * d <= 4096 else 8,
    )
    return dk, dv, cp.sum(dq_partial, axis=0)
