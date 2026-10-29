"""Experimental fused Model B scoring/softmax/weighted pooling forward.

This fuses the small learned score GEMM, positional bias, token-softmax and
weighted token reduction for each (block, summary).  All arithmetic accumulates
in FP32; input tokens and weights may be FP32/BF16/FP16.  This path is opt-in
until RTX 5080 forward AND backward parity and warmed full-model A/B pass.

The existing backward uses the returned probabilities.  The kernel reads each
block twice, so fusion is not *a priori* faster than a well-packed GEMM; report
measured speed and retain the reference path as a fallback.
"""

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


class _Pointer:
    def __init__(self, array, dtype):
        self.array = array
        self.dtype = dtype

    def data_ptr(self):
        return self.array.data.ptr


def _ptr(array):
    if tl is None:
        raise RuntimeError("Triton is required for fused matrix pooling")
    types = {
        "float32": tl.float32,
        "float16": tl.float16,
        "bfloat16": tl.bfloat16,
    }
    dtype = types.get(str(array.dtype))
    if dtype is None:
        raise TypeError(f"unsupported matrix pooling dtype: {array.dtype}")
    return _Pointer(array, dtype)


if triton is not None:
    @triton.jit
    def _forward(H, W, BIAS, OUT, PROB,
                 H0: tl.constexpr, H1: tl.constexpr, H2: tl.constexpr,
                 W0: tl.constexpr, W1: tl.constexpr,
                 B0: tl.constexpr, B1: tl.constexpr,
                 T: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
                 WITH_BIAS: tl.constexpr,
                 BT: tl.constexpr, BD: tl.constexpr):
        n = tl.program_id(0)
        r = tl.program_id(1)
        ts = tl.arange(0, BT)
        ds = tl.arange(0, BD)
        score = tl.full((BT,), 0, tl.float32)
        # Keep the intermediate live tile bounded: 128 x 64 values, rather
        # than an enormous 128 x 1024 x 4 tensor in one program.
        for start in range(0, D, BD):
            f = start + ds
            h = tl.load(H + n*H0 + ts[:, None]*H1 + f[None, :]*H2,
                        mask=(ts[:, None] < T) & (f[None, :] < D),
                        other=0).to(tl.float32)
            w = tl.load(W + f*W0 + r*W1, mask=f < D,
                        other=0).to(tl.float32)
            score += tl.sum(h * w[None, :], axis=1)
        if WITH_BIAS:
            bias = tl.load(BIAS + ts*B0 + r*B1, mask=ts < T,
                           other=0).to(tl.float32)
            score += bias
        score = tl.where(ts < T, score, -float('inf'))
        p = tl.exp(score - tl.max(score, axis=0))
        p /= tl.sum(p, axis=0)
        tl.store(PROB + n*T*R + ts*R + r, p, mask=ts < T)
        # Weighted pooling has no value projection: the 1024-dimensional
        # summary is projected only after the number of tokens is reduced.
        for start in range(0, D, BD):
            f = start + ds
            h = tl.load(H + n*H0 + ts[:, None]*H1 + f[None, :]*H2,
                        mask=(ts[:, None] < T) & (f[None, :] < D),
                        other=0).to(tl.float32)
            result = tl.sum(p[:, None] * h, axis=0)
            tl.store(OUT + n*R*D + r*D + f, result, mask=f < D)


def forward(blocks, score_weights, position_bias=None):
    """Pool contiguous [N,T,D] blocks, returning ([N,R,D], [N,T,R])."""
    import cupy as cp
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if blocks.ndim != 3 or score_weights.ndim != 2:
        raise ValueError("expected blocks [N,T,D], score weights [D,R]")
    n, t, d = map(int, blocks.shape)
    width, r = map(int, score_weights.shape)
    if n < 1 or t < 1 or d != width or r < 1:
        raise ValueError("invalid pooling dimensions")
    if position_bias is not None and position_bias.shape != (t, r):
        raise ValueError("pool position bias must be [T,R]")
    if not all(a.flags.c_contiguous for a in (blocks, score_weights)):
        raise ValueError("fused matrix pooling requires contiguous blocks/weights")
    if position_bias is not None and not position_bias.flags.c_contiguous:
        raise ValueError("fused matrix pooling requires contiguous position bias")
    output = cp.empty((n, r, d), dtype=cp.float32)
    probabilities = cp.empty((n, t, r), dtype=cp.float32)
    bias = position_bias if position_bias is not None else score_weights
    _forward[(n, r)](
        *map(_ptr, (blocks, score_weights, bias, output, probabilities)),
        *(int(s // blocks.itemsize) for s in blocks.strides),
        *(int(s // score_weights.itemsize) for s in score_weights.strides),
        *(int(s // bias.itemsize) for s in bias.strides),
        t, d, r, position_bias is not None,
        triton.next_power_of_2(t), 64,
        num_warps=8 if t >= 128 else 4,
    )
    return output, probabilities
