"""Experimental exact matrix-pooling backward for contiguous completed blocks.

Two Triton launches compute dLogits followed by dHidden and per-block dWeights.
FP32 accumulation is used; the final small parameter-gradient reductions remain
on GPU.  This is opt-in pending GPU compiler/parity and steady-state A/B tests.
"""

import os

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from .triton_matrix_pool_0102s import _ptr


def _r4_logits_enabled():
    return os.environ.get(
        "MINI_LLM_MATRIX_POOL_R4_LOGITS_BACKWARD", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}


if triton is not None:
    @triton.jit
    def _logits_backward(H, PROB, DS, DZ,
                         T: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
                         BT: tl.constexpr, BD: tl.constexpr):
        n = tl.program_id(0)
        r = tl.program_id(1)
        ts = tl.arange(0, BT)
        ds = tl.arange(0, BD)
        dp = tl.full((BT,), 0, tl.float32)
        for start in range(0, D, BD):
            f = start + ds
            h = tl.load(H + n*T*D + ts[:, None]*D + f[None, :],
                        (ts[:, None] < T) & (f[None, :] < D), 0).to(tl.float32)
            grad_s = tl.load(DS + n*R*D + r*D + f, f < D, 0).to(tl.float32)
            dp += tl.sum(h * grad_s[None, :], 1)
        p = tl.load(PROB + n*T*R + ts*R + r, ts < T, 0).to(tl.float32)
        weighted_mean = tl.sum(dp * p, 0)
        dz = (dp - weighted_mean) * p
        tl.store(DZ + n*T*R + ts*R + r, dz, ts < T)


    @triton.jit
    def _logits_backward_r4(H, PROB, DS, DZ,
                            T: tl.constexpr, D: tl.constexpr,
                            BT: tl.constexpr, BD: tl.constexpr):
        n = tl.program_id(0)
        ts = tl.arange(0, BT)
        ds = tl.arange(0, BD)
        dp0 = tl.full((BT,), 0, tl.float32)
        dp1 = tl.full((BT,), 0, tl.float32)
        dp2 = tl.full((BT,), 0, tl.float32)
        dp3 = tl.full((BT,), 0, tl.float32)
        for start in range(0, D, BD):
            f = start + ds
            h = tl.load(H + n*T*D + ts[:, None]*D + f[None, :],
                        (ts[:, None] < T) & (f[None, :] < D), 0).to(tl.float32)
            g0 = tl.load(DS + n*4*D + f, f < D, 0).to(tl.float32)
            g1 = tl.load(DS + n*4*D + D + f, f < D, 0).to(tl.float32)
            g2 = tl.load(DS + n*4*D + 2*D + f, f < D, 0).to(tl.float32)
            g3 = tl.load(DS + n*4*D + 3*D + f, f < D, 0).to(tl.float32)
            dp0 += tl.sum(h * g0[None, :], 1)
            dp1 += tl.sum(h * g1[None, :], 1)
            dp2 += tl.sum(h * g2[None, :], 1)
            dp3 += tl.sum(h * g3[None, :], 1)
        p0 = tl.load(PROB + n*T*4 + ts*4, ts < T, 0).to(tl.float32)
        p1 = tl.load(PROB + n*T*4 + ts*4 + 1, ts < T, 0).to(tl.float32)
        p2 = tl.load(PROB + n*T*4 + ts*4 + 2, ts < T, 0).to(tl.float32)
        p3 = tl.load(PROB + n*T*4 + ts*4 + 3, ts < T, 0).to(tl.float32)
        z0 = (dp0 - tl.sum(dp0 * p0, 0)) * p0
        z1 = (dp1 - tl.sum(dp1 * p1, 0)) * p1
        z2 = (dp2 - tl.sum(dp2 * p2, 0)) * p2
        z3 = (dp3 - tl.sum(dp3 * p3, 0)) * p3
        mask = ts < T
        base = n*T*4 + ts*4
        tl.store(DZ + base, z0, mask)
        tl.store(DZ + base + 1, z1, mask)
        tl.store(DZ + base + 2, z2, mask)
        tl.store(DZ + base + 3, z3, mask)


    @triton.jit
    def _token_backward(H, PROB, DS, DZ, W, DX, DW_PARTIAL,
                        T: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
                        ADD_INITIAL: tl.constexpr,
                        BT: tl.constexpr, BD: tl.constexpr):
        n = tl.program_id(0)
        fi = tl.program_id(1) * BD + tl.arange(0, BD)
        ts = tl.arange(0, BT)
        mask = (ts[:, None] < T) & (fi[None, :] < D)
        h = tl.load(H + n*T*D + ts[:, None]*D + fi[None, :],
                    mask, 0).to(tl.float32)
        grad_h = tl.full((BT, BD), 0, tl.float32)
        for r in range(R):
            p = tl.load(PROB + n*T*R + ts*R + r, ts < T, 0).to(tl.float32)
            dz = tl.load(DZ + n*T*R + ts*R + r, ts < T, 0).to(tl.float32)
            grad_s = tl.load(DS + n*R*D + r*D + fi, fi < D, 0).to(tl.float32)
            w = tl.load(W + fi*R + r, fi < D, 0).to(tl.float32)
            grad_h += p[:, None] * grad_s[None, :] + dz[:, None] * w[None, :]
            grad_w = tl.sum(h * dz[:, None], 0)
            tl.store(DW_PARTIAL + n*D*R + fi*R + r, grad_w, fi < D)
        off = n*T*D + ts[:, None]*D + fi[None, :]
        if ADD_INITIAL:
            grad_h += tl.load(DX + off, mask, 0).to(tl.float32)
        tl.store(DX + off, grad_h, mask)


def backward(blocks, probabilities, d_summaries, score_weights, dx,
             add_initial=False):
    """Return (dW, dPosition), and write dHidden directly into `dx`.

    All inputs must be C-contiguous, and `dx` must exactly cover the flattened
    contiguous complete blocks (no trailing partial blocks between batch rows).
    """
    import cupy as cp

    if triton is None:
        raise RuntimeError("Triton is required for fused matrix-pooling backward")
    if blocks.ndim != 3 or d_summaries.ndim != 3:
        raise ValueError("expected blocks [N,T,D] and summary grads [N,R,D]")
    n, t, d = map(int, blocks.shape)
    r = int(d_summaries.shape[1])
    if (n < 1 or r < 1 or t < 1 or
            d_summaries.shape != (n, r, d) or
            probabilities.shape != (n, t, r) or
            score_weights.shape != (d, r) or
            dx.shape != (n, t, d)):
        raise ValueError("matrix-pooling backward tensor shapes disagree")
    if dx.dtype != cp.float32:
        raise TypeError("destination gradient must be float32")
    arrays = (blocks, probabilities, d_summaries, score_weights, dx)
    if not all(a.flags.c_contiguous for a in arrays):
        raise ValueError("matrix-pooling backward requires C-contiguous arrays")
    dz = cp.empty((n, t, r), dtype=cp.float32)
    partial = cp.empty((n, d, r), dtype=cp.float32)
    if r == 4 and _r4_logits_enabled():
        _logits_backward_r4[(n,)](
            *map(_ptr, (blocks, probabilities, d_summaries, dz)),
            t, d, triton.next_power_of_2(t), 64,
            num_warps=4,
        )
    else:
        _logits_backward[(n, r)](
            *map(_ptr, (blocks, probabilities, d_summaries, dz)),
            t, d, r, triton.next_power_of_2(t), 64,
            num_warps=4,
        )
    _token_backward[(n, triton.cdiv(d, 32))](
        *map(_ptr, (blocks, probabilities, d_summaries, dz,
                    score_weights, dx, partial)),
        t, d, r, bool(add_initial), triton.next_power_of_2(t), 32,
        num_warps=4,
    )
    return cp.sum(partial, axis=0), cp.sum(dz, axis=0)
