"""Opt-in, standalone FP32 all-causal-cell QSA decode (no model dispatch).

The graph is softmax(scale * sum_d(q * K)) @ V, then sigmoid(qg[:, D:]).
Only cache[:n-1] is read; current K/V is supplied separately. One CTA per
(head, 128-token part) cooperates over D in 32-token tiles. A second CTA per
head merges the unnormalized (maximum, denominator, numerator) triples using
log-sum-exp rescaling. No dot/tensor-core instructions or reduced-precision
operands are used; FP fusion is disabled just as in qsa_decode_gpu.

Tree FP32 dimension/token reductions and split-context merging change rounding
relative to the serial FP32 reference, NOT the mathematical graph. This is not
bitwise parity. The standalone tests/probe quantify both against a NumPy FP64
oracle; their tolerances are acceptance criteria, not measured accuracy claims.
Scratch is FP32 H * ceil(n/128) * (D+2) elements, at most 64 parts for n<=8192
(1,585,152 bytes for H=24,D=256). There is no full KV gather, cache mutation,
host data read, synchronization, or process-global stream override.
"""

import operator

import numpy as np

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU import and optional-dependency safe; no driver probe.
    triton = None
    tl = None


MAX_CONTEXT = 8192
PART_TOKENS = 128
_KERNELS = {}


class _Pointer:
    """CuPy's allocation with Triton's minimal tensor ABI, without copying."""

    def __init__(self, array):
        self.array = array
        self.dtype = tl.float32

    def data_ptr(self):
        return self.array.data.ptr


if triton is not None:
    @triton.jit(do_not_specialize=["N", "SCALE"])
    def _parts_kernel(
        Q, K, V, KEYS, VALUES, S,
        N, SCALE,
        D: tl.constexpr, GROUPS: tl.constexpr, PARTS: tl.constexpr,
        QS0: tl.constexpr, QS1: tl.constexpr,
        K0: tl.constexpr, K1: tl.constexpr,
        V0: tl.constexpr, V1: tl.constexpr,
        KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
        VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
        BD: tl.constexpr,
    ):
        head = tl.program_id(0).to(tl.int64)
        part = tl.program_id(1)
        kh = head // GROUPS
        d = tl.arange(0, BD).to(tl.int64)
        rows = tl.arange(0, 32)
        query = tl.load(Q + head * QS0 + d * QS1, d < D, other=0.0)
        current_k = tl.load(K + kh * K0 + d * K1, d < D, other=0.0)
        current_v = tl.load(V + kh * V0 + d * V1, d < D, other=0.0)
        maximum = tl.full((), -float("inf"), tl.float32)
        denominator = tl.full((), 0.0, tl.float32)
        numerator = tl.zeros((BD,), tl.float32)
        for start in range(0, 128, 32):
            cell = (part * 128 + start + rows).to(tl.int64)
            # Mask the *load*, not just its result: n-1 is uncommitted storage.
            past_mask = (cell[:, None] < N - 1) & (d[None, :] < D)
            past_k = tl.load(
                KEYS + cell[:, None] * KS0 + kh * KS1 + d[None, :] * KS2,
                past_mask, other=0.0,
            )
            key = tl.where(cell[:, None] == N - 1, current_k[None, :], past_k)
            logits = tl.sum(key * query[None, :], axis=1) * SCALE
            logits = tl.where(cell < N, logits, -float("inf"))
            next_max = tl.maximum(maximum, tl.max(logits, axis=0))
            alpha = tl.exp(maximum - next_max)
            weight = tl.where(cell < N, tl.exp(logits - next_max), 0.0)
            past_v = tl.load(
                VALUES + cell[:, None] * VS0 + kh * VS1 + d[None, :] * VS2,
                past_mask, other=0.0,
            )
            value = tl.where(cell[:, None] == N - 1, current_v[None, :], past_v)
            numerator = numerator * alpha + tl.sum(weight[:, None] * value, axis=0)
            denominator = denominator * alpha + tl.sum(weight, axis=0)
            maximum = next_max
        offset = (head * PARTS + part) * (D + 2)
        tl.store(S + offset + d, numerator, d < D)
        tl.store(S + offset + D, maximum)
        tl.store(S + offset + D + 1, denominator)

    @triton.jit
    def _merge_kernel(
        S, QG, OUT,
        D: tl.constexpr, PARTS: tl.constexpr,
        GS0: tl.constexpr, GS1: tl.constexpr,
        BD: tl.constexpr, BP: tl.constexpr,
    ):
        head = tl.program_id(0).to(tl.int64)
        part = tl.arange(0, BP)
        d = tl.arange(0, BD).to(tl.int64)
        offset = (head * PARTS + part) * (D + 2)
        maximum = tl.load(S + offset + D, part < PARTS, other=-float("inf"))
        denominator = tl.load(S + offset + D + 1, part < PARTS, other=0.0)
        weight = tl.exp(maximum - tl.max(maximum, axis=0))
        numerator = tl.load(
            S + offset[:, None] + d[None, :],
            (part[:, None] < PARTS) & (d[None, :] < D), other=0.0,
        )
        total = tl.sum(denominator * weight, axis=0)
        context = tl.sum(numerator * weight[:, None], axis=0) / total
        gate = tl.load(QG + head * GS0 + (D + d) * GS1, d < D, other=0.0)
        z = tl.exp(-tl.abs(gate))
        sigmoid = tl.where(gate >= 0.0, 1.0 / (1.0 + z), z / (1.0 + z))
        tl.store(OUT + head * D + d, context * sigmoid, d < D)


def _validate(cp, q, qg, k, v, keys, values, n, scale):
    arrays = (q, qg, k, v, keys, values)
    if any(not isinstance(a, cp.ndarray) for a in arrays):
        raise TypeError("QSA cooperative operands must be CuPy arrays")
    if any(a.dtype != np.dtype("float32") for a in arrays):
        raise TypeError("QSA cooperative operands must all be FP32")
    if any(a.device.id != q.device.id for a in arrays):
        raise ValueError("QSA cooperative operands must be on the same device")
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError("q and k must have shape (heads, dim)")
    heads, dim = q.shape
    kv_heads = k.shape[0]
    if (heads < 1 or dim < 1 or kv_heads < 1 or heads % kv_heads
            or k.shape[1] != dim or v.shape != k.shape
            or qg.shape != (heads, 2 * dim)):
        raise ValueError("invalid QSA query/gate/current KV shapes or grouped heads")
    if (keys.ndim != 3 or values.ndim != 3
            or keys.shape[1:] != k.shape or values.shape[1:] != v.shape):
        raise ValueError("cache shapes must be (capacity, kv_heads, dim)")
    if isinstance(n, (bool, np.bool_)):
        raise TypeError("n must be an integer")
    n = operator.index(n)
    if not 1 <= n <= MAX_CONTEXT:
        raise ValueError(f"n must be in [1, {MAX_CONTEXT}]")
    # A capacity of n-1 (including zero for n=1) is sufficient: no n-1 read.
    if min(keys.shape[0], values.shape[0]) < n - 1:
        raise ValueError("cache capacity must cover n-1 committed tokens")
    if not isinstance(scale, (int, float, np.integer, np.floating)):
        raise TypeError("scale must be a host scalar")
    if not np.isfinite(scale) or abs(scale) > float(np.finfo(np.float32).max):
        raise ValueError("scale must be finite and representable in FP32")
    return n, float(np.float32(scale))


def _launch(cp, kernel, grid, args, signature):
    # warmup compiles without executing; JIT __getitem__ would choose Triton's
    # driver stream instead of CuPy's. Cache compiled kernels, never scratch.
    key = (cp.cuda.runtime.getDevice(), kernel, signature)
    compiled = _KERNELS.get(key)
    if compiled is None:
        compiled = kernel.warmup(
            *args, grid=grid, num_warps=4, num_stages=1, enable_fp_fusion=False,
        )
        _KERNELS[key] = compiled
    compiled[(*grid, 1)](*args, stream=int(cp.cuda.get_current_stream().ptr))


def qsa_cooperative_gpu(q, qg, k, v, keys, values, n, scale):
    """Return flattened FP32 gated context, with qsa_decode_gpu's argument order.

    All operands must be same-device CuPy FP32 arrays. Arbitrary element strides
    (including negative/zero strides) are honored. n is a host integer in
    [1,8192]; scale is rounded to FP32 as in the serial operator. Inputs and
    finite attention arithmetic are the caller's responsibility (no data scans).
    The q device's current CuPy stream is used; the caller's device and stream
    selections are preserved. Concurrent calls own separate bounded scratch.
    Calling this function explicitly is the only opt-in; no dispatch is changed.
    """
    import cupy as cp

    n, scale = _validate(cp, q, qg, k, v, keys, values, n, scale)
    if triton is None:
        raise RuntimeError("qsa_cooperative_gpu requires the optional Triton dependency")
    heads, dim = q.shape
    parts = (n + PART_TOKENS - 1) // PART_TOKENS
    bd = triton.next_power_of_2(dim)
    strides = tuple(int(s // a.itemsize) for a in (q, k, v, keys, values) for s in a.strides)
    gs = tuple(int(s // qg.itemsize) for s in qg.strides)
    geometry = (dim, heads // k.shape[0], parts, *strides, bd)
    merge_geometry = (dim, parts, *gs, bd, triton.next_power_of_2(parts))
    with cp.cuda.Device(q.device.id):
        scratch = cp.empty((heads, parts, dim + 2), dtype=cp.float32)
        output = cp.empty(heads * dim, dtype=cp.float32)
        _launch(cp, _parts_kernel, (heads, parts),
                (*map(_Pointer, (q, k, v, keys, values, scratch)), n, scale, *geometry),
                geometry)
        _launch(cp, _merge_kernel, (heads, 1),
                (*map(_Pointer, (scratch, qg, output)), *merge_geometry), merge_geometry)
    return output
