"""0102M experimental fused local/global attention gate (CuPy + Triton).

The gate projects a separate learned reliability logit for each query head.
One kernel replaces sigmoid, availability mask, and interpolation in forward;
one replaces local/global gradient splitting and sigmoid-logit gradient in
backward.  The W_gate GEMMs remain in the existing projection implementation.

Both kernels operate on FP32 attention contexts; logits may be BF16 or FP32.
No change to the gated mixture's math or causal availability mask.
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
        raise RuntimeError('Triton is required for the fused attention gate')
    dtype = {
        'float32': tl.float32,
        'float16': tl.float16,
        'bfloat16': tl.bfloat16,
        'bool': tl.int1,
    }.get(str(array.dtype))
    if dtype is None:
        raise TypeError(f'unsupported fused gate dtype: {array.dtype}')
    return _Pointer(array, dtype)


if triton is not None:
    @triton.jit
    def _gate_forward(LOCAL, GLOBAL, LOGITS, AVAILABLE,
                      OUT, GATE, EFFECTIVE,
                      ROWS: tl.constexpr, HEADS: tl.constexpr,
                      D: tl.constexpr, R: tl.constexpr, BD: tl.constexpr):
        row = tl.program_id(0) * R + tl.arange(0, R)
        d = tl.arange(0, BD)
        valid = row < ROWS
        valid_vec = valid[:, None] & (d[None, :] < D)
        offset = row[:, None] * D + d[None, :]
        a = tl.load(LOCAL + offset, mask=valid_vec, other=0).to(tl.float32)
        b = tl.load(GLOBAL + offset, mask=valid_vec, other=0).to(tl.float32)
        logit = tl.load(LOGITS + row, mask=valid, other=0).to(tl.float32)
        avail = tl.load(AVAILABLE + row // HEADS, mask=valid, other=0).to(tl.int1)
        gate = 1.0 / (1.0 + tl.exp(-logit))
        effective = tl.where(avail, gate, 0.0)
        mixed = a + effective[:, None] * (b - a)
        tl.store(OUT + offset, mixed, mask=valid_vec)
        tl.store(GATE + row, gate, mask=valid)
        tl.store(EFFECTIVE + row, effective, mask=valid)

    @triton.jit
    def _gate_backward(DC, LOCAL, GLOBAL, GATE, EFFECTIVE,
                       AVAILABLE, DL, DG, DLOGIT,
                       ROWS: tl.constexpr, HEADS: tl.constexpr,
                       D: tl.constexpr, R: tl.constexpr, BD: tl.constexpr):
        row = tl.program_id(0) * R + tl.arange(0, R)
        d = tl.arange(0, BD)
        valid = row < ROWS
        valid_vec = valid[:, None] & (d[None, :] < D)
        offset = row[:, None] * D + d[None, :]
        dc = tl.load(DC + offset, mask=valid_vec, other=0).to(tl.float32)
        local = tl.load(LOCAL + offset, mask=valid_vec, other=0).to(tl.float32)
        glob = tl.load(GLOBAL + offset, mask=valid_vec, other=0).to(tl.float32)
        gate = tl.load(GATE + row, mask=valid, other=0).to(tl.float32)
        eff = tl.load(EFFECTIVE + row, mask=valid, other=0).to(tl.float32)
        avail = tl.load(AVAILABLE + row // HEADS, mask=valid, other=0).to(tl.float32)
        d_gate = tl.sum(tl.where(d[None, :] < D, dc * (glob - local), 0.0), 1)
        dlogit = d_gate * avail * gate * (1.0 - gate)
        tl.store(DL + offset, dc * (1.0 - eff[:, None]), mask=valid_vec)
        tl.store(DG + offset, dc * eff[:, None], mask=valid_vec)
        tl.store(DLOGIT + row, dlogit, mask=valid)


def _check_shapes(local, glob, logits, available):
    if (local.ndim != 4 or local.shape != glob.shape
            or logits.shape != local.shape[:3]
            or available.shape != local.shape[:2]):
        raise ValueError('gate expects contexts [B,T,H,D], logits [B,T,H], availability [B,T]')
    if str(local.dtype) != 'float32' or str(glob.dtype) != 'float32':
        raise TypeError('gate contexts must be float32')
    if str(available.dtype) != 'bool':
        raise TypeError('gate availability must be bool')
    if not all(a.flags.c_contiguous for a in (local, glob, logits, available)):
        raise ValueError('fused gate expects contiguous input arrays')


def forward(local, glob, logits, available):
    import cupy as cp
    if triton is None:
        raise RuntimeError('Triton is not installed')
    _check_shapes(local, glob, logits, available)
    b, t, h, d = map(int, local.shape)
    context = cp.empty(local.shape, dtype=cp.float32)
    gate = cp.empty(logits.shape, dtype=cp.float32)
    effective = cp.empty_like(gate)
    rows = b * t * h
    r = 16
    _gate_forward[(triton.cdiv(rows, r),)](
        *map(_ptr, (local, glob, logits, available, context, gate, effective)),
        rows, h, d, r, triton.next_power_of_2(d), num_warps=4,
    )
    return context, gate, effective


def backward(d_context, local, glob, gate, effective, available):
    import cupy as cp
    if triton is None:
        raise RuntimeError('Triton is not installed')
    _check_shapes(local, glob, gate, available)
    if (d_context.shape != local.shape or str(d_context.dtype) != 'float32'
            or effective.shape != gate.shape):
        raise ValueError('invalid fused gate backward inputs')
    if not all(a.flags.c_contiguous for a in (d_context, effective)):
        raise ValueError('fused gate backward expects contiguous gradients')
    b, t, h, d = map(int, local.shape)
    d_local = cp.empty(local.shape, dtype=cp.float32)
    d_global = cp.empty_like(d_local)
    d_logits = cp.empty(gate.shape, dtype=cp.float32)
    rows = b * t * h
    r = 16
    _gate_backward[(triton.cdiv(rows, r),)](
        *map(_ptr, (d_context, local, glob, gate, effective,
                    available, d_local, d_global, d_logits)),
        rows, h, d, r, triton.next_power_of_2(d), num_warps=4,
    )
    return d_local, d_global, d_logits
