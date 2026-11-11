"""Library-level inference matmul with graph-capturable FP32 vector products.

Public contract is NumPy-style ``a @ b``, including batch broadcasting and an
optional caller-owned output. Single-row CuPy FP32 operands use a Triton GEMV;
other cases retain backend matmul. No weight conversion, reduced-precision dot,
training changes, architecture names, host value reads or synchronization.
"""

import math

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.quantized_linear import _TritonCuPyPointer, triton, tl


if triton is not None:
    @triton.jit
    def _vector_matmul(A, B, OUT, K: tl.constexpr, N: tl.constexpr,
                       BATCH: tl.constexpr, AS: tl.constexpr, BS: tl.constexpr,
                       AK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
                       TILE: tl.constexpr):
        row = tl.program_id(0)
        batch = tl.program_id(1)
        abase = tl.full((), 0, tl.int64)
        bbase = tl.full((), 0, tl.int64)
        remainder = batch
        for axis in tl.static_range(len(BATCH) - 1, -1, -1):
            index = remainder % BATCH[axis]
            remainder = remainder // BATCH[axis]
            abase += index * AS[axis]
            bbase += index * BS[axis]
        lane = tl.arange(0, TILE)
        acc = tl.full((TILE,), 0, tl.float32)
        for tile in range(tl.cdiv(K, TILE)):
            col = tile * TILE + lane
            a = tl.load(A + abase + col * AK, col < K, 0)
            b = tl.load(B + bbase + col * BK + row * BN, col < K, 0)
            acc = acc + a * b
        tl.store(OUT + batch * N + row, tl.sum(acc, axis=0))


def _geometry(a, b):
    if a.ndim < 2 or b.ndim < 2 or a.shape[-1] != b.shape[-2]:
        return None
    try:
        batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    except ValueError:
        return None
    return batch, (*batch, a.shape[-2], b.shape[-1])


def _batch_strides(array, batch):
    shape = (1,) * (len(batch) - array.ndim + 2) + array.shape[:-2]
    strides = (0,) * (len(batch) - array.ndim + 2) + tuple(
        int(s // array.itemsize) for s in array.strides[:-2])
    return tuple(0 if dim == 1 else stride for dim, stride in zip(shape, strides))


def supports_inference_gemv(a, b):
    """Metadata-only check for the single-row FP32 Triton implementation."""
    if (BACKEND_NAME != 'cupy' or triton is None or
            not isinstance(a, xp.ndarray) or not isinstance(b, xp.ndarray) or
            a.dtype != xp.float32 or b.dtype != xp.float32):
        return False
    geometry = _geometry(a, b)
    if geometry is None or a.shape[-2] != 1 or min(a.shape[-1], b.shape[-1]) < 1:
        return False
    batch, _ = geometry
    device = xp.cuda.runtime.getDevice()
    return (a.device.id == b.device.id == device and
            1 <= math.prod(batch) <= 65535 and b.shape[-1] < 2**31)


def matmul_inference(a, b, *, out=None):
    """Compute ``a @ b``; return fresh storage unless ``out`` is supplied.

    Triton supports arbitrary operand strides and broadcast batch dimensions.
    FP32 products/reductions use no TF32/FP16 approximation. Output is contiguous;
    aliased or noncontiguous caller outputs use backend matmul's established
    semantics. During graph capture, warm the exact geometry/strides beforehand,
    as with the existing library Triton operators.
    """
    if not supports_inference_gemv(a, b):
        return xp.matmul(a, b, out=out)
    batch, shape = _geometry(a, b)
    if out is not None:
        if (not isinstance(out, xp.ndarray) or out.shape != shape or out.dtype != xp.float32
                or out.device.id != a.device.id):
            raise ValueError('out must match the FP32 matmul shape/device')
        if not out.flags.c_contiguous or xp.may_share_memory(out, a) or xp.may_share_memory(out, b):
            return xp.matmul(a, b, out=out)
    else:
        out = xp.empty(shape, dtype=xp.float32)
    args = (_TritonCuPyPointer(a, tl.float32), _TritonCuPyPointer(b, tl.float32),
            _TritonCuPyPointer(out, tl.float32), a.shape[-1], b.shape[-1], tuple(batch),
            _batch_strides(a, batch), _batch_strides(b, batch),
            int(a.strides[-1] // a.itemsize), int(b.strides[-2] // b.itemsize),
            int(b.strides[-1] // b.itemsize), 512)
    grid = (b.shape[-1], math.prod(batch), 1)
    compiled = _vector_matmul.warmup(*args, grid=grid, num_warps=4, enable_fp_fusion=False)
    compiled[grid](*args, stream=int(xp.cuda.get_current_stream().ptr))
    return out
