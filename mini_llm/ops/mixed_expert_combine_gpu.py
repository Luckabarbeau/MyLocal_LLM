"""Two-launch ordered mixed-expert combine; no packed-weight access.

GPU terms are already weighted; CPU outputs are raw FP32 rows uploaded by the
caller. Both arrays must be nonempty. Host metadata describes ascending expert
order (not inferable from these arrays). All work uses CuPy's current stream;
the caller retains input/weight ownership through stream completion.
"""
from functools import lru_cache

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


_SOURCE = r'''
extern "C" __global__ void mixed_materialize(
    const float* gpu, const float* cpu, const float* routes,
    const int* table, float* terms, int d, int count, long long stride) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)count * d) return;
    const int expert = i / d, col = i % d;
    const int pos = table[2 * expert], slot = table[2 * expert + 1];
    if (pos >= 0) terms[i] = gpu[(long long)pos * d + col];
    else terms[i] = cpu[(long long)(-pos - 1) * d + col]
                    * routes[(long long)slot * stride];
}
extern "C" __global__ void mixed_ordered_reduce(
    const float* terms, float* out, int d, int count) {
    const long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= d) return;
    float acc = 0.0f;
    for (int expert = 0; expert < count; ++expert)
        acc = __fadd_rn(acc, terms[(long long)expert * d + col]);
    out[col] = acc;
}
'''


@lru_cache(maxsize=None)
def _module(device_id):
    with cp.cuda.Device(device_id):
        return cp.RawModule(code=_SOURCE, options=("--std=c++11", "--fmad=false"))


def combine(gpu_terms, cpu_outputs, routes, positions, route_slots):
    """Return fresh FP32 [1,D], summing selected experts in supplied order.

    ``positions`` and ``route_slots`` are host integer tuples of length E<=64.
    A nonnegative position indexes a weighted GPU row; a negative position p
    indexes raw CPU row -p-1, multiplied exactly once by routes[route_slots[e]].
    Routes are FP32 [E] and may have positive, negative or zero stride. Row
    arrays are contiguous FP32 [G,D]/[C,D], with G,C,D positive. Repeated row
    indices and route slots are legal. Validation only inspects host metadata;
    the sole H2D transfer is an E-by-2 int32 table (at most 512 bytes).
    """
    if cp is None:
        raise ValueError("mixed combine requires CuPy")
    for name, value in (("gpu_terms", gpu_terms), ("cpu_outputs", cpu_outputs)):
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float32
                or value.ndim != 2 or min(value.shape) <= 0
                or not value.flags.c_contiguous):
            raise ValueError(f"{name} must be nonempty contiguous CuPy FP32 [rows,D]")
    if gpu_terms.shape[1] != cpu_outputs.shape[1]:
        raise ValueError("GPU and CPU row widths must match")
    if not isinstance(positions, tuple) or not 1 <= len(positions) <= 64:
        raise ValueError("positions must be a host tuple of length 1..64")
    e, d = len(positions), gpu_terms.shape[1]
    integer = lambda v: isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
    if any(not integer(p) or not -cpu_outputs.shape[0] <= p < gpu_terms.shape[0]
           for p in positions):
        raise ValueError("positions must contain valid signed GPU/CPU row indices")
    if (not isinstance(route_slots, tuple) or len(route_slots) != e
            or any(not integer(s) or not 0 <= s < e for s in route_slots)):
        raise ValueError("route_slots must be a host tuple of E indices into routes")
    if (not isinstance(routes, cp.ndarray) or routes.dtype != cp.float32
            or routes.shape != (e,)):
        raise ValueError("routes must be a CuPy FP32 [E] array")
    device = cp.cuda.runtime.getDevice()
    if any(a.device.id != device for a in (gpu_terms, cpu_outputs, routes)):
        raise ValueError("all operands must be on the current CUDA device")
    if max(e * d, gpu_terms.shape[0], cpu_outputs.shape[0]) > 2147483647:
        raise ValueError("mixed combine dimensions exceed CUDA launch/index limits")
    table = cp.asarray(np.asarray(tuple(zip(positions, route_slots)), dtype=np.int32))
    terms = cp.empty((e, d), cp.float32)
    out = cp.empty((1, d), cp.float32)
    module = _module(device)
    module.get_function("mixed_materialize")(
        ((e * d + 255) // 256,), (256,),
        (gpu_terms, cpu_outputs, routes, table, terms, np.int32(d), np.int32(e),
         np.int64(routes.strides[0] // routes.itemsize)))
    module.get_function("mixed_ordered_reduce")(
        ((d + 255) // 256,), (256,), (terms, out, np.int32(d), np.int32(e)))
    return out
