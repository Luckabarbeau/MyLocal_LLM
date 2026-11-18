"""Batched FP32 finite validation without per-array device-to-host fences."""
from functools import lru_cache

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp


@lru_cache(maxsize=16)
def _kernel(device, count):
    parameters = ', '.join(f'const float* a{i}, long long n{i}' for i in range(count))
    checks = '\n'.join(
        f'for (long long j = tid; j < n{i}; j += stride) bad |= !isfinite(a{i}[j]);'
        for i in range(count))
    source = f'''
    extern "C" __global__ void finite_arrays({parameters}, unsigned int* status) {{
        const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        const long long stride = (long long)gridDim.x * blockDim.x;
        unsigned int bad = 0;
        {checks}
        if (__any_sync(0xffffffffu, bad) && (threadIdx.x % 32 == 0))
            atomicOr(status, 1u);
    }}
    '''
    with xp.cuda.Device(device):
        return xp.RawKernel(source, 'finite_arrays', options=('--std=c++11',))


def all_finite_gpu(*arrays):
    """Return a host bool after one current-stream validation/readback.

    Accepts 1..8 FP32 arrays on the current CuPy device. Strided views are
    copied to bounded contiguous validation scratch, not modified. All checks
    finish before returning; this routine is deliberately not graph-capturable.
    """
    if BACKEND_NAME != 'cupy':
        raise RuntimeError('Batched GPU validation requires CuPy')
    if not 1 <= len(arrays) <= 8:
        raise ValueError('Expected 1..8 arrays')
    device = xp.cuda.runtime.getDevice()
    if any(not isinstance(a, xp.ndarray) or a.dtype != xp.float32 for a in arrays):
        raise TypeError('Finite validation requires CuPy FP32 arrays')
    if any(a.device.id != device for a in arrays):
        raise ValueError('Finite validation requires the current device')
    contiguous = [xp.ascontiguousarray(a) for a in arrays]
    status = xp.zeros((), dtype=xp.uint32)
    args = tuple(item for a in contiguous for item in (a, np.int64(a.size))) + (status,)
    blocks = max(1, min(32, (max(a.size for a in contiguous) + 255) // 256))
    _kernel(device, len(arrays))((blocks,), (256,), args)
    return int(status.item()) == 0
