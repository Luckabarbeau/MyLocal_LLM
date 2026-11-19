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


def finite_status_gpu(*arrays, out=None):
    """Enqueue complete finite validation; return device uint32 (0 means valid).

    No host read. Consumers must check the status before committing state or
    exposing a successful request. The synchronous wrapper preserves its API.
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
    if out is None:
        status = xp.zeros((), dtype=xp.uint32)
    else:
        if (not isinstance(out,xp.ndarray) or out.shape != () or out.dtype != xp.uint32 or
                out.device.id != device or not out.flags.c_contiguous):
            raise ValueError('Finite status must be a current-device uint32 scalar')
        status = out
        status.fill(0)
    args = tuple(item for a in contiguous for item in (a, np.int64(a.size))) + (status,)
    blocks = max(1, min(32, (max(a.size for a in contiguous) + 255) // 256))
    _kernel(device, len(arrays))((blocks,), (256,), args)
    return status


def all_finite_gpu(*arrays):
    """Return a host bool after one complete current-stream validation/readback."""
    return int(finite_status_gpu(*arrays).item()) == 0
