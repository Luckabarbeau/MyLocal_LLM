"""Runtime-position adapters for the unchanged serial FP32 QSA graph.

The block adapter proves 0 <= position < min(capacity, all-cell width, INT_MAX).
Device positions are private operands, never accepted from an unchecked caller.
No kernel here performs a host read or changes the serial attention arithmetic.
"""
from functools import lru_cache
from numbers import Integral
import os

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp
from .qsa_decode_gpu import _SOURCE as _SERIAL_SOURCE


def _runtime_source(source):
    replacements = (
        ('void qsa_decode(', 'void qsa_decode_graph('),
        ('    int n, int dim, int groups, float scale,',
         '    const long long* position, int dim, int groups, float scale,'),
        ('    const int h = blockIdx.x, kh = h / groups, lane = threadIdx.x;',
         '    const int n = (int)(position[0] + 1);\n'
         '    const int h = blockIdx.x, kh = h / groups, lane = threadIdx.x;'),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Serial QSA source contract changed; review runtime adapter')
        source = source.replace(old, new, 1)
    return source


_SOURCE = _runtime_source(_SERIAL_SOURCE)
_AUX_SOURCE = r'''
extern "C" __global__ void set_qsa_position(long long* position, long long t) {
    position[0] = t;
}
extern "C" __global__ void pool_qsa_graph(const float* history, const float* raw,
    const long long* position, float* out, int rd, int block_size, float inv_block) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rd) return;
    long long t = position[0], n = t + 1;
    float sum = 0.0f;
    if (n % block_size == 0) {
        long long start = n - block_size;
        sum = block_size == 1 ? raw[i] : history[start * rd + i];
        for (long long j = start + 1; j < t; ++j) sum += history[j * rd + i];
        if (block_size != 1) sum += raw[i];
        sum *= inv_block;
    }
    out[i] = sum;
}
extern "C" __global__ void commit_qsa_graph(const unsigned int* bad,
    const float* k, const float* v, const float* raw, const float* pool,
    float* keys, float* values, float* history, float* pooled,
    const long long* position, int block_size, int kd, int rd) {
    if (*bad) return;
    long long t = position[0], n = t + 1, block = n / block_size - 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kd) { keys[t * kd + i] = k[i]; values[t * kd + i] = v[i]; }
    if (i < rd) {
        history[t * rd + i] = raw[i];
        if (n % block_size == 0) pooled[block * rd + i] = pool[i];
    }
}
'''


@lru_cache(maxsize=64)
def _kernel(device, name):
    with xp.cuda.Device(device):
        return xp.RawKernel(_SOURCE if name == 'qsa_decode_graph' else _AUX_SOURCE,
                            name, options=('--std=c++14', '--fmad=false'))


def _array(name, a, dtype, shape=None, *, contiguous=True):
    if BACKEND_NAME != 'cupy':
        raise RuntimeError('QSA graph kernels require CuPy')
    if not isinstance(a, xp.ndarray) or a.dtype != dtype:
        raise TypeError(f'{name} must be a CuPy {np.dtype(dtype).name} array')
    if (a.device.id != xp.cuda.runtime.getDevice() or
            (shape is not None and a.shape != shape) or
            (contiguous and not a.flags.c_contiguous)):
        raise ValueError(f'{name} shape/device/contiguous layout mismatch')
    return a


def _position(position):
    return _array('position', position, xp.int64, ())


def _block_size(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 1 <= value <= np.iinfo(np.int32).max:
        raise ValueError('block_size must be a positive int32 integer')
    return int(value)


def set_position_gpu(position, t, max_position):
    """Assign one checked host scalar using one tiny current-stream kernel."""
    _position(position)
    if (isinstance(t, (bool, np.bool_)) or not isinstance(t, Integral) or
            not 0 <= t <= max_position < np.iinfo(np.int32).max):
        raise ValueError('position must satisfy the runtime capacity/all-cell bound')
    _kernel(position.device.id, 'set_qsa_position')((1,), (1,), (position, np.int64(t)))


def qsa_decode_graph_gpu(q, qg, k, v, keys, values, position, scale):
    """Serial decode with n=position[0]+1; caller guarantees the runtime bound."""
    _position(position)
    for name, a in zip(('q', 'qg', 'k', 'v', 'keys', 'values'), (q, qg, k, v, keys, values)):
        _array(name, a, xp.float32, contiguous=False)
    if q.ndim != 2 or k.ndim != 2 or keys.ndim != 3:
        raise ValueError('QSA graph projection/cache ranks mismatch')
    heads, dim = q.shape
    kv = k.shape[0]
    if (min(heads, dim, kv) <= 0 or heads % kv or k.shape[1] != dim or
            qg.shape != (heads, 2*dim) or v.shape != k.shape or
            keys.shape[1:] != k.shape or values.shape != keys.shape or keys.shape[0] < 1):
        raise ValueError('QSA graph projection/cache shapes mismatch')
    if not np.isfinite(scale) or not 0 < scale <= np.finfo(np.float32).max:
        raise ValueError('scale must be positive finite FP32')
    if (os.environ.get('MINI_LLM_QSA_WARP_DOTS', '0') == '1' and
            dim == 256 and heads <= 24 and keys.shape[0] <= 8192):
        return _warp_decode_graph_gpu(q, qg, k, v, keys, values, position, scale)
    output = xp.empty(heads*dim, dtype=xp.float32)
    strides = tuple(np.int64(s // a.itemsize)
                    for a in (q, qg, k, v, keys, values) for s in a.strides)
    _kernel(q.device.id, 'qsa_decode_graph')((heads,), (128,),
        (q, qg, k, v, keys, values, output, position, np.int32(dim),
         np.int32(heads // kv), np.float32(scale), *strides))
    return output


@lru_cache(maxsize=16)
def _warp_graph_kernels(device):
    from .qsa_warp_dots_gpu import _DOTS_SOURCE, finalizer_source
    signature = '    int n, int groups, float scale, int logit_stride,'
    anchor = '    const int h = blockIdx.x, kh = h / groups;'
    if _DOTS_SOURCE.count(signature) != 1 or _DOTS_SOURCE.count(anchor) != 1:
        raise ValueError('Warp-dot source changed; review runtime-position adapter')
    dots = _DOTS_SOURCE.replace(signature,
        '    const long long* position, int groups, float scale, int logit_stride,', 1)
    dots = dots.replace(anchor, '    const int n = (int)(position[0] + 1);\n'+anchor, 1)
    with xp.cuda.Device(device):
        options = ('--std=c++14', '--fmad=false')
        return (xp.RawKernel(dots, 'qsa_warp_dots', options=options),
                xp.RawKernel(finalizer_source(_SOURCE, value_tile=32), 'qsa_decode_graph', options=options))


def _warp_decode_graph_gpu(q, qg, k, v, keys, values, position, scale):
    # Fixed capacity owns bounded scratch; kernels read only runtime n valid rows.
    heads, dim = q.shape
    capacity = keys.shape[0]
    logits = xp.empty(heads*capacity, dtype=xp.float32)
    output = xp.empty(heads*dim, dtype=xp.float32)
    strides = tuple(np.int64(s // a.itemsize)
                    for a in (q, qg, k, v, keys, values) for s in a.strides)
    qs0, qs1, _, _, k0, k1, _, _, ks0, ks1, ks2, _, _, _ = strides
    groups = np.int32(heads // k.shape[0])
    dots, finalize = _warp_graph_kernels(q.device.id)
    dots((heads, (capacity+3)//4), (128,),
        (q, k, keys, logits, position, groups, np.float32(scale), np.int32(capacity),
         qs0, qs1, k0, k1, ks0, ks1, ks2))
    finalize((heads, 8), (128,), (q, qg, k, v, keys, values, output, position,
        np.int32(dim), groups, np.float32(scale), logits, np.int32(capacity), *strides))
    return output


def pool_qsa_graph_gpu(history, raw, position, block_size):
    _position(position)
    block_size = _block_size(block_size)
    _array('raw', raw, xp.float32)
    if raw.ndim != 1 or not raw.size:
        raise ValueError('raw must be a nonempty vector')
    _array('history', history, xp.float32)
    if history.ndim != 2 or history.shape[1] != raw.size or history.shape[0] < 1:
        raise ValueError('history shape mismatch')
    out = xp.empty_like(raw)
    _kernel(raw.device.id, 'pool_qsa_graph')(((raw.size+255)//256,), (256,),
        (history, raw, position, out, np.int32(raw.size), np.int32(block_size),
         np.float32(1 / block_size)))
    return out


def commit_qsa_graph_gpu(status, keys, values, history, pooled, k, v, raw, pool,
                         position, block_size):
    _position(position)
    block_size = _block_size(block_size)
    _array('status', status, xp.uint32, ())
    for name, a in zip(('keys', 'values', 'history', 'pooled', 'k', 'v', 'raw', 'pool'),
                       (keys, values, history, pooled, k, v, raw, pool)):
        _array(name, a, xp.float32)
    if (keys.ndim != 3 or k.ndim != 2 or raw.ndim != 1 or not raw.size or not k.size or
            keys.shape[1:] != k.shape or values.shape != keys.shape or v.shape != k.shape or
            history.shape != (keys.shape[0], raw.size) or pool.shape != raw.shape or
            pooled.shape != (keys.shape[0] // block_size + 1, raw.size)):
        raise ValueError('QSA graph commit shapes mismatch')
    kd, rd = k.size, raw.size
    _kernel(k.device.id, 'commit_qsa_graph')(((max(kd, rd)+255)//256,), (256,),
        (status, k, v, raw, pool, keys, values, history, pooled, position,
         np.int32(block_size), np.int32(kd), np.int32(rd)))
