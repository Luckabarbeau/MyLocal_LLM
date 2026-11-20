"""Standalone D=256 QSA key staging, preserving the serial FP32 operation order.

Only global key reads change: four 32-cell groups are cooperatively loaded by
all 128 threads into a padded, transposed 33-KiB shared tile. Each cell's owner
still adds dimensions 0..255 left to right, with FMA disabled. The original
128-cell softmax, value accumulation and sigmoid source is not rewritten.
No model dispatch, cache mutation, quantization, or context-sized scratch.
"""

import operator

import numpy as np

from .qsa_decode_gpu import _SOURCE as _SERIAL_SOURCE


_SHARED = '    __shared__ float reduction[128], weights[128];'
_STAGED_SHARED = _SHARED + '\n    __shared__ float staged[256 * 33];'
_SERIAL_DOT = '''        if (cell < n) {
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float key = cell == n - 1 ? k[kh * k0 + d * k1]
                    : keys[cell * ks0 + kh * ks1 + d * ks2];
                dot += q[h * qs0 + d * qs1] * key;
            }
            logit = dot * scale;
        }'''
_STAGED_DOT = '''        for (int g = 0; g < 4; ++g) {
            for (int index = lane; index < 8192; index += 128) {
                const int token = base + g * 32 + index / 256;
                const int d = index % 256;
                float key = token < n ? (token == n - 1
                    ? k[kh * k0 + d * k1]
                    : keys[token * ks0 + kh * ks1 + d * ks2]) : 0.0f;
                staged[d * 33 + index / 256] = key;
            }
            __syncthreads();
            if (lane / 32 == g && cell < n) {
                float dot = 0.0f;
                for (int d = 0; d < 256; ++d) {
                    float key = staged[d * 33 + lane % 32];
                    dot += q[h * qs0 + d * qs1] * key;
                }
                logit = dot * scale;
            }
            __syncthreads();
        }'''


def staged_source(source):
    """Transform serial or dynamic-n QSA source; retain its entry/signature.

    Exact fragment counts fail closed on missing/duplicated/previously staged
    code. Geometry anchors protect the all-thread barriers and cell ownership.
    The caller must launch exactly 128 threads and guarantee dim == 256, also
    when using this adapter for the dynamic-position entry.
    """
    if not isinstance(source, str):
        raise TypeError('QSA source must be a string')
    anchors = (
        _SHARED, _SERIAL_DOT,
        '    const int h = blockIdx.x, kh = h / groups, lane = threadIdx.x;',
        '    for (int base = 0; base < n; base += 128) {',
        '        const int cell = base + lane, count = min(128, n - base);',
        '        float logit = -INFINITY;',
        '        reduction[lane] = logit;',
    )
    if 'staged[' in source or any(source.count(fragment) != 1 for fragment in anchors):
        raise ValueError('Serial QSA source contract changed; review staged adapter')
    return source.replace(_SHARED, _STAGED_SHARED, 1).replace(_SERIAL_DOT, _STAGED_DOT, 1)


_SOURCE = staged_source(_SERIAL_SOURCE)
_OPTIONS = ('--std=c++14', '--fmad=false')
_KERNELS = {}
# The unchanged int base += 128 must not overflow on its final increment.
_MAX_N = int(np.iinfo(np.int32).max) - 127


def _validate(cp, q, qg, k, v, keys, values, n, scale):
    arrays = (q, qg, k, v, keys, values)
    if any(not isinstance(a, cp.ndarray) for a in arrays):
        raise TypeError('QSA staged operands must be CuPy arrays')
    if any(a.dtype != np.dtype('float32') for a in arrays):
        raise TypeError('QSA staged operands must all be FP32')
    if any(a.device.id != q.device.id for a in arrays):
        raise ValueError('QSA staged operands must be on the same device')
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError('q and k must have shape (heads, dim)')
    heads, dim = q.shape
    if dim != 256:
        raise ValueError('QSA staged supports only dim=256; caller must select the serial fallback')
    kv = k.shape[0]
    if (heads < 1 or heads * dim > np.iinfo(np.int32).max or kv < 1 or heads % kv
            or k.shape[1] != dim or v.shape != k.shape or qg.shape != (heads, 2 * dim)):
        raise ValueError('invalid QSA query/gate/current KV shapes or grouped heads')
    if (keys.ndim != 3 or values.ndim != 3
            or keys.shape[1:] != k.shape or values.shape[1:] != v.shape):
        raise ValueError('cache shapes must be (capacity, kv_heads, dim)')
    if any(s % a.itemsize for a in arrays for s in a.strides):
        raise ValueError('QSA staged strides must be whole FP32 elements')
    if isinstance(n, (bool, np.bool_)):
        raise TypeError('n must be an integer')
    n = operator.index(n)
    if not 1 <= n <= _MAX_N:
        raise ValueError(f'n must be in [1, {_MAX_N}]')
    if min(keys.shape[0], values.shape[0]) < n - 1:
        raise ValueError('cache capacity must cover n-1 committed tokens')
    if isinstance(scale, (bool, np.bool_)) or not isinstance(
            scale, (int, float, np.integer, np.floating)):
        raise TypeError('scale must be a host scalar')
    if not np.isfinite(scale) or not 0 < scale <= float(np.finfo(np.float32).max):
        raise ValueError('scale must be positive finite FP32')
    scale = np.float32(scale)
    if scale == 0:
        raise ValueError('scale must be positive finite FP32')
    return n, scale


def qsa_staged_gpu(q, qg, k, v, keys, values, n, scale):
    """Return only flattened FP32 context, with qsa_decode_gpu's argument order.

    All operands are same-device FP32 CuPy arrays; positive, negative and zero
    element strides are supported. Only committed cache[:n-1] and current k/v
    are read, so capacity n-1 suffices. No input data scans or host transfers.
    Uses the query device's current stream and restores the caller's device.
    Unsupported D is an error, not an implicit dispatch/integration change.
    CuPy is imported lazily (the existing backend still follows its own env).
    """
    import cupy as cp

    n, scale = _validate(cp, q, qg, k, v, keys, values, n, scale)
    heads, dim = q.shape
    strides = tuple(np.int64(s // a.itemsize)
                    for a in (q, qg, k, v, keys, values) for s in a.strides)
    device = q.device.id
    with cp.cuda.Device(device):
        if device not in _KERNELS:
            _KERNELS[device] = cp.RawKernel(_SOURCE, 'qsa_decode', options=_OPTIONS)
        output = cp.empty(heads * dim, dtype=cp.float32)
        _KERNELS[device]((heads,), (128,),
                         (q, qg, k, v, keys, values, output, np.int32(n),
                          np.int32(dim), np.int32(heads // k.shape[0]), scale, *strides))
    return output
