"""Standalone exact FP32 D=256 QSA: warp-parallel dots, serial finalizer.

Four warps per CTA coalesce dimension loads; shuffles feed products back in
strict d=0..255 order, not a reduction tree. The second kernel preserves the
original 128-cell max/softmax/V/sigmoid source. No model integration or cache
writes. At most 24*8192 FP32 logits (768 KiB) plus the flattened output.
"""

import operator

import numpy as np

from .qsa_decode_gpu import _SOURCE as _SERIAL_SOURCE


_SERIAL_DOT = '''        if (cell < n) {
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float key = cell == n - 1 ? k[kh * k0 + d * k1]
                    : keys[cell * ks0 + kh * ks1 + d * ks2];
                dot += q[h * qs0 + d * qs1] * key;
            }
            logit = dot * scale;
        }'''
_LOGITS_DOT = '        if (cell < n) logit = logits[h * logit_stride + cell];'
_SIGNATURE = 'float scale,'
_LOGITS_SIGNATURE = 'float scale, const float* logits, int logit_stride,'
_ANCHORS = (
    _SERIAL_DOT, _SIGNATURE,
    '    const int h = blockIdx.x, kh = h / groups, lane = threadIdx.x;',
    '    __shared__ float reduction[128], weights[128];',
    '    for (int base = 0; base < n; base += 128) {',
    '        const int cell = base + lane, count = min(128, n - base);',
    '        float logit = -INFINITY;',
    '        reduction[lane] = logit;',
)


def warp_dots_source(source):
    """Adapt serial or dynamic-n source without importing the graph module.

    Only the scale-adjacent signature and original dot block are replaced.
    Missing/duplicate anchors and reapplication fail closed. For dynamic n,
    the parent must populate logits for that same n and prove n <= stride,
    stride <= 8192, heads <= 24, dim=256, and a 128-thread launch.
    """
    if not isinstance(source, str):
        raise TypeError('QSA source must be a string')
    if 'logits' in source or any(source.count(anchor) != 1 for anchor in _ANCHORS):
        raise ValueError('Serial QSA source contract changed; review warp dots adapter')
    return source.replace(_SIGNATURE, _LOGITS_SIGNATURE, 1).replace(_SERIAL_DOT, _LOGITS_DOT, 1)


_VALUE_LOOP = 'for (int d = lane; d < dim; d += 128)'
_TILED_VALUE_LOOP = ('for (int d = (int)blockIdx.y * {tile} + lane; '
                     'd < min(dim, ((int)blockIdx.y + 1) * {tile}); d += 128)')


def _validate_value_tile(value_tile):
    if isinstance(value_tile, (bool, np.bool_)):
        raise TypeError('value_tile must be an integer')
    value_tile = operator.index(value_tile)
    if value_tile not in (0, 32, 64, 128):
        raise ValueError('value_tile must be 0, 32, 64, or 128')
    return value_tile


def finalizer_source(source, *, value_tile=0):
    """Preserve the default finalizer; optionally partition only its three d loops.

    All 128 threads still compute each softmax tile. Only output initialization,
    ordered weighted-V accumulation and gating have disjoint dimension ranges.
    The parent must launch (heads, ceil(dim/value_tile)) for nonzero tiles.
    """
    value_tile = _validate_value_tile(value_tile)
    generated = warp_dots_source(source)
    if generated.count(_VALUE_LOOP) != 3:
        raise ValueError('Serial QSA source contract changed; require exactly three value loops')
    if value_tile:
        generated = generated.replace(_VALUE_LOOP, _TILED_VALUE_LOOP.format(tile=value_tile))
    return generated


_SOURCE = finalizer_source(_SERIAL_SOURCE)
_DOTS_SOURCE = r'''
extern "C" __global__ void qsa_warp_dots(
    const float* q, const float* k, const float* keys, float* logits,
    int n, int groups, float scale, int logit_stride,
    long long qs0, long long qs1, long long k0, long long k1,
    long long ks0, long long ks1, long long ks2) {
    const int h = blockIdx.x, kh = h / groups;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int cell = blockIdx.y * 4 + warp;
    // This exit is warp-uniform; every lane of each active warp shuffles.
    if (cell >= n) return;
    const unsigned int fullmask = 0xffffffffu;
    float dot = 0.0f;
    for (int baseD = 0; baseD < 256; baseD += 32) {
        const int d = baseD + lane;
        const float key = cell == n - 1 ? k[kh * k0 + d * k1]
            : keys[cell * ks0 + kh * ks1 + d * ks2];
        const float product = q[h * qs0 + d * qs1] * key;
        for (int j = 0; j < 32; ++j) {
            const float ordered_product = __shfl_sync(fullmask, product, j);
            dot += ordered_product;
        }
    }
    if (lane == 0) logits[h * logit_stride + cell] = dot * scale;
}
'''
_OPTIONS = ('--std=c++14', '--fmad=false')
_MAX_N = 8192
_MAX_HEADS = 24
_KERNELS = {}


def _validate(cp, q, qg, k, v, keys, values, n, scale):
    arrays = (q, qg, k, v, keys, values)
    if any(not isinstance(a, cp.ndarray) for a in arrays):
        raise TypeError('QSA warp dots operands must be CuPy arrays')
    if any(a.dtype != np.dtype('float32') for a in arrays):
        raise TypeError('QSA warp dots operands must all be FP32')
    if any(a.device.id != q.device.id for a in arrays):
        raise ValueError('QSA warp dots operands must be on the same device')
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError('q and k must have shape (heads, dim)')
    heads, dim = q.shape
    if dim != 256:
        raise ValueError('QSA warp dots supports only dim=256')
    kv = k.shape[0]
    if (not 1 <= heads <= _MAX_HEADS or kv < 1 or heads % kv
            or k.shape[1] != dim or v.shape != k.shape or qg.shape != (heads, 2 * dim)):
        raise ValueError('invalid QSA shapes or grouped heads; require 1..24 query heads')
    if (keys.ndim != 3 or values.ndim != 3
            or keys.shape[1:] != k.shape or values.shape[1:] != v.shape):
        raise ValueError('cache shapes must be (capacity, kv_heads, dim)')
    if any(s % a.itemsize for a in arrays for s in a.strides):
        raise ValueError('QSA warp dots strides must be whole FP32 elements')
    if isinstance(n, (bool, np.bool_)):
        raise TypeError('n must be an integer')
    n = operator.index(n)
    if not 1 <= n <= _MAX_N:
        raise ValueError('n must be in [1, 8192]')
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


def qsa_warp_dots_gpu(q, qg, k, v, keys, values, n, scale, *, value_tile=0):
    """Return flattened FP32 gated context, using exactly two kernel launches.

    Same-device FP32 operands may have arbitrary whole-element strides,
    including negative/zero strides. Only cache[:n-1] and current k/v are read.
    Metadata validation requires D=256, 1..24 heads and 1..8192 cells. No host
    synchronization, data scans, fallback dispatch, or input mutation. Uses
    the query device's current stream and restores the caller's device.
    value_tile=0 retains the original finalizer; 32/64/128 redundantly compute
    softmax per dimension tile without changing any dimension's FP32 V order.
    """
    value_tile = _validate_value_tile(value_tile)
    import cupy as cp

    n, scale = _validate(cp, q, qg, k, v, keys, values, n, scale)
    heads, dim = q.shape
    strides = tuple(np.int64(s // a.itemsize)
                    for a in (q, qg, k, v, keys, values) for s in a.strides)
    qs0, qs1, _, _, k0, k1, _, _, ks0, ks1, ks2, _, _, _ = strides
    groups = np.int32(heads // k.shape[0])
    device = q.device.id
    cache_key = (device, value_tile)
    with cp.cuda.Device(device):
        if cache_key not in _KERNELS:
            _KERNELS[cache_key] = (
                cp.RawKernel(_DOTS_SOURCE, 'qsa_warp_dots', options=_OPTIONS),
                cp.RawKernel(_SOURCE if value_tile == 0 else
                             finalizer_source(_SERIAL_SOURCE, value_tile=value_tile),
                             'qsa_decode', options=_OPTIONS),
            )
        dots, finalize = _KERNELS[cache_key]
        logits = cp.empty(heads * n, dtype=cp.float32)
        output = cp.empty(heads * dim, dtype=cp.float32)
        dots((heads, (n + 3) // 4), (128,),
             (q, k, keys, logits, np.int32(n), groups, scale, np.int32(n),
              qs0, qs1, k0, k1, ks0, ks1, ks2))
        final_grid = (heads,) if value_tile == 0 else (heads, (dim + value_tile - 1) // value_tile)
        finalize(final_grid, (128,),
                 (q, qg, k, v, keys, values, output, np.int32(n), np.int32(dim),
                  groups, scale, logits, np.int32(n), *strides))
    return output
