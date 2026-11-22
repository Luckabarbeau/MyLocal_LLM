"""Whole-chunk HC reads with the native singleton reduction order unchanged."""
from functools import lru_cache
import os

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp
from . import hyperconnection_read_gpu as native
from .hyperconnection_decode_gpu import _NORM, _OPTIONS


@lru_cache(maxsize=16)
def _kernels(device, bf16, parallel):
    if (_NORM.count('int c = blockIdx.x, lane = threadIdx.x;') != 1 or
            native._SOURCE.count('int wide, int rank) {') != 1 or
            native._SOURCE.count('float* mixed, int d, int rank) {') != 1):
        raise RuntimeError('Native HC source contract changed; review chunk adapter')
    norm = _NORM.replace('int c = blockIdx.x, lane = threadIdx.x;',
        'h += (long long)blockIdx.y*4*d; out += (long long)blockIdx.y*4*d;\n'
        '    int c = blockIdx.x, lane = threadIdx.x;')
    source = native._SOURCE.replace('int wide, int rank) {',
        'int wide, int rank) {\n'
        '    n += (long long)blockIdx.y*wide; low += (long long)blockIdx.y*rank;\n'
        '    gains += (long long)blockIdx.y*4;')
    source = source.replace('float* mixed, int d, int rank) {',
        'float* mixed, int d, int rank) {\n'
        '    n += (long long)blockIdx.y*4*d; low += (long long)blockIdx.y*rank;\n'
        '    mixed += (long long)blockIdx.y*d;')
    prefix = ('#define HC_PARALLEL\n' if parallel else '')
    prefix += '#define HC_W ' + ('unsigned short' if bf16 else 'float') + '\n'
    with xp.cuda.Device(device):
        module = xp.RawModule(code=prefix+norm+source, options=_OPTIONS)
        return tuple(module.get_function(name) for name in
                     ('hc_norm', 'hc_read_down_inject', 'hc_read_up_mix'))


def read_chunk(hc, streams):
    """Return batched (hidden, gains); decline unsupported metadata before work."""
    if BACKEND_NAME != 'cupy' or streams.ndim < 3 or not streams.flags.c_contiguous:
        return None
    rows = streams.size // (4*hc.width)
    if not 1 <= rows <= 65535:
        return None
    values = streams.reshape(rows, 4, hc.width)
    d, rank, epsilon, device = native._validate(values[0], hc.gamma, hc.down,
                                              hc.up, hc.inject, hc.epsilon)
    parallel = os.environ.get('MINI_LLM_HC_PARALLEL_READ', '0') == '1'
    norm, project, mix = _kernels(device, hc.down.dtype == xp.uint16, parallel)
    normalized = xp.empty((rows, 4*d), xp.float32)
    low = xp.empty((rows, rank), xp.float32)
    hidden = xp.empty(streams.shape[:-2] + (d,), xp.float32)
    gains = None if hc.inject is None else xp.empty(streams.shape[:-2] + (4,), xp.float32)
    norm((4, rows), (256,), (values, hc.gamma, normalized, np.int32(d), epsilon))
    project((rank + (4 if gains is not None else 0), rows), (256,),
        (normalized, hc.down, hc.down if gains is None else hc.inject, low,
         low if gains is None else gains, np.int32(4*d), np.int32(rank)))
    features = 2 if parallel else 8
    mix(((d+features-1)//features, rows), (256,),
        (normalized, low, hc.up, hidden, np.int32(d), np.int32(rank)))
    return hidden, gains
