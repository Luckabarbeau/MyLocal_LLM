"""Repository-owned sm80+ CuPy RawModule native packed expert kernels.

Port pinned at Strata 712fec6b4297c9cfd32ab43daaa2979a9aa01edf; the
sibling CUDA file carries its MIT license. No Strata source/build/import is
used at runtime. Importing this module does not import CuPy or touch a GPU.

All arrays are contiguous CuPy arrays on the current device/current stream.
Weight pointers are absolute device addresses, int64 [total_experts, 3], in
(gate, up, down) order; callers retain the allocations through stream completion.
Only finite inputs/scales are supported. Grouping order within an expert is
unspecified, as in the source. Invalid expert IDs are ignored (slot is set to 0).
H and Dm use grouped rows; slot maps original routes to those rows. H's last
8 bytes per 64 values are unspecified, exactly as in the source GU epilogue.

No orchestration, host staging, residency, route readback, or output combine is
performed here. Buffers can be allocated once and reused; expert waves write
only their grouped row ranges. Reuse across streams requires caller events.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REVISION = '712fec6b4297c9cfd32ab43daaa2979a9aa01edf'
# (local CUDA discriminator, GGUF block values, GGUF block bytes)
FORMATS = {'IQ2_S': (2, 256, 82), 'IQ3_XXS': (3, 256, 98),
           'IQ3_S': (4, 256, 110), 'IQ4_XS': (5, 256, 136),
           'IQ4_NL': (6, 32, 18), 'Q2_0': (7, 64, 18)}
GU_FORMATS = ('IQ2_S', 'IQ3_S', 'IQ3_XXS', 'IQ4_XS')
DOWN_FORMATS = ('IQ4_NL', 'Q2_0')
_MODULES = {}
_OCCUPANCY = {}


def pick_ww(total_routes, total_experts):
    """Source default: WW2/TR128 iff 56 < ALL routes / ALL experts <= 112."""
    if total_routes < 0 or not 1 <= total_experts <= 1024:
        raise ValueError('nonnegative routes and 1..1024 experts required')
    return 2 if 56 * total_experts < total_routes <= 112 * total_experts else 4


def shared_bytes(kind, ww):
    """Source double-buffered weights + four activation stages + rows + grid."""
    if ww not in (2, 4):
        raise ValueError('WW must be 2 or 4')
    grid = {'IQ2_S': 8192, 'IQ3_XXS': 1024, 'IQ3_S': 2048}.get(kind, 0)
    return 2 * (64 * ww) * 96 + 4 * (256 // ww) * 80 + (256 // ww) * 4 + grid


def group_bytes(total_routes, total_experts):
    pick_ww(total_routes, total_experts)
    start = ((4 * total_experts + 2) * 4 + 15) // 16 * 16
    return start + ((total_routes + 63) // 64 + total_experts) * 8


@dataclass(frozen=True)
class GroupTables:
    """GPU count/scan/place storage; all metadata below are device views."""
    scratch: object
    counts: object
    offsets: object
    fill: object
    tile_offsets: object
    tiles: object
    slot: object
    src: object
    total_routes: int
    total_experts: int
    top_k: int


def _cp():
    import cupy as cp
    return cp


def _array(a, dtype, name, shape=None):
    cp = _cp()
    if not isinstance(a, cp.ndarray) or a.dtype != np.dtype(dtype):
        raise TypeError(f'{name} must be a CuPy {np.dtype(dtype).name} array')
    if not a.flags.c_contiguous or a.device.id != cp.cuda.runtime.getDevice():
        raise ValueError(f'{name} must be contiguous on the current CUDA device')
    if shape is not None and a.shape != shape:
        raise ValueError(f'{name} must have shape {shape}, got {a.shape}')
    return a


def _module(embd=256, ff=128):
    cp = _cp()
    dev = cp.cuda.runtime.getDevice()
    key = (dev, embd, ff)
    if key not in _MODULES:
        props = cp.cuda.runtime.getDeviceProperties(dev)
        if props['major'] < 8 or cp.cuda.runtime.is_hip:
            raise RuntimeError('native prefill requires NVIDIA sm80+ CUDA')
        expressions = [f'native_kernel<{FORMATS[t][0]},{str(gu).lower()},{ww}>'
                       for gu, kinds in ((True, GU_FORMATS), (False, DOWN_FORMATS))
                       for t in kinds for ww in (2, 4)]
        expressions += [f'decode_weight<{v[0]}>' for v in FORMATS.values()]
        mod = cp.RawModule(code=Path(__file__).with_suffix('.cu').read_text(),
            options=('--std=c++17', '--prec-div=true', '--ftz=false',
                     f'-DNATIVE_EMBD={embd}', f'-DNATIVE_FF={ff}'),
            name_expressions=expressions)
        from . import gguf_codebooks as tables
        for name in ('iq2s_grid', 'iq3s_grid', 'iq3xxs_grid', 'kvalues_iq4nl'):
            host = getattr(tables, name)
            mem = mod.get_global(name)
            # CuPy's global pointer deliberately has a zero-sized owner;
            # the CUDA declarations fix the sizes of these byte tables.
            cp.ndarray((host.nbytes,), cp.uint8, memptr=mem).set(host.view(np.uint8).reshape(-1))
        occupancy = []
        for gu, kinds in ((True, GU_FORMATS), (False, DOWN_FORMATS)):
            for kind in kinds:
                for ww in (2, 4):
                    kernel = mod.get_function(f'native_kernel<{FORMATS[kind][0]},{str(gu).lower()},{ww}>')
                    smem = shared_bytes(kind, ww)
                    kernel.max_dynamic_shared_size_bytes = smem
                    occupancy.append(cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
                        kernel.kernel.ptr, 512, smem))
        # Source setup_one/setup_ww uses the minimum over both tile families
        # and all supported GU/down kernels, not a wave-local occupancy guess.
        if min(occupancy) < 1:
            raise RuntimeError('native prefill shared memory/register footprint does not fit this GPU')
        _OCCUPANCY[key] = min(occupancy)
        _MODULES[key] = mod
    return _MODULES[key]


def allocate_group(total_routes, total_experts, top_k):
    """Allocate reusable source-layout tables; no GPU synchronization."""
    cp = _cp()
    size = group_bytes(total_routes, total_experts)
    if top_k < 1 or total_routes % top_k:
        raise ValueError('routes must be divisible by positive top_k')
    scratch = cp.empty(size, cp.uint8)
    words = scratch[:(4 * total_experts + 2) * 4].view(cp.int32)
    e = total_experts
    start = ((4 * e + 2) * 4 + 15) // 16 * 16
    return GroupTables(scratch, words[:e], words[e:2*e+1],
        words[2*e+1:3*e+1], words[3*e+1:4*e+2],
        scratch[start:].view(cp.int32).reshape(-1, 2),
        cp.empty(total_routes, cp.int32), cp.empty(total_routes, cp.int32),
        total_routes, total_experts, top_k)


def group(ids, total_experts, *, top_k=None, out=None):
    """Run the pinned native count/scan/place. ids is int32 [tokens,k] or [R].

    For flat IDs supply top_k. out=allocate_group(...) avoids allocations.
    Returns counts, offsets, slot, src and 64-row tiles as GroupTables views.
    No route values or counts are read on the host.
    """
    cp = _cp()
    _array(ids, np.int32, 'ids')
    if ids.ndim not in (1, 2):
        raise ValueError('ids must be [R] or [tokens,top_k]')
    k = ids.shape[1] if ids.ndim == 2 else top_k
    if k is None or (top_k is not None and k != top_k):
        raise ValueError('top_k must agree with ids shape')
    n = ids.size
    tb = allocate_group(n, total_experts, k) if out is None else out
    if (tb.total_routes, tb.total_experts, tb.top_k) != (n, total_experts, k):
        raise ValueError('group output metadata must agree with ids')
    _array(tb.scratch, np.uint8, 'scratch', (group_bytes(n, total_experts),))
    _array(tb.slot, np.int32, 'slot', (n,))
    _array(tb.src, np.int32, 'src', (n,))
    mod = _module()
    tb.counts.fill(0)
    if n:
        sms = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())['multiProcessorCount']
        mod.get_function('count_kernel')((min((n+1023)//1024, 4*sms),), (1024,),
            (ids, np.int64(n), np.int32(total_experts), tb.counts), shared_mem=total_experts*4)
    mod.get_function('scan_kernel')((1,), (((total_experts+31)//32)*32,),
        (tb.scratch, np.int32(total_experts)))
    if n:
        mod.get_function('place_kernel')(((n+255)//256,), (256,),
            (ids, np.int64(n), np.int32(k), np.int32(total_experts), tb.scratch, tb.slot, tb.src))
    return tb


def quantize(x, *, out=None):
    """FP32 [tokens,K] -> uint8 [tokens,K/64,80], source RN-even INT8 layout."""
    cp = _cp()
    _array(x, np.float32, 'x')
    if x.ndim != 2 or x.shape[1] < 64 or x.shape[1] % 64:
        raise ValueError('x must be rank 2 with positive K divisible by 64')
    shape = (x.shape[0], x.shape[1]//64, 80)
    out = cp.empty(shape, cp.uint8) if out is None else out
    _array(out, np.uint8, 'out', shape)
    nblk = x.size//64
    if nblk:
        _module().get_function('quant_act_nat_kernel')(((nblk+7)//8,), (256,),
            (x, np.int64(nblk), out))
    return out


def execute_batch(e0, e1, total_routes, total_experts, ptrs, xa, grouped, *,
                  embd, ff, gu_kind, down_kind, H=None, Dm=None):
    """Execute expert wave [e0,e1); return packed H and FP32 grouped Dm.

    ptrs: int64 [ALL experts,3], xa: quantize(x), grouped: group(ids,E).
    H: uint8 [ALL routes,ff/64,80], Dm: float32 [ALL routes,embd].
    Supply H/Dm for allocation-free repeated waves. Row order is grouped.src;
    route output is Dm[grouped.slot]. Caller owns weight lifetimes and streams.
    Geometry must fit BOTH source tile families (embd%256=0, ff%128=0).
    Selection uses ALL routes/experts, never wave-local counts.
    """
    cp = _cp()
    ww = pick_ww(total_routes, total_experts)
    if not 0 <= e0 <= e1 <= total_experts:
        raise ValueError('invalid expert wave bounds')
    if embd <= 0 or embd % 256 or ff <= 0 or ff % 128:
        raise ValueError('source geometry requires embd divisible by 256 and ff by 128')
    if gu_kind not in GU_FORMATS or down_kind not in DOWN_FORMATS:
        raise ValueError('unsupported production gate/up or down format')
    if (grouped.total_routes, grouped.total_experts) != (total_routes, total_experts):
        raise ValueError('group metadata must agree with ALL routes/experts')
    _array(ptrs, np.int64, 'ptrs', (total_experts, 3))
    _array(xa, np.uint8, 'xa', (total_routes//grouped.top_k, embd//64, 80))
    _array(grouped.scratch, np.uint8, 'scratch', (group_bytes(total_routes, total_experts),))
    _array(grouped.src, np.int32, 'src', (total_routes,))
    H = cp.empty((total_routes, ff//64, 80), cp.uint8) if H is None else H
    Dm = cp.empty((total_routes, embd), cp.float32) if Dm is None else Dm
    _array(H, np.uint8, 'H', (total_routes, ff//64, 80))
    _array(Dm, np.float32, 'Dm', (total_routes, embd))
    if not total_routes or e0 == e1:
        return H, Dm
    mod = _module(embd, ff)
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    persistent = props['multiProcessorCount'] * _OCCUPANCY[(cp.cuda.runtime.getDevice(), embd, ff)]
    tiles = (total_routes+63)//64 + e1-e0
    gu_row = embd//FORMATS[gu_kind][1]*FORMATS[gu_kind][2]
    d_row = ff//FORMATS[down_kind][1]*FORMATS[down_kind][2]
    for gu, kind, act, nrows in ((True, gu_kind, xa, 2*ff), (False, down_kind, H, embd)):
        kernel = mod.get_function(f'native_kernel<{FORMATS[kind][0]},{str(gu).lower()},{ww}>')
        grid = min(tiles*(nrows//(64*ww)), persistent)
        kernel((grid,), (512,), (np.int32(e0), np.int32(e1), ptrs,
            np.uint64(gu_row), np.uint64(d_row), grouped.scratch, np.int32(total_experts),
            act, grouped.src, H, Dm), shared_mem=shared_bytes(kind, ww))
    return H, Dm


def decode_weights(raw, kind, n, k):
    """Diagnostic production load/convert bridge: signed codes and per16 scales."""
    cp = _cp()
    if kind not in FORMATS:
        raise ValueError('unsupported weight format')
    t, qk, qb = FORMATS[kind]
    if n < 0 or k <= 0 or k % qk:
        raise ValueError('weight dimensions must be whole GGUF blocks')
    _array(raw, np.uint8, 'raw', (n*(k//qk)*qb,))
    codes, scales = cp.empty((n, k), cp.int8), cp.empty((n, k//16), cp.float32)
    if n:
        _module().get_function(f'decode_weight<{t}>')((n, (k//32+127)//128), (128,),
            (raw, np.int32(k), np.int32(n), codes, scales))
    return codes, scales
