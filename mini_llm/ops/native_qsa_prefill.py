"""Pinned Strata projected-input QSA chunk mixer (no dense projections).

Authority: Niko1221/Strata 712fec6b4297c9cfd32ab43daaa2979a9aa01edf.
The sibling CUDA file preserves the upstream MIT notice and kernel arithmetic.
Only INT8 KV / FP16 scales per 64, 24 Q heads / 2 KV heads / D=256,
NVIDIA sm80+, analytic unscaled text RoPE, source 3xTF32 selection and v3 INT8
MMA attention are supported. The pin matches the installed source authority;
version labels alone do not establish a different runtime dispatch.
No dense projections, BF16 GEMM/image construction, residency resolver, decode,
FP16/Q4/hybrid KV, HIP, YaRN/linear/multimodal/table RoPE, alternate environment
dispatches, sliding window, or graph-safe changing host positions.
BF16 projection IMAGES are upstream inputs to bf16_proj, NOT its outputs:
raw index keys and projected index queries at this API boundary are FP32.
The indexer alone rounds raw keys through FP16 before pooling.
Pinned core/layer.cpp: kv_resident=0 means FULLY DEVICE RESIDENT. Only mode 1
uses a mapped host mirror + prefix staging; direct mapped-host attention is
not substituted for that source call chain.

CPU caller contract:
* K is ALREADY normed/rotated, V is not; Q is ALREADY normed/rotated.
* Append the entire chunk before attention. Earlier cells must be initialized.
* ids/steps come from the upstream selection graph. Each step is
  [pos, pos+1, (pos+1)//4, min(pos+1,2051)]; only the first width ids are read.
  They must be distinct, ascending, causal (0 <= id <= pos), and resident.
  No masking/selection is added inside attention. check=True synchronously
  validates these invariants (one aggregate device-to-host scalar).
* Arrays, host mirrors and streams must outlive all queued work. Same-stream
  calls are ordered; cross-stream dependencies/events belong to the caller.
  Synchronize before CPU reads/writes of mapped host buffers or close().
* Prefix staging copies ceil(n_cells/page_size) WHOLE identity pages, including
  padding of the last page, just like kv_stage_from_host. It never truncates
  history: capacity overflow raises. No automatic eviction/windowing is offered.
* All output buffers are caller-reusable. Apart from optional validation there
  are no D2H copies, per-query/head CPU loops or implicit synchronizations.
"""
from dataclasses import dataclass
from pathlib import Path
import ctypes
import numpy as np

PIN = '712fec6b4297c9cfd32ab43daaa2979a9aa01edf'
# GitHub API-retrieved full-file SHA256s; no shared provenance manifest is edited.
SOURCE_SHA256 = {
    'src/kernels/cuda/native_qsa_score.cu': '046dac614be068448e71d378150e9d1c5c39fa07d8fd27459611efd5e1566e5d',
    'src/kernels/cuda/native_qsa_indexer.cu': '8fa911b34164e03a4ada188a89eed5e1d5af7aff98ed6a94c270fa863c50764f',
    'src/kernels/cuda/qsa_select.cu': 'e8fd48784a1f98e9302b1b20d75ba1305530c08e7f0d1ef47e176ad62f07f39a',
    'src/kernels/cuda/qsa_prompt_attn.cu': 'f56e663115c80338d6e35c77491224cc259b1994c1b857e49b3ebe5497be0eca',
    'src/kernels/cuda/kv_stream.cu': '1e4b9c67666b4f460898429a684f4ce7a3c8fc506538450f99da46adaccd66fc',
    'include/strata/kernels/qsa_prompt_attn.hpp': 'e11c68aad92a4fac10dd1d42360aa0ea70ab3e602c95f0920ca6c2f233ec78fe',
    'include/strata/kernels/qsa_decode_attn.hpp': 'd15f962e0e40c98772e108e551fbc1233b6ec9b0d79d98709182126d3778a52a',
    'include/strata/kernels/qsa.hpp': '4f7a2dd563be7d15ffb306a450fcc74f7abb5816fe82dda0ae221a17864e3a74',
    'include/strata/kernels/kv_stream.hpp': '24a47cb5eca790fa35b13bbeb12cf3407389c41219899789880d965cf86d8abe',
    'include/strata/kernels/native_qsa_indexer.hpp': '3628d2cbedb29d296d959b88e45a8023eae37be725ecad904643eb4729933dc3',
    'include/strata/kernels/rope_scaling.hpp': '05a8dcf57c9b12a23128f22e4bef01445c3a37991df5e1ffacd01d077c46595d',
    'include/strata/kernels/mrope.hpp': 'f7add5611f54af2434daa1ad711fe441e214693bd33524a12c56058807fd381c',
    'src/core/layer.cpp': '86ce7cd55355f140efe8604a21dbb09505923f9432cd74c0418cff9799295f49',
}
_MODULES = {}
# sizeof(Smem3): kv + sc + part + p + alpha/lsum/mrow (all naturally 4-byte aligned).
_SHARED_BYTES = 4*2*32*64 + 4*2*32*4 + 4*12*33*4 + 12*33*4 + 3*16*4


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


def _module():
    cp = _cp()
    dev = cp.cuda.runtime.getDevice()
    if dev not in _MODULES:
        if cp.cuda.runtime.is_hip or cp.cuda.runtime.getDeviceProperties(dev)['major'] < 8:
            raise RuntimeError('native QSA requires NVIDIA sm80+ CUDA')
        # RawModule in CuPy 14 unconditionally appends -ftz=true. Compile
        # directly so the source's gradual underflow is not silently changed.
        from cupy.cuda import compiler, function, driver
        from cupy._core.core import assemble_cupy_compiler_options
        cubin, _ = compiler.compile_using_nvrtc(
            Path(__file__).with_suffix('.cu').read_text(),
            options=assemble_cupy_compiler_options(
                ('--std=c++17', '--prec-div=true', '--ftz=false')),
            arch=cp.cuda.Device(dev).compute_capability, cache_in_memory=True)
        mod = function.Module()
        mod.load(cubin)
        attn = mod.get_function('prompt_attn_i8v3_kernel')
        driver.funcSetAttribute(attn.ptr, driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                _SHARED_BYTES)
        driver.funcSetAttribute(attn.ptr, driver.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                100)
        scorer = mod.get_function('block_scores_tc_kernel')
        driver.funcSetAttribute(scorer.ptr, driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                _SCORE_SHARED_BYTES)
        _MODULES[dev] = mod
    return _MODULES[dev]


@dataclass(frozen=True)
class Q8Cache:
    """CuPy K/V int8 [pages,2,page_size,256], scales float16 [...,4].

    page_table: CuPy int32 [logical_pages], logical -> physical or -1.
    Mapping ownership and initialization belong to the caller. Append skips
    -1 resident pages, but attention requires every selected page resident.
    """
    k: object
    v: object
    k_scale: object
    v_scale: object
    page_table: object

    @property
    def page_size(self):
        return self.k.shape[2]


def _cache(c):
    if not isinstance(c, Q8Cache):
        raise TypeError('cache must be Q8Cache')
    _array(c.k, np.int8, 'cache.k')
    if c.k.ndim != 4 or c.k.shape[1] != 2 or c.k.shape[3] != 256 or min(c.k.shape) < 1:
        raise ValueError('cache codes must have shape [pages,2,page_size,256] with positive sizes')
    _array(c.v, np.int8, 'cache.v', c.k.shape)
    for name in ('k_scale', 'v_scale'):
        _array(getattr(c, name), np.float16, name, c.k.shape[:-1] + (4,))
    _array(c.page_table, np.int32, 'page_table')
    if c.page_table.ndim != 1 or c.page_table.size < 1:
        raise ValueError('page_table must be a nonempty vector')
    return c


def allocate_cache(n_pages, page_size=4, *, page_table=None):
    """Allocate uninitialized GPU pools; default mapping is identity."""
    if not isinstance(n_pages, int) or not isinstance(page_size, int) or min(n_pages, page_size) < 1:
        raise ValueError('positive integer page count and page size required')
    cp = _cp()
    shape = (n_pages, 2, page_size)
    table = cp.arange(n_pages, dtype=cp.int32) if page_table is None else page_table
    return _cache(Q8Cache(cp.empty(shape+(256,), cp.int8), cp.empty(shape+(256,), cp.int8),
                         cp.empty(shape+(4,), cp.float16), cp.empty(shape+(4,), cp.float16), table))


class HostQ8Cache:
    """Explicitly owned CUDA-mapped pinned NumPy identity-layout mirror.

    Construct on the target device, retain until work completes; close() ONLY
    after synchronizing all streams using this object. No destructor frees a
    possibly in-flight buffer: caller must close (or use its context manager,
    whose exit synchronizes the owning device before close).
    """
    def __init__(self, n_pages, page_size=4):
        if not isinstance(n_pages, int) or not isinstance(page_size, int) or min(n_pages, page_size) < 1:
            raise ValueError('positive integer page count and page size required')
        cp = _cp()
        if cp.cuda.runtime.is_hip:
            raise RuntimeError('mapped host cache requires CUDA UVA')
        self.device = cp.cuda.runtime.getDevice()
        self.page_size = page_size
        self.n_pages = n_pages
        self._ptrs = []
        try:
            for name, dtype, width in [('k', np.int8, 256), ('v', np.int8, 256),
                                       ('k_scale', np.float16, 4), ('v_scale', np.float16, 4)]:
                shape = (n_pages, 2, page_size, width)
                size = int(np.prod(shape)) * np.dtype(dtype).itemsize
                ptr = cp.cuda.runtime.hostAlloc(size, cp.cuda.runtime.hostAllocMapped |
                                               cp.cuda.runtime.hostAllocPortable)
                self._ptrs.append(ptr)
                # cudaHostAllocMapped without WriteCombined uses the same UVA address.
                raw = (ctypes.c_ubyte * size).from_address(ptr)
                setattr(self, name, np.frombuffer(raw, dtype=dtype).reshape(shape))
        except Exception:
            self.close()
            raise

    def close(self):
        """Free pinned storage; invalidates all views. Caller must synchronize first."""
        cp = _cp()
        for ptr in self._ptrs:
            cp.cuda.runtime.freeHost(ptr)
        self._ptrs.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        with _cp().cuda.Device(self.device):
            _cp().cuda.runtime.deviceSynchronize()
            self.close()


def _host(h, page_size, end):
    if not isinstance(h, HostQ8Cache) or len(h._ptrs) != 4:
        raise TypeError('live HostQ8Cache required')
    if h.device != _cp().cuda.runtime.getDevice() or h.page_size != page_size:
        raise ValueError('host mirror device/page_size mismatch')
    if end > h.n_pages * page_size:
        raise ValueError('host mirror capacity exceeded')
    return h


def make_steps(pos0, n_q):
    """Return CPU int32 [n_q,4]; upload once with cp.asarray on caller stream."""
    if not isinstance(pos0, int) or not isinstance(n_q, int) or min(pos0, n_q) < 0 or pos0+n_q > 2147483647:
        raise ValueError('nonnegative integer positions/count, end <= INT32_MAX required')
    pos = np.arange(pos0, pos0+n_q, dtype=np.int32)
    nkv = pos+1
    return np.stack((pos, nkv, nkv//4, np.minimum(nkv, 2051)), axis=1)


def kv_append(k, v, pos0, cache, *, host=None, stage=None, stream=None):
    """FP32 [T,2,256] -> resident INT8 pools, optional mapped host + stage.

    Exact source quantization: amax/127 rounded to FP16 FIRST, divide by that
    rounded scale, nearest-even int conversion, clamp [-127,127]; scale zero
    produces code zero. Inputs must be finite with representable FP16 scales.
    Stage mapping MUST be identity; its capacity must hold [0,pos0+T).
    page_table values must be -1 or valid nonaliased physical pages.
    """
    cp = _cp()
    _cache(cache)
    _array(k, np.float32, 'k')
    if k.ndim != 3 or k.shape[1:] != (2, 256):
        raise ValueError('k must have shape [T,2,256]')
    _array(v, np.float32, 'v', k.shape)
    t = k.shape[0]
    if not isinstance(pos0, int) or pos0 < 0 or pos0+t > cache.page_table.size*cache.page_size:
        raise ValueError('append exceeds logical cache capacity')
    hp = [np.uint64(0)] * 4
    sp = [np.uint64(0)] * 4
    if host is not None:
        _host(host, cache.page_size, pos0+t)
        hp = [np.uint64(p) for p in host._ptrs]
    if stage is not None:
        _cache(stage)
        if stage.page_size != cache.page_size or pos0+t > stage.k.shape[0]*stage.page_size:
            raise ValueError('staging capacity/page_size mismatch')
        sp = [stage.k, stage.v, stage.k_scale, stage.v_scale]
    if t:
        _module().get_function('kv_append_kernel')((t, 2, 8), (64,),
            (k, v, np.int64(pos0), cache.page_table, np.int64(cache.page_size),
             np.uint64(0), np.uint64(0), cache.k, cache.v, cache.k_scale, cache.v_scale,
             *hp, *sp), stream=stream)
    return cache


def stage_from_host(host, stage, n_cells, *, stream=None):
    """Source kv_stage_from_host: four prefix DMAs, no conversion or truncation.

    stage must use identity mapping. Raises if ceil(n_cells/P) exceeds its fixed
    capacity. Retain host until the stream completes; no CPU access meanwhile.
    """
    cp = _cp()
    _cache(stage)
    if not isinstance(n_cells, int) or n_cells < 0:
        raise ValueError('nonnegative integer n_cells required')
    _host(host, stage.page_size, n_cells)
    pages = (n_cells+stage.page_size-1)//stage.page_size
    if pages > stage.k.shape[0]:
        raise ValueError('staging capacity exceeded; history is never truncated')
    st = cp.cuda.get_current_stream() if stream is None else stream
    for name, ptr in zip(('k', 'v', 'k_scale', 'v_scale'), host._ptrs):
        dst = getattr(stage, name)
        cp.cuda.runtime.memcpyAsync(dst.data.ptr, ptr, pages*dst.strides[0],
                                    cp.cuda.runtime.memcpyDefault, st.ptr)
    return stage


def prompt_attn_batch(q, cache, ids, steps, *, out=None, stream=None, check=False):
    """Pinned v3 batched attention; FP32 [T,24,256] -> same shape/dtype.

    ids: CuPy int32 [T,cap], steps: CuPy int32 [T,4]. cap >= max width.
    Both are caller-owned SOURCE selection outputs, not recomputed here.
    One CUDA block per (query, KV head), 128 threads, 32-cell online softmax.
    check=False trusts the causal/residency contract; check=True validates it
    with a synchronous scalar transfer and is unsuitable for CUDA capture.
    """
    cp = _cp()
    _cache(cache)
    _array(q, np.float32, 'q')
    if q.ndim != 3 or q.shape[1:] != (24, 256):
        raise ValueError('q must have shape [T,24,256]')
    t = q.shape[0]
    _array(ids, np.int32, 'ids')
    if ids.ndim != 2 or ids.shape[0] != t or not 1 <= ids.shape[1] <= 2147483647:
        raise ValueError('ids must have shape [T,positive cap <= INT32_MAX]')
    _array(steps, np.int32, 'steps', (t, 4))
    if out is None:
        out = cp.empty_like(q)
    _array(out, np.float32, 'out', q.shape)
    st = cp.cuda.get_current_stream() if stream is None else stream
    with st:
        if check and t:
            pos, nkv, nbid, width = (steps[:, i] for i in range(4))
            active = cp.arange(ids.shape[1])[None, :] < width[:, None]
            safe = cp.clip(ids, 0, cache.page_table.size*cache.page_size-1)
            pages = cache.page_table[safe//cache.page_size]
            bad = cp.any((pos < 0) | (nkv != pos+1) | (nbid != nkv//4) |
                         (width != cp.minimum(nkv, 2051)) | (width < 1) |
                         (width > ids.shape[1]) | (nkv > cache.page_table.size*cache.page_size))
            bad |= cp.any(active & ((ids < 0) | (ids > pos[:, None]) |
                                    (pages < 0) | (pages >= cache.k.shape[0])))
            bad |= cp.any(active[:, 1:] & (ids[:, 1:] <= ids[:, :-1]))
            if bool(bad.item()):
                raise ValueError('invalid causal selection, step metadata, or residency')
        kernel = _module().get_function('prompt_attn_i8v3_kernel')
        # Source launch_i8v3 also limits each query grid to 65535.
        for start in range(0, t, 65535):
            count = min(t-start, 65535)
            kernel((count, 2), (128,),
                   (cache.k, cache.v, cache.k_scale, cache.v_scale, cache.page_table,
                    q[start:], ids[start:], steps[start:], np.int32(2),
                    np.int32(cache.page_size), np.float32(np.float32(1.4426950408889634)/np.float32(16)),
                    out[start:], np.int32(ids.shape[1])), shared_mem=_SHARED_BYTES, stream=st)
    return out


def gate_attn(attn, qf, *, out=None, stream=None):
    """Source CUDA gate: FP32 attn [T,24,256], qf [T,24,512] -> FP16.

    qf's last 256 elements per head are the gate; first 256 are ignored.
    out is FP16 [T,ld16], ld16 >= 6144; padding is untouched.
    """
    cp = _cp()
    _array(attn, np.float32, 'attn')
    if attn.ndim != 3 or attn.shape[1:] != (24, 256):
        raise ValueError('attn must have shape [T,24,256]')
    t = attn.shape[0]
    _array(qf, np.float32, 'qf', (t, 24, 512))
    if out is None:
        out = cp.empty((t, 6144), cp.float16)
    _array(out, np.float16, 'out')
    if out.ndim != 2 or out.shape[0] != t or out.shape[1] < 6144:
        raise ValueError('gate out must have shape [T,ld16 >= 6144]')
    if t:
        _module().get_function('gate_attn_kernel')(((t*6144+255)//256,), (256,),
            (attn, qf, out, np.int64(t), np.int64(out.shape[1])), stream=stream)
    return out


def prompt_attn_batch_gated(q, cache, ids, steps, qf, *, attn_out=None,
                            out=None, stream=None, check=False):
    """CUDA source call chain: batched attention THEN gate (not a fused kernel)."""
    attn = prompt_attn_batch(q, cache, ids, steps, out=attn_out, stream=stream, check=check)
    return gate_attn(attn, qf, out=out, stream=stream)


# Source host powf(1e7f, -2.0f/64), fixed artifact's unscaled text RoPE.
_THETA = np.float32(float.fromhex('0x1.356656p-1'))
_ROPE_ARGS = (_THETA, np.float32(1), np.float32(0), np.float32(0),
              np.float32(0), np.float32(1))
_SCORE_SHARED_BYTES = (16*(4*128+4) + 32*(128+4))*4


@dataclass
class IndexerState:
    """SOURCE state: tail FP32 [3,128] holding FP16-rounded raw values;
    dead FP32 [128] repeated-cell0 spare; pooled FP32 [max_cells//4+2,128];
    block_pos INT32 [1]. This is NOT the parent's existing FP32 raw history.
    next_pos/pos_base are host sequence ownership metadata, never GPU readbacks.
    Do not mutate them or reuse this state concurrently on unordered streams.
    """
    tail: object
    dead: object
    pooled: object
    block_pos: object
    max_cells: int
    next_pos: int = 0
    pos_base: int = 0


def allocate_indexer(max_cells, *, pos_base=0):
    """Clear source state once per sequence; capacity includes the spare row."""
    if (not isinstance(max_cells, int) or not isinstance(pos_base, int) or
            max_cells < 1 or pos_base < 0 or pos_base % 4 or max_cells+pos_base > 2147483647):
        raise ValueError('positive capacity, nonnegative four-aligned base, int32 positions required')
    cp = _cp()
    return IndexerState(cp.zeros((3, 128), cp.float32), cp.zeros(128, cp.float32),
                        cp.zeros((max_cells//4+2, 128), cp.float32),
                        cp.zeros(1, cp.int32), max_cells, pos_base=pos_base)


def _indexer(state):
    if not isinstance(state, IndexerState):
        raise TypeError('source IndexerState required (not a raw-key history buffer)')
    _array(state.tail, np.float32, 'tail', (3, 128))
    _array(state.dead, np.float32, 'dead', (128,))
    _array(state.pooled, np.float32, 'pooled', (state.max_cells//4+2, 128))
    _array(state.block_pos, np.int32, 'block_pos', (1,))
    return state


def _epsilon(epsilon):
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('finite positive epsilon required')
    return np.float32(epsilon)


def norm_rope(x, gamma, pos0, *, query_gate=False, out=None, epsilon=1e-6, stream=None):
    """Source fused weighted RMSNorm + analytic text RoPE (first 64 dims).

    x: CuPy FP32 [T,H,D], D=128 or 256, gamma FP32 [D]. query_gate=True
    takes [T,24,512] q|gate and returns [T,24,256], leaving the gate untouched.
    No BF16/FP16 rounding is inserted. pos0 is the absolute text position.
    Inputs must be finite; gamma is the source norm weight, not gamma+1.
    """
    cp = _cp()
    _array(x, np.float32, 'projected input')
    if x.ndim != 3:
        raise ValueError('projected input must be [T,H,D]')
    t, heads, width = x.shape
    dim = 256 if query_gate else width
    if (query_gate and (heads, width) != (24, 512)) or (not query_gate and dim not in (128, 256)):
        raise ValueError('source dimensions required: 128/256, or q|gate [T,24,512]')
    if not isinstance(pos0, int) or pos0 < 0 or pos0+t > 2147483647:
        raise ValueError('nonnegative int32 text positions required')
    eps = _epsilon(epsilon)
    _array(gamma, np.float32, 'norm gamma', (dim,))
    if out is None:
        out = cp.empty((t, heads, dim), cp.float32)
    _array(out, np.float32, 'norm/rope output', (t, heads, dim))
    if t:
        _module().get_function('rms_rope_rows_kernel')((t*heads,), (256,),
            (x, np.int64(heads*width), np.int64(width), out, np.int64(heads*dim),
             gamma, np.int64(heads), np.int64(dim), eps, np.int64(pos0),
             *_ROPE_ARGS, np.int32(0)), stream=stream)
    return out


def index_append(raw_keys, norm_gamma, pos0, state, *, epsilon=1e-6, stream=None):
    """Source native_qsa_indexer_append_batch, at most three kernel launches.

    raw_keys FP32 [T,128] is bf16_proj's FP32 OUTPUT, before norm/RoPE.
    norm_gamma FP32 [128]. pos0 is the contiguous relative CELL position.
    Completed blocks pool chronological FP16-rounded keys, then source FP32
    RMSNorm/gamma and RoPE at pos_base+4*b. The dead spare always rotates at 0.
    No per-token CPU work or GPU state readback. Returns the same mutated state.
    """
    _indexer(state)
    _array(raw_keys, np.float32, 'raw index keys')
    if raw_keys.ndim != 2 or raw_keys.shape[1] != 128:
        raise ValueError('raw index keys must have shape [T,128]')
    _array(norm_gamma, np.float32, 'index key gamma', (128,))
    n = raw_keys.shape[0]
    if not isinstance(pos0, int) or pos0 != state.next_pos or pos0+n > state.max_cells:
        raise ValueError('append must continue source indexer state within capacity')
    eps = _epsilon(epsilon)
    if not n:
        return state
    mod = _module()
    if pos0 == 0:
        mod.get_function('idx_append_first')((1,), (256,),
            (raw_keys, norm_gamma, eps, state.dead, state.pooled, *_ROPE_ARGS, np.int32(0)), stream=stream)
    first = 0 if pos0 <= 3 else pos0//4
    hi = (pos0+n-4)//4 if pos0+n >= 4 else -1
    if hi >= first:
        mod.get_function('idx_append_blocks')((hi-first+1,), (256,),
            (raw_keys, np.int64(n), np.int64(pos0), np.int32(state.pos_base), norm_gamma, eps,
             state.tail, state.dead, state.pooled, state.block_pos, np.int64(first), np.int64(hi),
             *_ROPE_ARGS, np.int32(0)), stream=stream)
    mod.get_function('idx_append_tail')((3,), (128,),
        (raw_keys, np.int64(n), np.int64(pos0), state.tail), stream=stream)
    state.next_pos = pos0+n
    return state


def make_steps_device(pos0, n_q, *, stream=None):
    """Source step equations on GPU, without CPU arrays or D2H readback."""
    if not isinstance(pos0, int) or not isinstance(n_q, int) or min(pos0, n_q) < 0 or pos0+n_q > 2147483647:
        raise ValueError('nonnegative integer positions/count, int32 end required')
    cp = _cp()
    st = cp.cuda.get_current_stream() if stream is None else stream
    with st:
        pos = cp.arange(pos0, pos0+n_q, dtype=cp.int32)
        nkv = pos+1
        return cp.stack((pos, nkv, nkv//4, cp.minimum(nkv, 2051)), axis=1)


def score_blocks(index_query, state, steps, *, out=None, active_blocks=None, stream=None):
    """Source qsa_select 3xTF32 scorer + source FP32 dead-tail scorer.

    index_query: FP32 [T,4,128], ALREADY weighted-normalized/rotated.
    Output FP32 [T,max_cells//4+2]. Entries b > steps[t,2] are unwritten.
    All completed blocks are present; each query reads ONLY b < its own n_bid,
    and the immutable cell0 dead key at b=n_bid (+1e9 for incomplete tails).
    This remains causal even after future blocks of this chunk are appended.
    active_blocks is a CPU upper bound, not a GPU-derived scalar (None=capacity).
    """
    cp = _cp()
    _indexer(state)
    _array(index_query, np.float32, 'normalized/rotated index query')
    if index_query.ndim != 3 or index_query.shape[1:] != (4, 128):
        raise ValueError('index query must have shape [T,4,128]')
    t = index_query.shape[0]
    if t > 65535*16:
        raise ValueError('score batch exceeds source query-grid bound')
    _array(steps, np.int32, 'steps', (t, 4))
    blocks = state.pooled.shape[0]
    reach = blocks if active_blocks is None else active_blocks
    if not isinstance(reach, int) or not 1 <= reach <= blocks:
        raise ValueError('active_blocks must be a positive capacity-bounded integer')
    if out is None:
        out = cp.empty((t, blocks), cp.float32)
    _array(out, np.float32, 'block scores', (t, blocks))
    if t:
        mod = _module()
        mod.get_function('block_scores_tc_kernel')(((reach+127)//128, (t+15)//16), (128,),
            (state.pooled, index_query, steps, np.int64(t), np.int64(blocks), np.int64(reach), out),
            shared_mem=_SCORE_SHARED_BYTES, stream=stream)
        mod.get_function('block_scores_tail_kernel')((t,), (32,),
            (state.dead, index_query, steps, np.int64(blocks), out), stream=stream)
    return out


def block_select(scores, steps, *, cap=2051, out=None, stream=None):
    """Original source radix selector, not an independent top-k algorithm.

    scores FP32 [T,max_blocks], steps INT32 [T,4]; output INT32 [T,cap].
    Caller guarantees 0 < width <= cap, n_bid < max_blocks. Weighted threshold
    ties go to lower CELL indices; selected IDs are emitted ascending. NaNs map
    to the source's lowest order_key; signed zeros are normalized by +0.
    Only width IDs are initialized. No CPU readback, sorting, or per-token loop.
    Uses the source reference radix kernel explicitly, not its dispatch autotuner.
    """
    cp = _cp()
    _array(scores, np.float32, 'block scores')
    if scores.ndim != 2 or scores.shape[1] < 1 or not isinstance(cap, int) or cap < 1:
        raise ValueError('scores [T,positive max_blocks] and positive cap required')
    t, blocks = scores.shape
    _array(steps, np.int32, 'steps', (t, 4))
    if out is None:
        out = cp.empty((t, cap), cp.int32)
    _array(out, np.int32, 'selected ids', (t, cap))
    if t:
        _module().get_function('block_topk_kernel')((t,), (256,),
            (scores, steps, np.int64(blocks), np.int64(cap), out), stream=stream)
    return out


@dataclass(frozen=True)
class Selection:
    ids: object
    steps: object
    score_scratch: object


def score_select(index_query, state, pos0, *, batch_size=128, ids=None, steps=None,
                 score_scratch=None, stream=None):
    """Source chunked batched scorer/radix pipeline. Query must be normed/rotated.

    Score scratch is bounded to min(batch_size,T) x (max_cells//4+2), reused
    across query tiles. Returned scratch contains only the LAST tile's scores.
    Steps/IDs cover the full chunk and conform to prompt_attn_batch. No D2H.
    Positions must describe an already-appended interval within source state.
    """
    cp = _cp()
    _indexer(state)
    _array(index_query, np.float32, 'index query')
    if index_query.ndim != 3 or index_query.shape[1:] != (4, 128):
        raise ValueError('index query must have shape [T,4,128]')
    t = index_query.shape[0]
    if not isinstance(pos0, int) or pos0 < 0 or pos0+t > state.next_pos:
        raise ValueError('selection interval must already be appended')
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 65535*16:
        raise ValueError('positive source-bounded selection batch_size required')
    st = cp.cuda.get_current_stream() if stream is None else stream
    with st:
        if steps is None:
            steps = make_steps_device(pos0, t, stream=st)
        _array(steps, np.int32, 'steps', (t, 4))
        cap = min(pos0+t, 2051) if t else 1
        if ids is None:
            ids = cp.empty((t, cap), cp.int32)
        _array(ids, np.int32, 'ids', (t, cap))
        rows = min(t, batch_size)
        if score_scratch is None:
            score_scratch = cp.empty((rows, state.pooled.shape[0]), cp.float32)
        _array(score_scratch, np.float32, 'score scratch', (rows, state.pooled.shape[0]))
        for start in range(0, t, batch_size):
            n = min(batch_size, t-start)
            active = (pos0+start+n)//4+1
            score_blocks(index_query[start:start+n], state, steps[start:start+n],
                         out=score_scratch[:n], active_blocks=active, stream=st)
            block_select(score_scratch[:n], steps[start:start+n], cap=cap,
                         out=ids[start:start+n], stream=st)
    return Selection(ids, steps, score_scratch)


def native_score_step(index_query, state, step, *, bias=None, out=None, stream=None):
    """Original native_qsa_score.cu CUDA diagnostic, NOT the prompt batch scorer.

    query FP32 [4,128], step INT32 [4], optional FP32 block bias [max_blocks].
    Caller must use CURRENT end-state step (pooled[n_bid] is the current spare).
    Output FP32 [max_cells], only [0,n_kv) initialized. The source's original
    raw-F32-bit TF32 MMA / two-warp accumulation is preserved independently of
    the batched 3xTF32 prompt arithmetic; equality between them is not promised.
    """
    cp = _cp()
    _indexer(state)
    _array(index_query, np.float32, 'native score query', (4, 128))
    _array(step, np.int32, 'native score step', (4,))
    if bias is not None:
        _array(bias, np.float32, 'native score bias', (state.pooled.shape[0],))
    if out is None:
        out = cp.empty(state.max_cells, cp.float32)
    _array(out, np.float32, 'native cell scores', (state.max_cells,))
    _module().get_function('native_score_kernel')(((state.max_cells//4+32)//32,), (32, 2),
        (state.pooled, index_query, np.uint64(0) if bias is None else bias,
         step, np.int32(state.max_cells), out), stream=stream)
    return out


@dataclass(frozen=True)
class QsaChunk:
    gated: object
    attention: object
    selection: Selection
    query: object
    key: object
    index_query: object


def qsa_chunk(qg, k, v, iq, raw, *, pos0, cache, indexer, q_gamma, k_gamma,
              iq_gamma, ik_gamma, host=None, stage=None, selection_batch=128,
              epsilon=1e-6, stream=None, check=False):
    """Complete source QSA mixer AFTER projections, BEFORE output projection.

    qg FP32 [T,24,512], k/v FP32 [T,2,256], iq FP32 [T,4,128], raw FP32
    [T,128]; q/k norm gamma FP32 [256], iq/ik gamma FP32 [128]. iq/raw MUST
    be FP32 outputs from the parent's BF16 projection graph, NOT BF16 images.
    Inputs are not overwritten. indexer must continue at pos0. All arrays and
    returned intermediate storage must outlive the stream's work.

    Resident mode (including source kv_resident=0): cache holds all history;
    no prefix upload. Optional host mirrors append writes. Streamed mode:
    stage requires host; source prefix DMA precedes append; capacity must hold
    the full prefix+chunk. No direct-host or truncated-history substitute.
    """
    cp = _cp()
    _indexer(indexer)
    _cache(cache)
    _array(qg, np.float32, 'projected q|gate')
    if qg.ndim != 3 or qg.shape[1:] != (24, 512):
        raise ValueError('projected q|gate must have shape [T,24,512]')
    t = qg.shape[0]
    for a, shape, name in ((k, (t, 2, 256), 'projected K'), (v, (t, 2, 256), 'projected V'),
                           (iq, (t, 4, 128), 'projected index query'), (raw, (t, 128), 'raw index key')):
        _array(a, np.float32, name, shape)
    for a, width, name in ((q_gamma, 256, 'Q gamma'), (k_gamma, 256, 'K gamma'),
                           (iq_gamma, 128, 'IQ gamma'), (ik_gamma, 128, 'IK gamma')):
        _array(a, np.float32, name, (width,))
    if not isinstance(pos0, int) or pos0 != indexer.next_pos or pos0+t > indexer.max_cells:
        raise ValueError('mixer must continue source indexer state within capacity')
    if pos0+t > cache.page_table.size*cache.page_size:
        raise ValueError('mixer cache capacity exceeded')
    if not isinstance(selection_batch, int) or not 1 <= selection_batch <= 65535*16:
        raise ValueError('positive source-bounded selection_batch required')
    eps = _epsilon(epsilon)
    if stage is not None:
        _cache(stage)
        _host(host, cache.page_size, pos0+t)
        if stage.page_size != cache.page_size or pos0+t > stage.k.shape[0]*stage.page_size:
            raise ValueError('staging capacity/page_size mismatch; history cannot be truncated')
    st = cp.cuda.get_current_stream() if stream is None else stream
    with st:
        key = norm_rope(k, k_gamma, indexer.pos_base+pos0, epsilon=eps, stream=st)
        if stage is not None:
            stage_from_host(host, stage, pos0, stream=st)
        kv_append(key, v, pos0, cache, host=host, stage=stage, stream=st)
        query = norm_rope(qg, q_gamma, indexer.pos_base+pos0, query_gate=True, epsilon=eps, stream=st)
        index_query = norm_rope(iq, iq_gamma, indexer.pos_base+pos0, epsilon=eps, stream=st)
        index_append(raw, ik_gamma, pos0, indexer, epsilon=eps, stream=st)
        selection = score_select(index_query, indexer, pos0, batch_size=selection_batch, stream=st)
        attention = prompt_attn_batch(query, cache if stage is None else stage,
                                     selection.ids, selection.steps, stream=st, check=check)
        gated = gate_attn(attention, qg, stream=st)
    return QsaChunk(gated, attention, selection, query, key, index_query)
