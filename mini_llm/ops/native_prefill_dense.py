"""Pinned Strata CUDA dense/HC prompt arithmetic, not decode arithmetic.

Source: 712fec6b4297c9cfd32ab43daaa2979a9aa01edf, prefill.cpp
3063-3093, 4155-4217; gemm.cu 603-627, 1178-1182, 1245-1263,
1477-1508; kernels.cu 133-172, 273-281. The sibling CUDA file carries
full MIT notice. No external engine/source tree is used at runtime.

CPU resolves GGUF metadata and uploads packed bytes; GPU dequantizes FP16,
then cuBLAS computes X W^T. HC uses FP32 norm -> BF16 image -> BF16 down
-> SiLU(x/4) -> BF16 image -> BF16 up, plus BF16 inject -> FP32 mix.
Mix reconstructs normalized FP32 streams, not their rounded BF16 image.

Only the unpadded NVIDIA sm80+ default path is supported: prompt_f16=False,
BF16X2 off, no req8, custom HC down/upmix, old-card emulation, or HIP.
No environment setting silently changes this contract. GEMM uses default
math (including the source's permission for reduced-precision reductions),
COMPUTE_32F, GEMM_DEFAULT; FP16 retries ALGO2 only on status 14, as source.
A fresh handle with a 32 MiB workspace is private to each GEMM call. This
is correctness scaffolding, not a performance/capture/residency API.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math

import numpy as np

REVISION = '712fec6b4297c9cfd32ab43daaa2979a9aa01edf'
# (GGML type, elements/block, bytes/block), exactly group32 in pinned dequant.
FORMATS = {'Q4_0': (2, 32, 18), 'Q5_0': (6, 32, 22),
           'Q5_1': (7, 32, 24), 'Q8_0': (8, 32, 34),
           'Q3_K': (11, 256, 110), 'Q4_K': (12, 256, 144),
           'Q5_K': (13, 256, 176), 'Q6_K': (14, 256, 210),
           'IQ4_NL': (20, 32, 18), 'IQ4_XS': (23, 256, 136),
           'Q2_0': (42, 64, 18)}


@dataclass(frozen=True)
class ConsumerImages:
    """Row-major uint16 BF16 bits and float16 FP16 image, no low image."""
    bf16: object
    fp16: object


@dataclass(frozen=True)
class HCRead:
    hidden: object
    gains: object
    images: ConsumerImages


def _cp():
    import cupy as cp
    return cp


def _array(a, dtype, name):
    cp = _cp()
    if not isinstance(a, cp.ndarray) or a.dtype != np.dtype(dtype):
        raise TypeError(f'{name} must be a CuPy {np.dtype(dtype).name} array')
    if not a.flags.c_contiguous or a.device.id != cp.cuda.runtime.getDevice():
        raise ValueError(f'{name} must be contiguous on the current device')
    return a


@lru_cache(maxsize=32)
def _module(device, width):
    cp = _cp()
    if cp.cuda.runtime.is_hip or cp.cuda.runtime.getDeviceProperties(device)['major'] < 8:
        raise NotImplementedError('native dense prefill requires NVIDIA sm80+')
    # CuPy 14 RawModule appends -ftz=true after caller options. Bypass that
    # wrapper; native cubin also avoids a newer NVRTC PTX/driver JIT mismatch.
    from cupy.cuda import compiler, function
    from cupy._core.core import assemble_cupy_compiler_options
    names = ['gr_norm_rs_kernel', 'gr_mix_r_kernel', 'gr_silu_kernel',
             'images', 'gains'] + [f'unpack<{v[0]}>' for v in FORMATS.values()]
    options = assemble_cupy_compiler_options((
        '--std=c++17', '--ftz=false', '--prec-div=true',
        f'--gpu-architecture=sm_{cp.cuda.Device(device).compute_capability}',
        f'-DNATIVE_WIDTH={width}'))
    cubin, mapping = compiler._NVRTCProgram(
        Path(__file__).with_suffix('.cu').read_text(),
        name_expressions=names, method='cubin').compile(options)
    mod = function.Module()
    mod.load(cubin)
    return mod, mapping


def _kernel(name, width=1):
    cp = _cp()
    mod, mapping = _module(cp.cuda.runtime.getDevice(), width)
    return mod.get_function(mapping[name])


def consumer_images(hidden):
    """Materialize both source consumer images from contiguous FP32 values."""
    cp = _cp()
    _array(hidden, np.float32, 'hidden')
    b, h = cp.empty(hidden.shape, cp.uint16), cp.empty(hidden.shape, cp.float16)
    if hidden.size:
        _kernel('images')(((hidden.size+255)//256,), (256,),
                         (hidden, b, h, np.int64(hidden.size)))
    return ConsumerImages(b, h)


def dequantize_fp16(raw, kind, shape):
    """Device packed GGUF bytes -> source FP16 matrix; no FP32 GEMM substitute."""
    cp = _cp()
    if kind not in FORMATS:
        raise NotImplementedError(f'native dense dequantization does not support {kind}')
    _array(raw, np.uint8, 'raw')
    if len(shape) != 2:
        raise ValueError('weight shape must be [out,in]')
    n, k = shape
    t, qk, qb = FORMATS[kind]
    if n <= 0 or k <= 0 or k % qk or raw.size != n*(k//qk)*qb:
        raise ValueError('weight dimensions/bytes must describe complete GGUF blocks')
    out = cp.empty((n, k), cp.float16)
    groups = n*k//32
    _kernel(f'unpack<{t}>')(((groups+255)//256,), (256,),
        (raw, np.int64(k//qk*qb), np.int64(n), np.int64(k), out))
    return out


def project_matrix(x, weight, *, bf16=False):
    """Source cublasGemmEx, row-major [T,K] x [N,K] -> FP32 [T,N]."""
    cp = _cp()
    from cupy_backends.cuda.libs import cublas as cb
    dtype = np.uint16 if bf16 else np.float16
    _array(x, dtype, 'x')
    _array(weight, dtype, 'weight')
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError('x/weight must be [T,K]/[N,K] with the same K')
    t, k = x.shape
    n = weight.shape[0]
    if min(n, k) <= 0 or max(t, n, k) > np.iinfo(np.int32).max:
        raise ValueError('GEMM dimensions must fit positive int32 (T may be zero)')
    _module(cp.cuda.runtime.getDevice(), 1)
    y = cp.empty((t, n), cp.float32)
    if not t:
        return y
    alpha, beta = np.array(1, np.float32), np.array(0, np.float32)
    handle = cb.create()
    workspace = cp.empty(32*1024*1024, cp.uint8)
    try:
        cb.setStream(handle, cp.cuda.get_current_stream().ptr)
        # CuPy exposes GemmEx but not cublasSetWorkspace in this installation.
        import ctypes
        import ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library('cublas') or 'libcublas.so')
        set_workspace = lib.cublasSetWorkspace_v2
        set_workspace.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
        set_workspace.restype = ctypes.c_int
        status = set_workspace(handle, workspace.data.ptr, workspace.nbytes)
        if status:
            raise RuntimeError(f'cublasSetWorkspace status {status}')
        cb.setMathMode(handle, cb.CUBLAS_DEFAULT_MATH)
        typ = 14 if bf16 else 2  # CUDA_R_16BF / CUDA_R_16F
        args = (handle, 1, 0, n, t, k, alpha.ctypes.data,
                weight.data.ptr, typ, k, x.data.ptr, typ, k,
                beta.ctypes.data, y.data.ptr, 0, n, cb.CUBLAS_COMPUTE_32F)
        try:
            cb.gemmEx(*args, cb.CUBLAS_GEMM_DEFAULT)
        except cb.CUBLASError as exc:
            if bf16 or exc.status != 14:
                raise
            cb.gemmEx(*args, 2)  # source FP16 internal-error retry, ALGO2
    finally:
        cb.destroy(handle)
    return y


def native_project(model, projectionname, hidden):
    """Project a model.weights.store tensor, using FP16 native or BF16 resident IO.

    hidden is contiguous FP32 [...,K] or ConsumerImages with the same layout.
    Supports the packed FORMATS and resident GGUF BF16.
    Does not reuse decode projections or mutate the model's placement/caches.
    Packed uploads and dequantized matrices are per-call, not persistent.
    """
    cp = _cp()
    store = model.weights.store
    desc = store.tensors[projectionname]
    if len(desc.shape) != 2:
        raise ValueError('projection must name a matrix')
    kind = desc.quant_type
    if kind not in FORMATS and kind != 'BF16':
        raise NotImplementedError(f'unsupported native projection format {kind}')
    imgs = hidden if isinstance(hidden, ConsumerImages) else consumer_images(hidden)
    x = imgs.bf16 if kind == 'BF16' else imgs.fp16
    if x.ndim < 1 or x.shape[-1] != desc.shape[1]:
        raise ValueError('hidden width does not agree with projection')
    raw = cp.asarray(store.tensor_bytes(projectionname))
    if kind in FORMATS:
        w = dequantize_fp16(raw, kind, desc.shape)
    else:
        w = raw.view(cp.uint16).reshape(desc.shape)
    y = project_matrix(x.reshape(-1, x.shape[-1]), w, bf16=kind == 'BF16')
    return y.reshape(x.shape[:-1] + (desc.shape[0],))


def _bf16_weight(w, shape, name):
    """HC constructors may retain exact BF16 widened to float32."""
    cp = _cp()
    if getattr(w, 'dtype', None) == np.dtype(np.float32):
        _array(w, np.float32, name)
        w = consumer_images(w).bf16
    _array(w, np.uint16, name)
    if w.shape != shape:
        raise ValueError(f'{name} must have shape {shape}')
    return w


def hc_read(hc, streams, mode='bf16'):
    """Return HCRead(hidden, gains, images) for [...,4,N] FP32 streams.

    mode='bf16' is the pinned CUDA default. Low images are deliberately absent.
    gains are 2*sigmoid(inject/4), ready for the repository HC write API.
    inject=None is supported for the output HC (gains=None).
    """
    if mode != 'bf16':
        raise NotImplementedError('only CUDA default BF16 HC, BF16X2 off, is supported')
    cp = _cp()
    _array(streams, np.float32, 'streams')
    if streams.ndim < 2 or streams.shape[-2] != 4 or streams.shape[-1] <= 0:
        raise ValueError('streams must have shape [...,4,N]')
    n = streams.shape[-1]
    t = streams.size//(4*n)
    gamma = _array(hc.gamma, np.float32, 'gamma')
    if gamma.shape != (4, n):
        raise ValueError('gamma must have shape [4,N]')
    if not math.isfinite(float(hc.epsilon)) or hc.epsilon <= 0:
        raise ValueError('epsilon must be finite and positive')
    rank = hc.down.shape[0]
    if rank <= 0:
        raise ValueError('HC rank must be positive')
    down = _bf16_weight(hc.down, (rank, 4*n), 'down')
    up = _bf16_weight(hc.up, (4*n, rank), 'up')
    inj = None if hc.inject is None else _bf16_weight(hc.inject, (4, 4*n), 'inject')
    outshape = streams.shape[:-2] + (n,)
    rs, xn = cp.empty((t, 4), cp.float32), cp.empty((t, 4*n), cp.uint16)
    if t:
        _kernel('gr_norm_rs_kernel', n)((t*4,), (256,),
            (streams, gamma, np.float32(hc.epsilon), rs, xn, np.uint64(0),
             np.int64(4*n), np.int32(0)))
    low = project_matrix(xn, down, bf16=True)
    lo16 = cp.empty(low.shape, cp.uint16)
    if low.size:
        _kernel('gr_silu_kernel', n)(((low.size+255)//256,), (256,),
            (low, lo16, np.uint64(0), np.int64(low.size)))
    gate = project_matrix(lo16, up, bf16=True)
    gains = None
    if inj is not None:
        logits = project_matrix(xn, inj, bf16=True)
        gains = cp.empty((t, 4), cp.float32)
        if gains.size:
            _kernel('gains')(((gains.size+255)//256,), (256,),
                (logits, gains, np.int64(gains.size)))
        gains = gains.reshape(streams.shape[:-2] + (4,))
    hidden = cp.empty(outshape, cp.float32)
    imgs = ConsumerImages(cp.empty(outshape, cp.uint16), cp.empty(outshape, cp.float16))
    if hidden.size:
        _kernel('gr_mix_r_kernel', n)(((hidden.size+255)//256,), (256,),
            (streams, rs, gamma, gate, hidden, imgs.bf16, np.int64(t),
             imgs.fp16, np.uint64(0)))
    return HCRead(hidden, gains, imgs)
