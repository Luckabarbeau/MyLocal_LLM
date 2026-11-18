"""Patch0213 direct packed IQ GEMV and fused gate/up SwiGLU for CuPy.

Contract: ``linear(x, W) = x @ W.T``; ``swiglu = silu(x @ G.T) * (x @ U.T)``.
Only contiguous 2D/3D CuPy FP32/FP16 activations and device uint8 packed
weights are accepted. Products, reductions and the activation are FP32; only
the final output is cast to the input dtype. No Q8 activation approximation or
floating weight tensor is used. These decode-oriented kernels are not a
large-prefill GEMM implementation.

Lane equations follow quantized_weight.py and the GGML tables vendored in
 gguf_codebooks.py (llama.cpp 171e8846b4af9766c354064cb776cb34a50f053f,
MIT license). The tables' full attribution/license is embedded in the CUDA
source with their constant initializers. Strata iq_kernels.cu at commit
c77572806996349b49995e4f98d41b5294d36a21 was consulted for layout context
(in particular raw Q2_0 type 42); no Strata kernel code is copied here.

RawKernel uses CuPy's current stream directly (no Triton stream bridge).
Modules are cached per device/dtype and tables are compiled read-only device
constants: there are no per-call table transfers. Weights are duck objects
with ``data``, ``shape`` and ``quant_type``; integration owns their residency.
"""

from functools import lru_cache
import math
import operator
import os

import numpy as np

from . import gguf_codebooks as _books
from .quantized_weight import BLOCK_FORMATS

try:
    import cupy as cp
except ImportError:
    cp = None


_FORMAT_ID = {name: i for i, name in enumerate(
    ("IQ2_S", "IQ3_S", "IQ3_XXS", "IQ4_NL", "IQ4_XS", "Q2_0"))}


def supports_type(qtype):
    """Report encoding support independently of CUDA/runtime availability."""
    return isinstance(qtype, str) and qtype in _FORMAT_ID


def _codebook_storage():
    """Resolve the opt-in storage experiment without touching the CUDA runtime.

    MINI_LLM_IQ_GPU_CODEBOOK_STORAGE=global selects read-only global tables.
    The verified constant-memory implementation remains the default.
    """
    mode = os.environ.get("MINI_LLM_IQ_GPU_CODEBOOK_STORAGE", "constant")
    if mode not in ("constant", "global"):
        raise ValueError("MINI_LLM_IQ_GPU_CODEBOOK_STORAGE must be constant or global")
    return mode


@lru_cache(maxsize=2)
def _cuda_source(storage="constant"):
    if storage not in ("constant", "global"):
        raise ValueError("codebook storage must be constant or global")
    qualifier = "__constant__" if storage == "constant" else "const"
    # Use the canonical host tables only to generate compilation constants.
    license_text = "\n".join("// " + line for line in _books.__doc__.splitlines())
    tables = []
    for name in ("iq2s_grid", "iq3s_grid", "iq3xxs_grid", "ksigns_iq2xs", "kvalues_iq4nl"):
        table = getattr(_books, name).reshape(-1)
        ctype = "signed char" if table.dtype == np.int8 else "unsigned char"
        values = ",".join(str(int(v)) for v in table)
        tables.append(f"__device__ {qualifier} {ctype} {name}[{table.size}] = {{{values}}};")
    return license_text + "\n#include <cuda_fp16.h>\n" + "\n".join(tables) + r'''
__device__ __forceinline__ unsigned int u16(const unsigned char* p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8);
}
__device__ __forceinline__ unsigned int u32(const unsigned char* p) {
    return u16(p) | (u16(p + 2) << 16);
}
__device__ __forceinline__ float lane_weight(
    const unsigned char* w, int col, int format, int qk, int qb) {
    const unsigned char* b = w + (long long)(col / qk) * qb;
    const int j = col % qk;
    const float d = __half2float(__ushort_as_half((unsigned short)u16(b)));
    if (format == 5) {  // Q2_0: adjacent two-bit lanes, not bit planes.
        return d * (float)(((b[2 + j / 4] >> (2 * (j % 4))) & 3) - 1);
    }
    if (format == 3) {  // IQ4_NL: low half, then high half.
        const int v = (b[2 + j % 16] >> (4 * (j / 16))) & 15;
        return d * (float)kvalues_iq4nl[v];
    }
    const int group = j / 32;
    const int within = j % 32;
    if (format == 4) {  // IQ4_XS: signed six-bit scale per 32.
        const int scale = ((b[4 + group / 2] >> (4 * (group % 2))) & 15)
                        | (((u16(b + 2) >> (2 * group)) & 3) << 4);
        const int v = (b[8 + 16 * group + within % 16] >> (4 * (within / 16))) & 15;
        return (d * (float)(scale - 32)) * (float)kvalues_iq4nl[v];
    }
    const int lane = within / 8;
    const int sub = within % 8;
    unsigned int signs;
    float dl;
    int value;
    if (format == 0) {  // IQ2_S: 10-bit index, independent eight-bit signs.
        const int index = b[2 + 4 * group + lane]
                        | (((b[66 + group] >> (2 * lane)) & 3) << 8);
        value = iq2s_grid[index * 8 + sub];
        signs = b[34 + 4 * group + lane];
        const int scale = (b[74 + group] >> (4 * (lane / 2))) & 15;
        dl = (d * (0.5f + (float)scale)) * 0.25f;
    } else {
        const int pos = 2 + 8 * group + 2 * lane + sub / 4;
        int index = b[pos];
        if (format == 1) {  // IQ3_S: two 9-bit grids per eight values.
            index |= ((b[66 + group] >> (2 * lane + sub / 4)) & 1) << 8;
            value = iq3s_grid[index * 4 + sub % 4];
            signs = b[74 + 4 * group + lane];
            const int scale = (b[106 + group / 2] >> (4 * (group % 2))) & 15;
            dl = d * (float)(1 + 2 * scale);
        } else {  // IQ3_XXS: seven-bit parity signs and a high scale nibble.
            const unsigned int aux = u32(b + 66 + 4 * group);
            value = iq3xxs_grid[index * 4 + sub % 4];
            signs = ksigns_iq2xs[(aux >> (7 * lane)) & 127];
            dl = (d * (0.5f + (float)(aux >> 28))) * 0.5f;
        }
    }
    const float v = dl * (float)value;
    return (signs & (1u << sub)) ? -v : v;
}

// One CTA per token/output pair. Only lane-local weights live in registers.
extern "C" __global__ void packed_linear(
    const INPUT_T* x, const unsigned char* w, INPUT_T* y,
    int k, int n, int format, int qk, int qb, int tile) {
    const long long pair = blockIdx.x;
    const long long token = pair / n;
    const int row = pair % n;
    const unsigned char* wr = w + (long long)row * (k / qk) * qb;
    float acc = 0.0f;
    for (long long start = 0; start < k; start += tile) {
        for (int offset = threadIdx.x; offset < tile; offset += blockDim.x) {
            const long long col = start + offset;
            if (col < k) acc += (float)x[token * k + col] * lane_weight(wr, (int)col, format, qk, qb);
        }
    }
    __shared__ float sums[256];
    sums[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[pair] = (INPUT_T)sums[0];
}
extern "C" __global__ void packed_swiglu(
    const INPUT_T* x, const unsigned char* wg, const unsigned char* wu, INPUT_T* y,
    int k, int n, int gf, int gqk, int gqb, int uf, int uqk, int uqb, int tile) {
    const long long pair = blockIdx.x;
    const long long token = pair / n;
    const int row = pair % n;
    const unsigned char* gr = wg + (long long)row * (k / gqk) * gqb;
    const unsigned char* ur = wu + (long long)row * (k / uqk) * uqb;
    float ag = 0.0f, au = 0.0f;
    for (long long start = 0; start < k; start += tile) {
        for (int offset = threadIdx.x; offset < tile; offset += blockDim.x) {
            const long long col = start + offset;
            if (col < k) {
                const float xv = (float)x[token * k + col];
                ag += xv * lane_weight(gr, (int)col, gf, gqk, gqb);
                au += xv * lane_weight(ur, (int)col, uf, uqk, uqb);
            }
        }
    }
    __shared__ float gs[256], us[256];
    gs[threadIdx.x] = ag; us[threadIdx.x] = au;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            gs[threadIdx.x] += gs[threadIdx.x + stride];
            us[threadIdx.x] += us[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float g = gs[0];
        y[pair] = (INPUT_T)((g / (1.0f + expf(-g))) * us[0]);
    }
}
'''


def _module(device_id, dtype):
    return _cached_module(device_id, dtype, _codebook_storage())


@lru_cache(maxsize=None)
def _cached_module(device_id, dtype, storage):
    # Storage is part of the key even when callers switch mode in-process.
    with cp.cuda.Device(device_id):
        source = _cuda_source(storage).replace("INPUT_T", "half" if dtype == "float16" else "float")
        return cp.RawModule(code=source, options=("--std=c++11", "--fmad=false"))


def _validate_x(x, block_k):
    if cp is None or not isinstance(x, cp.ndarray):
        raise TypeError("x must be a CuPy device array")
    if x.dtype not in (cp.dtype("float32"), cp.dtype("float16")):
        raise TypeError("packed IQ GPU supports only float32/float16 activations")
    if x.ndim not in (2, 3) or not x.flags.c_contiguous:
        raise ValueError("Expected contiguous 2D/3D x")
    if isinstance(block_k, (bool, np.bool_)):
        raise ValueError("block_k must be a power of two in [32, 1024]")
    try:
        block_k = operator.index(block_k)
    except TypeError:
        raise ValueError("block_k must be a power of two in [32, 1024]") from None
    if not 32 <= block_k <= 1024 or block_k & (block_k - 1):
        raise ValueError("block_k must be a power of two in [32, 1024]")
    return block_k


def _validate_weight(x, weight):
    if not all(hasattr(weight, field) for field in ("data", "shape", "quant_type")):
        raise TypeError("Expected packed device weight with data/shape/quant_type")
    if not supports_type(weight.quant_type):
        raise NotImplementedError(f"No IQ packed GPU kernel for {weight.quant_type}")
    data = weight.data
    if not isinstance(data, cp.ndarray) or data.dtype != cp.uint8:
        raise TypeError("packed data must be a CuPy uint8 array")
    if not data.flags.c_contiguous:
        raise ValueError("packed data must be contiguous")
    if data.device.id != x.device.id:
        raise ValueError("x and packed weight must be on the same device")
    shape = tuple(operator.index(v) for v in weight.shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("packed weight shape must be positive [N, K]")
    n, k = shape
    qk, qb = BLOCK_FORMATS[weight.quant_type]
    if k != x.shape[-1] or k % qk:
        raise ValueError("K mismatch or invalid GGML block alignment")
    if data.size != n * (k // qk) * qb:
        raise ValueError("Invalid packed byte length")
    if max(k, n) > np.iinfo(np.int32).max or math.prod(x.shape[:-1]) * n > 2147483647:
        raise ValueError("packed kernel dimensions exceed CUDA launch/index limits")
    return shape, (_FORMAT_ID[weight.quant_type], qk, qb)


def linear(x, weight, block_k=512):
    """Direct packed GEMV, returning ``x.shape[:-1] + (weight.shape[0],)``."""
    block_k = _validate_x(x, block_k)
    (n, k), fmt = _validate_weight(x, weight)
    with cp.cuda.Device(x.device.id):
        y = cp.empty(x.shape[:-1] + (n,), dtype=x.dtype)
        if y.size:
            kernel = _module(x.device.id, x.dtype.name).get_function("packed_linear")
            args = (x, weight.data, y) + tuple(np.int32(v) for v in (k, n, *fmt, block_k))
            kernel((y.size,), (min(block_k, 256),), args)
    return y


def swiglu(x, gate, up, block_k=512):
    """One-launch mixed-format gate/up projections and FP32 SiLU product.

    No intermediate gate/up storage rounding is performed for FP16 input.
    """
    block_k = _validate_x(x, block_k)
    (n, k), gfmt = _validate_weight(x, gate)
    ushape, ufmt = _validate_weight(x, up)
    if ushape != (n, k):
        raise ValueError("gate/up shapes must match")
    with cp.cuda.Device(x.device.id):
        y = cp.empty(x.shape[:-1] + (n,), dtype=x.dtype)
        if y.size:
            kernel = _module(x.device.id, x.dtype.name).get_function("packed_swiglu")
            args = (x, gate.data, up.data, y) + tuple(
                np.int32(v) for v in (k, n, *gfmt, *ufmt, block_k))
            kernel((y.size,), (min(block_k, 256),), args)
    return y


# Patch0231 is deliberately separate from the verified single-expert kernels.
# It reuses their exact lane equations/tables and per-CTA reduction order.
_GROUPED_CUDA = r'''
extern "C" __global__ void iq_grouped_gate_up(
    const float* x, const unsigned long long* table, float* hidden,
    int d, int f, int width, int gf, int gqk, int gqb,
    int uf, int uqk, int uqb, int tile) {
    const long long pair = blockIdx.x;
    const int expert = pair / f;
    const int row = pair % f;
    const unsigned char* gr = (const unsigned char*)table[(long long)expert * width]
                           + (long long)row * (d / gqk) * gqb;
    const unsigned char* ur = (const unsigned char*)table[(long long)expert * width + 1]
                           + (long long)row * (d / uqk) * uqb;
    float ag = 0.0f, au = 0.0f;
    for (long long start = 0; start < d; start += tile) {
        for (int offset = threadIdx.x; offset < tile; offset += blockDim.x) {
            const long long col = start + offset;
            if (col < d) {
                const float xv = x[col];
                ag += xv * lane_weight(gr, (int)col, gf, gqk, gqb);
                au += xv * lane_weight(ur, (int)col, uf, uqk, uqb);
            }
        }
    }
    __shared__ float gs[256], us[256];
    gs[threadIdx.x] = ag; us[threadIdx.x] = au;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            gs[threadIdx.x] += gs[threadIdx.x + stride];
            us[threadIdx.x] += us[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float g = gs[0];
        hidden[pair] = (g / (1.0f + expf(-g))) * us[0];
    }
}
extern "C" __global__ void iq_grouped_down_weighted(
    const float* hidden, const unsigned long long* table, const float* routes,
    float* terms, int d, int f, int width, long long route_stride,
    int df, int dqk, int dqb, int tile) {
    const long long pair = blockIdx.x;
    const int expert = pair / d;
    const int row = pair % d;
    const unsigned char* wr = (const unsigned char*)table[(long long)expert * width + 2]
                           + (long long)row * (f / dqk) * dqb;
    const float* h = hidden + (long long)expert * f;
    float acc = 0.0f;
    for (long long start = 0; start < f; start += tile) {
        for (int offset = threadIdx.x; offset < tile; offset += blockDim.x) {
            const long long col = start + offset;
            if (col < f) acc += h[col] * lane_weight(wr, (int)col, df, dqk, dqb);
        }
    }
    __shared__ float sums[256];
    sums[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const long long slot = width == 4 ? table[(long long)expert * width + 3] : expert;
        // Real FP32 storage boundary: this product cannot fuse with reduction.
        terms[pair] = sums[0] * routes[slot * route_stride];
    }
}
extern "C" __global__ void iq_grouped_ordered_reduce(
    const float* terms, float* out, int d, int count) {
    const long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (col < d) {
        float acc = 0.0f;
        for (int expert = 0; expert < count; ++expert)
            acc = __fadd_rn(acc, terms[(long long)expert * d + col]);
        out[col] = acc;
    }
}
'''


def _grouped_module(device_id):
    return _cached_grouped_module(device_id, _codebook_storage())


@lru_cache(maxsize=None)
def _cached_grouped_module(device_id, storage):
    with cp.cuda.Device(device_id):
        return cp.RawModule(code=_cuda_source(storage).replace("INPUT_T", "float") + _GROUPED_CUDA,
                            options=("--std=c++11", "--fmad=false"))


def _grouped_kernel():
    mode = os.environ.get('MINI_LLM_IQ_GPU_GROUPED_KERNEL', 'block')
    if mode not in ('block', 'warp', 'warp_specialized'):
        raise ValueError('MINI_LLM_IQ_GPU_GROUPED_KERNEL must be block or warp or warp_specialized')
    return mode


def grouped(x, experts, routes, slots=None, block_k=512):
    """Three-launch FP32 [1,D] IQ expert combine on CuPy's current stream.

    Handles must be in ascending expert-ID order and share each projection's
    shape/format; repeated handles and route slots are legal. Routes may have
    positive, negative or zero stride. The caller retains weight ownership
    through stream completion. Only an E-by-3/4 uint64 pointer/slot table is
    uploaded; scratch is E*F + E*D FP32 values and the [1,D] result.

    Standalone validation shares the integration seam's metadata-only contract.
    All three projection formats must be supported by this module; legacy/IQ
    hybrid triples are deliberately unsupported here.
    """
    from .grouped_quantized_experts import _unsupported_reason

    reason = _unsupported_reason(x, experts, routes, block_k, slots)
    if reason is not None:
        raise ValueError(f"Unsupported grouped IQ experts: {reason}")
    return _grouped_validated(x, experts, routes, slots, block_k)


def _grouped_validated(x, experts, routes, slots, block_k):
    """Internal launch seam; caller must have checked _unsupported_reason."""
    scheduling = _grouped_kernel()
    projections = (experts[0].gate, experts[0].up, experts[0].down)
    if not all(supports_type(w.quant_type) for w in projections):
        raise ValueError("grouped IQ requires all-IQ projection formats")
    e, d, f = len(experts), x.shape[1], projections[0].shape[0]
    if max(e * d, e * f, d, f) > 2147483647:
        raise ValueError("grouped IQ dimensions exceed CUDA launch/index limits")
    pointers = np.asarray([[int(w.data.data.ptr) for w in (h.gate, h.up, h.down)]
                           for h in experts], dtype=np.uint64)
    width = 3 if slots is None else 4
    if slots is not None:
        pointers = np.column_stack((pointers, np.asarray(slots, dtype=np.uint64)))
    table = cp.asarray(pointers)
    hidden = cp.empty((e, f), cp.float32)
    terms = cp.empty((e, d), cp.float32)
    out = cp.empty((1, d), cp.float32)
    formats = [(_FORMAT_ID[w.quant_type], *BLOCK_FORMATS[w.quant_type]) for w in projections]
    warp = scheduling in ('warp', 'warp_specialized')
    if warp:
        from .iq_grouped_warp_gpu import module as warp_module
        specialization = ((d, f, width, *formats[0], *formats[1], *formats[2])
                          if scheduling == 'warp_specialized' else None)
        module = warp_module(x.device.id, _codebook_storage(), specialization)
        gate_up = module.get_function('iq_warp_gate_up')
        down = module.get_function('iq_warp_down_weighted')
    else:
        module = _grouped_module(x.device.id)
        gate_up = module.get_function('iq_grouped_gate_up')
        down = module.get_function('iq_grouped_down_weighted')
    reduce = module.get_function("iq_grouped_ordered_reduce")
    ints = lambda values: tuple(np.int32(v) for v in values)
    extra = ints((e,)) if warp else ()
    block = (128,) if warp else (256,)
    rows_per_block = 4 if warp else 1
    gate_up(((e*f + rows_per_block - 1)//rows_per_block,), block,
            (x, table, hidden) + ints((d, f, width, *formats[0], *formats[1], block_k)) + extra)
    down(((e*d + rows_per_block - 1)//rows_per_block,), block,
         (hidden, table, routes, terms) + ints((d, f, width))
         + (np.int64(routes.strides[0] // routes.itemsize),) + ints((*formats[2], block_k)) + extra)
    reduce(((d + 255) // 256,), (256,), (terms, out) + ints((d, e)))
    return out
