"""Experimental tiled BF16 fused local-attention forward for CUDA.

The kernel is deliberately narrow: D=64, effective causal window <=512,
positive input strides with contiguous D, FP32 output and FP32 LSE. It is
opt-in through ``MINI_LLM_TRAIN_ATTN_BACKEND=fused`` and never replaces the
validated grouped-cuBLAS default implicitly.
"""

from __future__ import annotations

import os

import numpy as np

from ..backend import BACKEND_NAME, is_bfloat16_dtype, xp


_FUSED_LOCAL_MODULE = None
_FUSED_LOCAL_KERNELS = None
_FUSED_LOCAL_DISABLED = False
_FALSE = {"", "0", "false", "off", "no"}


def fused_local_requested():
    return os.environ.get("MINI_LLM_TRAIN_ATTN_BACKEND", "grouped").strip().lower() == "fused"


def fused_local_strict():
    return os.environ.get("MINI_LLM_TRAIN_ATTN_STRICT", "0").strip().lower() not in _FALSE


def fused_local_supported(q, k, v, window, kv_map, *, output_dtype):
    if BACKEND_NAME != "cupy":
        return False, "fused local attention requires the CuPy backend"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False, "Q/K/V must have rank four"
    if k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        return False, "fused local attention requires matching self-attention shapes"
    if not all(is_bfloat16_dtype(array.dtype) for array in (q, k, v)):
        return False, "fused local attention requires BF16 Q/K/V"
    if int(q.shape[-1]) != 64:
        return False, "fused local attention currently requires D=64"
    if int(window) < 1 or int(window) > 512:
        return False, "fused local attention currently supports windows up to 512"
    if np.dtype(output_dtype) != np.dtype("float32"):
        return False, "fused local attention currently produces FP32 context"
    itemsize = int(q.dtype.itemsize)
    for name, array in (("Q", q), ("K", k), ("V", v)):
        strides = tuple(int(stride) for stride in array.strides)
        if strides[-1] != itemsize:
            return False, f"{name} must have contiguous D elements"
        if any(stride <= 0 or stride % itemsize for stride in strides):
            return False, f"{name} must have positive element-aligned strides"
    if getattr(kv_map, "ndim", None) != 1 or int(kv_map.shape[0]) != int(q.shape[2]):
        return False, "kv_map must have one entry per supplied Q head"
    if str(kv_map.dtype) != "int64":
        return False, "kv_map must use the normalized int64 device representation"
    return True, "supported"


def _fused_tile_n():
    raw = os.environ.get("MINI_LLM_FUSED_LOCAL_TILE_N", "16").strip()
    try:
        tile_n = int(raw)
    except ValueError as exc:
        raise ValueError("MINI_LLM_FUSED_LOCAL_TILE_N must be 16 or 32") from exc
    if tile_n not in (16, 32):
        raise ValueError("MINI_LLM_FUSED_LOCAL_TILE_N must be 16 or 32")
    return tile_n


def _get_fused_local_kernels():
    global _FUSED_LOCAL_MODULE, _FUSED_LOCAL_KERNELS, _FUSED_LOCAL_DISABLED
    if BACKEND_NAME != "cupy" or _FUSED_LOCAL_DISABLED:
        return None
    if _FUSED_LOCAL_KERNELS is not None:
        return _FUSED_LOCAL_KERNELS

    code = r"""
    #include <cuda_bf16.h>
    #include <mma.h>
    using namespace nvcuda;

    #define NEG_INF_F __int_as_float((int)0xff800000u)

    static constexpr int BM = 32;
    static constexpr int DH = 64;

    template<int BN>
    __global__ void fused_local_fwd_bm32_d64(
        const __nv_bfloat16* Q,
        const __nv_bfloat16* K,
        const __nv_bfloat16* V,
        const long long* kv_map,
        float* O,
        float* LSE,
        int B,
        int T,
        int Hq,
        int Hkv,
        int window,
        float scale,
        long long q_sb,
        long long q_st,
        long long q_sh,
        long long k_sb,
        long long k_st,
        long long k_sh,
        long long v_sb,
        long long v_st,
        long long v_sh) {
        const int tid = (int)threadIdx.x;
        const int warp = tid >> 5;
        const int b = (int)blockIdx.z;
        const int h = (int)blockIdx.y;
        const int q0 = (int)blockIdx.x * BM;
        if (b >= B || h >= Hq || warp >= 2) return;

        const int kvh = (int)kv_map[h];
        if (kvh < 0 || kvh >= Hkv) return;

        __shared__ __align__(32) __nv_bfloat16 sq[BM][DH];
        __shared__ __align__(32) __nv_bfloat16 sk[BN][DH];
        __shared__ __align__(32) __nv_bfloat16 sv[BN][DH];
        __shared__ __align__(32) __nv_bfloat16 sp[BM][BN];
        __shared__ __align__(32) float ss[BM][BN];
        __shared__ __align__(32) float so[BM][DH];
        __shared__ float sm[BM];
        __shared__ float sl[BM];
        __shared__ float sa[BM];

        for (int index = tid; index < BM * DH; index += blockDim.x) {
            const int row = index / DH;
            const int d = index - row * DH;
            const int qi = q0 + row;
            sq[row][d] = qi < T
                ? Q[(long long)b * q_sb + (long long)qi * q_st
                    + (long long)h * q_sh + d]
                : __float2bfloat16_rn(0.0f);
            so[row][d] = 0.0f;
        }
        if (tid < BM) {
            sm[tid] = NEG_INF_F;
            sl[tid] = 0.0f;
            sa[tid] = 1.0f;
        }
        __syncthreads();

        const int q1 = min(T, q0 + BM);
        const int first_key = max(0, q0 - window + 1);
        const int k_begin = (first_key / BN) * BN;

        for (int k0 = k_begin; k0 < q1; k0 += BN) {
            for (int index = tid; index < BN * DH; index += blockDim.x) {
                const int row = index / DH;
                const int d = index - row * DH;
                const int kj = k0 + row;
                if (kj < T) {
                    sk[row][d] = K[(long long)b * k_sb + (long long)kj * k_st
                                      + (long long)kvh * k_sh + d];
                    sv[row][d] = V[(long long)b * v_sb + (long long)kj * v_st
                                      + (long long)kvh * v_sh + d];
                } else {
                    sk[row][d] = __float2bfloat16_rn(0.0f);
                    sv[row][d] = __float2bfloat16_rn(0.0f);
                }
            }
            __syncthreads();

            #pragma unroll
            for (int n0 = 0; n0 < BN; n0 += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16,
                               __nv_bfloat16, wmma::row_major> q_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16,
                               __nv_bfloat16, wmma::col_major> k_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> score_frag;
                wmma::fill_fragment(score_frag, 0.0f);
                #pragma unroll
                for (int d0 = 0; d0 < DH; d0 += 16) {
                    wmma::load_matrix_sync(q_frag, &sq[warp * 16][d0], DH);
                    wmma::load_matrix_sync(k_frag, &sk[n0][d0], DH);
                    wmma::mma_sync(score_frag, q_frag, k_frag, score_frag);
                }
                wmma::store_matrix_sync(&ss[warp * 16][n0], score_frag, BN,
                                        wmma::mem_row_major);
            }
            __syncthreads();

            if (tid < BM) {
                const int qi = q0 + tid;
                float tile_max = NEG_INF_F;
                if (qi < T) {
                    #pragma unroll
                    for (int n = 0; n < BN; ++n) {
                        const int kj = k0 + n;
                        const bool valid = kj < T && kj <= qi
                            && kj >= qi - window + 1;
                        const float value = valid ? ss[tid][n] * scale : NEG_INF_F;
                        ss[tid][n] = value;
                        tile_max = fmaxf(tile_max, value);
                    }
                } else {
                    #pragma unroll
                    for (int n = 0; n < BN; ++n) ss[tid][n] = NEG_INF_F;
                }

                if (tile_max == NEG_INF_F) {
                    sa[tid] = 1.0f;
                    #pragma unroll
                    for (int n = 0; n < BN; ++n)
                        sp[tid][n] = __float2bfloat16_rn(0.0f);
                } else {
                    const float old_max = sm[tid];
                    const float new_max = fmaxf(old_max, tile_max);
                    const float alpha = old_max == NEG_INF_F
                        ? 0.0f : expf(old_max - new_max);
                    float tile_sum = 0.0f;
                    #pragma unroll
                    for (int n = 0; n < BN; ++n) {
                        const float probability = ss[tid][n] == NEG_INF_F
                            ? 0.0f : expf(ss[tid][n] - new_max);
                        tile_sum += probability;
                        sp[tid][n] = __float2bfloat16_rn(probability);
                    }
                    sm[tid] = new_max;
                    sl[tid] = sl[tid] * alpha + tile_sum;
                    sa[tid] = alpha;
                }
            }
            __syncthreads();

            for (int index = tid; index < BM * DH; index += blockDim.x) {
                const int row = index / DH;
                so[row][index - row * DH] *= sa[row];
            }
            __syncthreads();

            #pragma unroll
            for (int d0 = 0; d0 < DH; d0 += 16) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> out_frag;
                wmma::load_matrix_sync(out_frag, &so[warp * 16][d0], DH,
                                       wmma::mem_row_major);
                #pragma unroll
                for (int n0 = 0; n0 < BN; n0 += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16,
                                   __nv_bfloat16, wmma::row_major> p_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16,
                                   __nv_bfloat16, wmma::row_major> v_frag;
                    wmma::load_matrix_sync(p_frag, &sp[warp * 16][n0], BN);
                    wmma::load_matrix_sync(v_frag, &sv[n0][d0], DH);
                    wmma::mma_sync(out_frag, p_frag, v_frag, out_frag);
                }
                wmma::store_matrix_sync(&so[warp * 16][d0], out_frag, DH,
                                        wmma::mem_row_major);
            }
            __syncthreads();
        }

        for (int index = tid; index < BM * DH; index += blockDim.x) {
            const int row = index / DH;
            const int d = index - row * DH;
            const int qi = q0 + row;
            if (qi < T) {
                O[(((long long)b * T + qi) * Hq + h) * DH + d] = so[row][d] / sl[row];
            }
        }
        if (tid < BM) {
            const int qi = q0 + tid;
            if (qi < T) {
                LSE[((long long)b * T + qi) * Hq + h] = sm[tid] + logf(sl[tid]);
            }
        }
    }
    """

    try:
        names = (
            "fused_local_fwd_bm32_d64<16>",
            "fused_local_fwd_bm32_d64<32>",
        )
        module = xp.RawModule(
            code=code,
            backend="nvrtc",
            options=("--std=c++17", "--gpu-architecture=compute_120"),
            name_expressions=names,
        )
        kernels = {
            16: module.get_function(names[0]),
            32: module.get_function(names[1]),
        }
    except Exception:
        if fused_local_strict():
            raise
        _FUSED_LOCAL_DISABLED = True
        return None
    _FUSED_LOCAL_MODULE = module
    _FUSED_LOCAL_KERNELS = kernels
    return kernels


def fused_local_forward(q, k, v, window, scale, kv_map, *, output_dtype):
    supported, reason = fused_local_supported(
        q, k, v, window, kv_map, output_dtype=output_dtype
    )
    if not supported:
        if fused_local_strict():
            raise RuntimeError(reason)
        return None
    kernels = _get_fused_local_kernels()
    if kernels is None:
        if fused_local_strict():
            raise RuntimeError("fused local CUDA module is unavailable")
        return None
    tile_n = _fused_tile_n()
    kernel = kernels[tile_n]

    batch, length, n_q_heads, _ = map(int, q.shape)
    n_kv_heads = int(k.shape[2])
    output = xp.empty(q.shape, dtype=xp.float32)
    lse = xp.empty((batch, length, n_q_heads), dtype=xp.float32)
    q_strides = tuple(int(stride // q.dtype.itemsize) for stride in q.strides)
    k_strides = tuple(int(stride // k.dtype.itemsize) for stride in k.strides)
    v_strides = tuple(int(stride // v.dtype.itemsize) for stride in v.strides)
    kernel(
        ((length + 31) // 32, n_q_heads, batch),
        (64,),
        (
            q,
            k,
            v,
            kv_map,
            output,
            lse,
            np.int32(batch),
            np.int32(length),
            np.int32(n_q_heads),
            np.int32(n_kv_heads),
            np.int32(window),
            np.float32(scale),
            np.int64(q_strides[0]),
            np.int64(q_strides[1]),
            np.int64(q_strides[2]),
            np.int64(k_strides[0]),
            np.int64(k_strides[1]),
            np.int64(k_strides[2]),
            np.int64(v_strides[0]),
            np.int64(v_strides[1]),
            np.int64(v_strides[2]),
        ),
    )
    return output, lse


def fused_local_kernel_attributes():
    kernels = _get_fused_local_kernels()
    if kernels is None:
        return None
    tile_n = _fused_tile_n()
    return {"tile_n": tile_n, **dict(kernels[tile_n].attributes)}
