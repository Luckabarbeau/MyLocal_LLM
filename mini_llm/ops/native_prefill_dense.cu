/*
Adapted from Strata 712fec6b4297c9cfd32ab43daaa2979a9aa01edf kernels.cu and dequant_bf16.cu.
Dequant arithmetic: Copyright (c) 2023-2026 The ggml authors (MIT).

MIT License

Copyright (c) 2026 Niko1221 and the Strata contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include <cuda_fp16.h>
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
constexpr int N = NATIVE_WIDTH, HC = 4, D = 4*N;
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ uint16_t bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
// The BF16 GEMMs' second operand (STRATA_PREFILL_BF16X2): what the BF16 image `hi` left out of f, itself in BF16.
// W.hi + W.lo carries ~16 mantissa bits of the activation - the decode path's FP32 x to within ~1e-5.
__device__ __forceinline__ uint16_t bf_lo(float f, uint16_t hi) { return bf(f - __uint_as_float((uint32_t) hi << 16)); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ uint16_t hf(float f) { return __half_as_ushort(__float2half_rn(f)); }
// A SwiGLU product for an FP16 GEMM: saturated, so a token with a massive activation cannot turn into inf and then
// NaN in the down projection (decode's q8_1 has room to ~8e6; FP16 ends at 65504).  A NaN stays NaN (fminf/fmaxf
// would make it -65504 and hide where it came from); finite values below 65504 round exactly as before.
// STRATA_F16_RANGE=1 (set_f16_range): what reaches FP16's range on this device - the largest |value| the activation
// images converted (hf_sat is their one funnel) and how many were beyond 65504 or NaN; W and Y are counted in gemm.cu.
// Off, the cost is one flag read per value.  The max is kept as float bits (positive floats order as unsigned ints),
// and a thread takes the atomic only when it beats the value it read, so the atomics stop once the max has settled.
#if defined(__HIPCC__)   // HIP only: the CUDA kernels stay exactly as they were (no flag read in hf_sat)
__device__ int g_f16_range_on = 0;
__device__ unsigned g_f16_range_max = 0;
__device__ unsigned long long g_f16_range_over = 0;
__device__ __forceinline__ void f16_range_note(float f) {
    const float a = fabsf(f);
    const unsigned bits = __float_as_uint(a);
    if (bits > g_f16_range_max) atomicMax(&g_f16_range_max, bits);
    if (a > 65504.0f || isnan(f)) atomicAdd(&g_f16_range_over, 1ull);
}
#endif
__device__ __forceinline__ uint16_t hf_sat(float f) {
#if defined(__HIPCC__)
    if (g_f16_range_on) f16_range_note(f);
#endif
    return hf(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f));
}

// Turing's FP16 image of the hyper-connection's xn16 (gemm.cu's bf16_to_f16_kernel applied to a BF16 value): the
// writers store what the BF16 tensor-core path would convert, so the GEMM reads it without a conversion pass
__device__ __forceinline__ uint16_t bf_to_hf_conv(uint16_t b) {
    float f = __uint_as_float((uint32_t) b << 16);
    if (f > 65504.0f && !isinf(f)) f = 65504.0f;
    else if (f < -65504.0f && !isinf(f)) f = -65504.0f;
    return __half_as_ushort(__float2half_rn(f));
}
// The prompt path's 16-bit activation image for the BF16-weight GEMMs: BF16, or FP16 where the GEMM library is fast
// only in FP16 (prompt_f16() in gemm.cu: rocBLAS on gfx103x).  Set once per device before the first prompt.
#if defined(__HIPCC__)   // HIP only (#835): the CUDA kernels stay exactly as they were, they never read the flag
__device__ int g_act_f16 = 0;
__device__ __forceinline__ uint16_t act16(float f) { return g_act_f16 ? hf_sat(f) : bf(f); }
#else
__device__ __forceinline__ uint16_t act16(float f) { return bf(f); }
#endif
// block-wide sum for blockDim.x <= 1024, result broadcast
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (w == 0) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    return sh[0];
}
__device__ __forceinline__ float h2f(const uint8_t* p) {
    return __half2float(__ushort_as_half((uint16_t) (p[0] | (p[1] << 8))));
}
__device__ __forceinline__ uint16_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    // a NaN (a NaN scale in the block) stays a quiet NaN, as in ggml_compute_fp32_to_bf16 and `bf16_from_f32`:
    // the rounding add below would carry it into -0 or inf
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 64u);
    u += 0x7fffu + ((u >> 16) & 1u);          // round to nearest even
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ void put(uint16_t* o, int i, float v) { o[i] = f2bf(v); }
struct H16 { uint16_t v; };
__device__ __forceinline__ void put(H16* o, int i, float v) { o[i].v = __half_as_ushort(__float2half_rn(v)); }
__device__ __forceinline__ void put(float* o, int i, float v) { o[i] = v; }

__constant__ int8_t kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// One 32-element group `g` (row-major over the whole slice); `out` points at that group's 32 outputs.
template <int TYPE, typename T>
__device__ __forceinline__ void group32(const uint8_t* row_blocks, int gi_in_row, T* out) {
    if constexpr (TYPE == 42) {                                   // Q2_0: 64 per block of 18 B
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 2) * 18;
        const float d = h2f(b);
        const int e0 = (gi_in_row % 2) * 32;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int e = e0 + j;
            const int q = (b[2 + e / 4] >> ((e % 4) * 2)) & 3;
            put(out, j, (float) (q - 1) * d);
        }
    } else if constexpr (TYPE == 2) {                              // Q4_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, j, (float) ((b[2 + j] & 0x0F) - 8) * d);
            put(out, j + 16, (float) ((b[2 + j] >> 4) - 8) * d);
        }
    } else if constexpr (TYPE == 6) {                              // Q5_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 22;
        const float d = h2f(b);
        const uint32_t qh = (uint32_t) b[2] | ((uint32_t) b[3] << 8) | ((uint32_t) b[4] << 16) | ((uint32_t) b[5] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = ((qh >> j) << 4) & 0x10;
            const int xh1 = (qh >> (j + 12)) & 0x10;
            put(out, j, (float) (((b[6 + j] & 0x0F) | xh0) - 16) * d);
            put(out, j + 16, (float) (((b[6 + j] >> 4) | xh1) - 16) * d);
        }
    } else if constexpr (TYPE == 7) {                              // Q5_1
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 24;
        const float d = h2f(b), m = h2f(b + 2);
        const uint32_t qh = (uint32_t) b[4] | ((uint32_t) b[5] << 8) |
                            ((uint32_t) b[6] << 16) | ((uint32_t) b[7] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = ((qh >> j) & 1) << 4;
            const int xh1 = ((qh >> (j + 16)) & 1) << 4;
            put(out, j, (float) ((b[8 + j] & 15) | xh0) * d + m);
            put(out, j + 16, (float) ((b[8 + j] >> 4) | xh1) * d + m);
        }
    } else if constexpr (TYPE == 8) {                              // Q8_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 34;
        const float d = h2f(b);
        for (int j = 0; j < 32; ++j) put(out, j, (float) (int8_t) b[2 + j] * d);
    } else if constexpr (TYPE == 20) {                             // IQ4_NL
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, j, d * (float) kv_iq4nl[b[2 + j] & 0xf]);
            put(out, j + 16, d * (float) kv_iq4nl[b[2 + j] >> 4]);
        }
    } else if constexpr (TYPE == 11) {                             // Q3_K: hmask[32] qs[64] scales[12] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 110;
        const int gi = gi_in_row % 8, n = gi / 4, jj = gi % 4;
        const uint8_t* hm = b;
        const uint8_t* q = b + 32 + n * 32;
        const uint8_t* sc = b + 96;
        const float d_all = h2f(b + 108);
        uint32_t aux[4];
        for (int i = 0; i < 3; ++i)
            aux[i] = (uint32_t)sc[4*i] | ((uint32_t)sc[4*i+1] << 8) |
                     ((uint32_t)sc[4*i+2] << 16) | ((uint32_t)sc[4*i+3] << 24);
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
        const int shift = 2 * jj;
        const uint8_t m = (uint8_t) (1u << (n * 4 + jj));
        for (int t = 0; t < 32; ++t) {
            const int is = n * 8 + jj * 2 + (t >= 16 ? 1 : 0);
            const float dl = d_all * (float) (scales[is] - 32);
            put(out, t, dl * (float) ((int) ((q[t] >> shift) & 3) - ((hm[t] & m) ? 0 : 4)));
        }
    } else if constexpr (TYPE == 12) {                             // Q4_K: d dmin scales[12] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 144;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* q = b + 16 + 32 * j64;
        for (int l = 0; l < 32; ++l) put(out, l, d1 * (float) (hi ? (q[l] >> 4) : (q[l] & 0xF)) - m1);
    } else if constexpr (TYPE == 13) {                             // Q5_K: d dmin scales[12] qh[32] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 176;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48 + 32 * j64;
        const uint8_t u = (uint8_t) (1u << (2 * j64 + hi));
        for (int l = 0; l < 32; ++l) {
            const int nib = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
            put(out, l, d1 * (float) (nib + ((qh[l] & u) ? 16 : 0)) - m1);
        }
    } else if constexpr (TYPE == 14) {                             // Q6_K: ql[128] qh[64] scales[16] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 210;
        const int gi = gi_in_row % 8, n = gi / 4, qu = gi % 4;
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
        const float d = h2f(b + 208);
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
            if (qu == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (qu == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (qu == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            put(out, l, d * (float) sc[is + 2 * qu] * (float) (q - 32));
        }
    } else if constexpr (TYPE == 23) {                             // IQ4_XS: d scales_h scales_l[4] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 136;
        const int ib = gi_in_row % 8;
        const float d = h2f(b);
        const uint16_t scales_h = (uint16_t) (b[2] | (b[3] << 8));
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float) (ls - 32);
        const uint8_t* qs = b + 8 + 16 * ib;
        for (int j = 0; j < 16; ++j) {
            put(out, j, dl * (float) kv_iq4nl[qs[j] & 0xf]);
            put(out, j + 16, dl * (float) kv_iq4nl[qs[j] >> 4]);
        }
    }
}

__global__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                                  float* __restrict__ rs_out, uint16_t* __restrict__ xn16,
                                  uint16_t* __restrict__ xn16_lo, int64_t ldx, int f16img) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    const int64_t xo = (row / HC) * ldx + (int64_t) c * N;   // the BF16 image's row (token stride ldx)
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    for (int d = threadIdx.x; d < N; d += blockDim.x) {
        const float v = r[d] * rs * w[c * N + d];
        const uint16_t h = act16(v);
        xn16[xo + d] = f16img ? bf_to_hf_conv(h) : h;
        if (xn16_lo) xn16_lo[xo + d] = bf_lo(v, h);
    }
}
__global__ void gr_mix_r_kernel(const float* __restrict__ R, const float* __restrict__ rs, const float* __restrict__ w,
                                const float* __restrict__ g, float* __restrict__ mixed, uint16_t* __restrict__ mixed16,
                                int64_t T, uint16_t* __restrict__ mixed_h, uint16_t* __restrict__ mixed16_lo) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        const float x = R[j] * rs[t * HC + c] * w[c * N + d];   // gr_norm_kernel's value, bit for bit
        s = fmaf(x, sigm(g[j]), s);
    }
    s /= (float) HC;
    mixed[i] = s;
    if (mixed16) {
        const uint16_t h = act16(s);
        mixed16[i] = h;
        if (mixed16_lo) mixed16_lo[i] = bf_lo(s, h);
    }
    if (mixed_h) mixed_h[i] = hf(s);
}
__global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo16, uint16_t* __restrict__ lo16_lo,
                               int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = lo[i] / (float) HC;
    const float v = x / (1.0f + __expf(-x));
    const uint16_t h = act16(v);
    lo16[i] = h;
    if (lo16_lo) lo16_lo[i] = bf_lo(v, h);
}

template<int TYPE> __global__ void unpack(const uint8_t* raw, int64_t rb, int64_t rows, int64_t k, H16* out) { int64_t g=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(g<rows*(k/32)) group32<TYPE>(raw+(g/(k/32))*rb,g%(k/32),out+g*32); }
extern "C" __global__ void images(const float* x, uint16_t* b, uint16_t* h, int64_t n) { int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) {b[i]=bf(x[i]);h[i]=hf(x[i]);} }
extern "C" __global__ void gains(const float* x,float* y,int64_t n) {int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=2.0f*sigm(x[i]/4.0f);}
