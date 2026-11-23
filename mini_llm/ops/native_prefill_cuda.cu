/*
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

Port of Strata 712fec6b4297c9cfd32ab43daaa2979a9aa01edf
moe_fused_iq.cu:60-880 (normal sm80+ INT8 path), moe_fused.cu:117-185.
Local adaptations: geometry macros, device pointer table, codebook globals,
flat grouping arguments. No external headers or runtime.
*/
#include <cuda_fp16.h>
// NVRTC has no host libc include path. CUDA device ABI uses these widths.
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned long long uintptr_t;
constexpr int kMoeEmbd = NATIVE_EMBD, kMoeFF = NATIVE_FF;
constexpr int kMoeGuRows = 2*kMoeFF, kTileRows = 64;
constexpr int GGML_TYPE_IQ2_XXS = 0;
constexpr int GGML_TYPE_IQ2_XS = 1;
constexpr int GGML_TYPE_IQ2_S = 2;
constexpr int GGML_TYPE_IQ3_XXS = 3;
constexpr int GGML_TYPE_IQ3_S = 4;
constexpr int GGML_TYPE_IQ4_XS = 5;
constexpr int GGML_TYPE_IQ4_NL = 6;
constexpr int GGML_TYPE_Q2_0 = 7;
constexpr int GGML_TYPE_Q4_K = 8;
constexpr int GGML_TYPE_Q5_K = 9;
constexpr int GGML_TYPE_Q5_0 = 10;
constexpr int GGML_TYPE_Q5_1 = 11;
constexpr int GGML_TYPE_Q8_0 = 12;
extern "C" {
__device__ uint32_t iq2s_grid[2048];
__device__ uint32_t iq3s_grid[512];
__device__ uint32_t iq3xxs_grid[256];
__device__ int8_t kvalues_iq4nl[16];
}
constexpr int AB = 80;                  // activation bytes per 64 values: 64 codes, {d0, d1}, 8 unused
constexpr int MAGIC = 0x4B400000;       // the bits of 1.5 * 2^23
constexpr float MAGICF = 12582912.0f;
// 16 warps, each 64 weight rows x 16 routed rows; WW of them along the weight rows and 16 / WW along the routed rows: a
// work item is (WR = 64 WW weight rows) x (TR = 256 / WW routed rows of one expert) - 256 x 64 or 128 x 128, chosen
// per launch by the rows per expert (pick_ww)
constexpr int THREADS = 512;
constexpr int WLD = 80;                 // bytes per decoded weight row of a stage: 64 int8 + 16 (no bank conflicts)
__host__ __device__ constexpr int tile_rows(int ww, int nt = THREADS) { return nt / 32 / ww * 16; }
__host__ __device__ constexpr int weight_rows(int ww) { return 64 * ww; }
constexpr int ASTAGES = 4;              // cp.async depth of the activations (64 values of K a stage)
// Turing: 2 (its copies are plain loads, so depth buys nothing, and 4 does not fit its 64 KB of shared memory)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
constexpr int ASTAGES_DEV = 2;
constexpr bool T75_DEV = true;    // the Q2_0 stage epilogue in native_kernel: fewer instructions, its sums in another order
#else
constexpr int ASTAGES_DEV = ASTAGES;
constexpr bool T75_DEV = false;
#endif
constexpr int GU_ROWS_K = kMoeEmbd, D_ROWS_K = kMoeFF;   // K of gate/up (n_embd) and of down (n_ff): the compiled geometry's (2560 and 640 for Flash-Next)
// the weight rows of a product: gate/up 2 n_ff (1280), down n_embd (2560)
constexpr int GU_ROWS_N = kMoeGuRows, D_ROWS_N = kMoeEmbd;
constexpr int H_ROW_BYTES = D_ROWS_K / 64 * AB;   // bytes of a row of H in the int8 activation layout (800 for Flash-Next)
static_assert(GU_ROWS_K % 256 == 0 && D_ROWS_K % 64 == 0, "the native expert kernels: K of gate/up is whole 256-value super-blocks, K of down whole 64-value stages");

// the formats (ggml type ids)
constexpr int T_IQ2_XXS = GGML_TYPE_IQ2_XXS, T_IQ2_XS = GGML_TYPE_IQ2_XS, T_IQ2_S = GGML_TYPE_IQ2_S,
              T_IQ3_XXS = GGML_TYPE_IQ3_XXS, T_IQ3_S = GGML_TYPE_IQ3_S, T_IQ4_XS = GGML_TYPE_IQ4_XS,
              T_IQ4_NL = GGML_TYPE_IQ4_NL, T_Q2_0 = GGML_TYPE_Q2_0,
              // Aurora (opt-in, HIP gfx11 only): Unsloth UD-Q4_K_XL's experts - gate/up Q4_K (one layer Q5_K), down Q5_1
              // (five layers Q8_0).  Their weights are w = d_w q + m_w (a minimum), so a sub-block's dot is
              //   sum w x = d_x (d_w sum q a + m_w sum a):
              // the activation blocks carry sum a per 32 values in their eight spare bytes (quant_act_nat_kernel and
              // the H epilogue write it for every path; no other format reads it)
              T_Q4_K = GGML_TYPE_Q4_K, T_Q5_K = GGML_TYPE_Q5_K, T_Q5_0 = GGML_TYPE_Q5_0, T_Q5_1 = GGML_TYPE_Q5_1, T_Q8_0 = GGML_TYPE_Q8_0;
// block bytes, scale per 16 values, codebook bytes in shared memory
__host__ __device__ constexpr int block_bytes(int t) {
    return t == T_IQ2_XXS ? 66 : t == T_IQ2_XS ? 74 : t == T_IQ2_S ? 82 : t == T_IQ3_XXS ? 98 : t == T_IQ3_S ? 110
         : t == T_IQ4_XS ? 136 : t == T_Q4_K ? 144 : t == T_Q5_K ? 176 : t == T_Q5_0 ? 22 : t == T_Q5_1 ? 24 : t == T_Q8_0 ? 34 : 18;
}
// the formats with a minimum (w = d q + m), and the raw words a sub-block's load stage holds
__host__ __device__ constexpr bool has_min(int t) { return t == T_Q4_K || t == T_Q5_K || t == T_Q5_1; }
__host__ __device__ constexpr int raw_words(int t) {
    return t == T_Q4_K ? 10 : t == T_Q5_K ? 18 : t == T_Q5_0 ? 6 : t == T_Q5_1 ? 6 : t == T_Q8_0 ? 9 : 5;
}
__host__ __device__ constexpr bool per16(int t) { return t == T_IQ2_XS || t == T_IQ2_S; }
__host__ __device__ constexpr int grid_bytes(int t) {
    return t == T_IQ2_XXS ? 256 * 8 : t == T_IQ2_XS ? 512 * 8 : t == T_IQ2_S ? 1024 * 8 : t == T_IQ3_XXS ? 256 * 4
         : t == T_IQ3_S ? 512 * 4 : 0;
}
constexpr int WLDH = 144;               // F16: bytes per decoded weight row of a stage: 64 halves + 16 (ldmatrix rows apart)
__host__ __device__ constexpr size_t smem_bytes(int t, int ww, int astages = ASTAGES, int nt = THREADS, bool f16 = false) {
    return (size_t) 2 * weight_rows(ww) * (f16 ? WLDH : WLD + 16) + (size_t) astages * tile_rows(ww, nt) * AB +
           (size_t) tile_rows(ww, nt) * 4 + grid_bytes(t);
}

// moe_fused.cu's grouping tables (the same layout: fused::group writes them)
struct Tables {
    int32_t *cnt, *off, *fill, *ts;
    int2* tiles;
};
__device__ size_t tiles_at(int n_expert) { return ((size_t) (4 * n_expert + 2) * 4 + 15) / 16 * 16; }
__device__ Tables tables(void* scratch, int n_expert) {
    int32_t* p = (int32_t*) scratch;
    return {p, p + n_expert, p + 2 * n_expert + 1, p + 3 * n_expert + 1,
            (int2*) ((uint8_t*) scratch + tiles_at(n_expert))};
}

// ---- activations in natural order: one warp per 64 values
extern "C" __global__ void quant_act_nat_kernel(const float* __restrict__ x, int64_t nblk, uint8_t* __restrict__ xa) {
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (w >= nblk) return;
    const float v0 = x[w * 64 + lane], v1 = x[w * 64 + 32 + lane];
    float a0 = fabsf(v0), a1 = fabsf(v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o));
    }
    uint8_t* out = xa + w * AB;
    const int c0 = a0 > 0.0f ? __float2int_rn(v0 * (127.0f / a0)) : 0, c1 = a1 > 0.0f ? __float2int_rn(v1 * (127.0f / a1)) : 0;
    out[lane] = (uint8_t) (int8_t) c0;
    out[32 + lane] = (uint8_t) (int8_t) c1;
    int s0 = c0, s1 = c1;   // the codes' sums per 32 values (the formats with a minimum read them)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, o);
        s1 += __shfl_xor_sync(0xffffffffu, s1, o);
    }
    if (lane == 0) *(float4*) (out + 64) = make_float4(a0 / 127.0f, a1 / 127.0f, (float) s0, (float) s1);
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p)); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// d = MAGIC + A (16 x 32 s8, row) * B (32 x 8 s8, col): the int32 dot with the magic bias already added (the C
// operand), so as_float(d) - 1.5 * 2^23 is the dot as a float
__device__ __forceinline__ void mma32(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(MAGIC));
}
// the same at m16n8k16 (A 16 x 16, B 16 x 8)
__device__ __forceinline__ void mma16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(MAGIC));
}
__device__ __forceinline__ float dotf(int d) { return __int_as_float(d) - MAGICF; }
// F16 (STRATA_PF_FUSED_F16): d += A (16 x 16 f16, row) * B (16 x 8 f16, col), FP32 accumulators in place
__device__ __forceinline__ void mma16h(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// four int8 (bytes 0-3) times a scale as two half2: lo = bytes 0, 1, hi = bytes 2, 3.  The flipped sign bit makes
// x + 128 the mantissa of 1024 (exact), minus 1152 is x, then one packed multiply.
__device__ __forceinline__ void i8x4_h2(uint32_t w, __half2 sc, uint32_t& lo, uint32_t& hi) {
    const uint32_t c = w ^ 0x80808080u;
    const __half2 m = __floats2half2_rn(1152.0f, 1152.0f);
    const uint32_t l = __byte_perm(c, 0x64646464u, 0x4140), h = __byte_perm(c, 0x64646464u, 0x4342);
    const __half2 lv = __hmul2(__hsub2(*(const __half2*) &l, m), sc), hv = __hmul2(__hsub2(*(const __half2*) &h, m), sc);
    lo = *(const uint32_t*) &lv;
    hi = *(const uint32_t*) &hv;
}
// four 8 x 16-byte matrices, row addresses from lanes 0-7, 8-15, 16-23, 24-31: lane (g, t) gets bytes 4t..4t+3 of row g
// of each - an mma fragment
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p))
                 : "memory");
}
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ float half_at(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }
// llama.cpp's sign unpacking: 7 bits of signs, the 8th their parity (bit 7 of v may be anything)
__device__ __forceinline__ uint32_t unpack_ksigns(uint32_t v) {
    v &= 0xFF;
    const uint32_t p = __popc(v) & 1;
    return (v ^ p << 7) * 0x01010101u;
}
// 8 bytes of a codebook entry with the sign byte `s` (broadcast) applied: bits 0-3 to .x, 4-7 to .y
__device__ __forceinline__ void signed8(uint32_t gx, uint32_t gy, uint32_t s, uint32_t& qx, uint32_t& qy) {
    const uint32_t m0 = __vcmpne4(s & 0x08040201u, 0), m1 = __vcmpne4(s & 0x80402010u, 0);
    qx = __vsub4(gx ^ m0, m0);
    qy = __vsub4(gy ^ m1, m1);
}
#if defined(__HIPCC__)
// gfx11 (Aurora): the signs from a shared-memory table instead of the compare / subtract emulation.  An entry (one per
// sign byte b; for the 7-bit sign formats per 7 bits, the parity bit folded in) is {m0, c0, m1, c1}: m = 0xFF in the
// bytes whose sign bit is set (bits 0-3 for .x, 4-7 for .y), c = m & 0x01010101.  q = (g ^ m) + c is -g where the sign
// is set - the two's complement, byte-wise - and g elsewhere; no byte carries into the next, because no codebook entry
// has a zero byte (256 - g with g in 1..127; the grids' bytes are 1..62), so it is the bytes signed8 gives.
__device__ __forceinline__ void signed8t(uint32_t gx, uint32_t gy, const uint4 sg, uint32_t& qx, uint32_t& qy) {
    qx = (gx ^ sg.x) + sg.y;
    qy = (gy ^ sg.z) + sg.w;
}
__host__ __device__ constexpr int sign_entries(int t) { return t == T_IQ2_S || t == T_IQ3_S ? 256 : 128; }
__device__ __forceinline__ uint4 sign_entry(int i, bool parity) {
    uint32_t b = (uint32_t) i;
    if (parity) b ^= (__popc(b) & 1) << 7;
    const uint32_t m0 = __vcmpne4(((b & 15) * 0x01010101u) & 0x08040201u, 0),
                   m1 = __vcmpne4(((b >> 4) * 0x01010101u) & 0x08040201u, 0);
    return make_uint4(m0, m0 & 0x01010101u, m1, m1 & 0x01010101u);
}
#endif
// 8 nibbles of q4 through a 16-entry int8 table (4 words): the low nibbles' values in .x, the high ones' in .y
// W (X5): direct-selector form on gfx: p = low 3 bits of each nibble picks within the table half, bit 3 picks the half;
// exhaustively bit-exact over all 2^32 inputs (X5). Disable with -DSTRATA_W_NO_T16.
#if defined(__HIPCC__) && !defined(STRATA_W_NO_T16)
__device__ __forceinline__ uint32_t table16_one(uint32_t x, const uint32_t (&t)[4]) {
    const uint32_t p = x & 0x07070707u;
    const uint32_t a = __builtin_amdgcn_perm(t[1], t[0], p), b = __builtin_amdgcn_perm(t[3], t[2], p);
    return __builtin_amdgcn_perm(b, a, ((x >> 1) & 0x04040404u) | 0x03020100u);
}
__device__ __forceinline__ void table16(uint32_t q4, const uint32_t (&t)[4], uint32_t& lo, uint32_t& hi) {
    lo = table16_one(q4, t);
    hi = table16_one(q4 >> 4, t);
}
#else
__device__ __forceinline__ void table16(uint32_t q4, const uint32_t (&t)[4], uint32_t& lo, uint32_t& hi) {
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t l = __byte_perm(t[0], t[1], q4 >> sh), h = __byte_perm(t[2], t[3], q4 >> sh);
        tmp[i] = __byte_perm(l, h, sel >> sh);
    }
    lo = __byte_perm(tmp[0], tmp[1], 0x6420);
    hi = __byte_perm(tmp[0], tmp[1], 0x7531);
}
#endif

// Raw bytes of sub-block `ib` of the block at `bp` (IQ: the 256-value super-block, ib 0..7; Q2_0: the 64-value block,
// ib 0..1; IQ4_NL: the 32-value block).
// 32 bits at a 2-byte aligned address (one load when it is 4-byte aligned)
__device__ __forceinline__ uint32_t ld32a(const uint8_t* p) {
    return ((uintptr_t) p & 3) == 0 ? *(const uint32_t*) p : ld32(p);
}
template <int T, int NW> __device__ __forceinline__ void load_unit(const uint8_t* bp, int ib, uint32_t (&w)[NW]) {
    static_assert(NW >= raw_words(T), "raw words");
    if constexpr (T == T_IQ2_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    } else if constexpr (T == T_IQ2_XS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16);
    } else if constexpr (T == T_IQ2_S) {
        w[0] = ld32(bp + 2 + 4 * ib); w[1] = ld32(bp + 34 + 4 * ib);
        w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) bp[74 + ib] << 24);
    } else if constexpr (T == T_IQ3_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 66 + 4 * ib); w[3] = ld16(bp);
    } else if constexpr (T == T_IQ3_S) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 74 + 4 * ib);
        w[3] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) ((bp[106 + ib / 2] >> (4 * (ib & 1))) & 15) << 24);
    } else if constexpr (T == T_IQ4_XS) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 8 + 16 * ib + 4 * k);
        const uint32_t ls = ((bp[4 + ib / 2] >> (4 * (ib & 1))) & 15) | (((ld16(bp + 2) >> (2 * ib)) & 3) << 4);
        w[4] = ld16(bp) | (ls << 16);
    } else if constexpr (T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 2 + 4 * k);
        w[4] = ld16(bp);
    } else if constexpr (T == T_Q4_K || T == T_Q5_K) {
        // qs: 32 bytes of the sub-block pair (ib / 2), the nibble chosen in convert; w[8] = d | dmin << 16; w[9]: the
        // scale/min bytes (get_scale_min_k4's: ib < 4: sc[ib], sc[ib + 4]; else sc[ib + 4], sc[ib - 4], sc[ib]), ib << 24
        constexpr int QS = T == T_Q4_K ? 16 : 48;
        const uint8_t* q = bp + QS + 32 * (ib >> 1);
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = ld32a(q + 4 * k);
        w[8] = ld32a(bp);
        const uint8_t* sc = bp + 4;
        w[9] = ib < 4 ? (uint32_t) sc[ib] | ((uint32_t) sc[ib + 4] << 8)
                      : (uint32_t) sc[ib + 4] | ((uint32_t) sc[ib - 4] << 8) | ((uint32_t) sc[ib] << 16);
        w[9] |= (uint32_t) ib << 24;
        if constexpr (T == T_Q5_K) {
#pragma unroll
            for (int k = 0; k < 8; ++k) w[10 + k] = ld32a(bp + 16 + 4 * k);   // qh
        }
    } else if constexpr (T == T_Q5_0) {   // d, qh, qs[16] (22 bytes: 2-byte aligned only)
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 6 + 4 * k);
        w[4] = ld32(bp + 2);
        w[5] = ld16(bp);
    } else if constexpr (T == T_Q5_1) {   // d, m, qh, qs[16]
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32a(bp + 8 + 4 * k);
        w[4] = ld32a(bp + 4);
        w[5] = ld32a(bp);
    } else if constexpr (T == T_Q8_0) {   // d, qs[32]
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = ld32(bp + 2 + 4 * k);
        w[8] = ld16(bp);
    } else {   // Q2_0
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    }
}

// load_unit's bytes in two passes (native_kernel's opt-in NT = 256 instances only, CUDA): `issue` loads the 16-bit and
// 8-bit pieces into registers and `take` assembles the words from them a stage later, so no instruction waits on the
// loads in the stage that issues them (in one pass the shifts and ORs right after the loads stalled every stage on a
// memory round trip).  The same bytes in the same words as load_unit.
constexpr int NP = 12;   // pieces: at most IQ4_XS's 10 halves and a byte
struct LdIssue {
    uint32_t* pc; int k;
    __device__ __forceinline__ uint32_t h(const uint8_t* p) { pc[k++] = *(const uint16_t*) p; return 0; }
    __device__ __forceinline__ uint32_t b(const uint8_t* p) { pc[k++] = *p; return 0; }
};
struct LdTake {
    const uint32_t* pc; int k;
    __device__ __forceinline__ uint32_t h(const uint8_t*) { return pc[k++]; }
    __device__ __forceinline__ uint32_t b(const uint8_t*) { return pc[k++]; }
};
template <class L> __device__ __forceinline__ uint32_t lw(L& l, const uint8_t* p) {
    const uint32_t lo = l.h(p);
    const uint32_t hi = l.h(p + 2);
    return lo | (hi << 16);
}
template <int T, class L, int NW> __device__ __forceinline__ void load_unit_l(L& l, const uint8_t* bp, int ib, uint32_t (&w)[NW]) {
    static_assert(NW >= raw_words(T), "raw words");
    if constexpr (T == T_IQ2_XXS || T == T_Q2_0) {
        w[0] = lw(l, bp + 2 + 8 * ib); w[1] = lw(l, bp + 6 + 8 * ib); w[2] = l.h(bp);
    } else if constexpr (T == T_IQ2_XS) {
        w[0] = lw(l, bp + 2 + 8 * ib); w[1] = lw(l, bp + 6 + 8 * ib);
        const uint32_t d = l.h(bp); const uint32_t s = l.b(bp + 66 + ib);
        w[2] = d | (s << 16);
    } else if constexpr (T == T_IQ2_S) {
        w[0] = lw(l, bp + 2 + 4 * ib); w[1] = lw(l, bp + 34 + 4 * ib);
        const uint32_t d = l.h(bp); const uint32_t qh = l.b(bp + 66 + ib); const uint32_t sc = l.b(bp + 74 + ib);
        w[2] = d | (qh << 16) | (sc << 24);
    } else if constexpr (T == T_IQ3_XXS) {
        w[0] = lw(l, bp + 2 + 8 * ib); w[1] = lw(l, bp + 6 + 8 * ib); w[2] = lw(l, bp + 66 + 4 * ib); w[3] = l.h(bp);
    } else if constexpr (T == T_IQ3_S) {
        w[0] = lw(l, bp + 2 + 8 * ib); w[1] = lw(l, bp + 6 + 8 * ib); w[2] = lw(l, bp + 74 + 4 * ib);
        const uint32_t d = l.h(bp); const uint32_t qh = l.b(bp + 66 + ib); const uint32_t sc = l.b(bp + 106 + ib / 2);
        w[3] = d | (qh << 16) | (((sc >> (4 * (ib & 1))) & 15) << 24);
    } else if constexpr (T == T_IQ4_XS) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = lw(l, bp + 8 + 16 * ib + 4 * k);
        const uint32_t lo = l.b(bp + 4 + ib / 2); const uint32_t hi = l.h(bp + 2); const uint32_t d = l.h(bp);
        const uint32_t ls = ((lo >> (4 * (ib & 1))) & 15) | (((hi >> (2 * ib)) & 3) << 4);
        w[4] = d | (ls << 16);
    } else {   // IQ4_NL
        static_assert(T == T_IQ4_NL, "the CUDA native formats");
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = lw(l, bp + 2 + 4 * k);
        w[4] = l.h(bp);
    }
}

// The sub-block as 32 int8 (q[0..7], natural order) and its scales (s0: values 0-15, s1: 16-31).
template <int T, int NW>
__device__ __forceinline__ void convert(const uint32_t (&w)[NW], const uint8_t* grid, const uint32_t (&kv)[4],
                                        uint32_t (&q)[8], float& s0, float& s1, const uint4* sgn = nullptr) {
    if constexpr (T == T_Q4_K || T == T_Q5_K) {
        const int ib = (int) (w[9] >> 24), sh = 4 * (ib & 1);
        uint32_t sc, mn;
        if (ib < 4) { sc = w[9] & 63; mn = (w[9] >> 8) & 63; }
        else {
            sc = ((w[9] & 0xF) | (((w[9] >> 8) & 0xC0) >> 2));   // sc[ib + 4] & 15 | (sc[ib - 4] >> 6) << 4
            mn = (((w[9]) >> 4) & 0xF) | (((w[9] >> 16) & 0xC0) >> 2);   // sc[ib + 4] >> 4 | (sc[ib] >> 6) << 4
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            q[k] = (w[k] >> sh) & 0x0F0F0F0Fu;
            if constexpr (T == T_Q5_K) q[k] |= ((w[10 + k] >> ib) & 0x01010101u) << 4;
        }
        s0 = half_at(w[8]) * (float) sc;
        s1 = -(half_at(w[8] >> 16) * (float) mn);
    } else if constexpr (T == T_Q5_0) {   // w = d (q - 16): the signed codes (like Q8_0's), no minimum
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t lo = ((w[4] >> (4 * k)) & 0xF) * 0x00204081u, hi = ((w[4] >> (16 + 4 * k)) & 0xF) * 0x00204081u;
            const uint32_t a = (w[k] & 0x0F0F0F0Fu) | ((lo & 0x01010101u) << 4);
            const uint32_t b = ((w[k] >> 4) & 0x0F0F0F0Fu) | ((hi & 0x01010101u) << 4);
            q[k] = ((a | 0x80808080u) - 0x10101010u) ^ 0x80808080u;       // a byte a in 0..31 -> the int8 a - 16
            q[4 + k] = ((b | 0x80808080u) - 0x10101010u) ^ 0x80808080u;
        }
        s0 = s1 = half_at(w[5]);
    } else if constexpr (T == T_Q5_1) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t lo = ((w[4] >> (4 * k)) & 0xF) * 0x00204081u, hi = ((w[4] >> (16 + 4 * k)) & 0xF) * 0x00204081u;
            q[k] = (w[k] & 0x0F0F0F0Fu) | ((lo & 0x01010101u) << 4);
            q[4 + k] = ((w[k] >> 4) & 0x0F0F0F0Fu) | ((hi & 0x01010101u) << 4);
        }
        s0 = half_at(w[5]);
        s1 = half_at(w[5] >> 16);
    } else if constexpr (T == T_Q8_0) {
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = w[k];
        s0 = s1 = half_at(w[8]);
    } else if constexpr (T == T_IQ2_XXS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[(w[0] >> (8 * l)) & 255];
#if defined(__HIPCC__)
            signed8t(e.x, e.y, sgn[(w[1] >> (7 * l)) & 127], q[2 * l], q[2 * l + 1]);
#else
            signed8(e.x, e.y, unpack_ksigns(w[1] >> (7 * l)), q[2 * l], q[2 * l + 1]);
#endif
        }
        s0 = s1 = half_at(w[2]) * (float) ((w[1] >> 27) | 1) * 0.125f;
    } else if constexpr (T == T_IQ2_XS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t c = (w[l >> 1] >> (16 * (l & 1))) & 0xFFFF;
            const uint2 e = g[c & 511];
#if defined(__HIPCC__)
            signed8t(e.x, e.y, sgn[(c >> 9) & 127], q[2 * l], q[2 * l + 1]);
#else
            signed8(e.x, e.y, unpack_ksigns(c >> 9), q[2 * l], q[2 * l + 1]);
#endif
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 16;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * ((sc >> 4) & 15) + 1) * 0.125f;
    } else if constexpr (T == T_IQ2_S) {
        const uint2* g = (const uint2*) grid;
        const uint32_t qh = (w[2] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[((w[0] >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300)];
#if defined(__HIPCC__)
            signed8t(e.x, e.y, sgn[(w[1] >> (8 * l)) & 255], q[2 * l], q[2 * l + 1]);
#else
            signed8(e.x, e.y, ((w[1] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
#endif
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 24;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * (sc >> 4) + 1) * 0.125f;
    } else if constexpr (T == T_IQ3_XXS) {
        const uint32_t* g = (const uint32_t*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
#if defined(__HIPCC__)
            signed8t(g[i0], g[i1], sgn[(w[2] >> (7 * l)) & 127], q[2 * l], q[2 * l + 1]);
#else
            signed8(g[i0], g[i1], unpack_ksigns(w[2] >> (7 * l)), q[2 * l], q[2 * l + 1]);
#endif
        }
        s0 = s1 = half_at(w[3]) * (float) (2 * (w[2] >> 28) + 1) * 0.25f;
    } else if constexpr (T == T_IQ3_S) {
        const uint32_t* g = (const uint32_t*) grid;
        const uint32_t qh = (w[3] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
#if defined(__HIPCC__)
            signed8t(g[i0 | ((qh << (8 - 2 * l)) & 256)], g[i1 | ((qh << (7 - 2 * l)) & 256)],
                     sgn[(w[2] >> (8 * l)) & 255], q[2 * l], q[2 * l + 1]);
#else
            signed8(g[i0 | ((qh << (8 - 2 * l)) & 256)], g[i1 | ((qh << (7 - 2 * l)) & 256)],
                    ((w[2] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
#endif
        }
        s0 = s1 = half_at(w[3]) * (float) (1 + 2 * (w[3] >> 24));
    } else if constexpr (T == T_IQ4_XS || T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) table16(w[k], kv, q[k], q[4 + k]);
        s0 = s1 = T == T_IQ4_XS ? half_at(w[4]) * (float) ((int) (w[4] >> 16) - 32) : half_at(w[4]);
    } else {   // Q2_0: code - 1 via a byte table {-1, 0, 1, 2}
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const uint32_t c = (w[h >> 1] >> (16 * (h & 1))) & 0xFFFF;
            const uint32_t qe = __byte_perm(0x020100FFu, 0x020100FFu, c & 0x7777);
            const uint32_t qo = __byte_perm(0x020100FFu, 0x020100FFu, (c >> 2) & 0x7777);
            q[2 * h] = __byte_perm(qe, qo, 0x5140);
            q[2 * h + 1] = __byte_perm(qe, qo, 0x7362);
        }
        s0 = s1 = half_at(w[2]);
    }
}

template <int T> __device__ __forceinline__ const void* grid_src() {
    if constexpr (T == T_IQ2_S) return iq2s_grid;
    else if constexpr (T == T_IQ3_XXS) return iq3xxs_grid;
    else if constexpr (T == T_IQ3_S) return iq3s_grid;
    else return nullptr;
}

extern "C" __global__ void count_kernel(const int32_t* __restrict__ ids, int64_t n, int E, int32_t* __restrict__ cnt) {
    extern __shared__ int32_t hist[];
    for (int e = threadIdx.x; e < E; e += blockDim.x) hist[e] = 0;
    __syncthreads();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int e = ids[i];
        if ((unsigned) e < (unsigned) E) atomicAdd(&hist[e], 1);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x)
        if (hist[e]) atomicAdd(&cnt[e], hist[e]);
}

// exclusive prefix sum over the block (blockDim a multiple of 32); *total = the block's sum
__device__ int block_excl_sum(int v, int* total, int* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) sh[w] = x;
    __syncthreads();
    if (w == 0) {
        int s = lane < nw ? sh[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        sh[lane] = s;
    }
    __syncthreads();
    const int r = x - v + (w > 0 ? sh[w - 1] : 0);
    *total = sh[nw - 1];
    __syncthreads();   // sh is reused by the next call
    return r;
}

// one block, a thread per expert (E <= 1024)
extern "C" __global__ void scan_kernel(uint8_t* scratch, int E) {
    const Tables tb = tables(scratch, E);
    __shared__ int sh[32];
    const int e = threadIdx.x;
    const int c = e < E ? tb.cnt[e] : 0, nt = (c + kTileRows - 1) / kTileRows;
    int rows = 0, tiles = 0;
    const int o = block_excl_sum(c, &rows, sh), to = block_excl_sum(nt, &tiles, sh);
    if (e < E) {
        tb.off[e] = o;
        tb.fill[e] = o;
        tb.ts[e] = to;
        for (int i = 0; i < nt; ++i) tb.tiles[to + i] = make_int2(e, o + i * kTileRows);
    }
    if (e == 0) {
        tb.off[E] = rows;
        tb.ts[E] = tiles;
    }
}

extern "C" __global__ void place_kernel(const int32_t* __restrict__ ids, int64_t n, int k, int E, uint8_t* scratch,
                             int32_t* __restrict__ slot, int32_t* __restrict__ src) {
    const Tables tb = tables(scratch, E);
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int e = ids[i];
    if ((unsigned) e >= (unsigned) E) { slot[i] = 0; return; }   // (the router never writes one)
    const int p = atomicAdd(&tb.fill[e], 1);
    slot[i] = p;
    src[p] = (int32_t) (i / k);
}

template <int WT, bool GU, int WW, int NT = THREADS, bool F16 = false>
__global__ void __launch_bounds__(NT, NT == THREADS ? 1 : 2)
native_kernel(int e0, int e1, const int64_t* __restrict__ ptrs,
              size_t gu_row, size_t d_row, uint8_t* scratch, int E,
              const uint8_t* __restrict__ act, const int32_t* __restrict__ src,
              uint8_t* __restrict__ out, float* __restrict__ dm) {
    const Tables tb = tables(scratch, E);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    constexpr int TR = tile_rows(WW, NT), WR = weight_rows(WW);
    static_assert(2 * WR <= NT, "every decode unit is a thread of the block");
    constexpr int WLDX = F16 ? WLDH : WLD;
    constexpr int WT_BYTES = WR * WLDX, WS_FLOATS = F16 ? 0 : WR * 4, ACT_STAGE = TR * AB;
    constexpr bool TWO_PASS = NT != THREADS;   // the two-pass loads (load_unit_l) only in the opt-in NT = 256 kernels
    constexpr int NS = (GU ? GU_ROWS_K : D_ROWS_K) / 64;   // 64-value stages along K
    constexpr int NFB = (GU ? GU_ROWS_N : D_ROWS_N) / WR;          // weight-row blocks per tile
    static_assert(GU_ROWS_N % WR == 0 && D_ROWS_N % WR == 0, "the weight rows of a product are whole work items (2 n_ff and n_embd are multiples of 64 WW)");
    constexpr int ACT_LD = NS * AB;                       // bytes per activation row: 3200 (a token), 800 (a row's H)
    constexpr int BS = block_bytes(WT);
    constexpr bool K16 = per16(WT);
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* wt = smem;                                               // [2][WR][WLD] decoded int8 weights
    float* ws = (float*) (smem + 2 * WT_BYTES);                       // [2][WR][4] their scales per 16 values
    uint8_t* stages = (uint8_t*) (ws + 2 * WS_FLOATS);                // [ASTAGES_DEV][TR][AB] activations
    int* srow = (int*) (stages + ASTAGES_DEV * ACT_STAGE);                // [TR] the activation row of each tile row
    uint8_t* sgrid = (uint8_t*) (srow + TR);                          // the codebook

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int wf = warp % WW, nb0 = 16 * (warp / WW);
    if constexpr (grid_bytes(WT) > 0) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += NT) ((uint32_t*) sgrid)[i] = gs[i];
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_XS || WT == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }
    // Local weight rows: m tile i of warp wf is rows 64 wf + 16 i.. (mma rows g and g + 8: rows +g and +8+g).  For
    // gate/up a tile is 8 features, its gate rows then its up rows, so a lane holds gate and up of one feature.  The
    // fragments come by ldmatrix: lane l gives the address of row (l & 7) of matrix l >> 3 - A: {rows 0-7, rows 8-15}
    // x {bytes 0-15, 16-31} of a 32-value chunk; B: {tokens nb0 + 0-7, + 8-15} x {bytes 0-15, 16-31}.
    const int lm = lane >> 3, l8 = lane & 7;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * WLDX + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * AB + 16 * (lm & 1);
    // this thread's decode unit (the first 2 WR threads): local weight row ur, sub-block uj of each stage
    const bool dec = tid < 2 * WR;
    const int ur = dec ? tid >> 1 : 0, uj = tid & 1;
    const int t0 = tb.ts[e0], nwork = (tb.ts[e1] - t0) * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y;
        if ((row0 - tb.off[e]) % TR != 0) continue;              // a 64-row tile inside an earlier item's TR rows
        const int nrows = min(TR, tb.off[e + 1] - row0);
        const int rbase = fb * WR;
        // local row ur: gate/up - feature (WR / 2) fb + 8 (ur >> 4) + (ur & 7), its up row when bit 3 is set
        const uint8_t* wrow = GU ? (const uint8_t*) ptrs[3 * e + ((ur & 8) ? 1 : 0)] +
                                       (size_t) (fb * (WR / 2) + 8 * (ur >> 4) + (ur & 7)) * gu_row
                                 : (const uint8_t*) ptrs[3 * e + 2] + (size_t) (rbase + ur) * d_row;
        // gate/up in Q2_0 (64-value blocks): a block per stage, as the down rows read theirs
        constexpr bool GU256 = GU && WT != T_Q2_0;
        auto unit = [&](int s) -> const uint8_t* {                  // the block of stage s's sub-block
            if (GU256) return wrow + (s >> 2) * BS;
            return WT == T_IQ4_NL ? wrow + (2 * s + uj) * BS : wrow + s * BS;
        };
        auto sub = [&](int s) { return GU256 ? 2 * (s & 3) + uj : uj; };
        // the weight bytes into L2 ahead of the register loads (one stage ahead only - less than a DRAM round trip):
        // gate/up PF super-blocks ahead (the row's two threads take a block's first and last line), down the whole row
        // slice at once (180 or 360 bytes a row)
        constexpr int PF = 2;
        constexpr int SBB = WT == T_Q2_0 ? 4 * BS : BS;              // bytes per 256 values of a row
        auto prefetch_sb = [&](int sb) {
            if (dec && sb < GU_ROWS_K / 256) pf_l2(wrow + sb * SBB + (uj ? SBB - 1 : 0));
        };
        if (GU) {
#pragma unroll
            for (int sb = 0; sb < PF; ++sb) prefetch_sb(sb);
        } else if (dec) {
            for (size_t o = 128 * (size_t) uj; o < d_row + 127; o += 256) pf_l2(wrow + min(o, d_row - 1));
        }
        __syncthreads();                                          // the previous item is done with the buffers
        if (tid < TR) srow[tid] = tid < nrows ? (GU ? src[row0 + tid] : row0 + tid) : -1;
        __syncthreads();                                          // srow
        // the rows past the item's end are not loaded: a warp entirely past it skips the products, and the columns
        // of a partial one are never written (no reduction mixes columns)
        auto load_act = [&](int s) {
            uint8_t* st = stages + (s % ASTAGES_DEV) * ACT_STAGE;
            for (int c = tid; c < TR * 5; c += NT) {
                const int r = c / 5, q = c % 5;
                if (r < nrows) cp16(st + r * AB + q * 16, act + (size_t) srow[r] * ACT_LD + s * AB + q * 16);
            }
        };
        auto put = [&](const uint32_t (&raw)[5], int buf) {
            if (!dec) return;
            uint32_t q[8];
            float s0, s1;
            convert<WT>(raw, sgrid, kv, q, s0, s1);
            if constexpr (F16) {   // two 16-value groups of 8 half2 words (the permuted order above)
                uint4* d = (uint4*) (wt + buf * WT_BYTES + ur * WLDX + 64 * uj);
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const __half2 sc = __float2half2_rn(k ? s1 : s0);
                    uint32_t lo[4], hi[4];
#pragma unroll
                    for (int w = 0; w < 4; ++w) i8x4_h2(q[4 * k + w], sc, lo[w], hi[w]);
                    d[2 * k] = make_uint4(lo[0], lo[1], lo[2], lo[3]);
                    d[2 * k + 1] = make_uint4(hi[0], hi[1], hi[2], hi[3]);
                }
            } else {
            uint4* d = (uint4*) (wt + buf * WT_BYTES + ur * WLD + 32 * uj);
            d[0] = make_uint4(q[0], q[1], q[2], q[3]);
            d[1] = make_uint4(q[4], q[5], q[6], q[7]);
            *(float2*) (ws + buf * WS_FLOATS + ur * 4 + 2 * uj) = make_float2(s0, s1);
            }
        };
#pragma unroll
        for (int s = 0; s < ASTAGES_DEV - 1; ++s) {
            if (s < NS) load_act(s);
            cp_commit();
        }
        uint32_t raw[5] = {0, 0, 0, 0, 0};
        uint32_t pc[NP];   // stage s + 2's pieces in flight (load_unit_l)
        auto issue = [&](int s) { LdIssue l{pc, 0}; uint32_t w[5]; load_unit_l<WT>(l, unit(s), sub(s), w); };
        auto take = [&](int s) { LdTake l{pc, 0}; load_unit_l<WT>(l, unit(s), sub(s), raw); };
        if (dec) load_unit<WT>(unit(0), sub(0), raw);
        if (dec && TWO_PASS && 2 < NS) issue(2);
        put(raw, 0);
        if (dec) load_unit<WT>(unit(1), sub(1), raw);
        // a warp whose rows are all past the tile's end only takes part in the loads
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
        float acc[4][2][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        for (int s = 0; s < NS; ++s) {
            cp_wait<ASTAGES_DEV - 2>();
            __syncthreads();
            if (s + ASTAGES_DEV - 1 < NS) load_act(s + ASTAGES_DEV - 1);
            cp_commit();
            if (GU && (s & 3) == 0) prefetch_sb((s >> 2) + PF);
            if (on0) {
                const uint8_t* W = wt + (s & 1) * WT_BYTES;
                const float* S = ws + (s & 1) * WS_FLOATS;
                const uint8_t* sa = stages + (s % ASTAGES_DEV) * ACT_STAGE;
                // the stage's scales: per weight row 4 (per 16 values), per routed row 2 (per 32)
                float2 dx[2][2];                                  // [n8 tile][column 2 tig + cc]
#pragma unroll
                for (int n = 0; n < 2; ++n)
#pragma unroll
                    for (int cc = 0; cc < 2; ++cc) dx[n][cc] = F16 ? float2{} : *(const float2*) (sa + (nb0 + 8 * n + 2 * tig + cc) * AB + 64);
                if constexpr (F16) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        uint32_t bq[4];                           // {n0 k 0-15, n0 16-31, n1 0-15, n1 16-31}: token nb0 + 8 n + g
                        ldsm4(bq, sa + b_off + 32 * h);
                        uint32_t bh[2][2][2];                     // [n][16-value group][b0, b1]
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            const __half2 dxh = __float2half2_rn(*(const float*) (sa + (nb0 + 8 * n + g) * AB + 64 + 4 * h));
#pragma unroll
                            for (int k = 0; k < 2; ++k) i8x4_h2(bq[2 * n + k], dxh, bh[n][k][0], bh[n][k][1]);
                        }
#pragma unroll
                        for (int i = 0; i < 4; ++i)
#pragma unroll
                            for (int k = 0; k < 2; ++k) {
                                uint32_t a[4];
                                ldsm4(a, W + a_off + 16 * i * WLDX + 64 * h + 32 * k);
                                mma16h(acc[i][0], a, bh[0][k][0], bh[0][k][1]);
                                if (on1) mma16h(acc[i][1], a, bh[1][k][0], bh[1][k][1]);
                            }
                    }
                } else
                if constexpr (WT == T_Q2_0 && T75_DEV) {
                    // Turing, Q2_0: one d per 64 values, so the stage's two halves share their weight scale - each row's dot
                    // is d (dx_0 dot_0 + dx_1 dot_1): the activation scales per half, then d once
                    uint32_t bq0[4], bq1[4];                      // the two halves' {n0 0-15, n0 16-31, n1 0-15, n1 16-31}
                    ldsm4(bq0, sa + b_off);
                    ldsm4(bq1, sa + b_off + 32);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        uint32_t a0[4], a1[4];
                        ldsm4(a0, W + a_off + 16 * i * WLD);
                        ldsm4(a1, W + a_off + 16 * i * WLD + 32);
                        const float wa = S[(64 * wf + 16 * i + g) * 4], wb = S[(64 * wf + 16 * i + 8 + g) * 4];
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            if (n == 1 && !on1) break;
                            int d0[4], d1[4];
                            mma32(d0, a0, bq0[2 * n], bq0[2 * n + 1]);
                            mma32(d1, a1, bq1[2 * n], bq1[2 * n + 1]);
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                const float t = fmaf(dx[n][q & 1].y, dotf(d1[q]), dx[n][q & 1].x * dotf(d0[q]));
                                acc[i][n][q] = fmaf(q < 2 ? wa : wb, t, acc[i][n][q]);
                            }
                        }
                    }
                } else
#pragma unroll
                for (int h = 0; h < 2; ++h) {                     // the stage's two 32-value halves
                    uint32_t bq[4];                               // {n0 bytes 0-15, n0 16-31, n1 0-15, n1 16-31}
                    ldsm4(bq, sa + b_off + 32 * h);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        uint32_t a[4];                            // {rows g, g + 8} x {bytes 0-15, 16-31}
                        ldsm4(a, W + a_off + 16 * i * WLD + 32 * h);
                        const float2 swa = *(const float2*) (S + (64 * wf + 16 * i + g) * 4 + 2 * h);
                        const float2 swb = *(const float2*) (S + (64 * wf + 16 * i + 8 + g) * 4 + 2 * h);
                        const float wa0 = swa.x, wa1 = swa.y, wb0 = swb.x, wb1 = swb.y;
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            if (n == 1 && !on1) break;
                            if constexpr (K16) {
                                // a scale per 16 weights: the two halves' dots, each times its scale, then d_x
                                int d0[4], d1[4];
                                mma16(d0, a[0], a[1], bq[2 * n]);
                                mma16(d1, a[2], a[3], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float v = fmaf(q < 2 ? wa1 : wb1, dotf(d1[q]), (q < 2 ? wa0 : wb0) * dotf(d0[q]));
                                    acc[i][n][q] = fmaf(h ? dx[n][q & 1].y : dx[n][q & 1].x, v, acc[i][n][q]);
                                }
                            } else {
                                int d[4];
                                mma32(d, a, bq[2 * n], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float p = (q < 2 ? wa0 : wb0) * (h ? dx[n][q & 1].y : dx[n][q & 1].x);
                                    acc[i][n][q] = fmaf(p, dotf(d[q]), acc[i][n][q]);
                                }
                            }
                        }
                    }
                }
            }
            // the next stage's weights (into the other buffer, read after the next barrier).  (Decoding them before
            // this stage's products instead was slower on the RTX 5070: a layer of 2048 / 3584 / 8192 tokens, IQ2_S
            // 5.2 / 9.5 / 17.5 ms -> 6.0 / 10.2 / 18.3.)
            if (s + 1 < NS) {
                put(raw, (s + 1) & 1);
                if (dec && s + 2 < NS) {
                    if constexpr (TWO_PASS) {
                        take(s + 2);
                        if (s + 3 < NS) issue(s + 3);
                    } else {
                        load_unit<WT>(unit(s + 2), sub(s + 2), raw);
                    }
                }
            }
        }
        if (!on0) continue;
        if (GU) {
            // SwiGLU of feature (WR / 2) fb + 32 wf + 8 i + g (lane g of m tile i: rows 64 wf + 16 i + g and + 8), tile
            // rows nb0 + 8 n + 2 tig + c; H to int8 per row over the warp's 32 features (one 32-value half-block of the
            // down product's K)
            const int blk = WW * fb + wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gt = acc[i][n][c], up = acc[i][n][2 + c];
                        h[i] = gt / (1.0f + __expf(-gt)) * up;
                        am = fmaxf(am, fabsf(h[i]));
                    }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        uint8_t* o = out + (size_t) (row0 + r) * H_ROW_BYTES + (blk >> 1) * AB;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) o[32 * hh + 8 * i + g] = (uint8_t) (int8_t) __float2int_rn(h[i] * inv);
                        if (g == 0) *(float*) (o + 64 + 4 * hh) = am / 127.0f;
                    }
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r = nb0 + 8 * n + 2 * tig + (q & 1);
                        if (r < nrows) dm[(size_t) (row0 + r) * kMoeEmbd + rbase + 64 * wf + 16 * i + 8 * (q >> 1) + g] = acc[i][n][q];
                    }
                }
        }
    }
#endif
}


template<int T> __global__ void decode_weight(const uint8_t* raw, int K, int N,
                                             int8_t* codes, float* scales) {
    int row = blockIdx.x, sub32 = blockIdx.y*blockDim.x+threadIdx.x;
    if (row >= N || sub32 >= K/32) return;
    constexpr int QK = T == T_IQ4_NL ? 32 : T == T_Q2_0 ? 64 : 256;
    const uint8_t* bp = raw + (size_t)row*(K/QK)*block_bytes(T) + (sub32/(QK/32))*block_bytes(T);
    uint32_t w[5]={}, q[8], kv[4]={}; float s0,s1;
    for(int j=0;j<16;++j) kv[j/4] |= (uint32_t)(uint8_t)kvalues_iq4nl[j] << (8*(j%4));
    load_unit<T>(bp, sub32%(QK/32), w);
    convert<T>(w, (const uint8_t*)grid_src<T>(), kv, q, s0,s1);
    for(int j=0;j<8;++j) ((uint32_t*)(codes+(size_t)row*K+sub32*32))[j]=q[j];
    scales[(size_t)row*(K/16)+2*sub32]=s0;
    scales[(size_t)row*(K/16)+2*sub32+1]=s1;
}
