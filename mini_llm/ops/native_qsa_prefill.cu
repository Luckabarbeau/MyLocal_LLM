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
*/
// Source port: Niko1221/Strata 712fec6b4297c9cfd32ab43daaa2979a9aa01edf.
// qsa_prompt_attn.cu SHA256 f56e663115c80338d6e35c77491224cc259b1994c1b857e49b3ebe5497be0eca
// Source ranges: 521:542, 1094:1399. Arithmetic and instruction order unchanged.
// prefill/kernels.cu ranges 2522:2574 (INT8 branch only exposed).
// Binding-only changes: flatten pool structs into pointer arguments; fixed artifact geometry.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#define INFINITY CUDART_INF_F
// NVRTC has CUDA device headers, but no host libc headers.
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef long long int64_t;
#define STRATA_PA_SM80 1
constexpr int HD=256, G=12, THREADS=128, CH2=32, KV_Q8_GROUP=64;
constexpr int kStepCount=4, kStepWidth=3;
struct QsaAttnPools {
    const int8_t *k_q, *v_q;
    const uint16_t *k_scale, *v_scale;
    const int32_t *page_table;
};
namespace strata { namespace kernels {
struct KvHostPools {
    uint16_t *k_pool=nullptr, *v_pool=nullptr;
    int8_t *k_q=nullptr, *v_q=nullptr;
    uint16_t *k_scale=nullptr, *v_scale=nullptr;
};
}}
__device__ __forceinline__ uint16_t hf(float f) { return __half_as_ushort(__float2half_rn(f)); }
__device__ __forceinline__ int swz(int cell, int byte) {   // byte offset of (cell, byte) in a stage slice
    return cell * 64 + ((((byte >> 4) ^ (cell >> 1)) & 3) << 4) + (byte & 15);
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
#if !STRATA_PA_SM80
    (void) smem; (void) gmem; (void) valid;
    __trap();   // not reached on Turing (the register-staged pipeline replaces it there)
#else
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if STRATA_PA_SM80
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void cp_async_wait1() {
#if STRATA_PA_SM80
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}
__device__ __forceinline__ void imma16832(int* c, const uint32_t* a, const uint32_t* b) {
#if !STRATA_PA_SM80
    __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}
// the low part's MMA: unsigned A (m in 0..255), signed B
__device__ __forceinline__ void imma16832_us(int* c, const uint32_t* a, const uint32_t* b) {
#if !STRATA_PA_SM80
    __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}
// x (|x| <= 127) as X = rint(x * 65536) = h * 65536 + l * 256 + m exactly: h, l signed bytes, m an unsigned one
__device__ __forceinline__ void split3(float x, int& h, int& l, int& m) {
    const int X = __float2int_rn(x * 65536.0f);
    h = (X + 32768) >> 16;
    const int r = X - (h << 16);   // -32768 .. 32767
    l = r >> 8;                    // -128 .. 127
    m = r & 255;                   // 0 .. 255
}
__device__ __forceinline__ uint32_t pack4(int a, int b, int c, int d) {
    return (uint32_t) (a & 0xff) | ((uint32_t) (b & 0xff) << 8) | ((uint32_t) (c & 0xff) << 16) | ((uint32_t) (d & 0xff) << 24);
}
__device__ __forceinline__ float join3(int h, int l, int m) {   // the parts' exact int32 sums -> h + (l*256 + m)/65536
    return (float) h + (float) ((l << 8) + m) * (1.0f / 65536.0f);
}
// four words of four bytes each (word i = cell i) -> four words of the same byte position across the cells
__device__ __forceinline__ void tr4x4(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, uint32_t* o) {
    const uint32_t t0 = __byte_perm(w0, w1, 0x5140), t1 = __byte_perm(w0, w1, 0x7362);
    const uint32_t t2 = __byte_perm(w2, w3, 0x5140), t3 = __byte_perm(w2, w3, 0x7362);
    o[0] = __byte_perm(t0, t2, 0x5410);
    o[1] = __byte_perm(t0, t2, 0x7632);
    o[2] = __byte_perm(t1, t3, 0x5410);
    o[3] = __byte_perm(t1, t3, 0x7632);
}

// v3's shared memory: ONE stage of K and one of V per warp (a warp gathers and reads only its own 64-dim slices, so
// reusing a buffer needs only that warp's own order): K(c + 1) is gathered while the block runs softmax(c) and
// p.v(c), V(c + 1) while it runs q.k(c + 1) and softmax(c + 1); and only the 12 real head rows of the partial sums
// and p.  25.5 KB instead of v2's 45.6 KB: three blocks an SM instead of two (one block an SM ran 1.54x slower than
// two - this kernel waits on latency, not on its MMAs).
struct Smem3 {
    int8_t kv[4][2][CH2][64];   // warp, K/V, cell, 64 dims in 16-byte pieces XOR-swizzled by the cell
    float sc[4][2][CH2];        // warp, K/V scale of the cell for the warp's group
    float part[4][G][CH2 + 1];  // q.k per dim group (the 12 head rows)
    float p[G][CH2 + 1];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

extern "C" __global__ void __launch_bounds__(THREADS, 3) prompt_attn_i8v3_kernel(
    const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale,
    const uint16_t* v_scale, const int32_t* page_table, const float* __restrict__ q,
                                                                      const int32_t* __restrict__ ids,
                                                                      const int32_t* __restrict__ steps, int n_kv_heads,
                                                                      int page_size, float scale_log2,
                                                                      float* __restrict__ attn, int cap) {
    QsaAttnPools p{k_q, v_q, k_scale, v_scale, page_table};
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem3& S = *reinterpret_cast<Smem3*>(smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int dim0 = warp * 64;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; S.alpha[t] = 0.0f; }

    // q: rows gid / gid + 8 (heads; 12..15 are pad), this warp's 64 dims as the A fragments of two k-steps of 32,
    // three 8-bit parts with a scale per (row, 64-dim group)
    uint32_t qa[3][2][4];
    float qinv[2];
    {
        float x[2][2][2][4];   // [row half][k-step][k half][4 dims]
        float mx[2] = {0.0f, 0.0f};
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
            const int row = gid + rh * 8;
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
#pragma unroll
                for (int kh = 0; kh < 2; ++kh) {
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (row < G) v = *reinterpret_cast<const float4*>(q + (size_t) row * HD + dim0 + ks * 32 + kh * 16 + tig * 4);
                    x[rh][ks][kh][0] = v.x; x[rh][ks][kh][1] = v.y; x[rh][ks][kh][2] = v.z; x[rh][ks][kh][3] = v.w;
                    mx[rh] = fmaxf(mx[rh], fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
                }
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], o));
        }
        float up[2];
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
            up[rh] = mx[rh] > 0.0f ? 127.0f / mx[rh] : 0.0f;
            qinv[rh] = mx[rh] * (1.0f / 127.0f);
        }
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
#pragma unroll
            for (int r = 0; r < 4; ++r) {   // a0: row gid, k 0..15 half; a1: row gid + 8; a2: row gid, k 16..31; a3
                const int rh = r & 1, kh = r >> 1;
                int hh[4], ll[4], mm[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) split3(x[rh][ks][kh][e] * up[rh], hh[e], ll[e], mm[e]);
                qa[0][ks][r] = pack4(hh[0], hh[1], hh[2], hh[3]);
                qa[1][ks][r] = pack4(ll[0], ll[1], ll[2], ll[3]);
                qa[2][ks][r] = pack4(mm[0], mm[1], mm[2], mm[3]);
            }
    }

    const int n_chunks = (n + CH2 - 1) / CH2;
    auto cell_of = [&](int c) -> int { return c < n ? __ldg(ids + c) : -1; };
    auto row_of = [&](int cell) -> long long {
        if (cell < 0) return -1;
        const long long page = (long long) __ldg(p.page_table + cell / page_size);
        return (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    };
    // this warp's 64-dim slice of K (kv = 0) or V (1) of the chunk whose rows the lanes hold, and the cells' scales
    auto issue = [&](long long r, int kv) {
        const int8_t* pool = kv == 0 ? p.k_q : p.v_q;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + 32 * j, cell = idx >> 2, pc = idx & 3;
            const long long rr = __shfl_sync(0xffffffffu, r, cell);
            const bool ok = rr >= 0;
            const size_t off = ok ? (size_t) rr * HD + dim0 + pc * 16 : 0;
            cp_async16(&S.kv[warp][kv][0][0] + swz(cell, pc * 16), pool + off, ok);
        }
        const uint16_t* sc = kv == 0 ? p.k_scale : p.v_scale;
        S.sc[warp][kv][lane] = r >= 0 ? __half2float(__ushort_as_half(__ldg(sc + r * (HD / KV_Q8_GROUP) + warp))) : 0.0f;
    };
    long long r_nxt = row_of(cell_of(lane));   // chunk 0
    issue(r_nxt, 0);
    cp_async_commit();
    issue(r_nxt, 1);
    cp_async_commit();
    r_nxt = row_of(cell_of(CH2 + lane));       // chunk 1
    int cell_nn = cell_of(2 * CH2 + lane);     // chunk 2
    __syncthreads();

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    const int8_t* K = &S.kv[warp][0][0][0];
    const int8_t* V = &S.kv[warp][1][0][0];

    for (int ci = 0; ci < n_chunks; ++ci) {
        const int c0 = ci * CH2;
        const bool more = ci + 1 < n_chunks;
        cp_async_wait1();   // K(ci) is in (pending: V(ci))
        __syncwarp();
        // q.k over this warp's 64 dims: n-tile nt holds cells nt * 8 .. + 7 (column n = cell nt * 8 + n)
#pragma unroll
        for (int nt = 0; nt < CH2 / 8; ++nt) {
            int ch[4] = {0, 0, 0, 0}, cl[4] = {0, 0, 0, 0}, cm[4] = {0, 0, 0, 0};
            const int cell = nt * 8 + gid;
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                uint32_t b[2];
                b[0] = *reinterpret_cast<const uint32_t*>(K + swz(cell, ks * 32 + tig * 4));
                b[1] = *reinterpret_cast<const uint32_t*>(K + swz(cell, ks * 32 + 16 + tig * 4));
                imma16832(ch, qa[0][ks], b);
                imma16832(cl, qa[1][ks], b);
                imma16832_us(cm, qa[2][ks], b);
            }
            const int c = nt * 8 + 2 * tig;
            const float s0 = S.sc[warp][0][c], s1 = S.sc[warp][0][c + 1];
            S.part[warp][gid][c] = join3(ch[0], cl[0], cm[0]) * qinv[0] * s0;
            S.part[warp][gid][c + 1] = join3(ch[1], cl[1], cm[1]) * qinv[0] * s1;
            if (gid + 8 < G) {
                S.part[warp][gid + 8][c] = join3(ch[2], cl[2], cm[2]) * qinv[1] * s0;
                S.part[warp][gid + 8][c + 1] = join3(ch[3], cl[3], cm[3]) * qinv[1] * s1;
            }
        }
        __syncwarp();   // every lane has read K(ci) and its scales
        if (more) issue(r_nxt, 0);
        cp_async_commit();
        __syncthreads();
        // online softmax over the four groups' sum (fixed order): rows 0..11, 8 threads a row (warp 3 idles)
        if (t < G * 8) {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = c0 + c < n ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * scale_log2
                                  : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        cp_async_wait1();   // V(ci) is in (pending: K(ci + 1))
        __syncwarp();
        // p.v over this warp's 64 dims: A = p * v_scale (rows gid / gid + 8, k = the chunk's 32 cells) in three parts
        // scaled per row; B = V (k = cells, n-tile j column n = dim n * 8 + j)
        {
            uint32_t pa[3][4];
            float pinv[2];
            {   // a scale per row: a chunk whose p are all small still gets its 24 bits (one per warp and chunk lost
                // ~3x of the accuracy against FP64)
                float a[4][4], mx[2] = {0.0f, 0.0f};
#pragma unroll
                for (int r = 0; r < 4; ++r) {   // a0: row gid, cells tig * 4 ..; a1: row gid + 8; a2 / a3: cells 16 + ...
                    const int row = gid + (r & 1) * 8, cb = (r >> 1) * 16 + tig * 4;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        a[r][e] = row < G ? S.p[row][cb + e] * S.sc[warp][1][cb + e] : 0.0f;
                        mx[r & 1] = fmaxf(mx[r & 1], a[r][e]);
                    }
                }
                float up[2];
#pragma unroll
                for (int rh = 0; rh < 2; ++rh) {
#pragma unroll
                    for (int o = 1; o < 4; o <<= 1) mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], o));
                    up[rh] = mx[rh] > 0.0f ? 127.0f / mx[rh] : 0.0f;
                    pinv[rh] = mx[rh] * (1.0f / 127.0f);
                }
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    int hh[4], ll[4], mm[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) split3(a[r][e] * up[r & 1], hh[e], ll[e], mm[e]);
                    pa[0][r] = pack4(hh[0], hh[1], hh[2], hh[3]);
                    pa[1][r] = pack4(ll[0], ll[1], ll[2], ll[3]);
                    pa[2][r] = pack4(mm[0], mm[1], mm[2], mm[3]);
                }
            }
            // B: this thread's 8 dims (gid * 8 .. + 7) of cells tig * 4 .. + 3 (b0) and 16 + tig * 4 .. + 3 (b1)
            uint32_t bv[2][8];   // [k half][dim j]
#pragma unroll
            for (int kh = 0; kh < 2; ++kh) {
                uint32_t lo[4], hi[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const uint2 w = *reinterpret_cast<const uint2*>(V + swz(kh * 16 + tig * 4 + e, gid * 8));
                    lo[e] = w.x;
                    hi[e] = w.y;
                }
                tr4x4(lo[0], lo[1], lo[2], lo[3], &bv[kh][0]);
                tr4x4(hi[0], hi[1], hi[2], hi[3], &bv[kh][4]);
            }
            __syncwarp();   // every lane has read V(ci) and its scales
            if (more) issue(r_nxt, 1);
            cp_async_commit();
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                int ch[4] = {0, 0, 0, 0}, cl[4] = {0, 0, 0, 0}, cm[4] = {0, 0, 0, 0};
                const uint32_t b[2] = {bv[0][j], bv[1][j]};
                imma16832(ch, pa[0], b);
                imma16832(cl, pa[1], b);
                imma16832_us(cm, pa[2], b);
                acc[j][0] = fmaf(acc[j][0], a0, join3(ch[0], cl[0], cm[0]) * pinv[0]);
                acc[j][1] = fmaf(acc[j][1], a0, join3(ch[1], cl[1], cm[1]) * pinv[0]);
                acc[j][2] = fmaf(acc[j][2], a1, join3(ch[2], cl[2], cm[2]) * pinv[1]);
                acc[j][3] = fmaf(acc[j][3], a1, join3(ch[3], cl[3], cm[3]) * pinv[1]);
            }
        }
        r_nxt = row_of(cell_nn);
        cell_nn = cell_of((ci + 3) * CH2 + lane);
    }
    __syncthreads();
    const float l0 = S.lsum[gid], l1 = gid + 8 < G ? S.lsum[gid + 8] : 0.0f;
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {   // column 2 * tig is dim 16 * tig + j, column 2 * tig + 1 dim 16 * tig + 8 + j
        const int d0 = dim0 + 16 * tig + j, d1 = d0 + 8;
        attn[(size_t) gid * HD + d0] = acc[j][0] * i0;
        attn[(size_t) gid * HD + d1] = acc[j][1] * i0;
        if (gid + 8 < G) {
            attn[(size_t) (gid + 8) * HD + d0] = acc[j][2] * i1;
            attn[(size_t) (gid + 8) * HD + d1] = acc[j][3] * i1;
        }
    }
}

extern "C" __global__ void kv_append_kernel(const float* __restrict__ K, const float* __restrict__ V, int64_t pos0,
                                 const int32_t* __restrict__ table, int64_t page_size, uint16_t* k_pool,
                                 uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                                 int8_t* hk, int8_t* hv, uint16_t* hks, uint16_t* hvs,
                                 int8_t* sk, int8_t* sv, uint16_t* sks, uint16_t* svs) {
    strata::kernels::KvHostPools host{nullptr,nullptr,hk,hv,hks,hvs};
    strata::kernels::KvHostPools stage{nullptr,nullptr,sk,sv,sks,svs};
    const int64_t t = blockIdx.x;
    const int kvh = blockIdx.y, g = blockIdx.z >> 1;
    const bool is_v = (blockIdx.z & 1) != 0;
    const int d = g * 64 + threadIdx.x;
    const float x = (is_v ? V : K)[t * 512 + kvh * 256 + d];
    const int64_t pos = pos0 + t;
    const int64_t page = table[pos / page_size];
    const int64_t row = (page * 2 + kvh) * page_size + pos % page_size;
    const int64_t row_id = ((pos / page_size) * 2 + kvh) * page_size + pos % page_size;
    if (k_pool != nullptr) {
        const uint16_t h = hf(x);
        if (page >= 0) (is_v ? v_pool : k_pool)[row * 256 + d] = h;
        if (host.k_pool != nullptr) (is_v ? host.v_pool : host.k_pool)[row_id * 256 + d] = h;
        if (stage.k_pool != nullptr) (is_v ? stage.v_pool : stage.k_pool)[row_id * 256 + d] = h;
        return;
    }
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float wm[2];
    if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(wm[0], wm[1]);
    const uint16_t sb = hf(amax / 127.0f);
    const float sf = __half2float(__ushort_as_half(sb));
    int q = 0;
    if (sf > 0.0f) { q = __float2int_rn(x / sf); q = q < -127 ? -127 : (q > 127 ? 127 : q); }
    if (page >= 0) {
        (is_v ? v_q : k_q)[row * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? v_scale : k_scale)[row * 4 + g] = sb;
    }
    if (host.k_q != nullptr) {
        (is_v ? host.v_q : host.k_q)[row_id * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? host.v_scale : host.k_scale)[row_id * 4 + g] = sb;
    }
    if (stage.k_q != nullptr) {
        (is_v ? stage.v_q : stage.k_q)[row_id * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? stage.v_scale : stage.k_scale)[row_id * 4 + g] = sb;
    }
}

extern "C" __global__ void gate_attn_kernel(const float* __restrict__ a, const float* __restrict__ qf, uint16_t* __restrict__ o16,
                                 int64_t T, int64_t ld16) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    o16[t * ld16 + h * 256 + d] = hf(a[i] * (1.0f / (1.0f + expf(-qf[t * 24 * 512 + h * 512 + 256 + d]))));
}

// ---- Extended source port (same pinned MIT authority) ----
// native_qsa_indexer.cu:39-130,153-254; qsa_select.cu:30-290,482-506;
// native_qsa_score.cu:32,35-145; prefill/kernels.cu:31-35,93-105,2474-2515.
// Binding changes only: names/namespaces, device ABI typedefs, and TAB=false
// specialization with null mRoPE/table arguments. Analytic text RoPE only.
constexpr int kStepPos=0, kStepNKv=1, kStepNBid=2;
namespace strata { namespace kernels {
struct RopeTab { const float *cos=nullptr, *sin=nullptr; int max_pos=0; };
__device__ __forceinline__ int mrope_pos(const int32_t* tab, int pos, int pair) {
    return tab ? __ldg(tab + (size_t) pos * 3 + pair % 3) : pos;
}
/// cos and sin of rotary position p, pair `pair` (0..31) from the table (scaling and magnitude included); false
/// when there is no table or p lies past it - then the caller computes the angle as it always did
__device__ __forceinline__ bool rope_tab_cs(const RopeTab& t, int p, int pair, float& c, float& s) {
    if (t.cos == nullptr || p < 0 || p >= t.max_pos) return false;
    c = __ldg(t.cos + (size_t) p * 32 + pair);
    s = __ldg(t.sin + (size_t) p * 32 + pair);
    return true;
}
__device__ inline float rope_yarn_ramp(float low, float high, int pair) {
    const float y = ((float) pair - low) / (high - low > 0.001f ? high - low : 0.001f);
    const float clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    return 1.0f - clamped;
}
__device__ inline void rope_scaled_angle(float theta_extrap, float freq_scale, float corr_low,
                                                     float corr_high, float ext_factor, float mscale_in,
                                                     int pair, float& cos_out, float& sin_out) {
    // The interpolation, always applied; the correction mixed in on top (ggml-cuda/rope.cu rope_yarn).
    float theta = freq_scale * theta_extrap;
    float mscale = mscale_in;
    if (ext_factor != 0.0f) {
        const float ramp_mix = rope_yarn_ramp(corr_low, corr_high, pair) * ext_factor;
        theta = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        // "Get n-d magnitude scaling corrected for interpolation" - the log term only when correcting.
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_out = cosf(theta) * mscale;
    sin_out = sinf(theta) * mscale;
}
}}
namespace indexer_port {
using namespace strata::kernels;
constexpr int D=128,R=4,ROT=64,THREADS=256;
constexpr bool TAB=false;
__device__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}
// TAB (#280, STRATA_ROPE_TABLE=1): the angles from the session's float64 table.  The host launches <false> whenever
// no table applies - the default - so the default kernel is 0.1.31's code exactly (the table read is not in it;
// with it merely skipped at run time, the compiled default path changed its results).
template <bool TAB>
__device__ __forceinline__ void append_body(const float* __restrict__ raw, const int32_t* __restrict__ pos_dev,
                        int pos_stride, int n_steps,
                        int pos_base, const float* __restrict__ gamma, float epsilon,
                        float* __restrict__ tail, float* __restrict__ dead,
                        float* __restrict__ pooled, int32_t* __restrict__ block_pos,
                        int max_cells, float theta_scale, float freq_scale, float corr_low,
                        float corr_high, float ext_factor, float mscale,
                        const int32_t* __restrict__ mtab, const RopeTab& rt) {
    __shared__ float values[D];
    __shared__ float partials[32];
    const int d = threadIdx.x;
    for (int step_idx = 0; step_idx < n_steps; ++step_idx) {
        const int pos = pos_dev[(size_t) step_idx * pos_stride];
        if (pos >= 0 && pos < max_cells) {
            const float* raw_step = raw + (size_t) step_idx * D;
            const int slot = pos % R;
            float incoming = 0.0f;
            if (d < D) {
                // SET_ROWS stores F16; GET_ROWS expands those exact values to F32.
                incoming = __half2float(__float2half_rn(raw_step[d]));
                if (slot < R - 1) tail[slot * D + d] = incoming;
            }
            if (pos == 0 || slot == R - 1) {
                float mean = 0.0f;
                if (d < D) {
                    // The spare's four gather indices all name cell zero. Completed blocks
                    // use chronological slices; each graph ADD materializes an F32 sum.
                    float sum = pos == 0 ? incoming : tail[d];
#pragma unroll
                    for (int j = 1; j < R; ++j)
                        sum = __fadd_rn(sum, pos == 0 || j == R - 1 ? incoming : tail[j * D + d]);
                    mean = __fmaf_rn(0.25f, sum, 0.0f); // SCALE includes a zero bias.
                }
                float square_sum = 0.0f;
                if (d < D) square_sum += mean * mean;
                square_sum = warp_sum(square_sum);
                const int lane = d % 32;
                if (lane == 0) partials[d / 32] = square_sum;
                __syncthreads();
                square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;
                square_sum = warp_sum(square_sum);
                const float scale = rsqrtf(square_sum / D + epsilon);
                if (d < D) values[d] = scale * mean * gamma[d];
                __syncthreads();
                if (d < D) {
                    const int b = pos / R;
                    const int rope_pos = pos == 0 ? 0 : pos_base + R * b;
                    float y = values[d];
                    if (d < ROT) {
                        const int pair = d % (ROT / 2);
                        // The spare (pos == 0) keeps its zero angle; under YaRN the mscale magnitude still rides in
                        // through cos(0) - which is exactly what the queries are scaled by, so the top-k is unmoved.
                        float c, s;
                        if (!(TAB && rope_tab_cs(rt, pos == 0 ? 0 : mrope_pos(mtab, rope_pos, pair), pair, c, s))) {
                            const float theta_extrap = (pos == 0 ? 0 : mrope_pos(mtab, rope_pos, pair)) * powf(theta_scale, float(pair));
                            rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
                        }
                        const float a = values[pair], z = values[pair + ROT / 2];
                        y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
                    }
                    pooled[size_t(b) * D + d] = y;
                    if (pos == 0) dead[d] = y;
                    else pooled[size_t(b + 1) * D + d] = dead[d];
                    if (d == 0 && pos != 0) *block_pos = rope_pos;
                }
            }
        }
        if (n_steps > 1) __syncthreads();
    }
}
extern "C" __global__ void idx_append_one(const float* __restrict__ raw, const int32_t* __restrict__ pos_dev,
                        int pos_stride, int n_steps,
                        int pos_base, const float* __restrict__ gamma, float epsilon,
                        float* __restrict__ tail, float* __restrict__ dead,
                        float* __restrict__ pooled, int32_t* __restrict__ block_pos,
                        int max_cells, float theta_scale, float freq_scale, float corr_low,
                        float corr_high, float ext_factor, float mscale,
                        int unused) {
    const int32_t* mtab=nullptr; RopeTab rt{};
    append_body<TAB>(raw, pos_dev, pos_stride, n_steps, pos_base, gamma, epsilon, tail, dead, pooled, block_pos, max_cells,
                     theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale, mtab, rt);
}
template <bool TAB>
__device__ __forceinline__ float pooled_value(const float* values, int d, int rope_pos, float theta_scale,
                                              float freq_scale, float corr_low, float corr_high,
                                              float ext_factor, float mscale, const int32_t* mtab, bool zero_pos,
                                              const RopeTab& rt) {
    float y = values[d];
    if (d < ROT) {
        const int pair = d % (ROT / 2);
        // The same rotation the single append applies, rope_scaling.hpp's helper shared: under none this is
        // today's cosf/sinf; under linear/YaRN the pooled keys move with the cache's K, and the spare's zero
        // angle carries the mscale magnitude through cos(0) exactly as the single append's does.
        float c, s;
        if (!(TAB && rope_tab_cs(rt, zero_pos ? 0 : mrope_pos(mtab, rope_pos, pair), pair, c, s))) {
            const float theta_extrap = (zero_pos ? 0 : mrope_pos(mtab, rope_pos, pair)) * powf(theta_scale, float(pair));
            rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
        }
        const float a = values[pair], z = values[pair + ROT / 2];
        y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
    }
    return y;
}
__device__ __forceinline__ float norm_scale(float mean, float* partials, int d) {
    float square_sum = d < D ? mean * mean : 0.0f;
    square_sum = warp_sum(square_sum);
    const int lane = d % 32;
    if (lane == 0) partials[d / 32] = square_sum;
    __syncthreads();
    square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;
    square_sum = warp_sum(square_sum);
    return square_sum;
}
// cell 0 of a sequence: the spare (every gather index names cell 0), written to pooled[0] and dead
extern "C" __global__ void idx_append_first(const float* __restrict__ raw, const float* __restrict__ gamma, float epsilon,
                             float* __restrict__ dead, float* __restrict__ pooled, float theta_scale,
                             float freq_scale, float corr_low, float corr_high, float ext_factor, float mscale,
                             int unused) {
    const int32_t* mtab=nullptr; RopeTab rt{};
    const int d = threadIdx.x;
    __shared__ float values[D];
    __shared__ float partials[32];
    float mean = 0.0f;
    if (d < D) {
        const float incoming = __half2float(__float2half_rn(raw[d]));
        float sum = incoming;
#pragma unroll
        for (int j = 1; j < R; ++j) sum = __fadd_rn(sum, incoming);
        mean = __fmaf_rn(0.25f, sum, 0.0f);
    }
    const float square_sum = norm_scale(mean, partials, d);
    const float scale = rsqrtf(square_sum / D + epsilon);
    if (d < D) values[d] = scale * mean * gamma[d];
    __syncthreads();
    if (d >= D) return;
    const float y = pooled_value<TAB>(values, d, 0, theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale,
                                 mtab, true, rt);
    pooled[d] = y;
    dead[d] = y;
}
extern "C" __global__ void idx_append_blocks(const float* __restrict__ raw, int64_t n, int64_t p0, int pos_base,
                              const float* __restrict__ gamma, float epsilon, const float* __restrict__ tail,
                              const float* __restrict__ dead, float* __restrict__ pooled, int32_t* __restrict__ block_pos,
                              int64_t first_block, int64_t last_block, float theta_scale, float freq_scale,
                              float corr_low, float corr_high, float ext_factor, float mscale,
                              int unused) {
    const int32_t* mtab=nullptr; RopeTab rt{};
    const int64_t b = first_block + blockIdx.x;
    const int d = threadIdx.x;
    __shared__ float values[D];
    __shared__ float partials[32];
    float mean = 0.0f;
    if (d < D) {
        auto key = [&](int j) -> float {
            const int64_t cell = b * R + j;
            return cell >= p0 ? __half2float(__float2half_rn(raw[(cell - p0) * D + d])) : tail[j * D + d];
        };
        float sum = key(0);
#pragma unroll
        for (int j = 1; j < R; ++j) sum = __fadd_rn(sum, key(j));
        mean = __fmaf_rn(0.25f, sum, 0.0f);
    }
    const float square_sum = norm_scale(mean, partials, d);
    const float scale = rsqrtf(square_sum / D + epsilon);
    if (d < D) values[d] = scale * mean * gamma[d];
    __syncthreads();
    if (d >= D) return;
    const int rope_pos = pos_base + R * (int) b;
    pooled[size_t(b) * D + d] = pooled_value<TAB>(values, d, rope_pos, theta_scale, freq_scale, corr_low, corr_high,
                                                  ext_factor, mscale, mtab, false, rt);
    if (b == last_block) {
        pooled[size_t(b + 1) * D + d] = dead[d];
        if (d == 0) *block_pos = rope_pos;
    }
}
// the tail after the batch: slot s holds the key of the batch's last cell with cell % 4 == s (s < 3), if any
extern "C" __global__ void idx_append_tail(const float* __restrict__ raw, int64_t n, int64_t p0, float* __restrict__ tail) {
    const int s = blockIdx.x, d = threadIdx.x;
    if (d >= D) return;
    const int64_t last = p0 + n - 1;
    int64_t cell = last - ((last % R) - s + R) % R;       // the last cell <= last with cell % 4 == s
    if (cell < p0) return;
    tail[s * D + d] = __half2float(__float2half_rn(raw[(cell - p0) * D + d]));
}
}
namespace select_port {
constexpr int IDX_DIM=128,IDX_HEADS=4,R=4,SCORE_WARPS=8,TOPK_T=256;
__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

extern "C" __global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_kernel(const float* __restrict__ pooled,
                                                                        const float* __restrict__ dead,
                                                                        const float* __restrict__ q_idx,
                                                                        const int32_t* __restrict__ steps,
                                                                        int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.y;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
    const float4 k4 = *reinterpret_cast<const float4*>(key + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}

extern "C" __global__ void __launch_bounds__(TOPK_T) block_topk_kernel(const float* __restrict__ scores,
                                                            const int32_t* __restrict__ steps, int64_t max_blocks,
                                                            int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[256];
    __shared__ int s_a[TOPK_T], s_b[TOPK_T];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;                      // blocks 0..n_bid, the last possibly empty
    const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;          // cells equal to thr that fit, lowest index first
    // ---- per-thread counts of cells above and at the threshold, then their exclusive prefixes
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    __syncthreads();
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag; s_b[i] = ae;
            ag += g; ae += e;
        }
    }
    __syncthreads();
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    __syncthreads();
    s_a[t] = sel;
    __syncthreads();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) { const int c = s_a[i]; s_a[i] = a; a += c; }
    }
    __syncthreads();
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}


// ---- QSA select on tensor cores (perf-review, after D-1): the block scores of many queries are one GEMM,
// rows (query, indexer head) x columns (blocks), K = 128, with relu per head summed. 3xTF32 (each operand split
// into a TF32 hi and lo part, hi*hi + hi*lo + lo*hi) keeps FP32-level accuracy; the summation order differs from
// the warp kernel, so a score can move in its last bits and a near-tie can select differently (not bitwise).
// Blocks < n_bid only; the tail block n_bid (the `dead` key, +1e9) is scored by the warp kernel's own code.
constexpr int TC_QT = 16;                 // queries per CTA (one m16 tile per indexer head)
constexpr int TC_NB = 32;                 // blocks per tile (4 warps x n8)
constexpr int TC_ITER = 4;                // tiles per CTA (the query tile is loaded once)
constexpr int TC_QS = IDX_HEADS * IDX_DIM + 4;   // query row stride in floats
constexpr int TC_KS = IDX_DIM + 4;               // key row stride

// TF32 conversion and MMA need sm_80: below it they compile to a trap and qsa_block_scores_tc refuses the device
#if defined(__HIPCC__)          // AMD: no mma.sync / cp.async; the host keeps the warp kernel (below)
#define STRATA_SEL_SM80 0
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define STRATA_SEL_SM80 1
#else
#define STRATA_SEL_SM80 0
#endif
__device__ __forceinline__ uint32_t tf32_hi(float x) {
#if STRATA_SEL_SM80
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
#else
    return __float_as_uint(x);
#endif
}
__device__ __forceinline__ void mma_tf32(float* c, const uint32_t* a, const uint32_t* b) {
#if !STRATA_SEL_SM80
    __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

extern "C" __global__ void __launch_bounds__(128) block_scores_tc_kernel(const float* __restrict__ pooled,
                                                              const float* __restrict__ q_idx,
                                                              const int32_t* __restrict__ steps, int64_t nq,
                                                              int64_t max_blocks, int64_t reach,
                                                              float* __restrict__ out) {
    extern __shared__ __align__(16) float sm[];
    float* sQ = sm;                                   // [TC_QT][TC_QS]
    float* sK = sm + TC_QT * TC_QS;                   // [TC_NB][TC_KS]
    __shared__ int s_nbid[TC_QT];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, gid = lane >> 2, tig = lane & 3;
    const int64_t q0 = (int64_t) blockIdx.y * TC_QT;
    for (int i = t; i < TC_QT * IDX_HEADS * IDX_DIM / 4; i += 128) {
        const int r = i / (IDX_HEADS * IDX_DIM / 4), c = i % (IDX_HEADS * IDX_DIM / 4);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + r < nq) v = reinterpret_cast<const float4*>(q_idx + (q0 + r) * IDX_HEADS * IDX_DIM)[c];
        *reinterpret_cast<float4*>(sQ + r * TC_QS + c * 4) = v;
    }
    if (t < TC_QT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : 0;
    int lo_nbid = 0x7fffffff, hi_nbid = 0;
    __syncthreads();
    for (int i = 0; i < TC_QT; ++i) {
        if (q0 + i >= nq) break;
        lo_nbid = min(lo_nbid, s_nbid[i]);
        hi_nbid = max(hi_nbid, s_nbid[i]);
    }
    for (int it = 0; it < TC_ITER; ++it) {
        const int64_t b0 = ((int64_t) blockIdx.x * TC_ITER + it) * TC_NB;
        if (b0 >= reach || b0 >= hi_nbid) break;      // blocks >= every query's n_bid: nothing to score
        __syncthreads();                              // the previous tile's reads are done
        for (int i = t; i < TC_NB * IDX_DIM / 4; i += 128) {
            const int r = i / (IDX_DIM / 4), c = i % (IDX_DIM / 4);
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (b0 + r < hi_nbid) v = reinterpret_cast<const float4*>(pooled + (b0 + r) * IDX_DIM)[c];
            *reinterpret_cast<float4*>(sK + r * TC_KS + c * 4) = v;
        }
        __syncthreads();
        float acc[IDX_HEADS][4];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.f;
        const float* kr = sK + (warp * 8 + gid) * TC_KS;
#pragma unroll 4
        for (int k0 = 0; k0 < IDX_DIM; k0 += 8) {
            const float kx0 = kr[k0 + tig], kx1 = kr[k0 + tig + 4];
            uint32_t bh[2], bl[2];
            bh[0] = tf32_hi(kx0);
            bh[1] = tf32_hi(kx1);
            bl[0] = tf32_hi(kx0 - __uint_as_float(bh[0]));
            bl[1] = tf32_hi(kx1 - __uint_as_float(bh[1]));
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float* qa = sQ + h * IDX_DIM + k0 + tig;
                const float x0 = qa[gid * TC_QS], x1 = qa[(gid + 8) * TC_QS];
                const float x2 = qa[gid * TC_QS + 4], x3 = qa[(gid + 8) * TC_QS + 4];
                uint32_t ah[4], al[4];
                ah[0] = tf32_hi(x0); ah[1] = tf32_hi(x1); ah[2] = tf32_hi(x2); ah[3] = tf32_hi(x3);
                al[0] = tf32_hi(x0 - __uint_as_float(ah[0]));
                al[1] = tf32_hi(x1 - __uint_as_float(ah[1]));
                al[2] = tf32_hi(x2 - __uint_as_float(ah[2]));
                al[3] = tf32_hi(x3 - __uint_as_float(ah[3]));
                mma_tf32(acc[h], al, bh);
                mma_tf32(acc[h], ah, bl);
                mma_tf32(acc[h], ah, bh);
            }
        }
        // relu per head, heads added in order (as the warp kernel), written where the block completed for the query
        const int64_t bc = b0 + warp * 8 + 2 * tig;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int qr = gid + half * 8;
            const int64_t qi = q0 + qr;
            if (qi >= nq) continue;
            const int nb = s_nbid[qr];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int64_t b = bc + j;
                if (b >= nb || b >= max_blocks) continue;
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float d = acc[h][half * 2 + j];
                    score += d > 0.0f ? d : 0.0f;
                }
                out[qi * max_blocks + b] = score;
            }
        }
    }
}
extern "C" __global__ void __launch_bounds__(32) block_scores_tail_kernel(const float* __restrict__ dead,
                                                               const float* __restrict__ q_idx,
                                                               const int32_t* __restrict__ steps,
                                                               int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    if (n_bid >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float4 k4 = *reinterpret_cast<const float4*>(dead + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + n_bid] = score;
    }
}
}
namespace native_score_port {
constexpr int D=128, HEADS=4, R=4, ROWS=32, WARPS=2, STRIDE=36, COMBINE=68;
struct TileA { uint32_t x[4]; };
struct TileB { uint32_t x[2]; };
struct TileC { float x[4]={0.0f,0.0f,0.0f,0.0f}; };
#if !defined(__HIPCC__)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__device__ __forceinline__ void load_a(TileA& a,const float* p) {
    const float* src=p+(threadIdx.x%16)*STRIDE+(threadIdx.x/16)*4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a.x[0]),"=r"(a.x[1]),"=r"(a.x[2]),"=r"(a.x[3]):"l"(src));
}
__device__ __forceinline__ void load_b(TileB& b,const float* p) {
    const float* src=p+(threadIdx.x%8)*STRIDE+((threadIdx.x/8)*4)%8;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];"
        : "=r"(b.x[0]),"=r"(b.x[1]):"l"(src));
}
__device__ __forceinline__ void mma(TileC& c,const TileA& a,const TileB& b) {
    // Deliberately no cvt.rn.tf32: pinned mma.cuh passes raw F32 bits directly.
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c.x[0]),"+f"(c.x[1]),"+f"(c.x[2]),"+f"(c.x[3])
        : "r"(a.x[0]),"r"(a.x[1]),"r"(a.x[2]),"r"(a.x[3]),"r"(b.x[0]),"r"(b.x[1]));
}
#endif
extern "C" __global__ __launch_bounds__(64,1) void native_score_kernel(
        const float* __restrict__ pooled,const float* __restrict__ query,
        const float* __restrict__ bias,const int32_t* __restrict__ step,
        int max_cells,float* __restrict__ cells) {
    const int n=step[kStepNKv],full=step[kStepNBid];
    if(n<1||n>max_cells||step[kStepPos]!=n-1||full!=n/R||
       step[kStepWidth]!=(n<2051?n:2051))return;
    const int row0=blockIdx.x*ROWS;
    if(row0>full)return;
    const int lane=threadIdx.x,warp=threadIdx.y;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    // Turing (STRATA_EXPERIMENTAL_SM75, a layer-split stage): no tf32 mma.  The same scores with FP32 FMAs, one row
    // per thread of the first warp - rounded differently from the tensor-core path (FP32 instead of TF32 inputs).
    if(warp!=0)return;
    const int row=row0+lane;
    if(row>full)return;
    float h[HEADS];
    for(int j=0;j<HEADS;++j){
        float acc=0.0f;
        for(int d=0;d<D;++d)acc=fmaf(pooled[size_t(row)*D+d],query[j*D+d],acc);
        h[j]=fmaxf(acc,0.0f);
    }
    float sum=__fadd_rn(__fadd_rn(__fadd_rn(h[0],h[1]),h[2]),h[3]);
    if(bias)sum=__fadd_rn(sum,bias[row]);
    sum=__fadd_rn(sum,row==full&&n%R?1e9f:0.0f);
    for(int i=row*R;i<n&&i<(row+1)*R;++i)cells[i]=sum;
#else
    __shared__ __align__(16) float shared[WARPS*16*STRIDE];
    float* tile=shared+warp*16*STRIDE;
    TileC c[2];
    // mmf's launch heuristic picks2warps for K128. Each warp accumulates
    // K[warp*32,warp*32+32), then K[warp*32+64,warp*32+96).
    for(int col=warp*32+lane;col<D;col+=WARPS*32){
        TileA a[2][4];
#pragma unroll
        for(int ia=0;ia<2;++ia){
            __syncwarp();
#pragma unroll
            for(int i=0;i<16;++i){
                const int row=row0+ia*16+i;
                tile[i*STRIDE+lane]=row<=full?pooled[size_t(row)*D+col]:0.0f;
            }
            __syncwarp();
#pragma unroll
            for(int k=0;k<4;++k)load_a(a[ia][k],tile+k*8);
        }
        __syncwarp();
#pragma unroll
        for(int h=0;h<8;++h)tile[h*STRIDE+lane]=h<HEADS?query[h*D+col]:0.0f;
        __syncwarp();
#pragma unroll
        for(int k=0;k<4;++k){
            TileB b;load_b(b,tile+k*8);
#pragma unroll
            for(int ia=0;ia<2;++ia)mma(c[ia],a[ia][k],b);
        }
    }
    __syncthreads();
#pragma unroll
    for(int ia=0;ia<2;++ia){
#pragma unroll
        for(int l=0;l<4;++l){
            const int i=warp*ROWS+ia*16+(l/2)*8+lane/4;
            const int h=(lane%4)*2+l%2;
            shared[h*COMBINE+i]=c[ia].x[l];
        }
    }
    __syncthreads();
    // In the reference: +0 then warp0 partial then warp1 partial, followed
    // by materialized ReLU, CONT(head0), ADD(head1), ADD(head2), ADD(head3).
    if(warp==0){
        const int row=row0+lane;
        if(row>full)return;
        float h[HEADS];
#pragma unroll
        for(int j=0;j<HEADS;++j){
            float v=__fadd_rn(0.0f,shared[j*COMBINE+lane]);
            v=__fadd_rn(v,shared[j*COMBINE+ROWS+lane]);
            h[j]=fmaxf(v,0.0f);
        }
        float sum=__fadd_rn(__fadd_rn(__fadd_rn(h[0],h[1]),h[2]),h[3]);
        if(bias)sum=__fadd_rn(sum,bias[row]);
        sum=__fadd_rn(sum,row==full&&n%R?1e9f:0.0f);
        // The live causal mask is +0. Invalid/padded cells are never exported.
        sum=__fadd_rn(sum,0.0f);
        for(int i=row*R;i<n&&i<(row+1)*R;++i)cells[i]=sum;
    }
#endif
}

#endif
}
namespace prep_port {
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
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
extern "C" __global__ void __launch_bounds__(256) rms_rope_rows_kernel(const float* in, int64_t in_ts, int64_t in_hs, float* out,
                                                            int64_t out_ts, const float* __restrict__ w, int64_t heads,
                                                            int64_t dim, float eps, int64_t pos0, float theta_scale,
                                                            float freq_scale, float corr_low, float corr_high,
                                                            float ext_factor, float mscale, int unused) {
    constexpr bool TAB=false; const int32_t* mtab=nullptr; strata::kernels::RopeTab rt{};
    __shared__ float sh[32];
    __shared__ float sv[256];
    const int64_t row = blockIdx.x;             // t * heads + h
    const int64_t t = row / heads, h = row % heads;
    const int tid = threadIdx.x;
    const float x = tid < dim ? in[t * in_ts + h * in_hs + tid] : 0.0f;
    // rms_rows_kernel: ss += r[c] * r[c] from 0, one column a thread (dim <= 256) = the rounded square.  The empty asm makes the
    // square a value the compiler cannot fold into the first add of block_sum's butterfly (it did: fma(x, x, other lane's
    // square) is another rounding than the sum of two rounded squares; hip_qsa_nr_parity found it on gfx1151, and __fmul_rn
    // does not stop it: it compiles to the same v_fmac).
    float ss = x * x;
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+v"(ss));
#endif
    const float sc = rsqrtf(block_sum(ss, sh) / (float) dim + eps);
    if (tid < dim) sv[tid] = sc * x * w[tid];
    __syncthreads();
    float* o = out + t * out_ts + h * dim;
    if (tid < 32) {                             // rope_kernel: one thread a (pair, pair + 32)
        const int pair = tid;
        float c, s;
        if (!(TAB && strata::kernels::rope_tab_cs(rt, strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair), pair, c,
                                                  s))) {
            const float theta_extrap =
                (float) strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair) * powf(theta_scale, (float) pair);
            strata::kernels::rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale,
                                               pair, c, s);
        }
        const float a = sv[pair], b = sv[pair + 32];
        o[pair] = a * c - b * s;
        o[pair + 32] = a * s + b * c;
    } else if (tid >= 64 && tid < dim) {
        o[tid] = sv[tid];
    }
}
}
