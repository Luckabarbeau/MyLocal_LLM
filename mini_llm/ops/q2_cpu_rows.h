// SPDX-License-Identifier: MIT
// Adapted singleton row_multi<1> from Strata, revision
// 712fec6b4297c9cfd32ab43daaa2979a9aa01edf, src/kernels/cpu/q2_avx2_rows.inl.
// Source SHA256: efc28c0975d03c276924dff7f29c26e3b6725c2718604db57e3dbe4250b8b7db
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
#pragma once
// Include AFTER q2_cpu_simd.h; reuse its GGML layout decoding helpers.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

struct MyLocalQ2RowsActivation {
    std::vector<int8_t> q;
    std::vector<float> scale, hx;
    std::vector<int32_t> sum;

    explicit MyLocalQ2RowsActivation(int k = 0) { resize(k); }
    void resize(int k) {
        if (k < 0 || k % 64) throw std::invalid_argument("Q2 rows width must be a multiple of 64");
        q.resize(k);
        scale.resize(k / 32);
        hx.resize(k / 32);
        sum.resize(k / 32);
    }
    // Caller owns setup/capacity and ISA validation. No allocation or requantization.
    void prepare(const void *q8, int k);
};

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define MYLOCAL_Q2_ROWS_TARGET __attribute__((target("avx2,f16c,fma")))
extern "C" int mylocal_q2_rows_available() noexcept {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c") &&
           __builtin_cpu_supports("fma");
}

MYLOCAL_Q2_ROWS_TARGET inline void MyLocalQ2RowsActivation::prepare(const void *q8, int k) {
    const auto *src = static_cast<const uint8_t *>(q8);
    for (int b = 0; b < k / 32; ++b) {
        std::memcpy(q.data() + 32*b, src + 34*b + 2, 32);
        int sumq = 0;
        for (int j = 0; j < 32; ++j) sumq += q[32*b + j];
        sum[b] = sumq;
        scale[b] = q2_half(src + 34*b);
        hx[b] = scale[b] * float(sumq);
    }
}

// Absolute row indices: only out[r0:r1] is written. Scratch is immutable here.
extern "C" MYLOCAL_Q2_ROWS_TARGET void mylocal_q2_rows(
    const uint8_t *weight, size_t stride, int k, const MyLocalQ2RowsActivation *act,
    int r0, int r1, float *out) {
    const __m256i ones = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; ++r) {
        const uint8_t *row = weight + size_t(r)*stride;
        __m256 acc = _mm256_setzero_ps();
        float corr = 0.0f;
        for (int b = 0; b < k / 64; ++b) {
            const float d = q2_half(row + size_t(b)*18);
            __m256i lo, hi;
            q2_codes(row + size_t(b)*18 + 2, lo, hi);
            const auto *q = reinterpret_cast<const __m256i *>(act->q.data() + 64*b);
            const __m256i s0 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(lo, _mm256_loadu_si256(q)), ones);
            const __m256i s1 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(hi, _mm256_loadu_si256(q + 1)), ones);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d * act->scale[2*b]), _mm256_cvtepi32_ps(s0), acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d * act->scale[2*b+1]), _mm256_cvtepi32_ps(s1), acc);
            corr += d * (act->hx[2*b] + act->hx[2*b+1]);
        }
        const __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        const __m128 s = _mm_add_ps(h, _mm_movehl_ps(h, h));
        out[r] = _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s))) - corr;
    }
}
#undef MYLOCAL_Q2_ROWS_TARGET
#else
extern "C" int mylocal_q2_rows_available() noexcept { return 0; }
inline void MyLocalQ2RowsActivation::prepare(const void *, int) {}
extern "C" void mylocal_q2_rows(const uint8_t *, size_t, int,
                                const MyLocalQ2RowsActivation *, int, int, float *) {}
#endif

// A distinct dispatch marker, not a new dot implementation. Direct linear/expert
// callers retain the existing scalar-order SIMD dot until RowPool handles it.
extern "C" void mylocal_q2_rows_marker(int n, float *out, size_t bs, const void *w,
                                     size_t bx, const void *x, size_t by, int nr) {
    mylocal_q2_avx2_dot(n, out, bs, w, bx, x, by, nr);
}

// Exact-order alternative. Compile with -ffp-contract=off, without fast-math:
// scalar FP32 additions, including the initial +0, match mylocal_q2_avx2_dot.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define MYLOCAL_Q2_ROWS_EXACT_TARGET __attribute__((target("avx2,f16c")))
MYLOCAL_Q2_ROWS_EXACT_TARGET static inline int32_t mylocal_q2_rows_sum4(
    __m256i codes, __m256i q) {
    const __m256i products = _mm256_madd_epi16(
        _mm256_maddubs_epi16(codes, q), _mm256_set1_epi16(1));
    __m128i total = _mm_add_epi32(_mm256_castsi256_si128(products),
                                _mm256_extracti128_si256(products, 1));
    total = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0x4e));
    total = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0xb1));
    return _mm_cvtsi128_si32(total);
}

template<int NR> MYLOCAL_Q2_ROWS_EXACT_TARGET static void mylocal_q2_rows_exact_group(
    const uint8_t *weight, size_t stride, int k, const MyLocalQ2RowsActivation *act,
    int r, float *out) {
    float result[NR] = {};
    for (int b = 0; b < k / 64; ++b) {
        const auto *q = reinterpret_cast<const __m256i *>(act->q.data() + 64*b);
        const __m256i q0 = _mm256_loadu_si256(q);
        const __m256i q1 = _mm256_loadu_si256(q + 1);
        const float scale0 = act->scale[2*b], scale1 = act->scale[2*b+1];
        const int32_t sum0 = act->sum[2*b], sum1 = act->sum[2*b+1];
        for (int t = 0; t < NR; ++t) {
            const uint8_t *block = weight + size_t(r+t)*stride + size_t(b)*18;
            __m256i lo, hi;
            q2_codes(block + 2, lo, hi);
            const int32_t dot0 = mylocal_q2_rows_sum4(lo, q0) - sum0;
            const int32_t dot1 = mylocal_q2_rows_sum4(hi, q1) - sum1;
            float subtotal = 0.0f;
            subtotal += scale0 * float(dot0);
            subtotal += scale1 * float(dot1);
            result[t] += q2_half(block) * subtotal;
        }
    }
    for (int t = 0; t < NR; ++t) out[r+t] = result[t];
}

extern "C" MYLOCAL_Q2_ROWS_EXACT_TARGET void mylocal_q2_rows_exact(
    const uint8_t *weight, size_t stride, int k, const MyLocalQ2RowsActivation *act,
    int r0, int r1, float *out) {
    int r = r0;
    for (; r1-r >= 4; r += 4)
        mylocal_q2_rows_exact_group<4>(weight, stride, k, act, r, out);
    for (; r < r1; ++r)
        mylocal_q2_rows_exact_group<1>(weight, stride, k, act, r, out);
}
#undef MYLOCAL_Q2_ROWS_EXACT_TARGET
#else
extern "C" void mylocal_q2_rows_exact(const uint8_t *, size_t, int,
                                      const MyLocalQ2RowsActivation *, int, int, float *) {}
#endif

extern "C" void mylocal_q2_rows_exact_marker(int n, float *out, size_t bs, const void *w,
                                           size_t bx, const void *x, size_t by, int nr) {
    mylocal_q2_avx2_dot(n, out, bs, w, bx, x, by, nr);
}

// Same diagnostic status/buffer contract as mylocal_q2_rows_test below.
extern "C" int mylocal_q2_rows_exact_test(int n, int k, const uint8_t *w, size_t stride,
                                          const uint8_t *q8, float *out) noexcept {
    if (!mylocal_q2_rows_available()) return 2;
    if (n <= 0 || k <= 0 || k % 64 || !w || !q8 || !out ||
        stride < size_t(k / 64)*18 ||
        size_t(n) > std::numeric_limits<size_t>::max()/stride) return 1;
    try {
        MyLocalQ2RowsActivation act(k);
        act.prepare(q8, k);
        mylocal_q2_rows_exact(w, stride, k, &act, 0, n, out);
        return 0;
    } catch (...) {
        return 3;
    }
}

// Diagnostic only: 0 success, 1 invalid dimensions/pointers, 2 unsupported ISA,
// 3 scratch allocation failure. Buffer sizes remain the caller's responsibility.
extern "C" int mylocal_q2_rows_test(int n, int k, const uint8_t *w, size_t stride,
                                    const uint8_t *q8, float *out) noexcept {
    if (!mylocal_q2_rows_available()) return 2;
    if (n <= 0 || k <= 0 || k % 64 || !w || !q8 || !out ||
        stride < size_t(k / 64)*18 ||
        size_t(n) > std::numeric_limits<size_t>::max()/stride) return 1;
    try {
        MyLocalQ2RowsActivation act(k);
        act.prepare(q8, k);
        mylocal_q2_rows(w, stride, k, &act, 0, n, out);
        return 0;
    } catch (...) {
        return 3;
    }
}
