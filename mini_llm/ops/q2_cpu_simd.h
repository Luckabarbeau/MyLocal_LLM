// SPDX-License-Identifier: LGPL-3.0-or-later
// Original Q2_0 integer SIMD dot; GGML Q8_0 bytes and scalar float ordering.
// Layout authority: pinned GGML 171e8846, block_q2_0 / block_q8_0.
// Strata 712fec6 CPU rows motivate SIMD decode, not a change to ActQ precision.
#pragma once
#include <cstring>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define MYLOCAL_Q2_TARGET __attribute__((target("avx2,f16c")))
extern "C" int mylocal_q2_avx2_available() noexcept {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
}

MYLOCAL_Q2_TARGET static float q2_half(const uint8_t *p) {
    uint16_t bits;
    std::memcpy(&bits, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

// Decode one 64-value block into two unsigned-code vectors, in value order.
MYLOCAL_Q2_TARGET static void q2_codes(const uint8_t *p, __m256i &a, __m256i &b) {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
    const __m128i mask = _mm_set1_epi8(3);
    const __m128i c0 = _mm_and_si128(packed, mask);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(packed, 2), mask);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi16(packed, 6), mask);
    const __m128i low01 = _mm_unpacklo_epi8(c0, c1), low23 = _mm_unpacklo_epi8(c2, c3);
    const __m128i high01 = _mm_unpackhi_epi8(c0, c1), high23 = _mm_unpackhi_epi8(c2, c3);
    a = _mm256_set_m128i(_mm_unpackhi_epi16(low01, low23), _mm_unpacklo_epi16(low01, low23));
    b = _mm256_set_m128i(_mm_unpackhi_epi16(high01, high23), _mm_unpacklo_epi16(high01, high23));
}

MYLOCAL_Q2_TARGET static int q2_integer_dot(__m256i codes, const uint8_t *q) {
    const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q));
    const __m256i ones = _mm256_set1_epi16(1);
    // Work with unsigned codes0..3, subtract sum(q) for the code offset -1.
    // Pair products <=768, so maddubs cannot saturate, even for q=-128.
    const __m256i weighted = _mm256_madd_epi16(_mm256_maddubs_epi16(codes, values), ones);
    const __m256i correction = _mm256_madd_epi16(
        _mm256_maddubs_epi16(_mm256_set1_epi8(1), values), ones);
    const __m256i sums = _mm256_sub_epi32(weighted, correction);
    __m128i total = _mm_add_epi32(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
    total = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0x4e));
    total = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0xb1));
    return _mm_cvtsi128_si32(total);
}

extern "C" MYLOCAL_Q2_TARGET void mylocal_q2_avx2_dot(
    int n, float *out, size_t, const void *w, size_t, const void *x, size_t, int) {
    const auto *weights = static_cast<const uint8_t *>(w);
    const auto *activation = static_cast<const uint8_t *>(x);
    float result = 0.0f;
    for (int block = 0; block < n/64; ++block) {
        __m256i low, high;
        q2_codes(weights+18*block+2, low, high);
        const uint8_t *a = activation+68*block;
        float subtotal = 0.0f;
        subtotal += q2_half(a)*float(q2_integer_dot(low, a+2));
        subtotal += q2_half(a+34)*float(q2_integer_dot(high, a+36));
        result += q2_half(weights+18*block)*subtotal;
    }
    *out = result;
}
#else
extern "C" int mylocal_q2_avx2_available() noexcept { return 0; }
extern "C" void mylocal_q2_avx2_dot(int, float *, size_t, const void *, size_t,
                                    const void *, size_t, int) {}
#endif
