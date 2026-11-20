/* SPDX-License-Identifier: MIT
 * Derived from llama.cpp171e8846 ggml_vec_dot_iq4_nl_q8_0 (x86 AVX2).
 * Copyright (c) 2023-2026 The ggml authors.
 * Full MIT notice is in the included gguf_codebooks.h and must accompany this
 * source. Retains paired-block FMA/hsum/scalar-tail order and GGML Q8_0 bytes.
 * VNNI combines the two exact non-saturating integer pair-sum instructions.
 */
#pragma once
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__) && __GNUC__>=11
#include <immintrin.h>
#include <cstring>
#include "gguf_codebooks.h"
#define MYLOCAL_IQ4_VNNI_TARGET __attribute__((target("avx2,f16c,fma,avxvnni")))

extern "C" int mylocal_iq4_vnni_available() noexcept {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c") &&
           __builtin_cpu_supports("fma") && __builtin_cpu_supports("avxvnni");
}
MYLOCAL_IQ4_VNNI_TARGET static inline float iq4_half(const uint8_t *p) {
    uint16_t bits; std::memcpy(&bits,p,2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}
MYLOCAL_IQ4_VNNI_TARGET static inline __m256i iq4_values(const uint8_t *p,__m128i table,__m128i mask) {
    __m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    __m128i lo=_mm_shuffle_epi8(table,_mm_and_si128(bits,mask));
    __m128i hi=_mm_shuffle_epi8(table,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
    return _mm256_insertf128_si256(_mm256_castsi128_si256(lo),hi,1);
}
MYLOCAL_IQ4_VNNI_TARGET static inline __m256i iq4_dot(__m256i weights,__m256i values) {
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(),_mm256_abs_epi8(weights),
                                   _mm256_sign_epi8(values,weights));
}
extern "C" MYLOCAL_IQ4_VNNI_TARGET void mylocal_iq4_vnni_dot(int n,float *out,size_t,
    const void *weights,size_t,const void *activation,size_t,int) {
    const uint8_t *x=static_cast<const uint8_t*>(weights);
    const uint8_t *y=static_cast<const uint8_t*>(activation);
    const __m128i table=_mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i mask=_mm_set1_epi8(15);
    __m256 a=_mm256_setzero_ps(),b=_mm256_setzero_ps();
    int block=0;
    for(;block+1<n/32;block+=2,x+=36,y+=68) {
        __m256i q1=iq4_values(x+2,table,mask),q2=iq4_values(x+20,table,mask);
        __m256i v1=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(y+2));
        __m256i v2=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(y+36));
        const float d1=iq4_half(y)*iq4_half(x);
        const float d2=iq4_half(y+34)*iq4_half(x+18);
        a=_mm256_fmadd_ps(_mm256_set1_ps(d1),_mm256_cvtepi32_ps(iq4_dot(q1,v1)),a);
        b=_mm256_fmadd_ps(_mm256_set1_ps(d2),_mm256_cvtepi32_ps(iq4_dot(q2,v2)),b);
    }
    const __m256 sum=_mm256_add_ps(a,b);
    __m128 r=_mm_add_ps(_mm256_extractf128_ps(sum,1),_mm256_castps256_ps128(sum));
    r=_mm_add_ps(r,_mm_movehl_ps(r,r)); r=_mm_add_ss(r,_mm_movehdup_ps(r));
    float result=_mm_cvtss_f32(r);
    if(block<n/32) {
        int s1=0,s2=0;
        const int8_t *q=reinterpret_cast<const int8_t*>(y+2);
        for(int j=0;j<16;++j) {
            s1+=int(q[j])*kvalues_iq4nl[x[j+2]&15];
            s2+=int(q[j+16])*kvalues_iq4nl[x[j+2]>>4];
        }
        const float d=iq4_half(y)*iq4_half(x);
        result+=d*float(s1+s2);
    }
    *out=result;
}
#else
extern "C" int mylocal_iq4_vnni_available() noexcept { return 0; }
extern "C" void mylocal_iq4_vnni_dot(int,float*,size_t,const void*,size_t,const void*,size_t,int) {}
#endif
