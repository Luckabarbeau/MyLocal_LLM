/* Paired IQ3_XXS gate/up, based on pinned GGML AVX2 arithmetic.
 * llama.cpp 171e8846b4af9766c354064cb776cb34a50f053f,
 * ggml/src/ggml-cpu/arch/x86/quants.c. License/attribution in
 * iq3xxs_cpu_simd.h applies (MIT, Copyright 2023-2026 The ggml authors).
 * Only Q8_K input loads are shared; each row retains its independent sums,
 * scale multiplication, FMA chain, horizontal reduction and final quarter.
 */
#pragma once
#include "iq3xxs_cpu_simd.h"
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
template<int Rows> MYLOCAL_IQ3XXS_TARGET static inline void mylocal_iq3xxs_rows(
    int n, float *out, const uint8_t *gate, const uint8_t *up, const uint8_t *q) {
    __m256 accum[Rows];
    for(int r=0;r<Rows;++r) accum[r]=_mm256_setzero_ps();
    for(int block=0;block<n/256;++block,q+=292) {
        float yd; std::memcpy(&yd,q,4);
        __m256i sums1[Rows],sums2[Rows];
        for(int r=0;r<Rows;++r) sums1[r]=sums2[r]=_mm256_setzero_si256();
        for(int group=0;group<8;group+=2) {
            const __m256i a1=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q+4+32*group));
            const __m256i a2=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q+36+32*group));
            #pragma GCC unroll 2
            for(int r=0;r<Rows;++r) {
                const uint8_t *x=(r?up:gate)+size_t(block)*98;
                const auto *grid=mylocal_iq3xxs_tables.grid;
                const auto *signs=mylocal_iq3xxs_tables.signs;
                const uint8_t *idx=x+2+8*group;
                const __m256i g1=_mm256_set_epi32(grid[idx[7]],grid[idx[6]],grid[idx[5]],grid[idx[4]],
                    grid[idx[3]],grid[idx[2]],grid[idx[1]],grid[idx[0]]);
                const __m256i g2=_mm256_set_epi32(grid[idx[15]],grid[idx[14]],grid[idx[13]],grid[idx[12]],
                    grid[idx[11]],grid[idx[10]],grid[idx[9]],grid[idx[8]]);
                uint32_t aux1,aux2;
                std::memcpy(&aux1,x+66+4*group,4); std::memcpy(&aux2,x+70+4*group,4);
                const __m256i s1=_mm256_set_epi64x(signs[(aux1>>21)&127],signs[(aux1>>14)&127],
                    signs[(aux1>>7)&127],signs[aux1&127]);
                const __m256i s2=_mm256_set_epi64x(signs[(aux2>>21)&127],signs[(aux2>>14)&127],
                    signs[(aux2>>7)&127],signs[aux2&127]);
                const __m256i dot1=_mm256_maddubs_epi16(g1,_mm256_sign_epi8(a1,s1));
                const __m256i dot2=_mm256_maddubs_epi16(g2,_mm256_sign_epi8(a2,s2));
                sums1[r]=_mm256_add_epi32(sums1[r],_mm256_madd_epi16(dot1,_mm256_set1_epi16(2*(aux1>>28)+1)));
                sums2[r]=_mm256_add_epi32(sums2[r],_mm256_madd_epi16(dot2,_mm256_set1_epi16(2*(aux2>>28)+1)));
            }
        }
        #pragma GCC unroll 2
        for(int r=0;r<Rows;++r) {
            uint16_t half; std::memcpy(&half,(r?up:gate)+size_t(block)*98,2);
            const float xd=_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(half)));
            accum[r]=_mm256_fmadd_ps(_mm256_set1_ps(xd*yd),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sums1[r],sums2[r])),accum[r]);
        }
    }
    for(int r=0;r<Rows;++r) {
        __m128 sum=_mm_add_ps(_mm256_extractf128_ps(accum[r],1),_mm256_castps256_ps128(accum[r]));
        sum=_mm_add_ps(sum,_mm_movehl_ps(sum,sum));
        sum=_mm_add_ss(sum,_mm_movehdup_ps(sum));
        out[r]=0.25f*_mm_cvtss_f32(sum);
    }
}
extern "C" MYLOCAL_IQ3XXS_TARGET void mylocal_iq3xxs_pair_single(int n,float *out,size_t,
    const void *weights,size_t,const void *activation,size_t,int) {
    mylocal_iq3xxs_rows<1>(n,out,static_cast<const uint8_t*>(weights),nullptr,static_cast<const uint8_t*>(activation));
}
#else
extern "C" void mylocal_iq3xxs_pair_single(int,float*,size_t,const void*,size_t,const void*,size_t,int) {}
#endif
