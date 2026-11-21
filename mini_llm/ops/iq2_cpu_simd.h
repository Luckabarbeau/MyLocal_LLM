/* IQ2_S AVX2 gather variant of ggml_vec_dot_iq2_s_q8_K.
 * Arithmetic/layout authority: llama.cpp 171e8846b4af9766c354064cb776cb34a50f053f,
 * ggml/src/ggml-cpu/arch/x86/quants.c. Preserves its integer lane sums,
 * per-block FMA, horizontal sum and final 1/8 scale; Q8_K is unchanged.
 *
 * MIT License
 * Copyright (c) 2023-2026 The ggml authors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#include <cstring>
#include "gguf_codebooks.h"
#define MYLOCAL_IQ2_TARGET __attribute__((target("avx2,f16c,fma")))

struct alignas(32) MyLocalIQ2Grid {
    long long values[1024];
    MyLocalIQ2Grid() {
        for (int i=0;i<1024;++i) std::memcpy(values+i,iq2s_grid[i],8);
    }
};
static const MyLocalIQ2Grid mylocal_iq2_grid;

extern "C" int mylocal_iq2_gather_available() noexcept {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c") && __builtin_cpu_supports("fma");
}

extern "C" MYLOCAL_IQ2_TARGET void mylocal_iq2_gather_dot(int n,float *out,size_t,
    const void *weights,size_t,const void *activation,size_t,int) {
    const uint8_t *x=static_cast<const uint8_t*>(weights);
    const uint8_t *y=static_cast<const uint8_t*>(activation);
    const __m256i shuf=_mm256_set_epi64x(0x0303030303030303ll,0x0202020202020202ll,
                                       0x0101010101010101ll,0);
    const __m256i bits=_mm256_setr_epi8(1,2,4,8,16,32,64,-128,1,2,4,8,16,32,64,-128,
                                      1,2,4,8,16,32,64,-128,1,2,4,8,16,32,64,-128);
    const __m128i shifts=_mm_setr_epi32(8,6,4,2), high=_mm_set1_epi32(0x300);
    __m256 accum=_mm256_setzero_ps();
    for(int block=0;block<n/256;++block,x+=82,y+=292) {
        uint16_t half; float yd;
        std::memcpy(&half,x,2); std::memcpy(&yd,y,4);
        const float xd=_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(half)));
        const float scale=xd*yd;
        __m256i sum1=_mm256_setzero_si256(),sum2=_mm256_setzero_si256();
        for(int group=0;group<8;group+=2) {
            int low1,low2; uint32_t sign1,sign2;
            std::memcpy(&low1,x+2+4*group,4); std::memcpy(&low2,x+6+4*group,4);
            const __m128i idx1=_mm_or_si128(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(low1)),
                _mm_and_si128(_mm_sllv_epi32(_mm_set1_epi32(x[66+group]),shifts),high));
            const __m128i idx2=_mm_or_si128(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(low2)),
                _mm_and_si128(_mm_sllv_epi32(_mm_set1_epi32(x[67+group]),shifts),high));
            const __m256i grid1=_mm256_i32gather_epi64(mylocal_iq2_grid.values,idx1,8);
            const __m256i grid2=_mm256_i32gather_epi64(mylocal_iq2_grid.values,idx2,8);
            std::memcpy(&sign1,x+34+4*group,4); std::memcpy(&sign2,x+38+4*group,4);
            const __m256i mask1=_mm256_cmpeq_epi8(_mm256_and_si256(
                _mm256_shuffle_epi8(_mm256_set1_epi32(sign1),shuf),bits),bits);
            const __m256i mask2=_mm256_cmpeq_epi8(_mm256_and_si256(
                _mm256_shuffle_epi8(_mm256_set1_epi32(sign2),shuf),bits),bits);
            const __m256i a1=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(y+4+32*group));
            const __m256i a2=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(y+36+32*group));
            const __m256i q1=_mm256_sub_epi8(_mm256_xor_si256(mask1,a1),mask1);
            const __m256i q2=_mm256_sub_epi8(_mm256_xor_si256(mask2,a2),mask2);
            const int s1=x[74+group],s2=x[75+group];
            const __m256i scales1=_mm256_set_m128i(_mm_set1_epi16(2*(s1>>4)+1),_mm_set1_epi16(2*(s1&15)+1));
            const __m256i scales2=_mm256_set_m128i(_mm_set1_epi16(2*(s2>>4)+1),_mm_set1_epi16(2*(s2&15)+1));
            sum1=_mm256_add_epi32(sum1,_mm256_madd_epi16(_mm256_maddubs_epi16(grid1,q1),scales1));
            sum2=_mm256_add_epi32(sum2,_mm256_madd_epi16(_mm256_maddubs_epi16(grid2,q2),scales2));
        }
        accum=_mm256_fmadd_ps(_mm256_set1_ps(scale),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sum1,sum2)),accum);
    }
    __m128 sum=_mm_add_ps(_mm256_extractf128_ps(accum,1),_mm256_castps256_ps128(accum));
    sum=_mm_add_ps(sum,_mm_movehl_ps(sum,sum));
    sum=_mm_add_ss(sum,_mm_movehdup_ps(sum));
    *out=0.125f*_mm_cvtss_f32(sum);
}
#else
extern "C" int mylocal_iq2_gather_available() noexcept { return 0; }
extern "C" void mylocal_iq2_gather_dot(int,float*,size_t,const void*,size_t,const void*,size_t,int) {}
#endif
