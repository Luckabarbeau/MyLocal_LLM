/* IQ3_XXS AVX2 gather derivative of ggml_vec_dot_iq3_xxs_q8_K.
 * Authority: llama.cpp 171e8846b4af9766c354064cb776cb34a50f053f,
 * ggml/src/ggml-cpu/arch/x86/quants.c. Original Q8_K producer, integer pair
 * sums, FMA order and final 0.25f scaling are preserved.
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
#define MYLOCAL_IQ3XXS_TARGET __attribute__((target("avx2,f16c,fma")))
struct alignas(32) MyLocalIQ3XXSTables {
    int grid[256];
    long long signs[128];
    MyLocalIQ3XXSTables() {
        for(int i=0;i<256;++i) std::memcpy(grid+i,iq3xxs_grid[i],4);
        for(int i=0;i<128;++i) {
            uint64_t v=0;
            for(int b=0;b<8;++b)
                v|=uint64_t((ksigns_iq2xs[i]&(1<<b))?255:1)<<(8*b);
            std::memcpy(signs+i,&v,8);
        }
    }
};
static const MyLocalIQ3XXSTables mylocal_iq3xxs_tables;
extern "C" int mylocal_iq3xxs_gather_available() noexcept {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c") && __builtin_cpu_supports("fma");
}
extern "C" MYLOCAL_IQ3XXS_TARGET void mylocal_iq3xxs_gather_dot(int n,float *out,size_t,
    const void *weights,size_t,const void *activation,size_t,int) {
    const uint8_t *x=static_cast<const uint8_t*>(weights);
    const uint8_t *y=static_cast<const uint8_t*>(activation);

    __m256 accum=_mm256_setzero_ps();
    for(int block=0;block<n/256;++block,x+=98,y+=292) {
        uint16_t half; float yd;
        std::memcpy(&half,x,2); std::memcpy(&yd,y,4);
        const float xd=_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(half)));
        const float scale=xd*yd;
        __m256i sum1=_mm256_setzero_si256(),sum2=_mm256_setzero_si256();
        for(int group=0;group<8;group+=2) {
            const __m256i idx1=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(x+2+8*group)));
            const __m256i idx2=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(x+10+8*group)));
            const __m256i grid1=_mm256_i32gather_epi32(mylocal_iq3xxs_tables.grid,idx1,4);
            const __m256i grid2=_mm256_i32gather_epi32(mylocal_iq3xxs_tables.grid,idx2,4);
            uint32_t aux1,aux2;
            std::memcpy(&aux1,x+66+4*group,4); std::memcpy(&aux2,x+70+4*group,4);
            const auto *signs=mylocal_iq3xxs_tables.signs;
            const __m256i signs1=_mm256_set_epi64x(signs[(aux1>>21)&127],signs[(aux1>>14)&127],
                signs[(aux1>>7)&127],signs[aux1&127]);
            const __m256i signs2=_mm256_set_epi64x(signs[(aux2>>21)&127],signs[(aux2>>14)&127],
                signs[(aux2>>7)&127],signs[aux2&127]);
            const __m256i a1=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(y+4+32*group));
            const __m256i a2=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(y+36+32*group));
            const __m256i dot1=_mm256_maddubs_epi16(grid1,_mm256_sign_epi8(a1,signs1));
            const __m256i dot2=_mm256_maddubs_epi16(grid2,_mm256_sign_epi8(a2,signs2));
            sum1=_mm256_add_epi32(sum1,_mm256_madd_epi16(dot1,_mm256_set1_epi16(2*(aux1>>28)+1)));
            sum2=_mm256_add_epi32(sum2,_mm256_madd_epi16(dot2,_mm256_set1_epi16(2*(aux2>>28)+1)));
        }
        accum=_mm256_fmadd_ps(_mm256_set1_ps(scale),_mm256_cvtepi32_ps(_mm256_add_epi32(sum1,sum2)),accum);
    }
    __m128 sum=_mm_add_ps(_mm256_extractf128_ps(accum,1),_mm256_castps256_ps128(accum));
    sum=_mm_add_ps(sum,_mm_movehl_ps(sum,sum));
    sum=_mm_add_ss(sum,_mm_movehdup_ps(sum));
    *out=0.25f*_mm_cvtss_f32(sum);
}
#else
extern "C" int mylocal_iq3xxs_gather_available() noexcept { return 0; }
extern "C" void mylocal_iq3xxs_gather_dot(int,float*,size_t,const void*,size_t,const void*,size_t,int) {}
#endif
