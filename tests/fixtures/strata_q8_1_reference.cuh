// Standalone extracted CUDA oracle; NOT generated from the Python implementation.
// Strata pinned 712fec6b4297c9cfd32ab43daaa2979a9aa01edf.
// Compile native with --use_fast_math; IEEE control without it.
// Only adaptation: includes/constants/layout supplied; PDL scheduling hooks are no-ops.
// https://raw.githubusercontent.com/Niko1221/Strata/712fec6b4297c9cfd32ab43daaa2979a9aa01edf/src/kernels/cuda/native_mmvq.cu
// SHA256 5fb3dde666b64380cc803618f25b079d80a47b29b1dadf7650afad3932fd6a9f
// https://raw.githubusercontent.com/Niko1221/Strata/712fec6b4297c9cfd32ab43daaa2979a9aa01edf/include/strata/kernels/q8_1_finite.hpp
// SHA256 346e8d2e17f015f7b4bd1234d811a866b545fb376d69b25e02bb7b50253f0ca7
// https://raw.githubusercontent.com/Niko1221/Strata/712fec6b4297c9cfd32ab43daaa2979a9aa01edf/LICENSE
// SHA256 949b9b0aada7476006469e58ee4ffb0e0f68c2d9137e90a356757b61a09d37f0
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

Additional native_mmvq.cu attribution/license:
 Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
 ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
 and ggml/src/ggml-common.h. See docs/native-mmvq.md for exact scope.

 MIT License
 Copyright (c) 2023-2026 The ggml authors

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
// NVRTC has no host libc include search path; use CUDA's CCCL headers.
#include <cuda/std/cstdint>
#include <cuda/std/cstddef>
using cuda::std::int8_t;
#include <cuda_fp16.h>
#define STRATA_PDL_RESTRICT
__device__ void pdl_trigger() {}
__device__ void pdl_wait() {}
// include/strata/kernels/q8_1_finite.hpp - #606: the q8_1 activation block's scale and sum, kept finite.
//
// A q8_1 block stores its scale d = amax / 127 and the sum of its 32 raw activations as fp16. A sum past 65504 (one
// massive-activation dimension among 32 values is enough) rounds to inf, and d does past amax = 8.3M; the dot
// products then read inf * 0 = NaN, and a NaN in the residual ends as one token repeated (ggml-org/llama.cpp#23606
// is the same defect). Clamped to the largest finite half, such a block gives an approximate product instead of NaN,
// and the values are held to the int8 range they then exceed. Every block that was finite before is stored bit for
// bit as before: values in (65504, 65520) round to 65504 anyway, and a finite block's |x / d| is at most 127. A NaN
// stays NaN (the comparisons are false for it), so STRATA_DBG_NAN still sees one.
#pragma once

#include <cuda_fp16.h>

namespace strata::kernels {

constexpr float kHalfMax = 65504.0f;

__device__ __forceinline__ float q8_1_finite(const float v) {
    return fabsf(v) > kHalfMax ? copysignf(kHalfMax, v) : v;
}

/// #3948: native_mmvq.cu (native_quantize_q8_1, the separate quantiser every consumer of a q8_1 image was written for)
/// is compiled with --use_fast_math, which makes `a / b` the fast full-range division (div.full.f32, up to 2 ulp off). The
/// fused producers (fused_gr.cu's hyper-connection read, verify_kernels.cu's GDN output norm: STRATA_QFUSE) are not, so
/// their IEEE division gave a different code at the rare element whose x / d lies next to a rounding boundary (found by
/// gr_parity: one code in about 10^6, off by one). A translation unit that defines STRATA_Q81_FASTMATH_DIV before including
/// this header divides the way the fast-math one does; every other one, and every HIP build, keeps `a / b` (so nothing
/// that compiled before this changes).
__device__ __forceinline__ float q81_div(const float a, const float b) {
#if defined(STRATA_Q81_FASTMATH_DIV) && defined(__CUDA_ARCH__) && !defined(__HIPCC__)
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
#else
    return a / b;
#endif
}

/// The block's quantized value of `xi` for the (finite) scale `d`; `amax` = 0 is an all-zero block.
__device__ __forceinline__ int8_t q8_1_quant(const float xi, const float d, const float amax) {
    if (amax == 0.0f) return 0;
    const float q = roundf(q81_div(xi, d));
    return (int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
}

__device__ __forceinline__ half2 q8_1_ds(const float d, const float sum) {
    return make_half2(__float2half(q8_1_finite(d)), __float2half(q8_1_finite(sum)));
}

}  // namespace strata::kernels

namespace strata::kernels {
constexpr int WARP=32, Q8K=32, QUANT_THREADS=256;
struct Q81Block { half2 ds; int8_t qs[32]; };
// NVRTC's offsetof expansion is not a constant expression. These two members
// fill all 36 bytes, so qs must immediately follow the 4-byte ds (no padding).
static_assert(sizeof(Q81Block)==36 && alignof(Q81Block)==4 &&
              sizeof(half2)==4 && sizeof(Q81Block::qs)==32,
              "Q8_1 layout: 4-byte ds followed by 32 codes");
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    }
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP));
    }
    return x;
}

extern "C"
__launch_bounds__(QUANT_THREADS, 1)
__global__ void native_quantize_q8_1_kernel(const float* x_, Q81Block* y_, int n_in) {
    const float* STRATA_PDL_RESTRICT x = x_;   // __restrict__ below sm_70 only (pdl.hpp, #1469)
    Q81Block* STRATA_PDL_RESTRICT y = y_;
    pdl_trigger();   // PDL (pdl.hpp): the projection after this one may start loading its weights
    pdl_wait();
    const int i = int(blockIdx.x) * QUANT_THREADS + int(threadIdx.x);
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float xi = x[i];
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);   // #606: q8_1_finite.hpp - the same bits for every finite block
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

}
