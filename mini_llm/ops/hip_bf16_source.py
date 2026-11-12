"""HIP BF16 spelling adapter for CuPy-generated CUDA kernel sources."""


_HIP_BF16_PREAMBLE = r"""
#include <hip/hip_bf16.h>
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
#define __float2bfloat16_rn __float2bfloat16
#define __floats2bfloat162_rn(a, b) \
    __halves2bfloat162(__float2bfloat16(a), __float2bfloat16(b))
"""


def adapt_bf16_source(code, *, is_hip):
    """Change only HIP's header/type/conversion names, preserving CUDA source."""
    if not is_hip:
        return code
    return code.replace("#include <cuda_bf16.h>", _HIP_BF16_PREAMBLE)
