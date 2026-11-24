#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
namespace {
__global__ void finite(const float* p, uint64_t count, int* status) {
    for (uint64_t i=blockIdx.x*blockDim.x+threadIdx.x; i<count; i+=gridDim.x*blockDim.x)
        if (!isfinite(p[i])) atomicExch(status,1);
}
}
cudaError_t ml_qsa_layer_finite(cudaStream_t stream, const float* p, uint64_t count, int* status) {
    finite<<<256,256,0,stream>>>(p,count,status);
    return cudaGetLastError();
}
