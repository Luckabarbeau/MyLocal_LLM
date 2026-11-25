#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>
constexpr uint64_t ML_KV_I8_ROW=1056;
struct KVPageRequests {
    uint64_t count, misses, error;
    uint64_t pages[2051], missing[2051];
};
cudaError_t ml_kvi8_quantize(cudaStream_t, const float*, const float*, uint64_t,
                            unsigned char*, int*);
cudaError_t ml_kvi8_requests(cudaStream_t, const uint64_t*, const int64_t*,
    const uint64_t*, uint64_t, uint64_t, KVPageRequests*);
cudaError_t ml_kvi8_attention_launch(cudaStream_t, const uint64_t*, const int64_t*, uint64_t,
    const uint64_t*, uint64_t, uint64_t, const unsigned char*, const unsigned char*,
    const float*, const float*, float*, uint64_t*, int64_t*, float*, float*, int*, float);
