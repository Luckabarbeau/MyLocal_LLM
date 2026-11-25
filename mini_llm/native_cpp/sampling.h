#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stddef.h>
#include "runtime.h"
// CUB sort/reduce own no allocator: caller supplies max(sort, reduce+header)
// temporary storage. Greedy ArgMax and stochastic sort reuse the same allocation.
size_t ml_sampling_temp_bytes(int n);
cudaError_t ml_sampling_launch(cudaStream_t stream, const float* logits, int n,
    float temperature, double top_p, int k, uint64_t seed, uint64_t counter,
    int supplied, double uniform, float* keys_in, float* keys_out,
    int* ids_in, int* ids_out, float* weights, float* cdf,
    void* temporary, size_t temporary_bytes, ml_sample_result* result);
