#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>
// Device IDs/n are owned by the retained selection workspace. No ID readback.
cudaError_t ml_qsa_attention_launch(cudaStream_t stream, const uint64_t* n,
    const int64_t* ids, const float* q, const float* gate,
    const float* k, const float* v, const float* keys, const float* values,
    float* contexts, float* gated, int* status, float scale);
