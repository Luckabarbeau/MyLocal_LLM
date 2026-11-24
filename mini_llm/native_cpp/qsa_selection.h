#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>
// Internal launch API: all pointers are engine-owned device storage.
cudaError_t ml_qsa_launch(cudaStream_t stream, uint64_t capacity, const uint64_t* n,
                         const float* query, const float* pooled, const float* dead,
                         float* scores, int64_t* ids, int* status, int policy, bool clear_status=true);
