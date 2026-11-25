#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>
cudaError_t ml_qsa_paged_pack(cudaStream_t, const float*, const float*, uint64_t,
    bool, unsigned char*, int*);
// logical=UINT64_MAX gathers current rows and initializes compact n/IDs.
// Each historical call gathers only IDs in one logical page into stable scratch.
cudaError_t ml_qsa_paged_gather(cudaStream_t, const uint64_t*, const int64_t*,
    uint64_t, uint64_t, bool, const unsigned char*, float*, uint64_t*, int64_t*);
