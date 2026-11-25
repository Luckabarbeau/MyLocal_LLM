#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>
// Table entries are physical slot + 1; zero means nonresident.
cudaError_t ml_kv_selected_launch(cudaStream_t stream, const uint64_t* n,
    const int64_t* ids, uint64_t active_n, const uint64_t* table, uint64_t page_size,
    uint64_t physical_pages, const float* pages, const float* q, const float* gate,
    const float* k, const float* v, float* scratch, uint64_t* compact_n,
    int64_t* compact_ids, float* contexts, float* gated, int* status, float scale);
