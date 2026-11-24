#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>
// Lossless stored F32/F16/BF16 or IQ4_NL row decode, finite checked, no allocation.
cudaError_t ml_qsa_decode_rows_launch(cudaStream_t stream, const void* source,
    uint32_t type, uint64_t rows, uint64_t columns, uint64_t row_bytes,
    float* output, int* status);
// reference_fp32: lane-local decoded weights times unquantized FP32 input.
cudaError_t ml_qsa_projection_launch(cudaStream_t stream, const void* weight,
    uint32_t type, uint64_t rows, uint64_t columns, uint64_t row_bytes, const float* input,
    float* output, int* status);
