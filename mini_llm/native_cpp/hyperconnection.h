#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>

// Exact stored F32(type=0) or BF16(type=30) -> FP32 conversion on the supplied
// stream. Every consumed value is finite-checked; no numerical rounding of BF16
// occurs (zero-extend its bit pattern). Separate caller buffers, count>0.
cudaError_t ml_hc_decode_launch(cudaStream_t stream, const void* source, int type,
    uint64_t count, float* destination, int* status);

// Isolated reference_fp32 single-token HC primitives. All arrays are contiguous
// FP32 device buffers on the caller's current device. No allocation, copies,
// synchronization, activation cache, BF16 rounding, or implicit precision mode.
// Output-first weights: gamma[4,D] (converted multiplicative scale, NO +1),
// down[R,4D], up[4D,R], inject[4,4D]. D>0, R>0, 4D<=INT_MAX, R<=INT_MAX-4.
// Epsilon is finite positive FP32. Four separate RMS groups, each of width D:
// n[c,j] = (h[c,j]/sqrt(mean_D(h[c]*h[c])+epsilon))*gamma[c,j].
// a = (down@n)*0.25; low = a*sigmoid(a).
// mixed[j] = sum(c=0..3, n[c,j]*sigmoid(up[c*D+j]@low))*0.25.
// gains[c] = 2*sigmoid((inject[c]@n)*0.25).
// All arithmetic stays FP32; separate multiply/add and divide/sqrt are explicit.
// Reduction order follows existing HC CUDA kernels, not necessarily GEMM order.
//
// Scratch normalized[4D], low[R]; outputs mixed[D], gains[4]. All scratch and
// outputs must be disjoint from inputs AND one another. All pointers required
// except inject and gains: BOTH null means read-only, exactly one null is invalid.
// Caller keeps buffers alive/stable until stream completion. Sizes, contiguity,
// device ownership, pointer validity and nonoverlap are caller obligations.
//
// status is a caller-owned device int, initialized to 0 on the same stream.
// Any nonfinite consumed input, arithmetic intermediate or output atomically
// sets status=1; never cleared here (even when sigmoid masks an infinity).
// Outputs are invalid if status!=0. Host argument errors return
// cudaErrorInvalidValue without launching or changing status. Return codes cover
// launch submission errors only. Caller also checks asynchronous CUDA errors
// and status at its existing completion gate. All launches use exactly stream.
cudaError_t ml_hc_read_launch(cudaStream_t stream, int d, int rank, float epsilon,
    const float* h, const float* gamma, const float* down, const float* up,
    const float* inject, float* normalized, float* low, float* mixed,
    float* gains, int* status);

// Read-only final HC, identical to read with inject=gains=nullptr. There is NO
// injection or additional conventional output RMSNorm after this collapse.
cudaError_t ml_hc_collapse_launch(cudaStream_t stream, int d, int rank,
    float epsilon, const float* h, const float* gamma, const float* down,
    const float* up, float* normalized, float* low, float* mixed, int* status);

// new_h[c,j] = ORIGINAL_h[c,j] + gains[c]*block_output[j], NOT normalized h
// and NOT h plus block_input. h/new_h[4D], block_output[D], gains[4]. All
// pointers required; new_h disjoint from all inputs (no in-place write).
// Pair gains with the read that consumed exactly these original streams.
cudaError_t ml_hc_write_launch(cudaStream_t stream, int d, const float* h,
    const float* block_output, const float* gains, float* new_h, int* status);
