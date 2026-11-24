#pragma once
#include <cuda_runtime_api.h>

// Internal C++ launch API, one token/sequence, D=2560, four streams.
// Contiguous device FP32, caller's current device and supplied stream only.
// No allocation/copy/synchronization, C ABI, state ownership or table decoding.
// All buffers must be mutually disjoint, including inputs vs scratch/outputs;
// read-only inputs may share storage. Keep all storage live until completion.
// status: caller device int initialized to zero on stream; sticky 1 on any
// consumed nonfinite value/intermediate/output. Outputs invalid if status!=0.
// Host invalid arguments return cudaErrorInvalidValue without launching/writes.
// Return covers submission errors; caller checks asynchronous errors + status.

// Optional dense output-first projections: embedding[D], key_weight[4D,D],
// value_weight[D,D] -> projected_key[4,D], value[D]. No biases or rounding.
// May be replaced by existing exact projection machinery at integration time.
cudaError_t ml_ple_project_launch(cudaStream_t stream, const float* embedding,
    const float* key_weight, const float* value_weight,
    float* projected_key, float* value, int* status);

// RMS(x,g)=(x/sqrt(mean_D(x*x)+epsilon))*g, separately for each stream.
// Gammas are converted multiplicative scales, NO added 1.
// Inputs: streams[4,D], projected_key[4,D], value[D], gamma_{key,query,conv}[4,D],
// weights[4,D,4] (same storage as reference [4D,4]), history[9,4,D] oldest first.
// Scratch: key_norm[4,D], query_norm[4,D], u[4,D].
// Outputs: gates[4], z[4,D], updated[4,D], next_history[9,4,D].
// score=sum_D(key_norm*query_norm)/sqrt(D);
// gates=sigmoid(sign(score)*sqrt(max(abs(score),1e-6))); sign(0)=0.
// u=gates[:,None]*value; z=RMS(u,gamma_conv).
// conv=history[0]*w[...,0]+history[3]*w[...,1]+history[6]*w[...,2]+z*w[...,3].
// updated=streams+(u+conv*sigmoid(conv)); next_history=concat(history[1:],z).
// EOS NEVER clears history. Fresh sequence: zero history + (-1,-1) hash context.
// epsilon must be finite positive FP32. FP32 arithmetic, no FMA/fast math;
// parallel reductions need not be bit-identical to Python GEMM/reductions.
cudaError_t ml_ple_apply_launch(cudaStream_t stream, float epsilon,
    const float* streams, const float* projected_key, const float* value,
    const float* gamma_key, const float* gamma_query, const float* gamma_conv,
    const float* weights, const float* history,
    float* key_norm, float* query_norm, float* u, float* gates, float* z,
    float* updated, float* next_history, int* status);
