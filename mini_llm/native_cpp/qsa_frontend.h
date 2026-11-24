#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>

// Internal FP32, contiguous, device-buffer API; no allocation or synchronization.
// Fixed Q24/K2, head width 256, index Q4 x 128, rotary width 64, base 1e7.
// n is the INCLUDING-current length (>=1); text position is n-1. Caller keeps
// n and all inputs stable through stream completion. All pointers are required,
// outputs must be disjoint from inputs and each other; sizes below are floats.
//
// Inputs: qg[24,512] = per-head [query, gate logits], k[2,256], raw[128],
// iq[4,128], first_raw[128] = historical raw key at token 0 (unused at n=1),
// partial[3,128] = chronological raw keys at n-4,n-3,n-2, read ONLY when n%4=0.
// Converted gamma_q/k[256], gamma_iq/ik[128] are used directly, NOT 1+gamma.
// RMS(x) = (x * (1/sqrt(mean_fp32(x*x)+epsilon))) * gamma, epsilon finite >0.
//
// Outputs: q[24,256], rotated_k[2,256], gate[24,256] (copied logits, NOT
// sigmoid), rotated_iq[4,128], dead[128] = RMS(raw if n=1 else first_raw),
// with NO RoPE on dead. Q/K/IQ use text position n-1. Half-layout RoPE pairs
// i and i+32 for i=0..31, angle=FP32(position)*powf(1e7,-i/32); suffix >=64
// is copied from the normalized row. Scalar text only, no multimodal sections.
//
// new_pool_ready is a device int, overwritten with 0 or 1 on EVERY launch.
// At n%4=0, new_pool[128] = RoPE(RMS((((partial[0]+partial[1])+partial[2])
// +raw)*0.25), position=n-4). Every addition is explicitly FP32 left-to-right.
// Otherwise new_pool is untouched and new_pool_ready=0. The flag describes
// block completion, NOT validity: caller must check status before using outputs.
//
// Caller initializes device status to 0. Nonfinite consumed inputs, intermediate
// or output values, or n=0 atomically set status=1 (never cleared here).
// Historical first_raw/partial are not inspected when unused. Invalid host
// pointers/epsilon return cudaErrorInvalidValue without launching or setting
// device status. Return value covers immediate launch errors only; caller checks
// asynchronous CUDA errors and status at its existing stream completion gate.
// No fast math/approximate rsqrt; FP32 reduction order can differ from Python.
cudaError_t ml_qsa_frontend_launch(cudaStream_t stream, const uint64_t* n,
    const float* qg, const float* k, const float* raw, const float* iq,
    const float* first_raw, const float* partial,
    const float* gamma_q, const float* gamma_k,
    const float* gamma_iq, const float* gamma_ik, float epsilon,
    float* q, float* rotated_k, float* gate, float* rotated_iq, float* dead,
    int* new_pool_ready, float* new_pool, int* status);
