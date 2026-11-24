#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>

// Isolated single-token reference_fp32 primitives. No allocation, copies, host
// routing, synchronization, weight ownership or runtime registration. The parent
// registers/retains immutable packed payloads and uploads Expert[512]. All array
// arguments are on the current GPU; stream ordering/lifetime belong to caller.
// Build with --fmad=false (also for qsa_projection.cu). No activation Q8 path.
struct MLMoeMatrix {
    const void* data;
    uint64_t row_bytes; // physical stride; >= packed row bytes
    uint32_t type;      // GGML: 8 Q8_0, 18 IQ3_XXS, 20 IQ4_NL, 21 IQ3_S,
                        // 22 IQ2_S, 23 IQ4_XS, 42 Q2_0; shared GU 12/13/14/23
    uint32_t reserved;  // must be zero
};
struct MLMoeExpert { MLMoeMatrix gate, up, down; };

// status is a caller-initialized, sticky-nonzero device int. These kernels OR
// 1 for nonfinite and 2 for bad descriptor/route; the reused shared projection
// helper may overwrite a nonzero value with 1 on nonfinite data. Never clear
// status here; only zero/nonzero is a cross-primitive contract.
// cudaError_t reports host/launch errors;
// device errors require parent synchronization/status observation. Failed output
// is not consumable. Disjoint buffers required unless explicitly stated.
// logits[512] -> probabilities[512], ids[10] ascending, scaled_weights[10].
// Full 512 softmax, top10 logits (ties: lower ID first), selected-mass floor
// 2^-14, expert_scale applied ONCE here. Scale must be finite and nonnegative.
cudaError_t ml_moe_route_launch(cudaStream_t stream, const float* logits,
    float expert_scale, float* probabilities, int* ids, float* scaled_weights,
    int* status);

// Registered GPU descriptor list [512]; unselected entries may be null.
// GU: IQ3_XXS/IQ2_S/IQ3_S/IQ4_XS; down: IQ4_NL/Q2_0.
// d in [256,6144], f in [256,12288], D multiple of 256, F multiple of 64 (including checkpoint F=640).
// ids must be strictly ascending valid IDs. Inputs x[d], scaled_weights[10].
// Workspace hidden[10*f], terms[10*d] (weighted individual expert outputs),
// output[d]. Each output sums terms sequentially in ascending ID, no FMA.
cudaError_t ml_moe_experts_launch(cudaStream_t stream, int d, int f,
    const MLMoeExpert* experts, const int* ids, const float* scaled_weights,
    const float* x, float* hidden, float* terms, float* output, int* status);

// Runtime mixed-policy seam: resident[512] is immutable explicit placement.
// Nonresident terms are untouched until finish receives CPU rows in route slots.
cudaError_t ml_moe_resident_launch(cudaStream_t, int d, int f,
    const MLMoeExpert*, const int* resident, const int* ids, const float* weights,
    const float* x, float* hidden, float* terms, int* status);
cudaError_t ml_moe_finish_launch(cudaStream_t, int d, const int* ids,
    const int* resident, const float* weights, const float* cpu_rows, float* terms,
    float* routed, const float* shared, float* output, int* status);

// Shared GU uses unchanged ml_qsa_projection_launch (exact contiguous strides),
// Q4_K/Q5_K/Q6_K/IQ4_XS. Down IQ4_NL/Q8_0 supports padded row stride.
// scalar_gate is contiguous F32[d], independently projected and sigmoid gated.
// Workspace gate[f], up[f], hidden[f], score[1]. raw[d] is ungated shared down;
// output[d] is sigmoid(score)*raw. No expert_scale in the shared branch.
cudaError_t ml_moe_shared_launch(cudaStream_t stream, int d, int f,
    MLMoeExpert weights, const float* scalar_gate, const float* x,
    float* gate, float* up, float* hidden, float* score, float* raw,
    float* output, int* status);

// output[d] = routed[d] + shared[d]; output may alias either input.
cudaError_t ml_moe_combine_launch(cudaStream_t stream, int d,
    const float* routed, const float* shared, float* output, int* status);
