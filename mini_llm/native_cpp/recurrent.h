#pragma once
#include <cuda_runtime_api.h>
// Qwen4Exp converted tiled head order; FP32 activations and logical [Hv,Dk,Dv]
// matrix storage, raw oldest-first convolution history. No implicit +1/-exp.
cudaError_t ml_recurrent_launch(cudaStream_t stream, int hk, int hv, int dim,
    int taps, float epsilon, const float* qkv, const float* gate,
    const float* beta_logits, const float* alpha, const float* conv_weight,
    const float* gamma, const float* a, const float* dt, float* history,
    float* memory, float* convolved, float* beta, float* log_decay,
    float* readout, float* gated, int* status);
