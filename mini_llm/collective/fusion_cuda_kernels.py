"""0082N: fused FP32 CuPy encoder/aggregation and member backward.

Only imported on the CuPy backend. The NumPy reference remains in
probability_fusion.py. One CUDA thread handles a token/hidden channel across
all participating members to avoid large broadcast intermediates and extra
launches in the vocabulary-wide encoder.
"""

from functools import lru_cache

import cupy as cp


_CUDA_SOURCE = r'''
extern "C" __global__
void fusion_member_encode_aggregate(
    const float* __restrict__ ratio,
    const float* __restrict__ model_token,
    const float* __restrict__ token_context,
    const float* __restrict__ entropy,
    const float* __restrict__ w_ratio,
    const float* __restrict__ w_identity,
    const float* __restrict__ w_context,
    const float* __restrict__ w_entropy,
    const float* __restrict__ bias,
    float* __restrict__ member,
    float* __restrict__ member_mean,
    float* __restrict__ member_var,
    int rows, int models, int vocab, int hidden
) {
    const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long count = (unsigned long long)rows * vocab * hidden;
    if (idx >= count) return;
    const int channel = (int)(idx % hidden);
    const int token = (int)((idx / hidden) % vocab);
    const int row = (int)(idx / ((unsigned long long)vocab * hidden));
    const float a = w_ratio[channel];
    const float b = w_identity[channel];
    const float c = w_context[channel];
    const float d = w_entropy[channel];
    const float intercept = bias[channel];
    float sum = 0.0f;
    // A model-major tensor is still cached for backward, but all its large
    // broadcast inputs are read directly without constructing [R,M,V,H]
    // temporary operands.
    for (int m = 0; m < models; ++m) {
        const unsigned long long input_i = ((unsigned long long)row * models + m) * vocab + token;
        const unsigned long long output_i = input_i * hidden + channel;
        const float x = ratio[input_i] * a + model_token[(unsigned long long)m * vocab + token] * b
                      + token_context[input_i] * c + entropy[(unsigned long long)row * models + m] * d
                      + intercept;
        const float value = tanhf(x);
        member[output_i] = value;
        sum += value;
    }
    const float avg = sum / (float)models;
    member_mean[idx] = avg;
    float var = 0.0f;
    for (int m = 0; m < models; ++m) {
        const unsigned long long output_i = (((unsigned long long)row * models + m) * vocab + token)
                                           * hidden + channel;
        const float delta = member[output_i] - avg;
        var += delta * delta;
    }
    member_var[idx] = var / (float)models;
}

extern "C" __global__
void fusion_member_backward(
    const float* __restrict__ member,
    const float* __restrict__ member_mean,
    const float* __restrict__ dfeatures,
    float* __restrict__ dpre,
    int rows, int models, int vocab, int hidden
) {
    const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long count = (unsigned long long)rows * models * vocab * hidden;
    if (idx >= count) return;
    const int channel = (int)(idx % hidden);
    const int token = (int)((idx / hidden) % vocab);
    const int model = (int)((idx / ((unsigned long long)vocab * hidden)) % models);
    const int row = (int)(idx / ((unsigned long long)models * vocab * hidden));
    const unsigned long long mean_i = ((unsigned long long)row * vocab + token) * hidden + channel;
    const unsigned long long feature_i = ((unsigned long long)row * vocab + token)
                                         * (2 * hidden + 4) + channel;
    const float dm = dfeatures[feature_i];
    const float dv = dfeatures[feature_i + hidden];
    const float value = member[idx];
    const float upstream = (dm + 2.0f * dv * (value - member_mean[mean_i])) / (float)models;
    dpre[idx] = upstream * (1.0f - value * value);

}

// 0082P: Gradients of five shared member-encoder vectors.  The previous
// CuPy path broadcast five [R,M,V] operands to [R,M,V,H], followed by five
// huge global reductions.  Reduce 1024 token positions per block/channel,
// write tiny partials, then finish the reduction over the tile axis.
// Each thread owns four positions and handles one hidden channel.  The
// result includes the same FP32 terms as the NumPy reference, with only
// floating-point reduction order changed.
extern "C" __global__
void fusion_member_parameter_partials(
    const float* __restrict__ dpre,
    const float* __restrict__ ratio,
    const float* __restrict__ model_token,
    const float* __restrict__ token_context,
    const float* __restrict__ entropy,
    float* __restrict__ partials,
    int rows, int models, int vocab, int hidden
) {
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const unsigned long long count = (unsigned long long)rows * models * vocab;
    const unsigned long long base = (unsigned long long)blockIdx.x * (blockDim.x * 4) + tid;
    float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f, bias = 0.0f;
    for (int j = 0; j < 4; ++j) {
        const unsigned long long idx = base + (unsigned long long)j * blockDim.x;
        if (idx >= count) continue;
        const int m = (int)((idx / vocab) % models);
        const int v = (int)(idx % vocab);
        const float g = dpre[idx * hidden + h];
        a += g * ratio[idx];
        b += g * model_token[(unsigned long long)m * vocab + v];
        c += g * token_context[idx];
        d += g * entropy[idx / vocab];
        bias += g;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        a += __shfl_down_sync(0xffffffff, a, offset);
        b += __shfl_down_sync(0xffffffff, b, offset);
        c += __shfl_down_sync(0xffffffff, c, offset);
        d += __shfl_down_sync(0xffffffff, d, offset);
        bias += __shfl_down_sync(0xffffffff, bias, offset);
    }
    __shared__ float warp_partial[5][8];
    if (lane == 0) {
        warp_partial[0][warp] = a;
        warp_partial[1][warp] = b;
        warp_partial[2][warp] = c;
        warp_partial[3][warp] = d;
        warp_partial[4][warp] = bias;
    }
    __syncthreads();
    if (warp == 0) {
        a = lane < 8 ? warp_partial[0][lane] : 0.0f;
        b = lane < 8 ? warp_partial[1][lane] : 0.0f;
        c = lane < 8 ? warp_partial[2][lane] : 0.0f;
        d = lane < 8 ? warp_partial[3][lane] : 0.0f;
        bias = lane < 8 ? warp_partial[4][lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xffffffff, a, offset);
            b += __shfl_down_sync(0xffffffff, b, offset);
            c += __shfl_down_sync(0xffffffff, c, offset);
            d += __shfl_down_sync(0xffffffff, d, offset);
            bias += __shfl_down_sync(0xffffffff, bias, offset);
        }
        if (lane == 0) {
            const unsigned long long out = ((unsigned long long)blockIdx.x * hidden + h) * 5;
            partials[out] = a;
            partials[out + 1] = b;
            partials[out + 2] = c;
            partials[out + 3] = d;
            partials[out + 4] = bias;
        }
    }
}

// Compute the two [R,M,V] feature derivatives in one streaming read of dpre.
// These were previously separate large broadcast/reduction operations.
extern "C" __global__
void fusion_member_feature_derivatives(
    const float* __restrict__ dpre,
    const float* __restrict__ w_identity,
    const float* __restrict__ w_context,
    float* __restrict__ did,
    float* __restrict__ dcontext,
    unsigned long long count, int hidden
) {
    const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    float identity = 0.0f, context = 0.0f;
    for (int h = 0; h < hidden; ++h) {
        const float g = dpre[idx * hidden + h];
        identity += g * w_identity[h];
        context += g * w_context[h];
    }
    did[idx] = identity;
    dcontext[idx] = context;
}

// 0082Q: propagate the output head gradient and tanh derivative in a
// single memory-streaming pass for the default one-layer decoder. Avoid
// allocating the 384-MiB [R,V,K] ddecoder and tanh-square temporaries.
extern "C" __global__
void fusion_decoder_first_layer_backward(
    const float* __restrict__ decoder,
    const float* __restrict__ dz,
    const float* __restrict__ output_weight,
    float* __restrict__ dpre,
    unsigned long long elements, int channels
) {
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const int channel = (int)(index % channels);
    const float activation = decoder[index];
    const float upstream = dz[index / channels] * output_weight[channel];
    dpre[index] = upstream * (1.0f - activation * activation);
}
'''


@lru_cache(maxsize=1)
def _kernels():
    return (
        cp.RawKernel(_CUDA_SOURCE, 'fusion_member_encode_aggregate'),
        cp.RawKernel(_CUDA_SOURCE, 'fusion_member_backward'),
        cp.RawKernel(_CUDA_SOURCE, 'fusion_member_parameter_partials'),
        cp.RawKernel(_CUDA_SOURCE, 'fusion_member_feature_derivatives'),
        cp.RawKernel(_CUDA_SOURCE, 'fusion_decoder_first_layer_backward'),
    )


def member_encode_aggregate(ratio, model_token, token_context, entropy,
                            w_ratio, w_identity, w_context, w_entropy, bias):
    """Return member [R,M,V,H], mean/variance [R,V,H] in one GPU launch."""
    # CuPy GEMM outputs may be column-major; raw pointer indexing is
    # intentionally C-order. ascontiguousarray is a no-op for C-order inputs.
    ratio = cp.ascontiguousarray(ratio)
    model_token = cp.ascontiguousarray(model_token)
    token_context = cp.ascontiguousarray(token_context)
    entropy = cp.ascontiguousarray(entropy)
    rows, models, vocab = map(int, ratio.shape)
    hidden = int(w_ratio.size)
    member = cp.empty((rows, models, vocab, hidden), dtype=cp.float32)
    mean = cp.empty((rows, vocab, hidden), dtype=cp.float32)
    var = cp.empty_like(mean)
    launch = (rows * vocab * hidden + 255) // 256
    _kernels()[0]((launch,), (256,), (
        ratio, model_token, token_context, entropy,
        w_ratio, w_identity, w_context, w_entropy, bias,
        member, mean, var,
        cp.int32(rows), cp.int32(models), cp.int32(vocab), cp.int32(hidden),
    ))
    return member, mean, var


def member_backward(member, mean, dfeatures):
    """Return exact d(preactivation) without large broadcast temporaries."""
    dfeatures = cp.ascontiguousarray(dfeatures)
    rows, models, vocab, hidden = map(int, member.shape)
    dpre = cp.empty_like(member)
    launch = (rows * models * vocab * hidden + 255) // 256
    _kernels()[1]((launch,), (256,), (
        member, mean, dfeatures, dpre,
        cp.int32(rows), cp.int32(models), cp.int32(vocab), cp.int32(hidden),
    ))
    return dpre


def member_parameter_grads(dpre, ratio, model_token, token_context, entropy,
                           w_identity, w_context):
    """Fused CuPy FP32 member parameter gradients and input feature derivatives.

    Returns (grad_ratio, grad_identity, grad_context, grad_entropy, grad_bias,
    d_identity, d_context). Only final [H,5] reduction is performed via CuPy;
    no [R,M,V,H] broadcast temporary is allocated.
    """
    dpre = cp.ascontiguousarray(dpre)
    ratio = cp.ascontiguousarray(ratio)
    model_token = cp.ascontiguousarray(model_token)
    token_context = cp.ascontiguousarray(token_context)
    entropy = cp.ascontiguousarray(entropy)
    rows, models, vocab, hidden = map(int, dpre.shape)
    positions = rows * models * vocab
    tiles = (positions + 1023) // 1024
    partials = cp.empty((tiles, hidden, 5), dtype=cp.float32)
    did = cp.empty((rows, models, vocab), dtype=cp.float32)
    dcontext = cp.empty_like(did)
    _kernels()[2]((tiles, hidden), (256,), (
        dpre, ratio, model_token, token_context, entropy, partials,
        cp.int32(rows), cp.int32(models), cp.int32(vocab), cp.int32(hidden)))
    _kernels()[3](((positions + 255) // 256,), (256,), (
        dpre, w_identity, w_context, did, dcontext,
        cp.uint64(positions), cp.int32(hidden)))
    reduced = cp.sum(partials, axis=0)
    return (reduced[:, 0], reduced[:, 1], reduced[:, 2], reduced[:, 3],
            reduced[:, 4], did, dcontext)


def decoder_first_layer_backward(decoder, dz, output_weight):
    """One CUDA pass for dCE/d(first-layer preactivation) [R,V,K].

    Only applies to a one-layer decoder. C-order flattened channel dimension
    is required by the raw CUDA kernel, as enforced here.
    """
    decoder = cp.ascontiguousarray(decoder)
    dz = cp.ascontiguousarray(dz)
    output_weight = cp.ascontiguousarray(output_weight)
    rows, vocab, width = map(int, decoder.shape)
    if dz.shape != (rows, vocab) or output_weight.shape != (width,):
        raise ValueError('decoder, dz and output_weight shapes do not align')
    out = cp.empty_like(decoder)
    size = decoder.size
    _kernels()[4](((size + 255) // 256,), (256,),
                  (decoder, dz, output_weight, out,
                   cp.uint64(size), cp.int32(width)))
    return out
