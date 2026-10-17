"""0080AJ3 experimental fused CUDA reductions for *all*-source vocabulary reads.

All four source logits are normalized independently, mixed exactly with the
learned target/source bias, and queried against the *entire* vocabulary.  The
expensive QK, AV, and backward embedding GEMMs remain batched cuBLAS calls.
RawKernel fuses short source mixing and online softmax (forward), then source
responsibilities and score VJP (backward).  No score/probability is cached on
the training tape; tile buffers scale with H * block_tile * K * vocab_tile.

This is a deliberately opt-in CUDA implementation. It must be benchmarked and
validated against batched_mixture on the target GPU before promotion.
"""

import math
import numpy as np

from mini_llm.backend import BACKEND_NAME, xp

if BACKEND_NAME != 'cupy':
    raise ImportError('fused_vocab_cuda requires the CuPy backend')


_CUDA = r'''
// NVRTC does not necessarily have access to the host C/C++ math headers.
// CUDA device math functions (fmaxf, expf, logf) are available directly.
// Keep this source self-contained, like the existing attention kernels.

// One block per (head, target-query) row. The input/output scores are FP32.
// Threads cooperate to reduce the tile maximum and normalization mass.
extern "C" __global__
void mix_online_forward(const float* __restrict__ logits,
                        const float* __restrict__ source_norm,
                        const float* __restrict__ bias,
                        float* __restrict__ scores,
                        float* __restrict__ maximum,
                        float* __restrict__ mass,
                        float* __restrict__ oldscale,
                        int G, int K, int H, int V, float invT) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int q = row % (G*K);
    const int h = row / (G*K);
    const int b = q / K;
    const int j = q % K;
    extern __shared__ float reduce[];
    float localmax = -3.402823466e+38F;
    for (int v = tid; v < V; v += blockDim.x) {
        float max_src = -3.402823466e+38F;
        for (int r=0; r<K; ++r) {
            float a = logits[(b*K+r)*V+v]*invT - source_norm[b*K+r]
                    + bias[(h*K+j)*K+r];
            max_src = fmaxf(max_src, a);
        }
        float src_sum = 0.0f;
        for (int r=0; r<K; ++r) {
            float a = logits[(b*K+r)*V+v]*invT - source_norm[b*K+r]
                    + bias[(h*K+j)*K+r];
            src_sum += expf(a-max_src);
        }
        float s = scores[row*V+v] + max_src + logf(src_sum);
        scores[row*V+v] = s;
        localmax = fmaxf(localmax, s);
    }
    reduce[tid] = localmax;
    __syncthreads();
    for (int step=blockDim.x/2; step>0; step>>=1) {
        if (tid<step) reduce[tid] = fmaxf(reduce[tid], reduce[tid+step]);
        __syncthreads();
    }
    float newmax = fmaxf(maximum[row], reduce[0]);
    float old = expf(maximum[row] - newmax);
    float partial = 0.0f;
    for (int v=tid; v<V; v+=blockDim.x) {
        float w = expf(scores[row*V+v]-newmax);
        scores[row*V+v] = w;
        partial += w;
    }
    reduce[tid] = partial;
    __syncthreads();
    for (int step=blockDim.x/2; step>0; step>>=1) {
        if (tid<step) reduce[tid] += reduce[tid+step];
        __syncthreads();
    }
    if (tid==0) {
        maximum[row] = newmax;
        mass[row] = mass[row]*old + reduce[0];
        oldscale[row] = old;
    }
}

// One thread per (head, target-query, vocabulary) position. Recompute the
// per-source responsibilities; never store K*V source probabilities in tape.
// Temporary source derivatives are kept to ONE vocabulary tile and reduced
// by CuPy after the kernel (no atomic races on shared parameters).
extern "C" __global__
void mix_score_backward(const float* __restrict__ logits,
                        const float* __restrict__ source_norm,
                        const float* __restrict__ bias,
                        const float* __restrict__ query_score,
                        const float* __restrict__ upstream_dot,
                        const float* __restrict__ attention_norm,
                        const float* __restrict__ row_dot,
                        float* __restrict__ weight,
                        float* __restrict__ dscore,
                        float* __restrict__ dsource,
                        int G, int K, int H, int V, float invT) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int N = H*G*K*V;
    if (idx >= N) return;
    const int v = idx % V;
    const int row = idx / V;
    const int q = row % (G*K);
    const int h = row / (G*K);
    const int b = q / K;
    const int j = q % K;
    float max_src = -3.402823466e+38F;
    for (int r=0; r<K; ++r) {
        float a = logits[(b*K+r)*V+v]*invT - source_norm[b*K+r]
                + bias[(h*K+j)*K+r];
        max_src = fmaxf(max_src, a);
    }
    float sum_src = 0.0f;
    for (int r=0; r<K; ++r) {
        float a = logits[(b*K+r)*V+v]*invT - source_norm[b*K+r]
                + bias[(h*K+j)*K+r];
        sum_src += expf(a-max_src);
    }
    const float logmix = max_src + logf(sum_src);
    const float w = expf(logmix + query_score[idx] - attention_norm[row]);
    const float ds = w*(upstream_dot[idx] - row_dot[row]);
    weight[idx] = w;
    dscore[idx] = ds;
    for (int r=0; r<K; ++r) {
        float a = logits[(b*K+r)*V+v]*invT - source_norm[b*K+r]
                + bias[(h*K+j)*K+r];
        dsource[(row*K+r)*V+v] = ds*expf(a-logmix);
    }
}
'''

_FORWARD = None
_BACKWARD = None


def _kernels():
    global _FORWARD, _BACKWARD
    if _FORWARD is None:
        _FORWARD = xp.RawKernel(_CUDA, 'mix_online_forward', options=('--std=c++11',))
        _BACKWARD = xp.RawKernel(_CUDA, 'mix_score_backward', options=('--std=c++11',))
    return _FORWARD, _BACKWARD


def _contig(a):
    return xp.ascontiguousarray(a, dtype=xp.float32)


def _tile_logits(read, logits, b0, b1, v0, v1):
    return _contig(logits[b0:b1, :, v0:v1])


def forward_fused_mixture(read, decoder_states, proposal_logits):
    from .streamed_vocab_read import _tiles, _source_log_norm

    g, k, v = proposal_logits.shape
    h, dh = read.n_heads, read.head_width
    states = decoder_states.astype(read.acc_dtype)
    q = (states.reshape(-1, read.decoder_width) @
         read.query_projection.W.data.astype(read.acc_dtype)).reshape(g, k, h, dh)
    source_norm = _source_log_norm(read, proposal_logits)
    bias = _contig(read.source_bias.data)
    output = xp.empty((g, k, read.model_width), dtype=read.acc_dtype)
    attention_norm = xp.empty((g, k, h), dtype=read.acc_dtype)
    kernel, _ = _kernels()
    for b0, b1 in _tiles(g, read.block_tile):
        gb = b1-b0
        qh = _contig(q[b0:b1].transpose(2, 0, 1, 3).reshape(h, gb*k, dh))
        maximum = xp.full((h, gb*k), -xp.inf, dtype=xp.float32)
        mass = xp.zeros((h, gb*k), dtype=xp.float32)
        value = xp.zeros((h, gb*k, dh), dtype=xp.float32)
        old = xp.empty_like(maximum)
        norm_tile = _contig(source_norm[b0:b1])
        for v0, v1 in _tiles(v, read.vocab_tile):
            vt = v1-v0
            e = _contig(read.embedding.data[v0:v1]).reshape(vt, h, dh).transpose(1, 0, 2).copy()
            e = _contig(e)
            score = _contig(xp.matmul(qh, e.transpose(0, 2, 1)))
            score *= read.read_scale/math.sqrt(dh)
            src_logits = _tile_logits(read, proposal_logits, b0, b1, v0, v1)
            kernel((h*gb*k,), (256,), (
                src_logits, norm_tile, bias, score, maximum, mass, old,
                np.int32(gb), np.int32(k), np.int32(h), np.int32(vt), np.float32(1.0/read.temperature)), shared_mem=256*4)
            value = value*old[:, :, None] + xp.matmul(score, e)
        output[b0:b1] = (value/mass[:, :, None]).reshape(h, gb, k, dh).transpose(
            1, 2, 0, 3).reshape(gb, k, read.model_width)
        attention_norm[b0:b1] = (maximum+xp.log(mass)).reshape(h, gb, k).transpose(1, 2, 0)
    correction = (output.reshape(-1, read.model_width) @
                  read.output_projection.W.data.astype(read.acc_dtype)).reshape(
                      g, k, read.decoder_width)
    return correction, q, output, attention_norm, source_norm, states


def backward_fused_mixture(read, d_correction, cache):
    from .streamed_vocab_read import _tiles

    logits = cache.proposal_logits
    g, k, v = logits.shape
    h, dh = read.n_heads, read.head_width
    dy = d_correction.astype(read.acc_dtype)
    dyflat = dy.reshape(-1, read.decoder_width)
    read.output_projection.W.grad += cache.read.reshape(-1, read.model_width).T @ dyflat
    d_read = (dyflat @ read.output_projection.W.data.astype(read.acc_dtype).T).reshape(
        g, k, h, dh)
    d_q = xp.zeros_like(cache.queries, dtype=read.acc_dtype)
    d_logits = xp.zeros((g, k, v), dtype=read.acc_dtype)
    source_total = xp.zeros((g, k), dtype=read.acc_dtype)
    scale = read.read_scale/math.sqrt(dh)
    bias = _contig(read.source_bias.data)
    _, kernel = _kernels()
    for b0, b1 in _tiles(g, read.block_tile):
        gb = b1-b0
        qflat = _contig(cache.queries[b0:b1].transpose(2, 0, 1, 3).reshape(h, gb*k, dh))
        upstream = _contig(d_read[b0:b1].transpose(2, 0, 1, 3).reshape(h, gb*k, dh))
        rval = _contig(cache.read[b0:b1].reshape(gb, k, h, dh).transpose(
            2, 0, 1, 3).reshape(h, gb*k, dh))
        row_dot = _contig(xp.sum(upstream*rval, axis=-1))
        norm = _contig(cache.attention_norm[b0:b1].transpose(2, 0, 1).reshape(h, gb*k))
        dq_flat = xp.zeros_like(qflat)
        source_norm = _contig(cache.source_norm[b0:b1])
        for v0, v1 in _tiles(v, read.vocab_tile):
            vt = v1-v0
            e = _contig(read.embedding.data[v0:v1]).reshape(vt, h, dh).transpose(1, 0, 2).copy()
            e = _contig(e)
            qscore = _contig(xp.matmul(qflat, e.transpose(0, 2, 1)))
            qscore *= scale
            dot = _contig(xp.matmul(upstream, e.transpose(0, 2, 1)))
            weight = xp.empty_like(qscore)
            dscore = xp.empty_like(qscore)
            dsource = xp.empty((h, gb, k, k, vt), dtype=xp.float32)
            src_logits = _tile_logits(read, logits, b0, b1, v0, v1)
            n = h*gb*k*vt
            kernel(((n+255)//256,), (256,), (
                src_logits, source_norm, bias, qscore, dot, norm, row_dot,
                weight, dscore, dsource, np.int32(gb), np.int32(k), np.int32(h), np.int32(vt),
                np.float32(1.0/read.temperature)))
            dq_flat += scale*xp.matmul(dscore, e)
            de = (xp.matmul(weight.transpose(0, 2, 1), upstream) +
                  scale*xp.matmul(dscore.transpose(0, 2, 1), qflat))
            read.embedding.grad[v0:v1] += de.transpose(1, 0, 2).reshape(
                vt, read.model_width)
            tile_grad = xp.sum(dsource, axis=(0, 2))  # [gb, source, vocab tile]
            d_logits[b0:b1, :, v0:v1] += tile_grad
            source_total[b0:b1] += xp.sum(tile_grad, axis=-1)
            read.source_bias.grad += xp.sum(dsource, axis=(1, 4))
        d_q[b0:b1] += dq_flat.reshape(h, gb, k, dh).transpose(1, 2, 0, 3)
    for b0, b1 in _tiles(g, read.block_tile):
        for v0, v1 in _tiles(v, read.vocab_tile):
            prior = xp.exp(logits[b0:b1, :, v0:v1].astype(read.acc_dtype) /
                           read.temperature - cache.source_norm[b0:b1, :, None])
            d_logits[b0:b1, :, v0:v1] -= prior*source_total[b0:b1, :, None]
    d_logits /= read.temperature
    dq_flat = d_q.reshape(-1, read.model_width)
    read.query_projection.W.grad += (
        cache.projection_input.reshape(-1, read.decoder_width).T @ dq_flat)
    d_input = (dq_flat @ read.query_projection.W.data.astype(read.acc_dtype).T).reshape(
        g, k, read.decoder_width)
    return d_input, d_logits
