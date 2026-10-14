"""0079A: exact full-vocabulary soft-token feedback and manual derivatives.

Reference mathematics (no top-k, no detached feedback)::

    P = softmax(L / T)
    S = P @ E
    dP = dS @ E.T
    dE += P.T @ dS
    dL = P * (dP - sum(P * dP, axis=-1)) / T

All saved contexts are per-*call*, not mutable layer state, so the same module
can participate in multiple differentiable refinement passes. The 2-D GEMM
paths follow the repository's CuPy BF16 limitation on N-D matmul.
"""

from dataclasses import dataclass
import os
from itertools import product

import numpy as np

from mini_llm.backend import xp, BACKEND_NAME, is_low_precision_dtype, is_bfloat16_dtype
from mini_llm.parameter import Parameter
from mini_llm.performance_profiler import block_proposal_detail_scope


def _acc_dtype(dtype):
    return xp.float32 if is_low_precision_dtype(dtype) else dtype



# 0080O5: the old stable_softmax forms multiple FP32 [groups*K,V]
# intermediates (promotion, scaled logits, centered logits, exp).  This
# optional CUDA implementation allocates only the FP32 output probabilities.
# It retains all vocabulary entries and computes each row's maximum/sum in
# FP32; the original implementation remains the default and the fallback.
_SOFTMAX_FORWARD_CUDA_KERNELS = None


def _softmax_forward_cuda_kernels():
    global _SOFTMAX_FORWARD_CUDA_KERNELS
    if _SOFTMAX_FORWARD_CUDA_KERNELS is None:
        code = r"""
        __device__ __forceinline__ float read_fwd_bf16(unsigned short bits) {
            union { unsigned int u; float f; } value;
            value.u = ((unsigned int)bits) << 16;
            return value.f;
        }
        __device__ __forceinline__ float read_fwd_f32(float x) { return x; }

        // Per-row stable softmax: max reduction, exp-sum reduction, output.
        // No full-vocabulary intermediates except the returned FP32 output.
        #define SOFTMAX_FWD(NAME, TYPE, READ) \
        extern "C" __global__ void NAME( \
            const TYPE* x, float* out, int rows, int vocab, float inv_t) { \
            const int row = blockIdx.x; \
            if (row >= rows) return; \
            const long long offset = ((long long)row) * vocab; \
            extern __shared__ float shared[]; \
            float m = -3.402823466e+38f; \
            for (int v = threadIdx.x; v < vocab; v += blockDim.x) { \
                const float a = READ(x[offset + v]) * inv_t; \
                m = fmaxf(m, a); \
            } \
            shared[threadIdx.x] = m; \
            __syncthreads(); \
            for (int stride = blockDim.x >> 1; stride; stride >>= 1) { \
                if (threadIdx.x < stride) \
                    shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]); \
                __syncthreads(); \
            } \
            const float row_max = shared[0]; \
            float sum = 0.0f; \
            for (int v = threadIdx.x; v < vocab; v += blockDim.x) \
                sum += expf(READ(x[offset + v]) * inv_t - row_max); \
            shared[threadIdx.x] = sum; \
            __syncthreads(); \
            for (int stride = blockDim.x >> 1; stride; stride >>= 1) { \
                if (threadIdx.x < stride) \
                    shared[threadIdx.x] += shared[threadIdx.x + stride]; \
                __syncthreads(); \
            } \
            const float inv_sum = 1.0f / shared[0]; \
            for (int v = threadIdx.x; v < vocab; v += blockDim.x) \
                out[offset + v] = expf(READ(x[offset + v]) * inv_t - row_max) * inv_sum; \
        }
        SOFTMAX_FWD(stable_softmax_f32, float, read_fwd_f32)
        SOFTMAX_FWD(stable_softmax_bf16, unsigned short, read_fwd_bf16)
        """
        _SOFTMAX_FORWARD_CUDA_KERNELS = (
            xp.RawKernel(code, 'stable_softmax_f32'),
            xp.RawKernel(code, 'stable_softmax_bf16'),
        )
    return _SOFTMAX_FORWARD_CUDA_KERNELS


def _stable_softmax_fused_gpu(logits, temperature):
    if BACKEND_NAME != 'cupy' or not logits.flags.c_contiguous:
        return None
    if is_bfloat16_dtype(logits.dtype):
        kernel = _softmax_forward_cuda_kernels()[1]
    elif logits.dtype == xp.float32:
        kernel = _softmax_forward_cuda_kernels()[0]
    else:
        return None
    vocab = int(logits.shape[-1])
    if vocab == 0:
        return None
    rows = int(logits.size // vocab)
    result = xp.empty(logits.shape, dtype=xp.float32)
    if rows:
        kernel((rows,), (256,), (
            logits, result, np.int32(rows), np.int32(vocab),
            np.float32(1.0 / float(temperature)),
        ), shared_mem=256 * 4)
    return result


def stable_softmax(logits, temperature=1.0):
    """Full-vocabulary softmax with FP32 reductions for FP16/BF16 inputs."""
    if not 0.0 < float(temperature) < float("inf"):
        raise ValueError("feedback temperature must be finite and positive")
    if logits.ndim < 2:
        raise ValueError("expected logits [..., vocab]")
    if (os.environ.get('MINI_LLM_FUSED_SOFTMAX_FORWARD', '0').strip().lower()
            not in ('0', 'false', 'off', 'no', '')):
        fused = _stable_softmax_fused_gpu(logits, temperature)
        if fused is not None:
            return fused
    z = logits.astype(_acc_dtype(logits.dtype), copy=False) / float(temperature)
    z = z - xp.max(z, axis=-1, keepdims=True)
    exp_z = xp.exp(z)
    return exp_z / xp.sum(exp_z, axis=-1, keepdims=True)


# 0080O4: this JVP is applied to [group * K, full_vocabulary] matrices.
# The old CuPy expression materializes multiple FP32 arrays of that size.
# GPU kernels are compiled lazily, only when explicitly enabled; no CUDA
# dependency or numerical change is introduced for the default AR/CPU paths.
_SOFTMAX_VJP_CUDA_KERNELS = None


def _softmax_vjp_cuda_kernels():
    global _SOFTMAX_VJP_CUDA_KERNELS
    if _SOFTMAX_VJP_CUDA_KERNELS is None:
        code = r"""
        __device__ __forceinline__ float read_bf16(unsigned short bits) {
            union { unsigned int u; float f; } value;
            value.u = ((unsigned int)bits) << 16;
            return value.f;
        }

        // Exactly one CUDA block per full-vocabulary row. Reduce p*g in FP32
        // using a small shared-memory array, then write the FP32 VJP without
        // allocating full-vocabulary (p*g) or (g-dot) workspaces.
        #define VJP_KERNEL(NAME, TYPE, READ) \
        extern "C" __global__ void NAME( \
            const float* p, const TYPE* g, float* out, \
            int n_rows, int vocab, float inv_temperature) { \
            const int row = blockIdx.x; \
            if (row >= n_rows) return; \
            const long long offset = ((long long)row) * vocab; \
            extern __shared__ float shared[]; \
            float accum = 0.0f; \
            for (int v = threadIdx.x; v < vocab; v += blockDim.x) { \
                const float grad = READ(g[offset + v]); \
                accum += p[offset + v] * grad; \
            } \
            shared[threadIdx.x] = accum; \
            __syncthreads(); \
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) { \
                if (threadIdx.x < stride) \
                    shared[threadIdx.x] += shared[threadIdx.x + stride]; \
                __syncthreads(); \
            } \
            const float dot = shared[0]; \
            for (int v = threadIdx.x; v < vocab; v += blockDim.x) { \
                const float grad = READ(g[offset + v]); \
                out[offset + v] = (p[offset + v] * (grad - dot)) * inv_temperature; \
            } \
        }
        __device__ __forceinline__ float read_fp32(float x) { return x; }
        VJP_KERNEL(softmax_vjp_grad_f32, float, read_fp32)
        VJP_KERNEL(softmax_vjp_grad_bf16, unsigned short, read_bf16)
        """
        # No default GPU-code compilation during import or ordinary training.
        _SOFTMAX_VJP_CUDA_KERNELS = (
            xp.RawKernel(code, 'softmax_vjp_grad_f32'),
            xp.RawKernel(code, 'softmax_vjp_grad_bf16'),
        )
    return _SOFTMAX_VJP_CUDA_KERNELS


def _softmax_vjp_fused_gpu(probs, upstream, temperature, *, out=None):
    """CUDA FP32 row-reduction + fused VJP, one full-sized output only.

    Returns None for layouts/dtypes not supported by the kernel. Those cases
    use the bounded tiled fallback rather than copying the entire input.
    """
    if (BACKEND_NAME != 'cupy' or probs.dtype != xp.float32
            or not probs.flags.c_contiguous or not upstream.flags.c_contiguous):
        return None
    if is_bfloat16_dtype(upstream.dtype):
        kernel = _softmax_vjp_cuda_kernels()[1]
    elif upstream.dtype == xp.float32:
        kernel = _softmax_vjp_cuda_kernels()[0]
    else:
        return None
    n_rows = int(probs.size // probs.shape[-1])
    vocab = int(probs.shape[-1])
    if n_rows == 0:
        return xp.empty(probs.shape, dtype=xp.float32)
    if out is None:
        output = xp.empty(probs.shape, dtype=xp.float32)
    else:
        if (out.shape != probs.shape or out.dtype != xp.float32
                or not out.flags.c_contiguous or out is not probs):
            raise ValueError('in-place VJP output must be the owned FP32 probability array')
        output = out
    # RawKernel scalar ABI must match C int32/float32 (not Python int/float).
    kernel((n_rows,), (256,),
           (probs, upstream, output, np.int32(n_rows), np.int32(vocab),
            np.float32(1.0 / float(temperature))),
           shared_mem=256 * 4)
    return output


def _softmax_vjp_tiled(probs, upstream, temperature, rows_per_chunk):
    """Bound intermediate memory to O(rows_per_chunk * vocab).

    Contiguous arrays use slices of packed rows. Non-contiguous layouts are
    deliberately processed one row at a time: reshaping them into a packed
    matrix could silently materialize a full-vocabulary contiguous copy.
    """
    dtype = _acc_dtype(probs.dtype)
    vocab = int(probs.shape[-1])
    n_rows = int(probs.size // vocab)
    out = xp.empty(probs.shape, dtype=dtype)
    if probs.flags.c_contiguous and upstream.flags.c_contiguous:
        p2 = probs.reshape(n_rows, vocab)
        g2 = upstream.reshape(n_rows, vocab)
        out2 = out.reshape(n_rows, vocab)
        for start in range(0, n_rows, rows_per_chunk):
            stop = min(n_rows, start + rows_per_chunk)
            p = p2[start:stop].astype(dtype, copy=False)
            g = g2[start:stop].astype(dtype, copy=False)
            row_dot = xp.sum(p * g, axis=-1, keepdims=True)
            out2[start:stop] = p * (g - row_dot) / float(temperature)
        return out
    for idx in product(*(range(int(size)) for size in probs.shape[:-1])):
        p = probs[idx].astype(dtype, copy=False)
        g = upstream[idx].astype(dtype, copy=False)
        dot = xp.sum(p * g)
        out[idx] = p * (g - dot) / float(temperature)
    return out


def softmax_vjp(probs, upstream, temperature=1.0):
    """Exact full-vocabulary softmax VJP, without a V-by-V Jacobian.

    `MINI_LLM_FUSED_SOFTMAX_VJP=1` selects a memory-efficient CUDA kernel for
    contiguous FP32 probability arrays and FP32/BF16 upstream gradients.
    Each row uses FP32 accumulation; parallel reduction order differs slightly
    from NumPy/CuPy sum. Unsupported GPU layouts fall back to bounded tiles.

    `MINI_LLM_SOFTMAX_VJP_CHUNK_ROWS=64` instead enables a portable tiled
    implementation, useful for isolating CUDA kernel issues. Both optimized
    paths retain full-vocabulary probabilities and full differentiability.
    The original vectorized reference remains the default.
    """
    if probs.shape != upstream.shape:
        raise ValueError("probability and upstream gradient shapes must match")
    if not 0.0 < float(temperature) < float("inf"):
        raise ValueError("feedback temperature must be finite and positive")
    if probs.ndim < 2:
        raise ValueError("expected probabilities [..., vocab]")
    chunk_raw = os.environ.get('MINI_LLM_SOFTMAX_VJP_CHUNK_ROWS', '0').strip()
    try:
        chunk_rows = int(chunk_raw)
    except ValueError as exc:
        raise ValueError('MINI_LLM_SOFTMAX_VJP_CHUNK_ROWS must be an integer') from exc
    if chunk_rows < 0:
        raise ValueError('MINI_LLM_SOFTMAX_VJP_CHUNK_ROWS must not be negative')
    use_fused = os.environ.get('MINI_LLM_FUSED_SOFTMAX_VJP', '0').strip().lower() not in ('0','false','off','no','')
    if use_fused and BACKEND_NAME == 'cupy':
        result = _softmax_vjp_fused_gpu(probs, upstream, temperature)
        if result is not None:
            return result
        # Requested fused mode remains memory-bounded for unsupported layouts.
        return _softmax_vjp_tiled(probs, upstream, temperature, chunk_rows or 64)
    if chunk_rows:
        return _softmax_vjp_tiled(probs, upstream, temperature, chunk_rows)
    dtype = _acc_dtype(probs.dtype)
    p = probs.astype(dtype, copy=False)
    g = upstream.astype(dtype, copy=False)
    return p * (g - xp.sum(p * g, axis=-1, keepdims=True)) / float(temperature)


@dataclass(frozen=True)
class ExpectationCache:
    probs: object
    embedding_weight: Parameter
    temperature: float
    original_shape: tuple


class SoftTokenExpectation:
    """Shared token-embedding expectation over **all** vocabulary entries.

    ``embedding_weight`` is the original ``model.embedding.W`` Parameter.
    Its ``grad`` is updated in-place in backward and therefore also collects
    gradients from ordinary fixed-token lookups and tied projections, if any.

    Precision: FP64/FP32 uses normal GEMM. On the GPU low-precision path we
    cast only the *small* [num_active_positions, V] probabilities/gradients to
    the embedding compute dtype, not the large embedding matrix to FP32.
    The FP32 probability/JVP accumulation is retained for softmax stability.
    """

    def __init__(self, embedding_weight: Parameter):
        if len(embedding_weight.shape) != 2:
            raise ValueError("embedding_weight must have shape [vocab, d_model]")
        self.embedding_weight = embedding_weight

    def forward(self, logits, *, temperature=1.0):
        if logits.shape[-1] != self.embedding_weight.shape[0]:
            raise ValueError("logit vocabulary dimension does not match embedding")
        with block_proposal_detail_scope('block_proposal.expectation.softmax.forward'):
            p = stable_softmax(logits, temperature)
        out_shape = (*p.shape[:-1], self.embedding_weight.shape[1])
        p2 = p.reshape(-1, p.shape[-1])
        e = self.embedding_weight.data
        p_compute = (
            p2.astype(e.dtype, copy=False)
            if is_low_precision_dtype(e.dtype) else p2
        )
        with block_proposal_detail_scope('block_proposal.expectation.gemm.forward'):
            s = (p_compute @ e).reshape(out_shape)
        return s, ExpectationCache(
            probs=p,
            embedding_weight=self.embedding_weight,
            temperature=float(temperature),
            original_shape=tuple(logits.shape),
        )

    def backward(self, upstream, cache: ExpectationCache):
        """Return gradient w.r.t. input logits; add weight gradient in-place."""
        weight = cache.embedding_weight
        if weight is not self.embedding_weight:
            raise ValueError("backward cache belongs to another embedding")
        if tuple(upstream.shape) != (
            *cache.original_shape[:-1], weight.shape[1]
        ):
            raise ValueError("superposition gradient shape mismatch")
        p2 = cache.probs.reshape(-1, weight.shape[0])
        dy2 = upstream.reshape(-1, weight.shape[1])
        e = weight.data
        if is_low_precision_dtype(e.dtype):
            # Two regular 2D BF16/FP16 GEMMs. FP32 gradient accumulation is
            # performed by Parameter.grad; never materialize E.astype(FP32).
            p_mat = p2.astype(e.dtype, copy=False)
            dy_mat = dy2.astype(e.dtype, copy=False)
        else:
            p_mat, dy_mat = p2, dy2
        with block_proposal_detail_scope('block_proposal.expectation.gemm.backward'):
            weight.grad += p_mat.T @ dy_mat
            dp = (dy_mat @ e.T).reshape(cache.original_shape)
        # Ownership contract: this expectation created and exclusively holds
        # its FP32 probability buffer. After accumulating dE, a fused CUDA
        # VJP can safely overwrite P with dLogits, eliminating one more large
        # [groups*K,V] FP32 output allocation. The tape is single-use in
        # this opt-in mode. General softmax_vjp remains non-destructive.
        consume = (os.environ.get('MINI_LLM_INPLACE_EXPECTATION_VJP', '0').strip().lower()
                   not in ('0', 'false', 'off', 'no', ''))
        fused_vjp = (os.environ.get('MINI_LLM_FUSED_SOFTMAX_VJP', '0').strip().lower()
                     not in ('0', 'false', 'off', 'no', ''))
        with block_proposal_detail_scope('block_proposal.expectation.softmax.backward'):
            if consume and fused_vjp and BACKEND_NAME == 'cupy':
                out = _softmax_vjp_fused_gpu(cache.probs, dp, cache.temperature,
                                             out=cache.probs)
                if out is not None:
                    return out
            return softmax_vjp(cache.probs, dp, cache.temperature)


class FuturePositionEmbeddings:
    """Separate learned unknown/soft/fixed state vectors for an active block.

    No additional absolute positional embedding is introduced: RoPE remains
    the sole positional mechanism in the existing Transformer.
    """

    def __init__(self, d_model, rng, *, std=0.02, dtype="float32"):
        if int(d_model) <= 0:
            raise ValueError("d_model must be positive")
        self.unknown = Parameter(
            rng.normal((int(d_model),), std=std, dtype=dtype),
            name="predictive_refinement.unknown", decay=False,
        )
        self.soft = Parameter(
            xp.zeros((int(d_model),), dtype=dtype),
            name="predictive_refinement.type_soft", decay=False,
        )
        self.fixed = Parameter(
            xp.zeros((int(d_model),), dtype=dtype),
            name="predictive_refinement.type_fixed", decay=False,
        )

    def parameters(self):
        return [self.unknown, self.soft, self.fixed]

    def unknown_block(self, batch_size, block_size):
        if batch_size <= 0 or block_size <= 0:
            raise ValueError("block dimensions must be positive")
        return xp.broadcast_to(
            self.unknown.data[None, None, :],
            (batch_size, block_size, self.unknown.shape[0]),
        ).copy()

    def soft_block(self, expected_embeddings):
        if expected_embeddings.shape[-1] != self.soft.shape[0]:
            raise ValueError("soft embedding dimension mismatch")
        return expected_embeddings + self.soft.data

    def fixed_block(self, token_ids, embedding_weight: Parameter):
        if len(embedding_weight.shape) != 2 or embedding_weight.shape[1] != self.fixed.shape[0]:
            raise ValueError("token embedding dimension mismatch")
        return embedding_weight.data[token_ids] + self.fixed.data

    def backward_unknown(self, dy):
        self.unknown.grad += xp.sum(dy.astype(self.unknown.grad.dtype, copy=False), axis=tuple(range(dy.ndim - 1)))

    def backward_soft(self, dy):
        self.soft.grad += xp.sum(dy.astype(self.soft.grad.dtype, copy=False), axis=tuple(range(dy.ndim - 1)))
        return dy

    def backward_fixed(self, dy, token_ids, embedding_weight: Parameter):
        self.fixed.grad += xp.sum(dy.astype(self.fixed.grad.dtype, copy=False), axis=tuple(range(dy.ndim - 1)))
        xp.add.at(
            embedding_weight.grad,
            token_ids.reshape(-1),
            dy.reshape(-1, dy.shape[-1]),
        )
