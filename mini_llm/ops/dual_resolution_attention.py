"""Trainable dual-resolution attention with compact document metadata.

The token branch is exact causal local GQA.  Complete same-document blocks are
compressed by a small bidirectional Transformer and learned multi-head pooling;
token queries can then attend to summaries whose block end strictly precedes
the query position.  Sequence attention is always evaluated in query chunks,
so this module never materializes a dense token-by-token ``T x T`` matrix.
"""

import math
import os

import numpy as np

from ..backend import BACKEND_NAME, is_low_precision_dtype, resolve_dtype, xp
from ..init import matrix_parameter, ones_parameter
from .attention import softmax_backward, softmax_forward
from .rope import rope_backward, rope_forward
from ..performance_profiler import performance_scope


def build_dual_resolution_metadata(token_ids, eos_token_id, block_size=128):
    """Build compact CPU metadata for packed, EOS-delimited documents.

    EOS belongs to the document it terminates.  The following token starts a
    new document at position zero.  Blocks restart at each document boundary;
    only complete blocks are emitted.
    """
    tokens = np.asarray(token_ids)
    if tokens.ndim != 2:
        raise ValueError("token_ids must have shape [B,T]")
    if tokens.dtype.kind not in {"i", "u"}:
        raise TypeError("token_ids must use an integer dtype")
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    batch, seq_len = tokens.shape
    document_ids = np.empty((batch, seq_len), dtype=np.int64)
    position_ids = np.empty((batch, seq_len), dtype=np.int64)
    row_blocks = []

    for row in range(batch):
        blocks = []
        document = 0
        document_start = 0
        position = 0
        for token_index in range(seq_len):
            document_ids[row, token_index] = document
            position_ids[row, token_index] = position
            position += 1
            if int(tokens[row, token_index]) == int(eos_token_id):
                document_stop = token_index + 1
                complete = (document_stop - document_start) // block_size
                for block in range(complete):
                    start = document_start + block * block_size
                    indices = np.arange(start, start + block_size, dtype=np.int64)
                    blocks.append(
                        (indices, document, int(position_ids[row, indices[-1]]))
                    )
                document += 1
                document_start = document_stop
                position = 0

        if document_start < seq_len:
            complete = (seq_len - document_start) // block_size
            for block in range(complete):
                start = document_start + block * block_size
                indices = np.arange(start, start + block_size, dtype=np.int64)
                blocks.append(
                    (indices, document, int(position_ids[row, indices[-1]]))
                )
        row_blocks.append(blocks)

    max_blocks = max((len(blocks) for blocks in row_blocks), default=0)
    block_indices = np.zeros(
        (batch, max_blocks, block_size), dtype=np.int64
    )
    block_valid = np.zeros((batch, max_blocks), dtype=np.bool_)
    block_document_ids = np.full((batch, max_blocks), -1, dtype=np.int64)
    block_end_positions = np.full((batch, max_blocks), -1, dtype=np.int64)

    for row, blocks in enumerate(row_blocks):
        for block_index, (indices, document, end_position) in enumerate(blocks):
            block_indices[row, block_index] = indices
            block_valid[row, block_index] = True
            block_document_ids[row, block_index] = document
            block_end_positions[row, block_index] = end_position

    return {
        "document_ids": document_ids,
        "position_ids": position_ids,
        "block_indices": block_indices,
        "block_valid": block_valid,
        "block_document_ids": block_document_ids,
        "block_end_positions": block_end_positions,
    }


def dual_resolution_loss_mask(inputs, eos_token_id, loss_mask=None):
    """Mask LM targets that would cross an EOS document boundary.

    For the usual shifted ``inputs``/``targets`` pair, the target following an
    EOS is scored at the position whose input is EOS.  Clearing that position
    prevents learning transitions between independently packed documents.
    """
    tokens = np.asarray(inputs)
    if tokens.ndim != 2:
        raise ValueError("inputs must have shape [B,T]")
    mask = (tokens != int(eos_token_id)).astype(np.float32)
    if loss_mask is None:
        return mask
    supplied = np.asarray(loss_mask)
    if supplied.shape != tokens.shape:
        raise ValueError("loss_mask must have the same shape as inputs")
    return mask * supplied.astype(np.float32, copy=False)


def _work_array(value):
    if is_low_precision_dtype(value.dtype):
        return value.astype("float32", copy=False)
    return value


def _add_parameter_grad(parameter, value):
    parameter.grad += value.astype(parameter.grad.dtype, copy=False)


def _linear_forward(value, parameter):
    value_2d = value.reshape(-1, value.shape[-1])
    result = value_2d @ parameter.data
    return result.reshape(value.shape[:-1] + (parameter.data.shape[1],))


def _linear_backward(gradient, value, parameter, accumulate_parameter_grads):
    value_2d = value.reshape(-1, value.shape[-1])
    gradient_2d = gradient.reshape(-1, gradient.shape[-1])
    if (_fast_projection_enabled() and _bf16_projection_matmul_enabled()
            and is_low_precision_dtype(parameter.data.dtype)):
        value_2d = value_2d.astype(parameter.data.dtype, copy=False)
        gradient_2d = gradient_2d.astype(parameter.data.dtype, copy=False)
        weight = parameter.data
    else:
        value_2d = _work_array(value_2d)
        gradient_2d = _work_array(gradient_2d)
        weight = _work_array(parameter.data)
    if accumulate_parameter_grads:
        _add_parameter_grad(parameter, value_2d.T @ gradient_2d)
    result = gradient_2d @ weight.T
    return result.reshape(value.shape)


def _fast_projection_enabled():
    # Patch 0102J: packed/BF16 is the default on CuPy GPUs, preserving the
    # pure NumPy reference (including numerical finite-difference tests).
    # Explicit 0/1 overrides the backend-specific default on either backend.
    default = "1" if BACKEND_NAME == "cupy" else "0"
    return os.environ.get("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", default).strip().lower() in {"1", "true", "yes"}


def _bf16_projection_matmul_enabled():
    return os.environ.get("MINI_LLM_DUAL_RESOLUTION_BF16_PROJECTION_GRADS", "1").strip().lower() in {"1", "true", "yes"}


def _bf16_output_dx_enabled():
    """Use tensor cores for the attention output-projection input gradient."""
    return (BACKEND_NAME == "cupy" and os.environ.get(
        "MINI_LLM_DUAL_RESOLUTION_BF16_OUTPUT_DX", "0"
    ).strip().lower() in {"1", "true", "yes"})


def _bf16_compressor_qkv_enabled():
    """Independent numerical/performance switch for Triton compressor QKV.

    Allows packed projections to be measured separately from the BF16
    compressor attention specialization introduced in patch 0102H.
    """
    return os.environ.get(
        "MINI_LLM_DUAL_RESOLUTION_BF16_COMPRESSOR_QKV", "1"
    ).strip().lower() in {"1", "true", "yes"}


def _fused_pooling_enabled():
    # Experimental: the reference CuPy pooling path stays the default until
    # numerical parity and *steady-state* RTX GPU speed have been established.
    return os.environ.get("MINI_LLM_DUAL_RESOLUTION_FUSED_POOLING", "0").strip().lower() in {"1", "true", "yes"}


class _PackedProjection:
    """Persistent packed weights and FP32 gradient views for related projections.

    Inference/checkpoint/optimizer still see the individual canonical Parameters.
    Refresh packed *weights* once after optimizer updates, not on every forward
    or checkpoint replay. Gradient views share contiguous FP32 accumulation.
    """

    def __init__(self, parameters):
        self.parameters = tuple(parameters)
        input_width = self.parameters[0].data.shape[0]
        assert all(p.data.ndim == 2 and p.data.shape[0] == input_width
                   for p in self.parameters)
        self.widths = tuple(int(p.data.shape[1]) for p in self.parameters)
        self.ends = np.cumsum((0,) + self.widths).tolist()
        self.weight = None
        self.gradient = None
        self.valid = False

    def refresh(self):
        params = self.parameters
        dtype = params[0].data.dtype
        total_width = self.ends[-1]
        shape = (int(params[0].data.shape[0]), total_width)
        # Frozen inference models legitimately have no gradient buffers.
        if params[0].grad is not None:
            if (self.gradient is None or self.gradient.shape != shape
                    or self.gradient.dtype != params[0].grad.dtype):
                packed_grad = xp.zeros(shape, dtype=params[0].grad.dtype)
                for i, param in enumerate(params):
                    packed_grad[:, self.ends[i]:self.ends[i + 1]] = param.grad
                self.gradient = packed_grad
            for i, param in enumerate(params):
                param.grad = self.gradient[:, self.ends[i]:self.ends[i + 1]]
        if self.weight is None or self.weight.shape != shape or self.weight.dtype != dtype:
            self.weight = xp.empty(shape, dtype=dtype)
        for i, param in enumerate(params):
            self.weight[:, self.ends[i]:self.ends[i + 1]] = param.data
        self.valid = True

    def forward_packed(self, value):
        """Return the contiguous 2-D packed projection for fused consumers."""
        if not self.valid:
            self.refresh()
        return value.reshape(-1, value.shape[-1]) @ self.weight

    def forward(self, value):
        out = self.forward_packed(value)
        out = out.reshape(value.shape[:-1] + (self.ends[-1],))
        return tuple(out[..., self.ends[i]:self.ends[i+1]] for i in range(len(self.widths)))

    def backward(self, grads, value, accumulate_parameter_grads):
        if not self.valid:
            self.refresh()
        pieces = [g.reshape(-1, g.shape[-1]) for g in grads]
        data = value.reshape(-1, value.shape[-1])
        # 0102V: at long contexts, packing all Q/K/V gradients requires a
        # temporary [B*T, total_width] buffer (192 MiB in BF16 at 65K).
        # Optionally stream this packing/GEMM work in row chunks instead.
        # This is deliberately opt-in until a real 65K GPU benchmark proves
        # the memory benefit outweighs the additional GEMM launches.
        chunk_rows = int(os.environ.get(
            "MINI_LLM_DUAL_RESOLUTION_PACKED_BACKWARD_CHUNK_TOKENS", "0"
        ))
        if chunk_rows < 0:
            raise ValueError("packed backward chunk size must be nonnegative")
        if 0 < chunk_rows < data.shape[0]:
            use_bf16 = (is_low_precision_dtype(self.weight.dtype)
                        and _bf16_projection_matmul_enabled())
            matmul_dtype = (self.weight.dtype if use_bf16 else
                            (xp.float32 if is_low_precision_dtype(data.dtype)
                             else data.dtype))
            weight = self.weight if use_bf16 else _work_array(self.weight)
            # Allocate the output once, but never allocate a full-sequence
            # packed gradient or cast the entire gradient sequence to BF16.
            dx = xp.empty(data.shape, dtype=matmul_dtype)
            for lo in range(0, data.shape[0], chunk_rows):
                hi = min(lo + chunk_rows, data.shape[0])
                packed = xp.empty((hi - lo, self.ends[-1]), dtype=matmul_dtype)
                for j, grad in enumerate(pieces):
                    part = grad[lo:hi]
                    if use_bf16:
                        part = part.astype(self.weight.dtype, copy=False)
                    else:
                        part = _work_array(part)
                    packed[:, self.ends[j]:self.ends[j + 1]] = part
                data_chunk = data[lo:hi]
                if use_bf16:
                    data_chunk = data_chunk.astype(self.weight.dtype, copy=False)
                else:
                    data_chunk = _work_array(data_chunk)
                if accumulate_parameter_grads:
                    dw = data_chunk.T @ packed
                    self.gradient += dw.astype(self.gradient.dtype, copy=False)
                dx[lo:hi] = packed @ weight.T
            return dx.reshape(value.shape)
        if is_low_precision_dtype(self.weight.dtype) and _bf16_projection_matmul_enabled():
            # Mirror the optimized 338M QKV projection backward: BF16 tensor-
            # core GEMMs, FP32 cumulative gradient buffers. Pack directly into
            # the BF16 destination so FP32 inputs do not create one cast
            # temporary per projection before concatenation.
            combined = xp.empty(
                (data.shape[0], self.ends[-1]), dtype=self.weight.dtype
            )
            for i, piece in enumerate(pieces):
                combined[:, self.ends[i]:self.ends[i + 1]] = piece
            data = data.astype(self.weight.dtype, copy=False)
        else:
            pieces = [_work_array(g) for g in pieces]
            data = _work_array(data)
            combined = xp.concatenate(pieces, axis=-1)
        if accumulate_parameter_grads:
            dw = data.T @ combined
            self.gradient += dw.astype(self.gradient.dtype, copy=False)
        dx = combined @ (self.weight if data.dtype == self.weight.dtype
                         else _work_array(self.weight)).T
        return dx.reshape(value.shape)


def _fused_gating_enabled():
    """Opt-in Triton gate for local/global context fusion, after 0102L."""
    return (BACKEND_NAME == "cupy" and os.environ.get(
        "MINI_LLM_DUAL_RESOLUTION_FUSED_GATE", "0"
    ).strip().lower() in {"1", "true", "yes"})


def _fused_compressor_rmsnorm_enabled():
    """Opt-in Triton compressor/pooling norms; separate from block RMSNorm."""
    return (BACKEND_NAME == "cupy" and os.environ.get(
        "MINI_LLM_DUAL_RESOLUTION_FUSED_RMSNORM", "0"
    ).strip().lower() in {"1", "true", "yes"})


def _rmsnorm_forward(value, gamma, eps=1e-6):
    if _fused_compressor_rmsnorm_enabled():
        from .triton_dual_resolution_rmsnorm import forward as fused_forward
        output, inverse_rms = fused_forward(value, gamma.data, float(eps))
        return output, {
            "value": value,
            "inverse_rms": inverse_rms,
            "triton_compressor_norm": True,
        }
    work = _work_array(value)
    mean_square = xp.mean(work * work, axis=-1, keepdims=True)
    inverse_rms = 1.0 / xp.sqrt(mean_square + float(eps))
    normalized = work * inverse_rms
    output = normalized * _work_array(gamma.data)
    if is_low_precision_dtype(value.dtype):
        output = output.astype(value.dtype)
    return output, {
        "value": value,
        "inverse_rms": inverse_rms,
        "normalized": normalized,
    }


def _rmsnorm_backward(
    gradient, cache, gamma, accumulate_parameter_grads, eps=1e-6
):
    if cache.get("triton_compressor_norm", False):
        from .triton_dual_resolution_rmsnorm import backward as fused_backward
        dx, dgamma = fused_backward(
            gradient, cache["value"], gamma.data, cache["inverse_rms"],
            accumulate_gamma=accumulate_parameter_grads,
        )
        if accumulate_parameter_grads:
            _add_parameter_grad(gamma, dgamma)
        return dx
    del eps
    value = _work_array(cache["value"])
    inverse_rms = cache["inverse_rms"]
    normalized = cache["normalized"]
    gradient_work = _work_array(gradient)
    if accumulate_parameter_grads:
        axes = tuple(range(gradient_work.ndim - 1))
        _add_parameter_grad(
            gamma, xp.sum(gradient_work * normalized, axis=axes)
        )
    dnormalized = gradient_work * _work_array(gamma.data)
    width = value.shape[-1]
    projection = xp.sum(dnormalized * value, axis=-1, keepdims=True)
    return (
        dnormalized * inverse_rms
        - value * (inverse_rms ** 3) * projection / float(width)
    )


def _sigmoid(value):
    work = _work_array(value)
    magnitude = xp.exp(-xp.abs(work))
    return xp.where(work >= 0, 1.0 / (1.0 + magnitude), magnitude / (1.0 + magnitude))


class DualResolutionAttention:
    """Handwritten local-token plus compressed-history grouped-query attention."""

    def __init__(
        self,
        d_model,
        n_q_heads,
        n_kv_heads,
        d_head,
        input_std,
        output_std,
        rng,
        rope_base=10_000.0,
        name="dual_resolution_attention",
        dtype="float32",
        config=None,
        dual_resolution_config=None,
    ):
        if config is None:
            config = dual_resolution_config
        elif dual_resolution_config is not None and dual_resolution_config is not config:
            raise ValueError("provide only one dual-resolution config object")
        if config is None:
            raise ValueError("a dual-resolution config object is required")

        self.d_model = int(d_model)
        self.n_q_heads = int(n_q_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.d_head = int(d_head)
        if self.n_q_heads * self.d_head != self.d_model:
            raise ValueError("n_q_heads * d_head must equal d_model")
        if self.n_q_heads % self.n_kv_heads:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")
        if self.d_head % 2:
            raise ValueError("RoPE requires an even d_head")

        self.group_size = self.n_q_heads // self.n_kv_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.rope_base = float(rope_base)
        self.dtype = resolve_dtype(dtype)
        self.config = config
        self.block_size = int(config.block_size)
        self.local_window = int(config.local_window)
        self.compression_dim = int(config.compression_dim)
        self.compression_heads = int(config.compression_heads)
        self.compression_depth = int(config.compression_depth)
        self.compression_block_chunk = int(
            getattr(config, "compression_block_chunk", 64)
        )
        self.global_query_chunk = int(config.global_query_chunk)
        self.recompute_attention_probs = bool(
            getattr(config, "recompute_attention_probs", True)
        )
        # Opt-in experimental 0102B implementation; all other branches remain CuPy.
        self.local_kernel = os.environ.get(
            "MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL",
            os.environ.get("MINI_LLM_DUAL_RESOLUTION_KERNEL", "cupy"),
        ).lower().strip()
        if self.local_kernel not in ("cupy", "triton"):
            raise ValueError("dual-resolution local kernel must be 'cupy' or 'triton'")
        # 0102C: independent opt-in implementations. CuPy remains the reference.
        self.global_kernel = os.environ.get(
            "MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL", "cupy"
        ).lower().strip()
        self.compression_kernel = os.environ.get(
            "MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL", "cupy"
        ).lower().strip()
        for component, backend in (("global", self.global_kernel),
                                   ("compression", self.compression_kernel)):
            if backend not in ("cupy", "triton"):
                raise ValueError(f"dual-resolution {component} kernel must be 'cupy' or 'triton'")
        if self.block_size <= 0 or self.local_window <= 0:
            raise ValueError("block_size and local_window must be positive")
        if self.compression_dim <= 0 or self.compression_heads <= 0:
            raise ValueError("compression dimensions must be positive")
        if self.compression_dim % self.compression_heads:
            raise ValueError("compression_dim must be divisible by compression_heads")
        if self.compression_depth < 0:
            raise ValueError("compression_depth must be non-negative")
        if self.compression_block_chunk < 1:
            raise ValueError("compression_block_chunk must be positive")
        if self.global_query_chunk <= 0:
            raise ValueError("global_query_chunk must be positive")
        self.compression_head_dim = self.compression_dim // self.compression_heads
        self.compression_scale = 1.0 / math.sqrt(self.compression_head_dim)
        if self.compression_kernel == "triton" and self.compression_head_dim != 64:
            raise ValueError("0102C Triton compression requires head dimension 64")

        q_width = self.n_q_heads * self.d_head
        kv_width = self.n_kv_heads * self.d_head
        self.Wq = matrix_parameter(
            (self.d_model, q_width), input_std, rng, f"{name}.Wq", dtype=dtype
        )
        self.Wk = matrix_parameter(
            (self.d_model, kv_width), input_std, rng, f"{name}.Wk", dtype=dtype
        )
        self.Wv = matrix_parameter(
            (self.d_model, kv_width), input_std, rng, f"{name}.Wv", dtype=dtype
        )
        self.Wo = matrix_parameter(
            (q_width, self.d_model), output_std, rng, f"{name}.Wo", dtype=dtype
        )

        self.W_compress_in = matrix_parameter(
            (self.d_model, self.compression_dim),
            input_std,
            rng,
            f"{name}.compress.in",
            dtype=dtype,
        )
        self.compression_layers = []
        for depth in range(self.compression_depth):
            prefix = f"{name}.compress.layer_{depth}"
            layer = {
                "norm": ones_parameter(
                    (self.compression_dim,), f"{prefix}.norm.gamma", dtype=dtype
                ),
                "Wq": matrix_parameter(
                    (self.compression_dim, self.compression_dim),
                    input_std,
                    rng,
                    f"{prefix}.Wq",
                    dtype=dtype,
                ),
                "Wk": matrix_parameter(
                    (self.compression_dim, self.compression_dim),
                    input_std,
                    rng,
                    f"{prefix}.Wk",
                    dtype=dtype,
                ),
                "Wv": matrix_parameter(
                    (self.compression_dim, self.compression_dim),
                    input_std,
                    rng,
                    f"{prefix}.Wv",
                    dtype=dtype,
                ),
                "Wo": matrix_parameter(
                    (self.compression_dim, self.compression_dim),
                    output_std,
                    rng,
                    f"{prefix}.Wo",
                    dtype=dtype,
                ),
            }
            self.compression_layers.append(layer)

        self.pool_norm = ones_parameter(
            (self.compression_dim,), f"{name}.compress.pool_norm.gamma", dtype=dtype
        )
        self.pool_query = matrix_parameter(
            (self.compression_heads, self.compression_head_dim),
            input_std,
            rng,
            f"{name}.compress.pool_query",
            dtype=dtype,
            decay=False,
        )
        self.W_pool_k = matrix_parameter(
            (self.compression_dim, self.compression_dim),
            input_std,
            rng,
            f"{name}.compress.pool_k",
            dtype=dtype,
        )
        self.W_pool_v = matrix_parameter(
            (self.compression_dim, self.compression_dim),
            input_std,
            rng,
            f"{name}.compress.pool_v",
            dtype=dtype,
        )
        self.W_pool_o = matrix_parameter(
            (self.compression_dim, self.compression_dim),
            output_std,
            rng,
            f"{name}.compress.pool_o",
            dtype=dtype,
        )
        self.W_compress_up = matrix_parameter(
            (self.compression_dim, self.d_model),
            output_std,
            rng,
            f"{name}.compress.up",
            dtype=dtype,
        )
        self.W_global_k = matrix_parameter(
            (self.d_model, kv_width),
            input_std,
            rng,
            f"{name}.global.Wk",
            dtype=dtype,
        )
        self.W_global_v = matrix_parameter(
            (self.d_model, kv_width),
            input_std,
            rng,
            f"{name}.global.Wv",
            dtype=dtype,
        )
        self.W_gate = matrix_parameter(
            (self.d_model, self.n_q_heads),
            input_std,
            rng,
            f"{name}.gate.W",
            dtype=dtype,
        )

    def _ensure_projection_groups(self):
        if not hasattr(self, "_packed_projection_groups"):
            self._packed_main = _PackedProjection((self.Wq, self.Wk, self.Wv))
            self._packed_compress = [
                _PackedProjection((layer["Wq"], layer["Wk"], layer["Wv"]))
                for layer in self.compression_layers
            ]
            self._packed_pool = _PackedProjection((self.W_pool_k, self.W_pool_v))
            self._packed_global = _PackedProjection((self.W_global_k, self.W_global_v))
            self._packed_projection_groups = (
                self._packed_main, *self._packed_compress,
                self._packed_pool, self._packed_global,
            )

    def parameters(self):
        parameters = [self.Wq, self.Wk, self.Wv]
        parameters.append(self.W_compress_in)
        for layer in self.compression_layers:
            parameters.extend(
                [layer["norm"], layer["Wq"], layer["Wk"], layer["Wv"], layer["Wo"]]
            )
        parameters.extend(
            [
                self.pool_norm,
                self.pool_query,
                self.W_pool_k,
                self.W_pool_v,
                self.W_pool_o,
                self.W_compress_up,
                self.W_global_k,
                self.W_global_v,
                self.W_gate,
                self.Wo,
            ]
        )
        return parameters

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def refresh_compute_buffers(self):
        """Repack QKV/KV weights after optimizer updates, like the 338M model."""
        if _fast_projection_enabled():
            self._ensure_projection_groups()
            for group in self._packed_projection_groups:
                group.refresh()

    def _regular_metadata(self, batch, seq_len, position_ids):
        document_ids = xp.zeros((batch, seq_len), dtype=xp.int64)
        if position_ids is None:
            positions = xp.broadcast_to(
                xp.arange(seq_len, dtype=xp.int64)[None, :], (batch, seq_len)
            )
        else:
            positions = xp.asarray(position_ids)
            if positions.shape == (seq_len,):
                positions = xp.broadcast_to(positions[None, :], (batch, seq_len))
            if positions.shape != (batch, seq_len):
                raise ValueError("position_ids must have shape [T] or [B,T]")
            if positions.dtype.kind not in {"i", "u"}:
                raise TypeError("position_ids must use an integer dtype")
        block_count = seq_len // self.block_size
        if block_count:
            indices = xp.arange(
                block_count * self.block_size, dtype=xp.int64
            ).reshape(1, block_count, self.block_size)
            block_indices = xp.broadcast_to(
                indices, (batch, block_count, self.block_size)
            )
            block_end_positions = positions[:, block_indices[0, :, -1]]
        else:
            block_indices = xp.zeros(
                (batch, 0, self.block_size), dtype=xp.int64
            )
            block_end_positions = xp.zeros((batch, 0), dtype=xp.int64)
        return {
            "document_ids": document_ids,
            "position_ids": positions,
            "block_indices": block_indices,
            "block_valid": xp.ones((batch, block_count), dtype=xp.bool_),
            "block_document_ids": xp.zeros((batch, block_count), dtype=xp.int64),
            "block_end_positions": block_end_positions,
        }

    def _prepare_metadata(self, metadata, batch, seq_len, position_ids):
        # The same document batch is consumed by 20 layers, then replayed in
        # reverse with block checkpointing. GPU conversions and synchronizing
        # bounds checks on every replay were needlessly serializing the stream.
        # Cache ONLY the CPU-produced immutable metadata and only in the case
        # without an explicit position_ids override.
        cache_key = (batch, seq_len, self.block_size, _fast_projection_enabled())
        can_cache = (
            isinstance(metadata, dict) and position_ids is None
            and isinstance(metadata.get("block_indices"), np.ndarray)
            and isinstance(metadata.get("block_valid"), np.ndarray)
        )
        if can_cache:
            cached = metadata.get("_dual_resolution_device_metadata_0102f")
            if cached is not None and cached[0] == cache_key:
                return cached[1]
        if metadata is None:
            prepared = self._regular_metadata(batch, seq_len, position_ids)
            host_bounds_checked = False
        else:
            required = (
                "document_ids", "position_ids", "block_indices", "block_valid",
                "block_document_ids", "block_end_positions",
            )
            missing = [key for key in required if key not in metadata]
            if missing:
                raise ValueError(f"dual_resolution_metadata missing fields: {missing}")
            host_bounds_checked = (
                isinstance(metadata["block_indices"], np.ndarray)
                and isinstance(metadata["block_valid"], np.ndarray)
            )
            if host_bounds_checked:
                host_indices = metadata["block_indices"]
                host_valid = metadata["block_valid"]
                if host_indices.ndim == 3 and host_valid.shape == host_indices.shape[:2]:
                    selected = host_indices[host_valid.astype(np.bool_, copy=False)]
                    if selected.size and (
                        np.any(selected < 0) or np.any(selected >= seq_len)
                    ):
                        raise IndexError("valid block_indices must refer to source tokens")
                else:
                    host_bounds_checked = False
            prepared = {key: xp.asarray(metadata[key]) for key in required}
            if position_ids is not None:
                supplied = xp.asarray(position_ids)
                if supplied.shape == (seq_len,):
                    supplied = xp.broadcast_to(supplied[None, :], (batch, seq_len))
                if supplied.shape != (batch, seq_len):
                    raise ValueError("position_ids must have shape [T] or [B,T]")
                prepared["position_ids"] = supplied

        if prepared["document_ids"].shape != (batch, seq_len):
            raise ValueError("document_ids must have shape [B,T]")
        if prepared["position_ids"].shape != (batch, seq_len):
            raise ValueError("metadata position_ids must have shape [B,T]")
        block_indices = prepared["block_indices"]
        if block_indices.ndim != 3 or block_indices.shape[0] != batch:
            raise ValueError("block_indices must have shape [B,M,block_size]")
        if block_indices.shape[2] != self.block_size:
            raise ValueError("metadata block size does not match config.block_size")
        block_shape = block_indices.shape[:2]
        for key in ("block_valid", "block_document_ids", "block_end_positions"):
            if prepared[key].shape != block_shape:
                raise ValueError(f"{key} must have shape [B,M]")
        for key in (
            "document_ids", "position_ids", "block_indices",
            "block_document_ids", "block_end_positions",
        ):
            if prepared[key].dtype.kind not in {"i", "u"}:
                raise TypeError(f"{key} must use an integer dtype")
        prepared["block_valid"] = prepared["block_valid"].astype(xp.bool_, copy=False)
        # Model B fast-path eligibility.  When all complete blocks occupy the
        # initial contiguous token range of every batch row, the compressor
        # may reshape the original hidden tensor rather than gather a full
        # [B, blocks, 128, D] copy.  Decide from CPU metadata; never transfer
        # a GPU predicate to the host during forward/backward execution.
        if metadata is None:
            prepared["contiguous_pool_blocks"] = (
                batch == 1 or seq_len % self.block_size == 0
            )
        elif host_bounds_checked:
            host_indices = metadata["block_indices"]
            host_valid = metadata["block_valid"]
            n_blocks = host_indices.shape[1]
            expected = np.arange(n_blocks * self.block_size, dtype=np.int64)
            prepared["contiguous_pool_blocks"] = bool(
                np.all(host_valid)
                and n_blocks * self.block_size <= seq_len
                and (batch == 1 or n_blocks * self.block_size == seq_len)
                and np.array_equal(
                    host_indices,
                    np.broadcast_to(
                        expected.reshape(1, n_blocks, self.block_size),
                        host_indices.shape,
                    ),
                )
            )
        else:
            # GPU-produced, irregular or unvalidated indexing must retain the
            # exact indexed-gather reference implementation.
            prepared["contiguous_pool_blocks"] = False
        # Stronger invariant used by bounded Model B global attention.  It is
        # deliberately established from host metadata only: packed documents,
        # noncontiguous coordinates, missing blocks, and GPU-produced layouts
        # must retain the fully metadata-aware kernel bounds.
        if metadata is None:
            prepared["single_contiguous_document"] = position_ids is None
        elif host_bounds_checked:
            host_docs = np.asarray(metadata["document_ids"])
            host_positions = np.asarray(metadata["position_ids"])
            host_block_docs = np.asarray(metadata["block_document_ids"])
            host_ends = np.asarray(metadata["block_end_positions"])
            n_blocks = host_ends.shape[1]
            expected_positions = np.broadcast_to(
                np.arange(seq_len, dtype=host_positions.dtype)[None, :],
                host_positions.shape,
            )
            expected_ends = np.broadcast_to(
                (np.arange(1, n_blocks + 1, dtype=host_ends.dtype)
                 * self.block_size - 1)[None, :],
                host_ends.shape,
            )
            prepared["single_contiguous_document"] = bool(
                prepared["contiguous_pool_blocks"]
                and np.all(metadata["block_valid"])
                and np.all(host_docs == host_docs[:, :1])
                and np.array_equal(host_positions, expected_positions)
                and np.array_equal(
                    host_block_docs,
                    np.broadcast_to(host_docs[:, :1], host_block_docs.shape),
                )
                and np.array_equal(host_ends, expected_ends)
            )
        else:
            prepared["single_contiguous_document"] = False
        if _fast_projection_enabled():
            # Index vectors are invariant across all 20 layers and checkpoint
            # replay. Build them once on the host when EOS metadata originates
            # from the CPU; never use CuPy boolean-index compaction to discover
            # the number of valid blocks during every attention invocation.
            if host_bounds_checked:
                row_host, slot_host = np.nonzero(metadata["block_valid"])
                starts = metadata["block_indices"][row_host, slot_host]
                prepared["valid_block_rows"] = xp.asarray(row_host, dtype=xp.int64)
                prepared["valid_block_slots"] = xp.asarray(slot_host, dtype=xp.int64)
                prepared["scatter_rows"] = xp.asarray(
                    np.repeat(row_host, self.block_size), dtype=xp.int64
                )
                prepared["scatter_tokens"] = xp.asarray(
                    starts.reshape(-1), dtype=xp.int64
                )
            else:
                if metadata is None:
                    # Regular metadata creates only valid complete blocks.
                    rows = xp.repeat(xp.arange(batch, dtype=xp.int64), block_shape[1])
                    slots = xp.tile(xp.arange(block_shape[1], dtype=xp.int64), batch)
                    indices = block_indices.reshape(-1, self.block_size)
                else:
                    # Explicit GPU-provided metadata may contain invalid slots.
                    # Preserve the exact mask (rare slower fallback).
                    rows, slots = xp.nonzero(prepared["block_valid"])
                    indices = block_indices[rows, slots]
                prepared["valid_block_rows"] = rows
                prepared["valid_block_slots"] = slots
                prepared["scatter_rows"] = xp.repeat(rows, self.block_size)
                prepared["scatter_tokens"] = indices.reshape(-1)
        # Rows with fewer complete blocks have -1 sentinels in
        # block_end_positions.  Keep those sentinels for the causal/global
        # visibility test, but never pass them to RoPE: masked summary keys
        # still undergo the positional rotation before attention masking.
        # This also avoids repeating the conversion on every layer/replay,
        # since 0102F shares the prepared metadata within a microbatch.
        prepared["block_rope_positions"] = xp.where(
            prepared["block_valid"],
            prepared["block_end_positions"],
            xp.zeros_like(prepared["block_end_positions"]),
        )
        if not host_bounds_checked:
            valid_indices = block_indices[prepared["block_valid"]]
            if valid_indices.size and (
                bool(xp.any(valid_indices < 0))
                or bool(xp.any(valid_indices >= seq_len))
            ):
                raise IndexError("valid block_indices must refer to source tokens")
        if can_cache:
            metadata["_dual_resolution_device_metadata_0102f"] = (cache_key, prepared)
        return prepared

    def _compress_attention_forward(self, q, k, v, retain_probabilities):
        """Full bidirectional attention, batched in bounded groups of blocks.

        A complete 65K context has 512 blocks. Building a single FP32
        [512, heads, 128, 128] attention matrix costs hundreds of MiB even
        when the backward cache is checkpointed. Chunking bounds that peak.
        """
        if self.compression_kernel == "triton":
            from ..backend import BACKEND_NAME
            if BACKEND_NAME != "cupy":
                raise RuntimeError("Triton compression requires the CuPy backend")
            from .triton_dual_resolution_dense import forward
            # Internal Q/K/V are [blocks, heads, 128, 64].
            # Dense Triton core expects [blocks, 128, heads, 64].
            result, triton_cache = forward(
                q.transpose(0, 2, 1, 3),
                k.transpose(0, 2, 1, 3),
                v.transpose(0, 2, 1, 3),
                scale=self.compression_scale,
            )
            return result.transpose(0, 2, 1, 3), triton_cache
        context = xp.empty_like(q)
        probabilities_saved = [] if retain_probabilities else None
        for start in range(0, q.shape[0], self.compression_block_chunk):
            stop = min(q.shape[0], start + self.compression_block_chunk)
            scores = xp.matmul(
                q[start:stop] * self.compression_scale,
                k[start:stop].swapaxes(-2, -1),
            )
            probabilities = softmax_forward(scores, axis=-1)
            context[start:stop] = xp.matmul(probabilities, v[start:stop])
            if retain_probabilities:
                probabilities_saved.append(probabilities)
        if retain_probabilities:
            return context, xp.concatenate(probabilities_saved, axis=0)
        return context, None

    def _compress_attention_backward(self, d_context, q, k, v, cached_probs):
        """Blockwise Q/K/V gradient with optional softmax recomputation."""
        if self.compression_kernel == "triton":
            from .triton_dual_resolution_dense import backward
            if cached_probs is None:
                raise RuntimeError("Triton compression backward requires forward LSE cache")
            dq, dk, dv = backward(
                d_context.transpose(0, 2, 1, 3).astype(xp.float32, copy=False),
                cached_probs,
            )
            return (dq.transpose(0, 2, 1, 3),
                    dk.transpose(0, 2, 1, 3),
                    dv.transpose(0, 2, 1, 3))
        d_q = xp.empty_like(q)
        d_k = xp.empty_like(k)
        d_v = xp.empty_like(v)
        for start in range(0, q.shape[0], self.compression_block_chunk):
            stop = min(q.shape[0], start + self.compression_block_chunk)
            sl = slice(start, stop)
            if cached_probs is None:
                scores = xp.matmul(
                    q[sl] * self.compression_scale,
                    k[sl].swapaxes(-2, -1),
                )
                probabilities = softmax_forward(scores, axis=-1)
            else:
                probabilities = cached_probs[sl]
            dp = xp.matmul(d_context[sl], v[sl].swapaxes(-2, -1))
            d_v[sl] = xp.matmul(probabilities.swapaxes(-2, -1), d_context[sl])
            d_scores = softmax_backward(dp, probabilities, axis=-1)
            d_q[sl] = xp.matmul(d_scores, k[sl]) * self.compression_scale
            d_k[sl] = xp.matmul(
                d_scores.swapaxes(-2, -1), q[sl]
            ) * self.compression_scale
        return d_q, d_k, d_v

    def _compress_forward(self, x, metadata, return_cache=True):
        batch, _, _ = x.shape
        indices = metadata["block_indices"]
        valid = metadata["block_valid"]
        block_count = indices.shape[1]
        safe_indices = xp.where(valid[..., None], indices, 0)
        batch_indices = xp.broadcast_to(
            xp.arange(batch, dtype=xp.int64)[:, None, None], safe_indices.shape
        )
        fast_scatter = _fast_projection_enabled() and "scatter_rows" in metadata
        if fast_scatter:
            scatter_rows = metadata["scatter_rows"]
            scatter_tokens = metadata["scatter_tokens"]
            blocks = x[scatter_rows, scatter_tokens].reshape(
                -1, self.block_size, self.d_model
            )
        else:
            gathered = x[batch_indices, safe_indices]
            blocks = gathered[valid]
        valid_count = blocks.shape[0]
        summaries = xp.zeros((batch, block_count, self.d_model), dtype=x.dtype)
        if valid_count == 0:
            return summaries, {
                "empty": True,
                "x_shape": x.shape,
                "safe_indices": safe_indices,
                "valid": valid,
            }

        hidden = _linear_forward(blocks, self.W_compress_in)
        layer_caches = []
        for compressor_index, layer in enumerate(self.compression_layers):
            normalized, norm_cache = _rmsnorm_forward(hidden, layer["norm"])
            if _fast_projection_enabled():
                self._ensure_projection_groups()
                q_projection, k_projection, v_projection = (
                    self._packed_compress[compressor_index].forward(normalized)
                )
            else:
                q_projection = _linear_forward(normalized, layer["Wq"])
                k_projection = _linear_forward(normalized, layer["Wk"])
                v_projection = _linear_forward(normalized, layer["Wv"])
            attention_dtype = (
                q_projection.dtype
                if (_fast_projection_enabled()
                    and _bf16_compressor_qkv_enabled()
                    and self.compression_kernel == "triton")
                else _work_array(q_projection).dtype
            )
            q = q_projection.astype(attention_dtype, copy=False).reshape(
                valid_count, self.block_size, self.compression_heads,
                self.compression_head_dim,
            ).transpose(0, 2, 1, 3)
            k = k_projection.astype(attention_dtype, copy=False).reshape(
                valid_count, self.block_size, self.compression_heads,
                self.compression_head_dim,
            ).transpose(0, 2, 1, 3)
            v = v_projection.astype(attention_dtype, copy=False).reshape(
                valid_count, self.block_size, self.compression_heads,
                self.compression_head_dim,
            ).transpose(0, 2, 1, 3)
            context, probabilities = self._compress_attention_forward(
                q, k, v,
                retain_probabilities=return_cache and not self.recompute_attention_probs,
            )
            merged = context.transpose(0, 2, 1, 3).reshape(
                valid_count, self.block_size, self.compression_dim
            )
            merged_compute = (
                merged.astype(hidden.dtype)
                if is_low_precision_dtype(hidden.dtype)
                else merged
            )
            attention_output = _linear_forward(merged_compute, layer["Wo"])
            next_hidden = hidden + attention_output
            if return_cache:
                layer_caches.append({
                    "hidden": hidden,
                    "normalized": normalized,
                    "norm_cache": norm_cache,
                    "q_projection": q_projection,
                    "k_projection": k_projection,
                    "v_projection": v_projection,
                    "q": q,
                    "k": k,
                    "v": v,
                    "probabilities": (
                        probabilities if self.compression_kernel == "triton"
                        else (None if self.recompute_attention_probs else probabilities)
                    ),
                    "merged_compute": merged_compute,
                })
            hidden = next_hidden

        pool_input, pool_norm_cache = _rmsnorm_forward(hidden, self.pool_norm)
        if _fast_projection_enabled():
            self._ensure_projection_groups()
            pool_k_projection, pool_v_projection = self._packed_pool.forward(pool_input)
        else:
            pool_k_projection = _linear_forward(pool_input, self.W_pool_k)
            pool_v_projection = _linear_forward(pool_input, self.W_pool_v)
        pool_k = _work_array(pool_k_projection).reshape(
            valid_count, self.block_size, self.compression_heads,
            self.compression_head_dim,
        ).transpose(0, 2, 1, 3)
        pool_v = _work_array(pool_v_projection).reshape(
            valid_count, self.block_size, self.compression_heads,
            self.compression_head_dim,
        ).transpose(0, 2, 1, 3)
        pool_query = _work_array(self.pool_query.data)
        fused_pooling = _fused_pooling_enabled()
        with performance_scope("dual_resolution.pooling.forward"):
            if fused_pooling:
                from .triton_dual_resolution_pooling import forward as pooling_forward
                pool_context, pool_probabilities = pooling_forward(
                    pool_k, pool_v, pool_query, self.compression_scale
                )
            else:
                pool_scores = xp.sum(
                    pool_k * pool_query[None, :, None, :], axis=-1
                ) * self.compression_scale
                pool_probabilities = softmax_forward(pool_scores, axis=-1)
                pool_context = xp.sum(pool_probabilities[..., None] * pool_v, axis=2)
        pool_merged = pool_context.reshape(valid_count, self.compression_dim)
        pool_merged_compute = (
            pool_merged.astype(hidden.dtype)
            if is_low_precision_dtype(hidden.dtype)
            else pool_merged
        )
        pooled = _linear_forward(pool_merged_compute, self.W_pool_o)
        summary_valid = _linear_forward(pooled, self.W_compress_up)
        if fast_scatter:
            summaries[metadata["valid_block_rows"], metadata["valid_block_slots"]] = summary_valid
        else:
            summaries[valid] = summary_valid
        if not return_cache:
            return summaries, None
        return summaries, {
            "empty": False,
            "x_shape": x.shape,
            "safe_indices": safe_indices,
            "valid": valid,
            "blocks": blocks,
            "scatter_rows": metadata.get("scatter_rows") if fast_scatter else None,
            "scatter_tokens": metadata.get("scatter_tokens") if fast_scatter else None,
            "summary_rows": metadata.get("valid_block_rows") if fast_scatter else None,
            "summary_slots": metadata.get("valid_block_slots") if fast_scatter else None,
            "hidden_initial": layer_caches[0]["hidden"] if layer_caches else hidden,
            "layers": layer_caches,
            "pool_hidden": hidden,
            "pool_input": pool_input,
            "pool_norm_cache": pool_norm_cache,
            "pool_k_projection": pool_k_projection,
            "pool_v_projection": pool_v_projection,
            "pool_k": pool_k,
            "pool_v": pool_v,
            "pool_probabilities": pool_probabilities,
            "pool_fused": fused_pooling,
            "pool_merged_compute": pool_merged_compute,
            "pooled": pooled,
        }

    def _compress_backward(self, d_summaries, cache, accumulate_parameter_grads):
        dx = xp.zeros(cache["x_shape"], dtype=_work_array(d_summaries).dtype)
        if cache["empty"]:
            return dx
        valid = cache["valid"]
        if cache.get("summary_rows") is not None:
            d_summary_valid = _work_array(
                d_summaries[cache["summary_rows"], cache["summary_slots"]]
            )
        else:
            d_summary_valid = _work_array(d_summaries[valid])
        d_pooled = _linear_backward(
            d_summary_valid,
            cache["pooled"],
            self.W_compress_up,
            accumulate_parameter_grads,
        )
        d_pool_merged = _linear_backward(
            d_pooled,
            cache["pool_merged_compute"],
            self.W_pool_o,
            accumulate_parameter_grads,
        )
        valid_count = d_pool_merged.shape[0]
        d_pool_context = d_pool_merged.reshape(
            valid_count, self.compression_heads, self.compression_head_dim
        )
        probabilities = cache["pool_probabilities"]
        pool_v = cache["pool_v"]
        pool_k = cache["pool_k"]
        pool_query = _work_array(self.pool_query.data)
        with performance_scope("dual_resolution.pooling.backward"):
            if cache.get("pool_fused", False):
                from .triton_dual_resolution_pooling import backward as pooling_backward
                d_pool_k, d_pool_v, d_pool_query = pooling_backward(
                    d_pool_context.astype(xp.float32, copy=False), pool_k, pool_v,
                    pool_query, probabilities, self.compression_scale,
                )
            else:
                d_pool_v = probabilities[..., None] * d_pool_context[:, :, None, :]
                d_probabilities = xp.sum(
                    d_pool_context[:, :, None, :] * pool_v, axis=-1
                )
                d_scores = softmax_backward(d_probabilities, probabilities, axis=-1)
                d_pool_k = (
                    d_scores[..., None]
                    * pool_query[None, :, None, :]
                    * self.compression_scale
                )
                d_pool_query = xp.sum(
                    d_scores[..., None] * pool_k, axis=(0, 2)
                ) * self.compression_scale
        if accumulate_parameter_grads:
            _add_parameter_grad(self.pool_query, d_pool_query)
        d_pool_k_projection = d_pool_k.transpose(0, 2, 1, 3).reshape(
            valid_count, self.block_size, self.compression_dim
        )
        d_pool_v_projection = d_pool_v.transpose(0, 2, 1, 3).reshape(
            valid_count, self.block_size, self.compression_dim
        )
        if _fast_projection_enabled():
            d_pool_input = self._packed_pool.backward(
                (d_pool_k_projection, d_pool_v_projection),
                cache["pool_input"], accumulate_parameter_grads,
            )
        else:
            d_pool_input = _linear_backward(
                d_pool_k_projection, cache["pool_input"],
                self.W_pool_k, accumulate_parameter_grads,
            )
            d_pool_input += _linear_backward(
                d_pool_v_projection, cache["pool_input"],
                self.W_pool_v, accumulate_parameter_grads,
            )
        d_hidden = _rmsnorm_backward(
            d_pool_input,
            cache["pool_norm_cache"],
            self.pool_norm,
            accumulate_parameter_grads,
        )

        for compressor_index in reversed(range(len(self.compression_layers))):
            layer = self.compression_layers[compressor_index]
            layer_cache = cache["layers"][compressor_index]
            d_attention_output = d_hidden
            d_hidden_residual = d_hidden
            d_merged = _linear_backward(
                d_attention_output,
                layer_cache["merged_compute"],
                layer["Wo"],
                accumulate_parameter_grads,
            )
            d_context = d_merged.reshape(
                valid_count, self.block_size, self.compression_heads,
                self.compression_head_dim,
            ).transpose(0, 2, 1, 3)
            probabilities = layer_cache["probabilities"]
            q = layer_cache["q"]
            k = layer_cache["k"]
            v = layer_cache["v"]
            d_q, d_k, d_v = self._compress_attention_backward(
                d_context, q, k, v, probabilities
            )
            d_q_projection = d_q.transpose(0, 2, 1, 3).reshape(
                valid_count, self.block_size, self.compression_dim
            )
            d_k_projection = d_k.transpose(0, 2, 1, 3).reshape(
                valid_count, self.block_size, self.compression_dim
            )
            d_v_projection = d_v.transpose(0, 2, 1, 3).reshape(
                valid_count, self.block_size, self.compression_dim
            )
            if _fast_projection_enabled():
                d_normalized = self._packed_compress[compressor_index].backward(
                    (d_q_projection, d_k_projection, d_v_projection),
                    layer_cache["normalized"], accumulate_parameter_grads,
                )
            else:
                d_normalized = _linear_backward(
                    d_q_projection, layer_cache["normalized"], layer["Wq"],
                    accumulate_parameter_grads,
                )
                d_normalized += _linear_backward(
                    d_k_projection, layer_cache["normalized"], layer["Wk"],
                    accumulate_parameter_grads,
                )
                d_normalized += _linear_backward(
                    d_v_projection, layer_cache["normalized"], layer["Wv"],
                    accumulate_parameter_grads,
                )
            d_hidden = d_hidden_residual + _rmsnorm_backward(
                d_normalized,
                layer_cache["norm_cache"],
                layer["norm"],
                accumulate_parameter_grads,
            )

        d_blocks = _linear_backward(
            d_hidden,
            cache["blocks"],
            self.W_compress_in,
            accumulate_parameter_grads,
        )
        safe_indices = cache["safe_indices"]
        batch = cache["x_shape"][0]
        batch_indices = xp.broadcast_to(
            xp.arange(batch, dtype=xp.int64)[:, None, None], safe_indices.shape
        )
        if cache.get("scatter_rows") is not None:
            # Valid completed blocks never overlap within a document: direct
            # scatter is exact and needs neither an all-block-sized temporary
            # nor atomic add.at. Invalid slots are absent from these indices.
            dx[cache["scatter_rows"], cache["scatter_tokens"]] = (
                d_blocks.reshape(-1, self.d_model)
            )
        else:
            d_all_blocks = xp.zeros(
                safe_indices.shape + (self.d_model,), dtype=d_blocks.dtype
            )
            d_all_blocks[valid] = d_blocks
            xp.add.at(
                dx,
                (batch_indices.reshape(-1), safe_indices.reshape(-1)),
                d_all_blocks.reshape(-1, self.d_model),
            )
        return dx

    def _group_queries(self, query):
        length = query.shape[0]
        return query.reshape(
            length, self.n_kv_heads, self.group_size, self.d_head
        ).transpose(1, 2, 0, 3)

    def _ungroup_queries(self, grouped):
        length = grouped.shape[2]
        return grouped.transpose(2, 0, 1, 3).reshape(
            length, self.n_q_heads, self.d_head
        )

    def _local_forward(self, q, k, v, document_ids, return_cache=True):
        if self.local_kernel == "triton":
            from ..backend import BACKEND_NAME
            if BACKEND_NAME != "cupy":
                raise RuntimeError(
                    "Triton dual-resolution local attention requires MINI_LLM_BACKEND=cupy"
                )
            # The GPU implementation is deliberately opt-in and strict. Reject
            # unsupported geometries rather than silently benchmark CuPy.
            from .triton_dual_resolution_local import local_attention_forward
            context, triton_cache = local_attention_forward(
                q, k, v, document_ids, window=self.local_window, scale=self.scale,
            )
            return context, ({"triton": triton_cache} if return_cache else None)
        batch, seq_len = q.shape[:2]
        output = xp.zeros(
            (batch, seq_len, self.n_q_heads, self.d_head),
            dtype=_work_array(q).dtype,
        )
        caches = []
        chunk_size = min(self.global_query_chunk, max(1, seq_len - 1))
        for row in range(batch):
            for query_start in range(0, seq_len, chunk_size):
                query_stop = min(seq_len, query_start + chunk_size)
                key_start = max(0, query_start - self.local_window + 1)
                key_stop = query_stop
                query_indices = xp.arange(query_start, query_stop)[:, None]
                key_indices = xp.arange(key_start, key_stop)[None, :]
                allowed = (
                    (key_indices <= query_indices)
                    & (key_indices >= query_indices - self.local_window + 1)
                    & (
                        document_ids[row, query_start:query_stop, None]
                        == document_ids[row, None, key_start:key_stop]
                    )
                )
                query_grouped = self._group_queries(
                    _work_array(q[row, query_start:query_stop])
                )
                key_heads = _work_array(k[row, key_start:key_stop]).transpose(1, 0, 2)
                value_heads = _work_array(v[row, key_start:key_stop]).transpose(1, 0, 2)
                scores = xp.matmul(
                    query_grouped * self.scale,
                    key_heads[:, None, :, :].swapaxes(-2, -1),
                )
                scores = xp.where(allowed[None, None, :, :], scores, -xp.inf)
                probabilities = softmax_forward(scores, axis=-1)
                context = xp.matmul(
                    probabilities, value_heads[:, None, :, :]
                )
                output[row, query_start:query_stop] = self._ungroup_queries(context)
                if return_cache:
                    caches.append({
                        "row": row,
                        "query_start": query_start,
                        "query_stop": query_stop,
                        "key_start": key_start,
                        "key_stop": key_stop,
                        "probabilities": (
                            None if self.recompute_attention_probs else probabilities
                        ),
                    })
        return output, caches

    def _local_backward(self, d_output, q, k, v, caches, document_ids):
        if isinstance(caches, dict) and "triton" in caches:
            from .triton_dual_resolution_local import local_attention_backward
            return local_attention_backward(
                _work_array(d_output).astype(xp.float32, copy=False),
                caches["triton"],
            )
        dq = xp.zeros(q.shape, dtype=_work_array(d_output).dtype)
        dk = xp.zeros(k.shape, dtype=dq.dtype)
        dv = xp.zeros(v.shape, dtype=dq.dtype)
        for entry in caches:
            row = entry["row"]
            qs, qe = entry["query_start"], entry["query_stop"]
            ks, ke = entry["key_start"], entry["key_stop"]
            probabilities = entry["probabilities"]
            query_grouped = self._group_queries(_work_array(q[row, qs:qe]))
            key_heads = _work_array(k[row, ks:ke]).transpose(1, 0, 2)
            if probabilities is None:
                qi = xp.arange(qs, qe)[:, None]
                ki = xp.arange(ks, ke)[None, :]
                allowed = (
                    (ki <= qi) & (ki >= qi - self.local_window + 1)
                    & (document_ids[row, qs:qe, None] == document_ids[row, None, ks:ke])
                )
                scores = xp.matmul(
                    query_grouped * self.scale,
                    key_heads[:, None, :, :].swapaxes(-2, -1),
                )
                probabilities = softmax_forward(
                    xp.where(allowed[None, None, :, :], scores, -xp.inf),
                    axis=-1,
                )
            value_heads = _work_array(v[row, ks:ke]).transpose(1, 0, 2)
            d_context = self._group_queries(_work_array(d_output[row, qs:qe]))
            d_probabilities = xp.matmul(
                d_context, value_heads[:, None, :, :].swapaxes(-2, -1)
            )
            d_value_heads = xp.matmul(
                probabilities.swapaxes(-2, -1), d_context
            ).sum(axis=1)
            d_scores = softmax_backward(d_probabilities, probabilities, axis=-1)
            d_query = xp.matmul(
                d_scores, key_heads[:, None, :, :]
            ) * self.scale
            d_key_heads = xp.matmul(
                d_scores.swapaxes(-2, -1), query_grouped
            ).sum(axis=1) * self.scale
            dq[row, qs:qe] += self._ungroup_queries(d_query)
            dk[row, ks:ke] += d_key_heads.transpose(1, 0, 2)
            dv[row, ks:ke] += d_value_heads.transpose(1, 0, 2)
        return dq, dk, dv

    def _global_forward(
        self, q, summary_k, summary_v, document_ids, position_ids,
        block_valid, block_document_ids, block_end_positions,
        return_cache=True,
    ):
        batch, seq_len = q.shape[:2]
        summary_count = summary_k.shape[1]
        if self.global_kernel == "triton":
            from ..backend import BACKEND_NAME
            if BACKEND_NAME != "cupy":
                raise RuntimeError("Triton global attention requires the CuPy backend")
            from .triton_dual_resolution_dense import forward
            # The tiled kernel calculates availability from its *exact* mask,
            # including packed documents and explicitly invalid blocks.
            context, triton_cache = forward(
                q, summary_k, summary_v, scale=self.scale,
                documents=document_ids, positions=position_ids,
                block_documents=block_document_ids,
                block_ends=block_end_positions,
                block_valid=block_valid,
            )
            return (context, triton_cache.available,
                    ({"triton": triton_cache} if return_cache else None))
        output = xp.zeros(
            (batch, seq_len, self.n_q_heads, self.d_head),
            dtype=_work_array(q).dtype,
        )
        available = xp.zeros((batch, seq_len), dtype=xp.bool_)
        caches = []
        if summary_count == 0:
            return output, available, caches
        chunk_size = min(self.global_query_chunk, max(1, seq_len - 1))
        for row in range(batch):
            key_heads = _work_array(summary_k[row]).transpose(1, 0, 2)
            value_heads = _work_array(summary_v[row]).transpose(1, 0, 2)
            for query_start in range(0, seq_len, chunk_size):
                query_stop = min(seq_len, query_start + chunk_size)
                allowed = (
                    block_valid[row, None, :]
                    & (
                        document_ids[row, query_start:query_stop, None]
                        == block_document_ids[row, None, :]
                    )
                    & (
                        block_end_positions[row, None, :]
                        < position_ids[row, query_start:query_stop, None]
                    )
                )
                row_available = xp.any(allowed, axis=-1)
                available[row, query_start:query_stop] = row_available
                query_grouped = self._group_queries(
                    _work_array(q[row, query_start:query_stop])
                )
                scores = xp.matmul(
                    query_grouped * self.scale,
                    key_heads[:, None, :, :].swapaxes(-2, -1),
                )
                scores = xp.where(allowed[None, None, :, :], scores, -xp.inf)
                scores = xp.where(
                    row_available[None, None, :, None], scores, 0.0
                )
                probabilities = softmax_forward(scores, axis=-1)
                probabilities = probabilities * allowed[None, None, :, :]
                context = xp.matmul(
                    probabilities, value_heads[:, None, :, :]
                )
                output[row, query_start:query_stop] = self._ungroup_queries(context)
                if return_cache:
                    caches.append({
                        "row": row,
                        "query_start": query_start,
                        "query_stop": query_stop,
                        "probabilities": (
                            None if self.recompute_attention_probs else probabilities
                        ),
                    })
        return output, available, caches

    def _global_backward(
        self, d_output, q, summary_k, summary_v, caches, metadata,
    ):
        if isinstance(caches, dict) and "triton" in caches:
            from .triton_dual_resolution_dense import backward
            return backward(_work_array(d_output).astype(xp.float32, copy=False),
                            caches["triton"])
        dq = xp.zeros(q.shape, dtype=_work_array(d_output).dtype)
        dk = xp.zeros(summary_k.shape, dtype=dq.dtype)
        dv = xp.zeros(summary_v.shape, dtype=dq.dtype)
        for entry in caches:
            row = entry["row"]
            qs, qe = entry["query_start"], entry["query_stop"]
            probabilities = entry["probabilities"]
            query_grouped = self._group_queries(_work_array(q[row, qs:qe]))
            key_heads = _work_array(summary_k[row]).transpose(1, 0, 2)
            value_heads = _work_array(summary_v[row]).transpose(1, 0, 2)
            if probabilities is None:
                allowed = (
                    metadata["block_valid"][row, None, :]
                    & (metadata["document_ids"][row, qs:qe, None]
                       == metadata["block_document_ids"][row, None, :])
                    & (metadata["block_end_positions"][row, None, :]
                       < metadata["position_ids"][row, qs:qe, None])
                )
                active = xp.any(allowed, axis=-1)
                scores = xp.matmul(
                    query_grouped * self.scale,
                    key_heads[:, None, :, :].swapaxes(-2, -1),
                )
                scores = xp.where(allowed[None, None, :, :], scores, -xp.inf)
                scores = xp.where(active[None, None, :, None], scores, 0.0)
                probabilities = softmax_forward(scores, axis=-1)
                probabilities *= allowed[None, None, :, :]
            d_context = self._group_queries(_work_array(d_output[row, qs:qe]))
            d_probabilities = xp.matmul(
                d_context, value_heads[:, None, :, :].swapaxes(-2, -1)
            )
            d_value_heads = xp.matmul(
                probabilities.swapaxes(-2, -1), d_context
            ).sum(axis=1)
            d_scores = softmax_backward(d_probabilities, probabilities, axis=-1)
            d_query = xp.matmul(
                d_scores, key_heads[:, None, :, :]
            ) * self.scale
            d_key_heads = xp.matmul(
                d_scores.swapaxes(-2, -1), query_grouped
            ).sum(axis=1) * self.scale
            dq[row, qs:qe] += self._ungroup_queries(d_query)
            dk[row] += d_key_heads.transpose(1, 0, 2)
            dv[row] += d_value_heads.transpose(1, 0, 2)
        return dq, dk, dv

    def forward(
        self,
        x,
        return_cache=True,
        position_ids=None,
        dual_resolution_metadata=None,
        **ignored_compatible,
    ):
        del ignored_compatible
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("attention input must have shape [B,T,d_model]")
        batch, seq_len, _ = x.shape
        if seq_len < 1:
            raise ValueError("attention sequence length must be positive")
        metadata = self._prepare_metadata(
            dual_resolution_metadata, batch, seq_len, position_ids
        )
        positions = metadata["position_ids"]
        x_2d = x.reshape(-1, self.d_model)
        if _fast_projection_enabled():
            self._ensure_projection_groups()
            q_projection, k_projection, v_projection = self._packed_main.forward(x)
        else:
            q_projection = x_2d @ self.Wq.data
            k_projection = x_2d @ self.Wk.data
            v_projection = x_2d @ self.Wv.data
        q_pre = q_projection.reshape(batch, seq_len, self.n_q_heads, self.d_head)
        k_pre = k_projection.reshape(batch, seq_len, self.n_kv_heads, self.d_head)
        v = v_projection.reshape(batch, seq_len, self.n_kv_heads, self.d_head)
        q, q_rope_cache = rope_forward(q_pre, self.rope_base, positions)
        k, k_rope_cache = rope_forward(k_pre, self.rope_base, positions)
        with performance_scope("dual_resolution.local.forward"):
            local_context, local_cache = self._local_forward(
                q, k, v, metadata["document_ids"], return_cache=return_cache
            )

        # For T <= block_size no query can ever see an earlier completed
        # summary. Skip the ENTIRE compressor/global/gate branch, including
        # replay and backward, not just its final gated contribution.
        if seq_len <= self.block_size:
            merged = local_context.reshape(batch, seq_len, self.d_model)
            merged_compute = (
                merged.astype(x.dtype) if is_low_precision_dtype(x.dtype)
                else merged
            )
            output = _linear_forward(merged_compute, self.Wo)
            if not return_cache:
                return output
            return output, {
                "local_only": True,
                "x": x, "metadata": metadata,
                "q": q, "k": k, "v": v,
                "q_rope_cache": q_rope_cache,
                "k_rope_cache": k_rope_cache,
                "local_cache": local_cache,
                "local_context": local_context,
                "summaries": xp.zeros((batch, 0, self.d_model), dtype=x.dtype),
                "global_available": xp.zeros((batch, seq_len), dtype=xp.bool_),
                "effective_gate": xp.zeros((batch, seq_len, self.n_q_heads), dtype=_work_array(q).dtype),
                "merged": merged,
                "merged_compute": merged_compute,
            }

        with performance_scope("dual_resolution.compressor.forward"):
            summaries, compressor_cache = self._compress_forward(
                x, metadata, return_cache=return_cache
            )
        summary_shape = summaries.shape[:2]
        if _fast_projection_enabled():
            self._ensure_projection_groups()
            summary_k_projection, summary_v_projection = self._packed_global.forward(summaries)
        else:
            summary_k_projection = _linear_forward(summaries, self.W_global_k)
            summary_v_projection = _linear_forward(summaries, self.W_global_v)
        summary_k_pre = summary_k_projection.reshape(
            summary_shape + (self.n_kv_heads, self.d_head)
        )
        summary_v = summary_v_projection.reshape(
            summary_shape + (self.n_kv_heads, self.d_head)
        )
        summary_k, summary_rope_cache = rope_forward(
            summary_k_pre, self.rope_base, metadata["block_rope_positions"]
        )
        with performance_scope("dual_resolution.global.forward"):
            global_context, global_available, global_cache = self._global_forward(
                q,
                summary_k,
                summary_v,
                metadata["document_ids"],
                positions,
                metadata["block_valid"],
                metadata["block_document_ids"],
                metadata["block_end_positions"],
                return_cache=return_cache,
            )
        gate_logits = _linear_forward(x, self.W_gate)
        fused_gate = _fused_gating_enabled()
        if fused_gate:
            from .triton_dual_resolution_gate import forward as gate_forward
            with performance_scope("dual_resolution.gate.forward"):
                context, gate, effective_gate = gate_forward(
                    xp.ascontiguousarray(local_context),
                    xp.ascontiguousarray(global_context),
                    xp.ascontiguousarray(gate_logits),
                    xp.ascontiguousarray(global_available),
                )
        else:
            with performance_scope("dual_resolution.gate.forward"):
                gate = _sigmoid(gate_logits)
                effective_gate = gate * global_available[..., None]
                context = (
                    local_context
                    + effective_gate[..., None] * (global_context - local_context)
                )
        merged = context.reshape(batch, seq_len, self.d_model)
        merged_compute = (
            merged.astype(x.dtype) if is_low_precision_dtype(x.dtype) else merged
        )
        output = _linear_forward(merged_compute, self.Wo)
        if not return_cache:
            return output
        return output, {
            "x": x,
            "metadata": metadata,
            "q": q,
            "k": k,
            "v": v,
            "q_rope_cache": q_rope_cache,
            "k_rope_cache": k_rope_cache,
            "local_context": local_context,
            "local_cache": local_cache,
            "summaries": summaries,
            "compressor_cache": compressor_cache,
            "summary_k": summary_k,
            "summary_v": summary_v,
            "summary_rope_cache": summary_rope_cache,
            "global_context": global_context,
            "global_available": global_available,
            "global_cache": global_cache,
            "gate": gate,
            "effective_gate": effective_gate,
            "gate_fused": fused_gate,
            "merged": merged,
            "merged_compute": merged_compute,
        }

    def backward(
        self,
        dy,
        cache,
        output_scale=None,
        return_output_scale_grad=False,
        shared_context_kv_grad=None,
        return_shared_context_kv_grad=False,
        accumulate_parameter_grads=True,
    ):
        x = cache["x"]
        if dy.shape != x.shape:
            raise ValueError("dy must have the same shape as the attention input")
        with performance_scope("dual_resolution.output_projection.backward"):
            dy_work = _work_array(dy)
            merged_compute = cache["merged_compute"]
            merged_2d = _work_array(merged_compute.reshape(-1, self.d_model))
            dy_2d = dy_work.reshape(-1, self.d_model)
            raw_output_wgrad = merged_2d.T @ dy_2d
            output_scale_grad = None
            if return_output_scale_grad:
                output_scale_grad = xp.sum(
                    raw_output_wgrad * _work_array(self.Wo.data),
                    dtype=xp.float32 if is_low_precision_dtype(self.Wo.data.dtype) else None,
                )
            scale = 1.0 if output_scale is None else output_scale
            if accumulate_parameter_grads:
                _add_parameter_grad(self.Wo, raw_output_wgrad * scale)
            if (_bf16_output_dx_enabled()
                    and is_low_precision_dtype(dy.dtype)
                    and dy.dtype == self.Wo.data.dtype):
                d_merged = (
                    dy.reshape(-1, self.d_model) @ self.Wo.data.T
                ).astype("float32", copy=False) * scale
            else:
                d_merged = (dy_2d @ _work_array(self.Wo.data).T) * scale
            d_context = d_merged.reshape(
                x.shape[0], x.shape[1], self.n_q_heads, self.d_head
            )

        if cache.get("local_only", False):
            with performance_scope("dual_resolution.local.backward"):
                dq, dk, dv = self._local_backward(
                    d_context, cache["q"], cache["k"], cache["v"],
                    cache["local_cache"], cache["metadata"]["document_ids"],
                )
            dx_gate = 0.0
            dx_compressor = 0.0
        else:
            local_context = cache["local_context"]
            global_context = cache["global_context"]
            effective_gate = cache["effective_gate"]
            gate = cache["gate"]
            with performance_scope("dual_resolution.gate.backward"):
                if cache.get("gate_fused", False):
                    from .triton_dual_resolution_gate import backward as gate_backward
                    d_local, d_global, d_gate_logits = gate_backward(
                        xp.ascontiguousarray(d_context),
                        xp.ascontiguousarray(local_context),
                        xp.ascontiguousarray(global_context),
                        xp.ascontiguousarray(gate),
                        xp.ascontiguousarray(effective_gate),
                        xp.ascontiguousarray(cache["global_available"]),
                    )
                else:
                    d_local = d_context * (1.0 - effective_gate[..., None])
                    d_global = d_context * effective_gate[..., None]
                    d_effective_gate = xp.sum(
                        d_context * (global_context - local_context), axis=-1
                    )
                    d_gate_logits = (
                        d_effective_gate
                        * cache["global_available"][..., None]
                        * gate
                        * (1.0 - gate)
                    )
            dx_gate = _linear_backward(
                d_gate_logits,
                x,
                self.W_gate,
                accumulate_parameter_grads,
            )

            with performance_scope("dual_resolution.local.backward"):
                dq_local, dk, dv = self._local_backward(
                    d_local, cache["q"], cache["k"], cache["v"],
                    cache["local_cache"], cache["metadata"]["document_ids"],
                )
            with performance_scope("dual_resolution.global.backward"):
                dq_global, d_summary_k, d_summary_v = self._global_backward(
                    d_global,
                    cache["q"],
                    cache["summary_k"],
                    cache["summary_v"],
                    cache["global_cache"], cache["metadata"],
                )
            dq = dq_local + dq_global
            d_summary_k_pre = rope_backward(
                d_summary_k, cache["summary_rope_cache"]
            )
            kgrad = d_summary_k_pre.reshape(
                cache["summaries"].shape[:2] + (self.n_kv_heads * self.d_head,)
            )
            vgrad = d_summary_v.reshape(
                cache["summaries"].shape[:2] + (self.n_kv_heads * self.d_head,)
            )
            if _fast_projection_enabled():
                d_summaries = self._packed_global.backward(
                    (kgrad, vgrad), cache["summaries"], accumulate_parameter_grads,
                )
            else:
                d_summaries = _linear_backward(
                    kgrad, cache["summaries"], self.W_global_k, accumulate_parameter_grads,
                )
                d_summaries += _linear_backward(
                    vgrad, cache["summaries"], self.W_global_v, accumulate_parameter_grads,
                )
            with performance_scope("dual_resolution.compressor.backward"):
                dx_compressor = self._compress_backward(
                    d_summaries,
                    cache["compressor_cache"],
                    accumulate_parameter_grads,
                )

        if shared_context_kv_grad is not None:
            if len(shared_context_kv_grad) != 2:
                raise ValueError("shared_context_kv_grad must be a (dk,dv) pair")
            dk += shared_context_kv_grad[0]
            dv += shared_context_kv_grad[1]

        dq_pre = rope_backward(dq, cache["q_rope_cache"])
        dk_pre = rope_backward(dk, cache["k_rope_cache"])
        q_width = self.n_q_heads * self.d_head
        kv_width = self.n_kv_heads * self.d_head
        if _fast_projection_enabled():
            dx_main = self._packed_main.backward(
                (dq_pre.reshape(-1, q_width),
                 dk_pre.reshape(-1, kv_width),
                 dv.reshape(-1, kv_width)),
                x, accumulate_parameter_grads,
            )
        else:
            dq_flat = _work_array(dq_pre.reshape(-1, q_width))
            dk_flat = _work_array(dk_pre.reshape(-1, kv_width))
            dv_flat = _work_array(dv.reshape(-1, kv_width))
            x_flat = _work_array(x.reshape(-1, self.d_model))
            if accumulate_parameter_grads:
                _add_parameter_grad(self.Wq, x_flat.T @ dq_flat)
                _add_parameter_grad(self.Wk, x_flat.T @ dk_flat)
                _add_parameter_grad(self.Wv, x_flat.T @ dv_flat)
            dx_main = (
                dq_flat @ _work_array(self.Wq.data).T
                + dk_flat @ _work_array(self.Wk.data).T
                + dv_flat @ _work_array(self.Wv.data).T
            ).reshape(x.shape)
        dx = dx_main + dx_gate + dx_compressor
        if is_low_precision_dtype(x.dtype):
            dx = dx.astype(x.dtype)

        if return_shared_context_kv_grad:
            context_grad = (None, None)
            if return_output_scale_grad:
                return dx, output_scale_grad, context_grad
            return dx, context_grad
        if return_output_scale_grad:
            return dx, output_scale_grad
        return dx
