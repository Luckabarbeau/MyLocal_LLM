"""Trainable dual-resolution attention with compact document metadata.

The token branch is exact causal local GQA.  Complete same-document blocks are
compressed by a small bidirectional Transformer and learned multi-head pooling;
token queries can then attend to summaries whose block end strictly precedes
the query position.  Sequence attention is always evaluated in query chunks,
so this module never materializes a dense token-by-token ``T x T`` matrix.
"""

import math

import numpy as np

from ..backend import is_low_precision_dtype, resolve_dtype, xp
from ..init import matrix_parameter, ones_parameter
from .attention import softmax_backward, softmax_forward
from .rope import rope_backward, rope_forward


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
    value_2d = _work_array(value.reshape(-1, value.shape[-1]))
    gradient_2d = _work_array(gradient.reshape(-1, gradient.shape[-1]))
    weight = _work_array(parameter.data)
    if accumulate_parameter_grads:
        _add_parameter_grad(parameter, value_2d.T @ gradient_2d)
    result = gradient_2d @ weight.T
    return result.reshape(value.shape)


def _rmsnorm_forward(value, gamma, eps=1e-6):
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
        self.global_query_chunk = int(config.global_query_chunk)
        if self.block_size <= 0 or self.local_window <= 0:
            raise ValueError("block_size and local_window must be positive")
        if self.compression_dim <= 0 or self.compression_heads <= 0:
            raise ValueError("compression dimensions must be positive")
        if self.compression_dim % self.compression_heads:
            raise ValueError("compression_dim must be divisible by compression_heads")
        if self.compression_depth < 0:
            raise ValueError("compression_depth must be non-negative")
        if self.global_query_chunk <= 0:
            raise ValueError("global_query_chunk must be positive")
        self.compression_head_dim = self.compression_dim // self.compression_heads
        self.compression_scale = 1.0 / math.sqrt(self.compression_head_dim)

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
        """Compatibility hook; this implementation has no derived weight buffers."""
        return None

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
        if metadata is None:
            prepared = self._regular_metadata(batch, seq_len, position_ids)
        else:
            required = (
                "document_ids",
                "position_ids",
                "block_indices",
                "block_valid",
                "block_document_ids",
                "block_end_positions",
            )
            missing = [key for key in required if key not in metadata]
            if missing:
                raise ValueError(f"dual_resolution_metadata missing fields: {missing}")
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
            "document_ids",
            "position_ids",
            "block_indices",
            "block_document_ids",
            "block_end_positions",
        ):
            if prepared[key].dtype.kind not in {"i", "u"}:
                raise TypeError(f"{key} must use an integer dtype")
        prepared["block_valid"] = prepared["block_valid"].astype(xp.bool_, copy=False)
        valid_indices = block_indices[prepared["block_valid"]]
        if valid_indices.size and (
            bool(xp.any(valid_indices < 0))
            or bool(xp.any(valid_indices >= seq_len))
        ):
            raise IndexError("valid block_indices must refer to source tokens")
        return prepared

    def _compress_forward(self, x, metadata):
        batch, _, _ = x.shape
        indices = metadata["block_indices"]
        valid = metadata["block_valid"]
        block_count = indices.shape[1]
        safe_indices = xp.where(valid[..., None], indices, 0)
        batch_indices = xp.broadcast_to(
            xp.arange(batch, dtype=xp.int64)[:, None, None], safe_indices.shape
        )
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
        for layer in self.compression_layers:
            normalized, norm_cache = _rmsnorm_forward(hidden, layer["norm"])
            q_projection = _linear_forward(normalized, layer["Wq"])
            k_projection = _linear_forward(normalized, layer["Wk"])
            v_projection = _linear_forward(normalized, layer["Wv"])
            q = _work_array(q_projection).reshape(
                valid_count, self.block_size, self.compression_heads,
                self.compression_head_dim,
            ).transpose(0, 2, 1, 3)
            k = _work_array(k_projection).reshape(
                valid_count, self.block_size, self.compression_heads,
                self.compression_head_dim,
            ).transpose(0, 2, 1, 3)
            v = _work_array(v_projection).reshape(
                valid_count, self.block_size, self.compression_heads,
                self.compression_head_dim,
            ).transpose(0, 2, 1, 3)
            scores = xp.matmul(q * self.compression_scale, k.swapaxes(-2, -1))
            probabilities = softmax_forward(scores, axis=-1)
            context = xp.matmul(probabilities, v)
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
            layer_caches.append(
                {
                    "hidden": hidden,
                    "normalized": normalized,
                    "norm_cache": norm_cache,
                    "q_projection": q_projection,
                    "k_projection": k_projection,
                    "v_projection": v_projection,
                    "q": q,
                    "k": k,
                    "v": v,
                    "probabilities": probabilities,
                    "merged_compute": merged_compute,
                }
            )
            hidden = next_hidden

        pool_input, pool_norm_cache = _rmsnorm_forward(hidden, self.pool_norm)
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
        summaries[valid] = summary_valid
        return summaries, {
            "empty": False,
            "x_shape": x.shape,
            "safe_indices": safe_indices,
            "valid": valid,
            "blocks": blocks,
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
            "pool_merged_compute": pool_merged_compute,
            "pooled": pooled,
        }

    def _compress_backward(self, d_summaries, cache, accumulate_parameter_grads):
        dx = xp.zeros(cache["x_shape"], dtype=_work_array(d_summaries).dtype)
        if cache["empty"]:
            return dx
        valid = cache["valid"]
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
        d_pool_input = _linear_backward(
            d_pool_k_projection,
            cache["pool_input"],
            self.W_pool_k,
            accumulate_parameter_grads,
        )
        d_pool_input += _linear_backward(
            d_pool_v_projection,
            cache["pool_input"],
            self.W_pool_v,
            accumulate_parameter_grads,
        )
        d_hidden = _rmsnorm_backward(
            d_pool_input,
            cache["pool_norm_cache"],
            self.pool_norm,
            accumulate_parameter_grads,
        )

        for layer, layer_cache in zip(
            reversed(self.compression_layers), reversed(cache["layers"])
        ):
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
            d_probabilities = xp.matmul(d_context, v.swapaxes(-2, -1))
            d_v = xp.matmul(probabilities.swapaxes(-2, -1), d_context)
            d_scores = softmax_backward(d_probabilities, probabilities, axis=-1)
            d_q = xp.matmul(d_scores, k) * self.compression_scale
            d_k = xp.matmul(d_scores.swapaxes(-2, -1), q) * self.compression_scale
            d_q_projection = d_q.transpose(0, 2, 1, 3).reshape(
                valid_count, self.block_size, self.compression_dim
            )
            d_k_projection = d_k.transpose(0, 2, 1, 3).reshape(
                valid_count, self.block_size, self.compression_dim
            )
            d_v_projection = d_v.transpose(0, 2, 1, 3).reshape(
                valid_count, self.block_size, self.compression_dim
            )
            d_normalized = _linear_backward(
                d_q_projection,
                layer_cache["normalized"],
                layer["Wq"],
                accumulate_parameter_grads,
            )
            d_normalized += _linear_backward(
                d_k_projection,
                layer_cache["normalized"],
                layer["Wk"],
                accumulate_parameter_grads,
            )
            d_normalized += _linear_backward(
                d_v_projection,
                layer_cache["normalized"],
                layer["Wv"],
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

    def _local_forward(self, q, k, v, document_ids):
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
                caches.append(
                    {
                        "row": row,
                        "query_start": query_start,
                        "query_stop": query_stop,
                        "key_start": key_start,
                        "key_stop": key_stop,
                        "probabilities": probabilities,
                    }
                )
        return output, caches

    def _local_backward(self, d_output, q, k, v, caches):
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
    ):
        batch, seq_len = q.shape[:2]
        summary_count = summary_k.shape[1]
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
                caches.append(
                    {
                        "row": row,
                        "query_start": query_start,
                        "query_stop": query_stop,
                        "probabilities": probabilities,
                    }
                )
        return output, available, caches

    def _global_backward(self, d_output, q, summary_k, summary_v, caches):
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
        q_pre = (x_2d @ self.Wq.data).reshape(
            batch, seq_len, self.n_q_heads, self.d_head
        )
        k_pre = (x_2d @ self.Wk.data).reshape(
            batch, seq_len, self.n_kv_heads, self.d_head
        )
        v = (x_2d @ self.Wv.data).reshape(
            batch, seq_len, self.n_kv_heads, self.d_head
        )
        q, q_rope_cache = rope_forward(q_pre, self.rope_base, positions)
        k, k_rope_cache = rope_forward(k_pre, self.rope_base, positions)
        local_context, local_cache = self._local_forward(
            q, k, v, metadata["document_ids"]
        )

        summaries, compressor_cache = self._compress_forward(x, metadata)
        summary_shape = summaries.shape[:2]
        summary_k_pre = _linear_forward(summaries, self.W_global_k).reshape(
            summary_shape + (self.n_kv_heads, self.d_head)
        )
        summary_v = _linear_forward(summaries, self.W_global_v).reshape(
            summary_shape + (self.n_kv_heads, self.d_head)
        )
        summary_k, summary_rope_cache = rope_forward(
            summary_k_pre, self.rope_base, metadata["block_end_positions"]
        )
        global_context, global_available, global_cache = self._global_forward(
            q,
            summary_k,
            summary_v,
            metadata["document_ids"],
            positions,
            metadata["block_valid"],
            metadata["block_document_ids"],
            metadata["block_end_positions"],
        )
        gate_logits = _linear_forward(x, self.W_gate)
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
        d_merged = (dy_2d @ _work_array(self.Wo.data).T) * scale
        d_context = d_merged.reshape(
            x.shape[0], x.shape[1], self.n_q_heads, self.d_head
        )

        local_context = cache["local_context"]
        global_context = cache["global_context"]
        effective_gate = cache["effective_gate"]
        d_local = d_context * (1.0 - effective_gate[..., None])
        d_global = d_context * effective_gate[..., None]
        d_effective_gate = xp.sum(
            d_context * (global_context - local_context), axis=-1
        )
        gate = cache["gate"]
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

        dq_local, dk, dv = self._local_backward(
            d_local, cache["q"], cache["k"], cache["v"], cache["local_cache"]
        )
        dq_global, d_summary_k, d_summary_v = self._global_backward(
            d_global,
            cache["q"],
            cache["summary_k"],
            cache["summary_v"],
            cache["global_cache"],
        )
        dq = dq_local + dq_global
        if shared_context_kv_grad is not None:
            if len(shared_context_kv_grad) != 2:
                raise ValueError("shared_context_kv_grad must be a (dk,dv) pair")
            dk += shared_context_kv_grad[0]
            dv += shared_context_kv_grad[1]

        d_summary_k_pre = rope_backward(
            d_summary_k, cache["summary_rope_cache"]
        )
        d_summaries = _linear_backward(
            d_summary_k_pre.reshape(
                cache["summaries"].shape[:2] + (self.n_kv_heads * self.d_head,)
            ),
            cache["summaries"],
            self.W_global_k,
            accumulate_parameter_grads,
        )
        d_summaries += _linear_backward(
            d_summary_v.reshape(
                cache["summaries"].shape[:2] + (self.n_kv_heads * self.d_head,)
            ),
            cache["summaries"],
            self.W_global_v,
            accumulate_parameter_grads,
        )
        dx_compressor = self._compress_backward(
            d_summaries,
            cache["compressor_cache"],
            accumulate_parameter_grads,
        )

        dq_pre = rope_backward(dq, cache["q_rope_cache"])
        dk_pre = rope_backward(dk, cache["k_rope_cache"])
        q_width = self.n_q_heads * self.d_head
        kv_width = self.n_kv_heads * self.d_head
        dq_flat = _work_array(dq_pre.reshape(-1, q_width))
        dk_flat = _work_array(dk_pre.reshape(-1, kv_width))
        dv_flat = _work_array(dv.reshape(-1, kv_width))
        x_flat = _work_array(x.reshape(-1, self.d_model))
        if accumulate_parameter_grads:
            _add_parameter_grad(self.Wq, x_flat.T @ dq_flat)
            _add_parameter_grad(self.Wk, x_flat.T @ dk_flat)
            _add_parameter_grad(self.Wv, x_flat.T @ dv_flat)
        dx_flat = (
            dq_flat @ _work_array(self.Wq.data).T
            + dk_flat @ _work_array(self.Wk.data).T
            + dv_flat @ _work_array(self.Wv.data).T
        )
        dx = dx_flat.reshape(x.shape) + dx_gate + dx_compressor
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
