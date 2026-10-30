"""Incremental dual-resolution attention for native NumPy/CuPy inference.

The operator keeps exact token K/V only for a bounded local ring.  Completed
same-document blocks are compressed once and their projected K/V are retained
in a second bounded ring.  Compression happens after the completing token has
attended, so a block summary first becomes visible to the following token.
"""

import math

from mini_llm.backend import (
    xp, BACKEND_NAME, is_bfloat16_dtype, is_low_precision_dtype,
)
from mini_llm.ops.rope import get_rope_cos_sin
from mini_llm.ops.silu import sigmoid_stable


def _data(value):
    """Unwrap this repository's Parameter/Linear objects without copying."""
    if value is None:
        return None
    if hasattr(value, "data"):
        value = value.data
    elif hasattr(value, "W"):
        value = value.W
        if hasattr(value, "data"):
            value = value.data
    if BACKEND_NAME == "cupy" and not hasattr(value, "__cuda_array_interface__"):
        return xp.asarray(value)
    return value


def _first_attr(obj, names):
    if obj is None:
        return None
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def _work_array(value):
    return (
        value.astype(xp.float32, copy=False)
        if is_low_precision_dtype(value.dtype) else value
    )


def _softmax(scores, axis=-1):
    scores = _work_array(scores)
    scores = scores - xp.max(scores, axis=axis, keepdims=True)
    probs = xp.exp(scores)
    return probs / xp.sum(probs, axis=axis, keepdims=True)


def _sigmoid(value):
    work = _work_array(value)
    magnitude = xp.exp(-xp.abs(work))
    return xp.where(
        work >= 0, 1.0 / (1.0 + magnitude), magnitude / (1.0 + magnitude)
    )


class DualResolutionAttentionInference:
    """One-token dual-resolution GQA with per-row bounded rings."""

    def __init__(
        self, d_model, n_q_heads, n_kv_heads, d_head, *, dual_config,
        rope_base=10_000.0, dtype="float32", rms_eps=1e-6,
        rope_layout="interleaved",
    ):
        if n_q_heads * d_head != d_model:
            raise ValueError("n_q_heads * d_head must equal d_model")
        if n_q_heads % n_kv_heads:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")
        if rope_layout not in {"interleaved", "half"}:
            raise ValueError("rope_layout must be interleaved or half")
        self.d_model = int(d_model)
        self.n_q_heads = int(n_q_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.d_head = int(d_head)
        self.group_size = self.n_q_heads // self.n_kv_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.rope_base = float(rope_base)
        self.dtype = dtype
        self.rms_eps = float(rms_eps)
        self.rope_layout = rope_layout
        self.block_size = int(dual_config.block_size)
        self.local_window = int(dual_config.local_window)
        self.max_blocks = int(dual_config.max_blocks)
        self.matrix_pool = (
            getattr(dual_config, "compression_mode", "transformer") == "matrix_pool"
        )
        self.long_local_window = int(
            getattr(dual_config, "long_local_window", self.local_window)
        )
        self.long_local_heads = int(getattr(dual_config, "long_local_heads", 0))
        self.sparse_stride = int(getattr(dual_config, "sparse_retrieval_stride", 0))
        self.sparse_head = int(getattr(dual_config, "sparse_retrieval_head", 1))
        self.sparse_phase = int(getattr(dual_config, "sparse_retrieval_phase", -1))
        self.sparse_enabled = self.sparse_stride > 0 and self.sparse_phase >= 0
        self.summaries_per_block = int(
            getattr(dual_config, "summaries_per_block", 1)
        )
        self.global_heads = int(getattr(dual_config, "global_heads", n_q_heads))
        self.global_head_dim = int(getattr(dual_config, "global_head_dim", d_head))
        self.global_enabled = bool(getattr(dual_config, "global_enabled", True))
        if not 1 <= self.block_size <= 128:
            raise ValueError("dual-resolution block_size must be in [1, 128]")
        if not 1 <= self.local_window <= 512:
            raise ValueError("dual-resolution local_window must be in [1, 512]")
        if not 1 <= self.max_blocks <= 512:
            raise ValueError("dual-resolution max_blocks must be in [1, 512]")

        self.Wq = self.Wk = self.Wv = self.Wo = None
        self.Wqkv = None
        self.W_pool_score = self.pool_position = None
        self.W_global_q = self.W_global_o = None
        self.Wg = self.bg = self.Wdown = None
        self.W_global_k = self.W_global_v = self.W_gate = None
        self.W_compress_in = self.W_compress_up = None
        self.compression_layers = None
        self.compression_heads = self.compression_head_dim = None
        self.compression_scale = None
        self.pool_norm_gamma = self.pool_query = None
        self.W_pool_k = self.W_pool_v = self.W_pool_o = None
        self.training_attention = None
        self.compressor = None
        self._compress_callable = None
        self._compress_norm_gamma = None
        self._compress_norm_eps = self.rms_eps
        self._compress_attention = None
        self._pool = None
        self._up = None
        self._rope_capacity = 0
        self._rope_cos = self._rope_sin = None

    def ensure_rope_capacity(self, length):
        length = int(length)
        if length <= self._rope_capacity:
            return
        self._rope_cos, self._rope_sin = get_rope_cos_sin(
            length, self.d_head, self.rope_base, self.dtype
        )
        self._rope_capacity = length

    def set_weights(self, training_attention):
        """Attach a dual-resolution training attention object.

        The base projections have a fixed public contract.  Compressor nesting
        is deliberately discovered by capability because the training operator
        is developed independently: an explicit ``compress_block_inference``
        method is preferred, while common norm/self-attention/pool/up layouts
        are supported by the native fallback below.
        """
        self.training_attention = training_attention
        required = ("Wq", "Wk", "Wv", "Wo")
        missing = [name for name in required if not hasattr(training_attention, name)]
        if missing:
            raise ValueError(f"dual-resolution attention is missing {missing}")
        self.Wq = _data(training_attention.Wq)
        self.Wk = _data(training_attention.Wk)
        self.Wv = _data(training_attention.Wv)
        self.Wo = _data(training_attention.Wo)
        self.Wqkv = xp.concatenate((self.Wq, self.Wk, self.Wv), axis=1)
        if self.matrix_pool:
            self.W_pool_score = _data(training_attention.W_pool_score)
            self.pool_position = _data(getattr(training_attention, "pool_position", None))
            self.W_global_q = _data(getattr(training_attention, "W_global_q", None))
            self.W_global_k = _data(getattr(training_attention, "W_global_k", None))
            self.W_global_v = _data(getattr(training_attention, "W_global_v", None))
            self.W_global_o = _data(getattr(training_attention, "W_global_o", None))
            if self.global_enabled and any(weight is None for weight in (
                self.W_global_q, self.W_global_k,
                self.W_global_v, self.W_global_o,
            )):
                raise ValueError("matrix-pool global attention weights are incomplete")
            return
        self.Wg = _data(getattr(training_attention, "Wg", None))
        self.bg = _data(getattr(training_attention, "bg", None))
        self.Wdown = _data(getattr(training_attention, "Wdown", None))

        # Native layout of mini_llm.ops.dual_resolution_attention.  Keep the
        # capability-based fallback below for independently developed variants.
        self.W_global_k = _data(getattr(training_attention, "W_global_k", None))
        self.W_global_v = _data(getattr(training_attention, "W_global_v", None))
        self.W_gate = _data(getattr(training_attention, "W_gate", None))
        self.W_compress_in = _data(getattr(training_attention, "W_compress_in", None))
        self.W_compress_up = _data(getattr(training_attention, "W_compress_up", None))
        raw_layers = getattr(training_attention, "compression_layers", None)
        if raw_layers is not None:
            self.compression_layers = [
                {
                    "norm": _data(layer["norm"]),
                    "Wq": _data(layer["Wq"]),
                    "Wk": _data(layer["Wk"]),
                    "Wv": _data(layer["Wv"]),
                    "Wo": _data(layer["Wo"]),
                }
                for layer in raw_layers
            ]
            self.compression_heads = int(training_attention.compression_heads)
            self.compression_head_dim = int(training_attention.compression_head_dim)
            self.compression_scale = float(training_attention.compression_scale)
            self.pool_norm_gamma = _data(training_attention.pool_norm)
            self.pool_query = _data(training_attention.pool_query)
            self.W_pool_k = _data(training_attention.W_pool_k)
            self.W_pool_v = _data(training_attention.W_pool_v)
            self.W_pool_o = _data(training_attention.W_pool_o)

        self.compressor = _first_attr(
            training_attention,
            ("compressor", "compression", "block_compressor", "summary_compressor"),
        )
        for owner in (training_attention, self.compressor):
            method = _first_attr(
                owner,
                ("compress_block_inference", "inference_compress_block", "inference_forward"),
            )
            if callable(method):
                self._compress_callable = method
                break

        norm = _first_attr(
            self.compressor,
            ("norm", "input_norm", "compressor_norm", "compression_norm"),
        )
        if norm is None:
            norm = _first_attr(
                training_attention, ("compressor_norm", "compression_norm")
            )
        self._compress_norm_gamma = _data(_first_attr(norm, ("gamma", "weight")))
        if norm is not None:
            self._compress_norm_eps = float(getattr(norm, "eps", self.rms_eps))
        self._compress_attention = _first_attr(
            self.compressor,
            ("self_attention", "attention", "attn", "compression_attention"),
        )
        if self._compress_attention is None:
            self._compress_attention = _first_attr(
                training_attention,
                ("compression_self_attention", "compressor_attention"),
            )
        self._pool = _first_attr(
            self.compressor, ("pool", "pooling", "learned_pool", "pooler")
        )
        if self._pool is None:
            self._pool = _first_attr(
                training_attention, ("compression_pool", "learned_pool", "pooler")
            )
        self._up = _first_attr(
            self.compressor, ("up", "up_projection", "Wup", "W_up")
        )
        if self._up is None:
            self._up = _first_attr(
                training_attention, ("compression_up", "Wup", "W_up")
            )

    def _apply_rope(self, x, position):
        position = int(position)
        self.ensure_rope_capacity(position + 1)
        cos = self._rope_cos[:, position:position + 1]
        sin = self._rope_sin[:, position:position + 1]
        if self.rope_layout == "half":
            half = self.d_head // 2
            x0 = x[..., :half].copy()
            x1 = x[..., half:]
            x[..., :half] = x0 * cos - x1 * sin
            x[..., half:] = x0 * sin + x1 * cos
        else:
            x0 = x[..., 0::2].copy()
            x1 = x[..., 1::2]
            x[..., 0::2] = x0 * cos - x1 * sin
            x[..., 1::2] = x0 * sin + x1 * cos
        return x

    @staticmethod
    def _ordered_ring(cache_row, start, count):
        capacity = int(cache_row.shape[1])
        if count == 0:
            return cache_row[:, :0, :]
        indices = (int(start) + xp.arange(int(count), dtype=xp.int64)) % capacity
        return cache_row[:, indices, :]

    @staticmethod
    def _insert_ring(cache_row, value, start, count):
        capacity = int(cache_row.shape[1])
        if count < capacity:
            slot = (int(start) + int(count)) % capacity
            count += 1
        else:
            slot = int(start)
            start = (int(start) + 1) % capacity
        cache_row[:, slot, :] = value
        return start, count

    def _single_row_attention(self, q, keys, values):
        # q [1,Hq,D], keys/values [Hkv,T,D]
        t_key = int(keys.shape[1])
        qg = q.reshape(self.n_kv_heads, self.group_size, self.d_head)
        q_compute = _work_array(qg)
        k_compute = _work_array(keys)
        scores = xp.matmul(q_compute * self.scale, k_compute.swapaxes(-2, -1))
        probs = _softmax(scores, axis=-1)
        v_compute = _work_array(values)
        context = xp.matmul(probs, v_compute).reshape(1, self.n_q_heads, self.d_head)
        return context.astype(q.dtype, copy=False) if is_bfloat16_dtype(q.dtype) else context

    def _generic_self_attention(self, x, attention):
        wq = _data(_first_attr(attention, ("Wq", "wq", "query")))
        wk = _data(_first_attr(attention, ("Wk", "wk", "key")))
        wv = _data(_first_attr(attention, ("Wv", "wv", "value")))
        wo = _data(_first_attr(attention, ("Wo", "wo", "output")))
        if any(weight is None for weight in (wq, wk, wv, wo)):
            raise RuntimeError(
                "compressor self-attention needs Wq/Wk/Wv/Wo or an explicit "
                "compress_block_inference method"
            )
        b, t, width = x.shape
        q_width = int(wq.shape[1])
        k_width = int(wk.shape[1])
        heads = int(getattr(attention, "n_q_heads", 1))
        kv_heads = int(getattr(attention, "n_kv_heads", heads))
        if q_width % heads or k_width % kv_heads:
            raise ValueError("invalid compressor attention projection shapes")
        head_dim = q_width // heads
        if k_width // kv_heads != head_dim or heads % kv_heads:
            raise ValueError("compressor attention has incompatible GQA dimensions")
        q = (x.reshape(-1, width) @ wq).reshape(b, t, heads, head_dim)
        k = (x.reshape(-1, width) @ wk).reshape(b, t, kv_heads, head_dim)
        v = (x.reshape(-1, width) @ wv).reshape(b, t, kv_heads, head_dim)
        group = heads // kv_heads
        qg = q.transpose(0, 2, 1, 3).reshape(b, kv_heads, group, t, head_dim)
        kh = k.transpose(0, 2, 1, 3)
        vh = v.transpose(0, 2, 1, 3)
        scores = xp.matmul(
            qg.astype(xp.float32, copy=False) / math.sqrt(head_dim),
            kh[:, :, None].astype(xp.float32, copy=False).swapaxes(-2, -1),
        )
        probs = _softmax(scores, axis=-1)
        context = xp.matmul(probs, vh[:, :, None].astype(xp.float32, copy=False))
        merged = context.reshape(b, heads, t, head_dim).transpose(0, 2, 1, 3)
        merged = merged.reshape(b * t, q_width).astype(x.dtype, copy=False)
        return (merged @ wo).reshape(b, t, -1)

    @staticmethod
    def _rms_norm(value, gamma, eps=1e-6):
        work = _work_array(value)
        inverse = 1.0 / xp.sqrt(
            xp.mean(work * work, axis=-1, keepdims=True) + float(eps)
        )
        return (work * inverse * _work_array(gamma)).astype(value.dtype, copy=False)

    def _compress_repository_block(self, block):
        """Exact no-backward form of DualResolutionAttention._compress_forward."""
        hidden = block @ self.W_compress_in
        batch, tokens, width = hidden.shape
        heads = int(self.compression_heads)
        head_dim = int(self.compression_head_dim)
        for layer in self.compression_layers:
            normalized = self._rms_norm(hidden, layer["norm"])
            q = (normalized @ layer["Wq"]).reshape(
                batch, tokens, heads, head_dim
            ).transpose(0, 2, 1, 3)
            k = (normalized @ layer["Wk"]).reshape(
                batch, tokens, heads, head_dim
            ).transpose(0, 2, 1, 3)
            v = (normalized @ layer["Wv"]).reshape(
                batch, tokens, heads, head_dim
            ).transpose(0, 2, 1, 3)
            scores = xp.matmul(
                _work_array(q) * self.compression_scale,
                _work_array(k).swapaxes(-2, -1),
            )
            probabilities = _softmax(scores, axis=-1)
            context = xp.matmul(probabilities, _work_array(v))
            merged = context.transpose(0, 2, 1, 3).reshape(batch, tokens, width)
            attention_output = merged.astype(hidden.dtype, copy=False) @ layer["Wo"]
            hidden = hidden + attention_output

        pool_input = self._rms_norm(hidden, self.pool_norm_gamma)
        pool_k = (pool_input @ self.W_pool_k).reshape(
            batch, tokens, heads, head_dim
        ).transpose(0, 2, 1, 3)
        pool_v = (pool_input @ self.W_pool_v).reshape(
            batch, tokens, heads, head_dim
        ).transpose(0, 2, 1, 3)
        pool_scores = xp.sum(
            _work_array(pool_k)
            * _work_array(self.pool_query)[None, :, None, :],
            axis=-1,
        ) * self.compression_scale
        pool_probabilities = _softmax(pool_scores, axis=-1)
        pool_context = xp.sum(
            pool_probabilities[..., None] * _work_array(pool_v),
            axis=2,
        )
        pool_merged = pool_context.reshape(batch, width).astype(hidden.dtype, copy=False)
        pooled = pool_merged @ self.W_pool_o
        return pooled @ self.W_compress_up

    def _native_compress_block(self, block):
        """Run a discovered gated-down/bidirectional/pool/up compressor."""
        if self.compression_layers is not None:
            return self._compress_repository_block(block)
        z = block
        if self.Wdown is not None and int(self.Wdown.shape[0]) == int(z.shape[-1]):
            down = z @ self.Wdown
            if (
                self.Wg is not None
                and int(self.Wg.shape[0]) == int(z.shape[-1])
                and int(self.Wg.shape[1]) == int(down.shape[-1])
            ):
                gate = z @ self.Wg
                if self.bg is not None:
                    gate = gate + self.bg
                down = down * sigmoid_stable(gate)
            z = down
        if self._compress_norm_gamma is not None:
            work = z.astype(xp.float32, copy=False)
            inv = 1.0 / xp.sqrt(
                xp.mean(work * work, axis=-1, keepdims=True) + self._compress_norm_eps
            )
            z = (work * inv).astype(z.dtype, copy=False) * self._compress_norm_gamma
        if self._compress_attention is None:
            raise RuntimeError(
                "dual-resolution compressor needs bidirectional self-attention or "
                "compress_block_inference"
            )
        attended = self._generic_self_attention(z, self._compress_attention)
        if attended.shape == z.shape:
            z = z + attended
        else:
            z = attended

        query = _data(_first_attr(
            self._pool,
            ("query", "pool_query", "score", "score_vector", "weight", "W"),
        ))
        if query is None:
            raise RuntimeError(
                "dual-resolution compressor needs a learned pooling query or "
                "compress_block_inference"
            )
        query = query.reshape(-1, int(z.shape[-1]))
        scores = xp.matmul(z.astype(xp.float32, copy=False), query.T.astype(xp.float32, copy=False))
        scores = xp.mean(scores, axis=-1) / math.sqrt(int(z.shape[-1]))
        weights = _softmax(scores, axis=1)
        pooled = xp.sum(weights[..., None] * z.astype(xp.float32, copy=False), axis=1)
        up = _data(self._up)
        if up is not None:
            pooled = pooled.astype(up.dtype, copy=False) @ up
        if int(pooled.shape[-1]) != self.d_model:
            raise ValueError("compressed summary must project back to d_model")
        return pooled.astype(block.dtype, copy=False)

    def _compress_block(self, block):
        if self.compression_layers is not None:
            return self._compress_repository_block(block)
        if self._compress_callable is not None:
            summary = self._compress_callable(block)
            if isinstance(summary, tuple):
                summary = summary[0]
            if summary.ndim == 3 and int(summary.shape[1]) == 1:
                summary = summary[:, 0, :]
            if summary.ndim != 2 or tuple(summary.shape) != (int(block.shape[0]), self.d_model):
                raise ValueError("compress_block_inference must return [B,d_model]")
            return summary
        return self._native_compress_block(block)

    def _matrix_local_attention(self, q, keys, values, *, sparse_k=None, sparse_v=None):
        """Apply the configured window independently to each local query head."""
        contexts = xp.empty_like(q)
        available = int(keys.shape[1])
        for head in range(self.n_q_heads):
            if self.sparse_enabled and head == self.sparse_head:
                if sparse_k is None or int(sparse_k.shape[1]) == 0:
                    contexts[:, :, head, :] = 0
                else:
                    kvh = head // self.group_size
                    # The sparse buffers hold only the one selected KV group.
                    qh = _work_array(q[:, :, head, :])
                    kh = _work_array(sparse_k[:, :, 0, :])
                    vh = _work_array(sparse_v[:, :, 0, :])
                    logits = (qh * self.scale) @ kh.transpose(0, 2, 1)
                    contexts[:, :, head, :] = (
                        _softmax(logits, axis=-1) @ vh
                    ).astype(q.dtype, copy=False)
                continue
            window = (
                self.long_local_window
                if head < self.long_local_heads else self.local_window
            )
            count = min(available, window)
            kv_head = head // self.group_size
            qh = _work_array(q[:, :, head, :])
            kh = _work_array(keys[kv_head, -count:, :])
            vh = _work_array(values[kv_head, -count:, :])
            scores = (qh * self.scale) @ kh.T
            probabilities = _softmax(scores, axis=-1)
            contexts[:, :, head, :] = (probabilities @ vh).astype(
                q.dtype, copy=False
            )
        return contexts

    def _matrix_global_attention(self, q, keys, values):
        qh = _work_array(q[0]).transpose(1, 0, 2)
        kh = _work_array(keys)
        vh = _work_array(values)
        scores = (qh * self.scale) @ kh.swapaxes(-2, -1)
        probabilities = _softmax(scores, axis=-1)
        context = probabilities @ vh
        return context.transpose(1, 0, 2)[None].astype(q.dtype, copy=False)

    def _matrix_pool_block(self, block):
        work = _work_array(block)
        scores = _work_array(
            block.reshape(-1, self.d_model) @ self.W_pool_score
        ).reshape(int(block.shape[0]), int(block.shape[1]), self.summaries_per_block)
        if self.pool_position is not None:
            scores = scores + _work_array(self.pool_position)[None, :, :]
        probabilities = _softmax(scores, axis=1)
        return probabilities.swapaxes(1, 2) @ work

    def _matrix_decode_one(
        self, x, residual_input, local_k, local_v, summary_k, summary_v,
        incomplete_hidden, local_start, local_count, summary_start,
        summary_count, incomplete_count, document_positions,
        sparse_k=None, sparse_v=None, sparse_count=None, absolute_position=None,
        host_counters=None,
    ):
        if residual_input is None or residual_input.shape != x.shape:
            raise ValueError("matrix-pool inference requires residual_input [B,1,d_model]")
        batch = int(x.shape[0])
        qkv = (x[:, 0, :] @ self.Wqkv)[:, None, :]
        q_end = self.n_q_heads * self.d_head
        k_end = q_end + self.n_kv_heads * self.d_head
        q_pre = qkv[..., :q_end].reshape(batch, 1, self.n_q_heads, self.d_head)
        k_pre = qkv[..., q_end:k_end].reshape(batch, 1, self.n_kv_heads, self.d_head)
        values = qkv[..., k_end:].reshape(batch, 1, self.n_kv_heads, self.d_head)
        output = xp.empty((batch, 1, self.d_model), dtype=x.dtype)
        completed_rows = []

        for row in range(batch):
            counters = host_counters if batch == 1 and row == 0 else None
            position = (
                int(counters["document_position"])
                if counters is not None else int(document_positions[row].item())
            )
            q_row = self._apply_rope(q_pre[row:row + 1], position)
            k_row = self._apply_rope(k_pre[row:row + 1], position)[0, 0]
            v_row = values[row, 0]
            local_start_value = (
                int(counters["local_start"])
                if counters is not None else int(local_start[row].item())
            )
            local_count_value = (
                int(counters["local_count"])
                if counters is not None else int(local_count[row].item())
            )
            ls, lc = self._insert_ring(
                local_k[row], k_row, local_start_value, local_count_value,
            )
            if counters is not None:
                counters["local_start"], counters["local_count"] = ls, lc
            local_capacity = int(local_v[row].shape[1])
            local_v[row, :, (ls + lc - 1) % local_capacity, :] = v_row
            local_start[row], local_count[row] = ls, lc
            lk = self._ordered_ring(local_k[row], ls, lc)
            lv = self._ordered_ring(local_v[row], ls, lc)
            sampled_k = sampled_v = None
            if self.sparse_enabled:
                if sparse_k is None or sparse_v is None or sparse_count is None or absolute_position is None:
                    raise ValueError("sparse retrieval inference requires explicit KV cache")
                count = (
                    int(counters["sparse_count"])
                    if counters is not None else int(sparse_count[row].item())
                )
                if int(absolute_position) % self.sparse_stride == self.sparse_phase:
                    if count >= sparse_k.shape[2]:
                        raise ValueError("sparse retrieval KV capacity exceeded")
                    kvh = self.sparse_head // self.group_size
                    sparse_k[row, 0, count, :] = k_row[kvh]
                    sparse_v[row, 0, count, :] = v_row[kvh]
                    count += 1
                    sparse_count[row] = count
                if counters is not None:
                    counters["sparse_count"] = count
                sampled_k = sparse_k[row:row + 1, :, :count, :].transpose(0, 2, 1, 3)
                sampled_v = sparse_v[row:row + 1, :, :count, :].transpose(0, 2, 1, 3)
            local_context = self._matrix_local_attention(
                q_row, lk, lv, sparse_k=sampled_k, sparse_v=sampled_v
            )
            local_delta = (
                local_context.reshape(1, self.d_model).astype(x.dtype, copy=False)
                @ self.Wo
            ).reshape(1, 1, self.d_model)
            local_state = (
                _work_array(residual_input[row:row + 1])
                + _work_array(local_delta)
            )
            global_delta = xp.zeros_like(local_delta)
            ss = (
                int(counters["summary_start"])
                if counters is not None else int(summary_start[row].item())
            )
            sc = (
                int(counters["summary_count"])
                if counters is not None else int(summary_count[row].item())
            )
            if self.global_enabled and sc:
                state_compute = local_state.astype(x.dtype, copy=False)
                qg = (
                    state_compute.reshape(1, self.d_model) @ self.W_global_q
                ).reshape(1, 1, self.global_heads, self.global_head_dim)
                qg = self._apply_rope(qg, position)
                sk = self._ordered_ring(summary_k[row], ss, sc)
                sv = self._ordered_ring(summary_v[row], ss, sc)
                global_context = self._matrix_global_attention(qg, sk, sv)
                global_delta = (
                    global_context.reshape(1, -1).astype(x.dtype, copy=False)
                    @ self.W_global_o
                ).reshape(1, 1, self.d_model)
            output[row:row + 1] = (
                local_delta + global_delta.astype(local_delta.dtype, copy=False)
            ).astype(output.dtype, copy=False)

            if self.global_enabled:
                offset = (
                    int(counters["incomplete_count"])
                    if counters is not None else int(incomplete_count[row].item())
                )
                incomplete_hidden[row, offset, :] = local_state[0, 0].astype(
                    incomplete_hidden.dtype, copy=False
                )
                offset += 1
                incomplete_count[row] = offset
                if counters is not None:
                    counters["incomplete_count"] = offset
                if offset == self.block_size:
                    completed_rows.append((row, position))

        for row, position in completed_rows:
            block = incomplete_hidden[row:row + 1, :self.block_size, :]
            summaries = self._matrix_pool_block(block).astype(x.dtype, copy=False)
            summary_rows = summaries.reshape(-1, self.d_model)
            keys = (summary_rows @ self.W_global_k).reshape(
                1, self.summaries_per_block,
                self.global_heads, self.global_head_dim,
            )
            keys = self._apply_rope(keys, position)[0]
            values = (summary_rows @ self.W_global_v).reshape(
                self.summaries_per_block, self.global_heads, self.global_head_dim
            )
            counters = host_counters if batch == 1 and row == 0 else None
            ss = (
                int(counters["summary_start"])
                if counters is not None else int(summary_start[row].item())
            )
            sc = (
                int(counters["summary_count"])
                if counters is not None else int(summary_count[row].item())
            )
            capacity = int(summary_k[row].shape[1])
            for index in range(self.summaries_per_block):
                ss, sc = self._insert_ring(
                    summary_k[row], keys[index], ss, sc
                )
                slot = (ss + sc - 1) % capacity
                summary_v[row, :, slot, :] = values[index]
            summary_start[row], summary_count[row] = ss, sc
            incomplete_count[row] = 0
            if counters is not None:
                counters["summary_start"], counters["summary_count"] = ss, sc
                counters["incomplete_count"] = 0
        return output

    def decode_one(
        self, x, local_k, local_v, summary_k, summary_v, incomplete_hidden,
        local_start, local_count, summary_start, summary_count, incomplete_count,
        document_positions, residual_input=None,
        sparse_k=None, sparse_v=None, sparse_count=None, absolute_position=None,
        host_counters=None,
    ):
        """Consume one normalized token for every independent batch row."""
        if x.ndim != 3 or int(x.shape[1]) != 1 or int(x.shape[2]) != self.d_model:
            raise ValueError("dual-resolution decode expects [B,1,d_model]")
        if self.matrix_pool:
            return self._matrix_decode_one(
                x, residual_input, local_k, local_v, summary_k, summary_v,
                incomplete_hidden, local_start, local_count, summary_start,
                summary_count, incomplete_count, document_positions,
                sparse_k=sparse_k, sparse_v=sparse_v, sparse_count=sparse_count,
                absolute_position=absolute_position, host_counters=host_counters,
            )
        batch = int(x.shape[0])
        qkv = (x[:, 0, :] @ self.Wqkv)[:, None, :]
        q_end = self.n_q_heads * self.d_head
        k_end = q_end + self.n_kv_heads * self.d_head
        q_pre = qkv[..., :q_end].reshape(batch, 1, self.n_q_heads, self.d_head)
        k_pre = qkv[..., q_end:k_end].reshape(batch, 1, self.n_kv_heads, self.d_head)
        values = qkv[..., k_end:].reshape(batch, 1, self.n_kv_heads, self.d_head)
        output = xp.empty((batch, 1, self.d_model), dtype=x.dtype)
        completed_rows = []

        for row in range(batch):
            position = int(document_positions[row].item())
            q_row = self._apply_rope(q_pre[row:row + 1], position)
            k_row = self._apply_rope(k_pre[row:row + 1], position)[0, 0]
            v_row = values[row, 0]
            ls, lc = self._insert_ring(
                local_k[row], k_row, int(local_start[row].item()), int(local_count[row].item())
            )
            local_capacity = int(local_v[row].shape[1])
            local_v[row, :, (ls + lc - 1) % local_capacity, :] = v_row
            local_start[row], local_count[row] = ls, lc

            lk = self._ordered_ring(local_k[row], ls, lc)
            lv = self._ordered_ring(local_v[row], ls, lc)
            ss, sc = int(summary_start[row].item()), int(summary_count[row].item())
            local_context = self._single_row_attention(q_row, lk, lv)
            if sc and self.W_gate is not None:
                sk = self._ordered_ring(summary_k[row], ss, sc)
                sv = self._ordered_ring(summary_v[row], ss, sc)
                global_context = self._single_row_attention(q_row, sk, sv)
                gate = _sigmoid(x[row, 0] @ self.W_gate).reshape(
                    1, self.n_q_heads, 1
                )
                context = local_context + gate * (global_context - local_context)
            elif sc:
                sk = self._ordered_ring(summary_k[row], ss, sc)
                sv = self._ordered_ring(summary_v[row], ss, sc)
                context = self._single_row_attention(
                    q_row, xp.concatenate((sk, lk), axis=1),
                    xp.concatenate((sv, lv), axis=1),
                )
            else:
                context = local_context
            output[row:row + 1] = (
                context.reshape(1, 1, self.d_model).astype(x.dtype, copy=False)
                @ self.Wo
            ).astype(output.dtype, copy=False)

            offset = int(incomplete_count[row].item())
            incomplete_hidden[row, offset, :] = x[row, 0, :]
            offset += 1
            incomplete_count[row] = offset
            if offset == self.block_size:
                completed_rows.append((row, position))

        # Compression is intentionally after every row's current-token attention.
        for row, position in completed_rows:
            block = incomplete_hidden[row:row + 1, :self.block_size, :]
            summary = self._compress_block(block)
            summary_wk = self.W_global_k if self.W_global_k is not None else self.Wk
            summary_wv = self.W_global_v if self.W_global_v is not None else self.Wv
            key = (summary @ summary_wk).reshape(
                1, 1, self.n_kv_heads, self.d_head
            )
            key = self._apply_rope(key, position)[0, 0]
            value = (summary @ summary_wv).reshape(self.n_kv_heads, self.d_head)
            ss, sc = self._insert_ring(
                summary_k[row], key,
                int(summary_start[row].item()), int(summary_count[row].item()),
            )
            summary_v[row, :, (ss + sc - 1) % self.max_blocks, :] = value
            summary_start[row], summary_count[row] = ss, sc
            incomplete_count[row] = 0
        return output
