"""Inference-only grouped-query attention with a preallocated KV cache."""

import numpy as _np

from mini_llm.backend import xp, BACKEND_NAME
from mini_llm.ops.rope import get_rope_cos_sin


class GQAAttentionInference:
    """Grouped-query attention for autoregressive inference with KV cache.

    K/V remain stored only for ``n_kv_heads``.  Query heads are viewed as
    ``[n_kv_heads, group_size]`` and batched matmul broadcasts shared K/V
    heads across each group, avoiding ``xp.repeat`` of the active KV cache.
    """

    def __init__(
        self, d_model, n_q_heads, n_kv_heads, d_head,
        rope_base=10_000.0, dtype="float32", max_context=None
    ):
        if n_q_heads * d_head != d_model:
            raise ValueError("n_q_heads * d_head must equal d_model.")
        if n_q_heads % n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads.")

        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.group_size = n_q_heads // n_kv_heads
        self.scale = 1.0 / (d_head ** 0.5)
        self.rope_base = float(rope_base)
        self.dtype = dtype

        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.Wo = None

        self._rope_capacity = 0
        self._rope_cos = None
        self._rope_sin = None
        if max_context is not None:
            self.ensure_rope_capacity(max_context)

    def set_weights(self, Wq, Wk, Wv, Wo):
        """Set attention weights (shared with the training model)."""
        self.Wq = xp.asarray(Wq) if BACKEND_NAME == "cupy" else Wq
        self.Wk = xp.asarray(Wk) if BACKEND_NAME == "cupy" else Wk
        self.Wv = xp.asarray(Wv) if BACKEND_NAME == "cupy" else Wv
        self.Wo = xp.asarray(Wo) if BACKEND_NAME == "cupy" else Wo

    def ensure_rope_capacity(self, length):
        """Ensure absolute-position RoPE tables exist through ``length``."""
        length = int(length)
        if length <= self._rope_capacity:
            return
        self._rope_cos, self._rope_sin = get_rope_cos_sin(
            length, self.d_head, self.rope_base, self.dtype
        )
        self._rope_capacity = length

    def _weight_data(self):
        if BACKEND_NAME == "cupy":
            return self.Wq, self.Wk, self.Wv, self.Wo
        return tuple(
            w.data if hasattr(w, "data") else w
            for w in (self.Wq, self.Wk, self.Wv, self.Wo)
        )

    def _group_queries(self, q):
        """[B,T,Hq,D] -> [B,Hkv,G,T,D]."""
        b, t, _, d = q.shape
        return q.reshape(
            b, t, self.n_kv_heads, self.group_size, d
        ).transpose(0, 2, 3, 1, 4)

    def _ungroup_queries(self, q_grouped):
        """[B,Hkv,G,T,D] -> [B,T,Hq,D]."""
        b, _, _, t, d = q_grouped.shape
        return q_grouped.transpose(0, 3, 1, 2, 4).reshape(
            b, t, self.n_q_heads, d
        )

    def _apply_rope_absolute(self, x, start_pos):
        """Apply cached RoPE for contiguous positions beginning at start_pos."""
        _, t_new, _, d = x.shape
        end_pos = int(start_pos) + t_new
        self.ensure_rope_capacity(end_pos)

        cos = self._rope_cos[:, int(start_pos):end_pos, :, :]
        sin = self._rope_sin[:, int(start_pos):end_pos, :, :]

        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        y = xp.empty_like(x)
        y[..., 0::2] = x0 * cos - x1 * sin
        y[..., 1::2] = x0 * sin + x1 * cos
        return y

    def _attention(self, q, k_native, v_native, causal_mask=None):
        """Attend with native Hkv K/V tensors; never physically repeat them."""
        b, t_query, _, _ = q.shape
        t_key = k_native.shape[2]

        q_grouped = self._group_queries(q)  # [B,Hkv,G,Tq,D]
        q_scaled = q_grouped * self.scale
        scores_grouped = xp.matmul(
            q_scaled,
            k_native[:, :, xp.newaxis, :, :].swapaxes(-2, -1),
        )                                  # [B,Hkv,G,Tq,Tk]
        scores = scores_grouped.reshape(
            b, self.n_q_heads, t_query, t_key
        )

        if causal_mask is not None:
            scores = xp.where(
                causal_mask[None, None, :, :], scores, -xp.inf
            )

        scores_f32 = scores.astype(xp.float32)
        scores_f32 -= xp.max(scores_f32, axis=-1, keepdims=True)
        probs = xp.exp(scores_f32)
        probs /= xp.sum(probs, axis=-1, keepdims=True)
        probs = probs.astype(self.dtype, copy=False)

        probs_grouped = probs.reshape(
            b, self.n_kv_heads, self.group_size, t_query, t_key
        )
        context_grouped = xp.matmul(
            probs_grouped,
            v_native[:, :, xp.newaxis, :, :],
        )
        return self._ungroup_queries(context_grouped)

    def prefill(self, x, k_cache, v_cache, start_pos, return_all=False):
        """Prefill the cache for a prompt and run causal attention."""
        b, t_prompt, _ = x.shape
        cache_capacity = k_cache.shape[2]
        end_pos = int(start_pos) + t_prompt
        if end_pos > cache_capacity:
            raise ValueError(
                f"Prompt length {t_prompt} at position {start_pos} "
                f"exceeds cache capacity {cache_capacity}."
            )

        if BACKEND_NAME == "cupy" and isinstance(x, _np.ndarray):
            x = xp.asarray(x)

        Wq, Wk, Wv, Wo = self._weight_data()
        q_pre = (x @ Wq).reshape(
            b, t_prompt, self.n_q_heads, self.d_head
        )
        k_pre = (x @ Wk).reshape(
            b, t_prompt, self.n_kv_heads, self.d_head
        )
        v = (x @ Wv).reshape(
            b, t_prompt, self.n_kv_heads, self.d_head
        )

        q = self._apply_rope_absolute(q_pre, start_pos)
        k = self._apply_rope_absolute(k_pre, start_pos)

        k_cache[:, :, int(start_pos):end_pos, :] = k.transpose(0, 2, 1, 3)
        v_cache[:, :, int(start_pos):end_pos, :] = v.transpose(0, 2, 1, 3)

        # Native cache views: [B,Hkv,Tk,D].
        k_all = k_cache[:, :, :end_pos, :]
        v_all = v_cache[:, :, :end_pos, :]

        # Query absolute positions are contiguous; build only the small boolean
        # mask needed for prefill. decode_one needs no causal mask.
        q_positions = xp.arange(
            int(start_pos), end_pos, dtype=xp.int32
        )[:, None]
        k_positions = xp.arange(end_pos, dtype=xp.int32)[None, :]
        causal_mask = k_positions <= q_positions

        context = self._attention(q, k_all, v_all, causal_mask=causal_mask)
        merged = context.reshape(
            b, t_prompt, self.n_q_heads * self.d_head
        )
        y = merged @ Wo
        return y if return_all else y[:, -1, :]

    def decode_one(self, x, k_cache, v_cache, start_pos):
        """Decode exactly one token using cached native-Hkv K/V tensors."""
        b, t_new, _ = x.shape
        if t_new != 1:
            raise AssertionError("decode_one processes exactly one token")

        if BACKEND_NAME == "cupy" and isinstance(x, _np.ndarray):
            x = xp.asarray(x)

        Wq, Wk, Wv, Wo = self._weight_data()
        q_pre = (x @ Wq).reshape(
            b, 1, self.n_q_heads, self.d_head
        )
        k_pre = (x @ Wk).reshape(
            b, 1, self.n_kv_heads, self.d_head
        )
        v = (x @ Wv).reshape(
            b, 1, self.n_kv_heads, self.d_head
        )

        q = self._apply_rope_absolute(q_pre, start_pos)
        k = self._apply_rope_absolute(k_pre, start_pos)

        end_pos = int(start_pos) + 1
        k_cache[:, :, int(start_pos):end_pos, :] = k.transpose(0, 2, 1, 3)
        v_cache[:, :, int(start_pos):end_pos, :] = v.transpose(0, 2, 1, 3)

        context = self._attention(
            q,
            k_cache[:, :, :end_pos, :],
            v_cache[:, :, :end_pos, :],
            causal_mask=None,
        )
        merged = context.reshape(b, 1, self.n_q_heads * self.d_head)
        y = merged @ Wo
        return y[:, -1, :]
