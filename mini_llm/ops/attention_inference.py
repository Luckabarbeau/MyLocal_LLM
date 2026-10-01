"""Inference-only grouped-query attention with a preallocated KV cache."""

import numpy as _np

from mini_llm.backend import xp, BACKEND_NAME, is_bfloat16_dtype, is_low_precision_dtype
from mini_llm.ops.rope import get_rope_cos_sin


_CAUSAL_ALLOW_MASK_CACHE = {}


def _get_causal_allow_mask(length):
    """Return cached lower-triangular allow mask [length,length]."""
    length = int(length)
    mask = _CAUSAL_ALLOW_MASK_CACHE.get(length)
    if mask is None:
        pos = xp.arange(length, dtype=xp.int32)
        mask = pos[None, :] <= pos[:, None]
        _CAUSAL_ALLOW_MASK_CACHE[length] = mask
    return mask


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
        self.Wqkv = None
        self._q_width = n_q_heads * d_head
        self._kv_width = n_kv_heads * d_head

        self._rope_capacity = 0
        self._rope_cos = None
        self._rope_sin = None
        if max_context is not None:
            self.ensure_rope_capacity(max_context)

    def set_weights(self, Wq, Wk, Wv, Wo):
        """Set weights and build one persistent inference QKV matrix."""
        self.Wq = xp.asarray(Wq) if BACKEND_NAME == "cupy" else Wq
        self.Wk = xp.asarray(Wk) if BACKEND_NAME == "cupy" else Wk
        self.Wv = xp.asarray(Wv) if BACKEND_NAME == "cupy" else Wv
        self.Wo = xp.asarray(Wo) if BACKEND_NAME == "cupy" else Wo

        # Inference repeatedly multiplies the same hidden state by Q, K and V.
        # Pack the weights once at model construction/load time so prefill and
        # especially decode issue one GEMM instead of three tiny GEMMs.
        self.Wqkv = xp.concatenate(
            (self.Wq, self.Wk, self.Wv), axis=1
        )

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

        # q_pre/k_pre are inference temporaries and are never reused after
        # RoPE. Rotate them in place to avoid allocating a second full tensor.
        # Preserve the even lanes because they are needed when writing odds.
        x0 = x[..., 0::2].copy()
        x1 = x[..., 1::2]
        x[..., 0::2] = x0 * cos - x1 * sin
        x[..., 1::2] = x0 * sin + x1 * cos
        return x

    def _attention(self, q, k_native, v_native, causal_mask=None):
        """Attend with native Hkv K/V tensors; never physically repeat them."""
        b, t_query, _, _ = q.shape
        t_key = k_native.shape[2]

        bf16_attention = is_bfloat16_dtype(q.dtype)

        if t_query == 1:
            # Decode fast path. CuPy's N-D BF16 matmul currently fails, so
            # promote only the batched attention products to FP32.
            q_decode = q[:, 0, :, :].reshape(
                b, self.n_kv_heads, self.group_size, self.d_head
            )
            q_scores = q_decode.astype(xp.float32, copy=False) if bf16_attention else q_decode
            k_scores = k_native.astype(xp.float32, copy=False) if bf16_attention else k_native
            scores_decode = xp.matmul(
                q_scores * self.scale,
                k_scores.swapaxes(-2, -1),
            )  # [B,Hkv,G,Tk]
            scores = scores_decode.reshape(
                b, self.n_q_heads, 1, t_key
            )
        else:
            q_grouped = self._group_queries(q)  # [B,Hkv,G,Tq,D]
            q_scores = q_grouped.astype(xp.float32, copy=False) if bf16_attention else q_grouped
            k_scores = k_native.astype(xp.float32, copy=False) if bf16_attention else k_native
            q_scaled = q_scores * self.scale
            scores_grouped = xp.matmul(
                q_scaled,
                k_scores[:, :, xp.newaxis, :, :].swapaxes(-2, -1),
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

        # Keep CuPy's unsupported N-D BF16 matmul out of P@V as well.
        # The result is cast back to the branch compute dtype before the large
        # output projection GEMM.
        if t_query == 1:
            probs_decode = probs.reshape(
                b, self.n_kv_heads, self.group_size, t_key
            )
            v_compute = v_native.astype(xp.float32, copy=False) if bf16_attention else v_native
            probs_compute = probs_decode if bf16_attention else probs_decode.astype(self.dtype, copy=False)
            context_decode = xp.matmul(
                probs_compute,
                v_compute,
            )  # [B,Hkv,G,D]
            context = context_decode.reshape(
                b, 1, self.n_q_heads, self.d_head
            )
            return context.astype(q.dtype, copy=False) if bf16_attention else context

        probs_grouped = probs.reshape(
            b, self.n_kv_heads, self.group_size, t_query, t_key
        )
        v_compute = v_native.astype(xp.float32, copy=False) if bf16_attention else v_native
        probs_compute = probs_grouped if bf16_attention else probs_grouped.astype(self.dtype, copy=False)
        context_grouped = xp.matmul(
            probs_compute,
            v_compute[:, :, xp.newaxis, :, :],
        )
        context = self._ungroup_queries(context_grouped)
        return context.astype(q.dtype, copy=False) if bf16_attention else context

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

        _, _, _, Wo = self._weight_data()
        qkv = (x.reshape(-1, self.d_model) @ self.Wqkv).reshape(
            b, t_prompt, -1
        )
        q_end = self._q_width
        k_end = q_end + self._kv_width
        q_pre = qkv[..., :q_end].reshape(
            b, t_prompt, self.n_q_heads, self.d_head
        )
        k_pre = qkv[..., q_end:k_end].reshape(
            b, t_prompt, self.n_kv_heads, self.d_head
        )
        v = qkv[..., k_end:].reshape(
            b, t_prompt, self.n_kv_heads, self.d_head
        )

        q = self._apply_rope_absolute(q_pre, start_pos)
        k = self._apply_rope_absolute(k_pre, start_pos)

        k_cache[:, :, int(start_pos):end_pos, :] = k.transpose(0, 2, 1, 3)
        v_cache[:, :, int(start_pos):end_pos, :] = v.transpose(0, 2, 1, 3)

        # Native cache views: [B,Hkv,Tk,D].
        k_all = k_cache[:, :, :end_pos, :]
        v_all = v_cache[:, :, :end_pos, :]

        # Reuse a shared absolute-position causal mask rather than rebuilding
        # arange/comparison tensors for every layer during prompt prefill.
        # Rows are the absolute query positions [start_pos:end_pos], columns
        # are keys [0:end_pos]. decode_one still needs no causal mask.
        causal_full = _get_causal_allow_mask(cache_capacity)
        causal_mask = causal_full[int(start_pos):end_pos, :end_pos]

        context = self._attention(q, k_all, v_all, causal_mask=causal_mask)
        merged = context.reshape(
            b, t_prompt, self.n_q_heads * self.d_head
        )
        y = (merged.reshape(-1, self.d_model) @ Wo).reshape(
            b, t_prompt, self.d_model
        )
        return y if return_all else y[:, -1, :]

    def decode_one(self, x, k_cache, v_cache, start_pos):
        """Decode exactly one token using cached native-Hkv K/V tensors."""
        b, t_new, _ = x.shape
        if t_new != 1:
            raise AssertionError("decode_one processes exactly one token")

        if BACKEND_NAME == "cupy" and isinstance(x, _np.ndarray):
            x = xp.asarray(x)

        _, _, _, Wo = self._weight_data()
        qkv = (x.reshape(-1, self.d_model) @ self.Wqkv).reshape(
            b, 1, -1
        )
        q_end = self._q_width
        k_end = q_end + self._kv_width
        q_pre = qkv[..., :q_end].reshape(
            b, 1, self.n_q_heads, self.d_head
        )
        k_pre = qkv[..., q_end:k_end].reshape(
            b, 1, self.n_kv_heads, self.d_head
        )
        v = qkv[..., k_end:].reshape(
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
        y = (merged.reshape(-1, self.d_model) @ Wo).reshape(
            b, 1, self.d_model
        )
        return y[:, -1, :]
