from ..backend import xp
from ..init import matrix_parameter
from .rope import rope_forward, rope_backward, clear_rope_cache


def softmax_forward(x, axis=-1, logit_multiplier=1.0):
    """Stable softmax, optionally restoring a pre-scaled logit magnitude.

    ``logit_multiplier`` is applied only after subtracting the row maximum.
    This is algebraically equivalent to softmax(logit_multiplier * x), while
    avoiding positive overflow because the shifted values are never > 0.
    """
    x_max = xp.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    if logit_multiplier != 1.0:
        # After max subtraction all finite logits are <= 0.  When restoring
        # a large pre-scale factor in FP16, extremely negative values may
        # overflow to -inf.  That is harmless for softmax mathematically, but
        # it emits a runtime warning.  Saturate only those already-negligible
        # tails before rescaling so the product stays representable.
        if shifted.dtype == xp.float16:
            lower = -float(xp.finfo(xp.float16).max) / float(logit_multiplier)
            shifted = xp.maximum(shifted, lower)
        shifted = shifted * logit_multiplier
    e = xp.exp(shifted)
    return e / xp.sum(e, axis=axis, keepdims=True)


def softmax_backward(dp, p, axis=-1):
    dot = xp.sum(dp * p, axis=axis, keepdims=True)
    return p * (dp - dot)


class GQAAttention:
    """Explicit grouped-query causal self-attention.

    K/V stay in their native ``n_kv_heads`` representation.  Query heads are
    viewed as ``[n_kv_heads, group_size]`` and batched matmul broadcasts the
    shared K/V heads across each query group.  This avoids physically repeating
    K/V tensors in forward and their gradients in backward.
    """

    _causal_mask_cache = {}

    def __init__(
        self, d_model, n_q_heads, n_kv_heads, d_head,
        input_std, output_std, rng, rope_base=10_000.0,
        name="attention", dtype="float32"
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

        self.Wq = matrix_parameter(
            (d_model, n_q_heads * d_head), input_std, rng, f"{name}.Wq", dtype=dtype
        )
        self.Wk = matrix_parameter(
            (d_model, n_kv_heads * d_head), input_std, rng, f"{name}.Wk", dtype=dtype
        )
        self.Wv = matrix_parameter(
            (d_model, n_kv_heads * d_head), input_std, rng, f"{name}.Wv", dtype=dtype
        )
        self.Wo = matrix_parameter(
            (n_q_heads * d_head, d_model), output_std, rng, f"{name}.Wo", dtype=dtype
        )

    def parameters(self):
        return [self.Wq, self.Wk, self.Wv, self.Wo]

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    @classmethod
    def clear_caches(cls):
        cls._causal_mask_cache.clear()
        clear_rope_cache()

    def _group_queries(self, q):
        """[B,T,Hq,D] -> [B,Hkv,G,T,D] as a view+transpose."""
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

    def _get_causal_mask(self, t):
        if t not in self._causal_mask_cache:
            self._causal_mask_cache[t] = xp.triu(
                xp.ones((t, t), dtype=bool), k=1
            )
        return self._causal_mask_cache[t]

    def forward(self, x, return_cache=True):
        if x.ndim != 3:
            raise ValueError("attention input must have shape [B,T,D].")
        b, t, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError("input final dimension != d_model.")

        q_pre = (x @ self.Wq.data).reshape(
            b, t, self.n_q_heads, self.d_head
        )
        k_pre = (x @ self.Wk.data).reshape(
            b, t, self.n_kv_heads, self.d_head
        )
        v = (x @ self.Wv.data).reshape(
            b, t, self.n_kv_heads, self.d_head
        )

        q, q_rope_cache = rope_forward(q_pre, self.rope_base)
        k, k_rope_cache = rope_forward(k_pre, self.rope_base)

        # Native GQA representation -- no xp.repeat(K/V).
        q_grouped = self._group_queries(q)             # [B,Hkv,G,T,Dh]
        k_heads = k.transpose(0, 2, 1, 3)             # [B,Hkv,T,Dh]
        v_heads = v.transpose(0, 2, 1, 3)             # [B,Hkv,T,Dh]

        # Scale Q before the dot product.  For FP16, additionally pre-scale
        # the attention logits before the GEMM.  This keeps the expensive QK^T
        # matmul on the fast FP16 tensor-core path while increasing its overflow
        # headroom by 32x.  The exact attention temperature is restored inside
        # the stable softmax *after* subtracting the row maximum, where all
        # values are non-positive and therefore cannot overflow to +Inf.
        #
        # Algebraically, for s = scale * QK^T and r = 32:
        #   softmax(s) = softmax(r * (s/r))
        # and subtracting max(s/r) before multiplying by r is exact up to FP16
        # roundoff.  Backward continues to use ds/dQ for the original s, so no
        # gradient rescaling is required below.
        score_prescale = 1.0 / 32.0 if q_grouped.dtype == xp.float16 else 1.0
        q_scaled = q_grouped * (self.scale * score_prescale)
        scores_grouped = xp.matmul(
            q_scaled,
            k_heads[:, :, xp.newaxis, :, :].swapaxes(-2, -1),
        )                                               # [B,Hkv,G,T,T]
        scores = scores_grouped.reshape(b, self.n_q_heads, t, t)

        causal = self._get_causal_mask(t)
        scores_masked = xp.where(causal[None, None, :, :], -xp.inf, scores)
        probs = softmax_forward(
            scores_masked,
            axis=-1,
            logit_multiplier=(1.0 / score_prescale),
        )

        probs_grouped = probs.reshape(
            b, self.n_kv_heads, self.group_size, t, t
        )
        context_grouped = xp.matmul(
            probs_grouped,
            v_heads[:, :, xp.newaxis, :, :],
        )                                               # [B,Hkv,G,T,Dh]
        context = self._ungroup_queries(context_grouped)
        merged = context.reshape(b, t, self.n_q_heads * self.d_head)
        y = merged @ self.Wo.data

        if not return_cache:
            return y

        return y, {
            "x": x,
            "q": q,
            "k": k,
            "v": v,
            "probs": probs,
            "merged": merged,
            "q_rope_cache": q_rope_cache,
            "k_rope_cache": k_rope_cache,
        }

    def backward(self, dy, cache):
        x = cache["x"]
        q, k, v = cache["q"], cache["k"], cache["v"]
        probs, merged = cache["probs"], cache["merged"]
        b, t, _ = x.shape

        self.Wo.grad += (
            merged.reshape(-1, merged.shape[-1]).T
            @ dy.reshape(-1, dy.shape[-1])
        )
        dmerged = dy @ self.Wo.data.T
        dcontext = dmerged.reshape(
            b, t, self.n_q_heads, self.d_head
        )

        q_grouped = self._group_queries(q)              # [B,Hkv,G,T,D]
        dcontext_grouped = self._group_queries(dcontext)
        k_heads = k.transpose(0, 2, 1, 3)              # [B,Hkv,T,D]
        v_heads = v.transpose(0, 2, 1, 3)
        probs_grouped = probs.reshape(
            b, self.n_kv_heads, self.group_size, t, t
        )

        dprobs_grouped = xp.matmul(
            dcontext_grouped,
            v_heads[:, :, xp.newaxis, :, :].swapaxes(-2, -1),
        )                                               # [B,Hkv,G,T,T]

        # V is shared by all query heads in a group; sum group contributions.
        dv_heads = xp.matmul(
            probs_grouped.swapaxes(-2, -1),
            dcontext_grouped,
        ).sum(axis=2)                                   # [B,Hkv,T,D]

        dprobs = dprobs_grouped.reshape(
            b, self.n_q_heads, t, t
        )
        dscores = softmax_backward(dprobs, probs, axis=-1)
        dscores_grouped = dscores.reshape(
            b, self.n_kv_heads, self.group_size, t, t
        )

        dq_grouped = xp.matmul(
            dscores_grouped,
            k_heads[:, :, xp.newaxis, :, :],
        ) * self.scale

        # K is also shared by every query head in the group.
        dk_heads = xp.matmul(
            dscores_grouped.swapaxes(-2, -1),
            q_grouped * self.scale,
        ).sum(axis=2)

        dq = self._ungroup_queries(dq_grouped)
        dk = dk_heads.transpose(0, 2, 1, 3)
        dv = dv_heads.transpose(0, 2, 1, 3)

        dq_pre = rope_backward(dq, cache["q_rope_cache"])
        dk_pre = rope_backward(dk, cache["k_rope_cache"])

        x2 = x.reshape(-1, self.d_model)
        dq2 = dq_pre.reshape(-1, self.n_q_heads * self.d_head)
        dk2 = dk_pre.reshape(-1, self.n_kv_heads * self.d_head)
        dv2 = dv.reshape(-1, self.n_kv_heads * self.d_head)

        self.Wq.grad += x2.T @ dq2
        self.Wk.grad += x2.T @ dk2
        self.Wv.grad += x2.T @ dv2

        return (
            dq_pre.reshape(b, t, -1) @ self.Wq.data.T
            + dk_pre.reshape(b, t, -1) @ self.Wk.data.T
            + dv.reshape(b, t, -1) @ self.Wv.data.T
        )
