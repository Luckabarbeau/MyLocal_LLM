from ..backend import xp, is_low_precision_dtype, is_bfloat16_dtype
from ..init import matrix_parameter
from .rope import rope_forward, rope_backward, clear_rope_cache
from .attention_selection import build_local_causal_plan
from .indexed_attention import indexed_attention_forward, indexed_attention_backward
from .retrieval_attention import ContextRetrievalAttention
from ..config import (
    AttentionLayerConfig,
    DenseAttentionConfig,
    LocalAttentionConfig,
    RetrievalAttentionConfig,
)


def softmax_forward(x, axis=-1, logit_multiplier=1.0):
    """Stable softmax with FP32 reductions for low-precision logits.

    Attention score GEMMs stay FP16/BF16, while max/subtract/exp/sum/divide
    execute in FP32.  ``logit_multiplier`` restores the FP16 score pre-scale
    only after max subtraction, so positive overflow cannot occur.
    """
    work = x.astype("float32", copy=False) if is_low_precision_dtype(x.dtype) else x
    x_max = xp.max(work, axis=axis, keepdims=True)
    shifted = work - x_max
    if logit_multiplier != 1.0:
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
        name="attention", dtype="float32", attention_config=None
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
        if attention_config is not None and not isinstance(
            attention_config, AttentionLayerConfig
        ):
            raise TypeError("attention_config must be an AttentionLayerConfig or None")
        self.attention_config = attention_config
        self._static_head_groups = []
        self._retrieval_groups = []

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

        if self.attention_config is not None:
            if len(self.attention_config.heads) != self.n_q_heads:
                raise ValueError(
                    "attention_config must define exactly one topology per query head"
                )
            self._configure_attention_patterns(
                self.attention_config, rng, input_std, name, dtype
            )

    def parameters(self):
        params = [self.Wq, self.Wk, self.Wv, self.Wo]
        for group in self._retrieval_groups:
            params.extend(group["module"].parameters())
        return params

    def _configure_attention_patterns(self, attention_config, rng, input_std, name, dtype):
        static_groups = {}
        retrieval_groups = {}

        for head_index, head_config in enumerate(attention_config.heads):
            if isinstance(head_config, DenseAttentionConfig):
                static_groups.setdefault(("dense", None), []).append(head_index)
            elif isinstance(head_config, LocalAttentionConfig):
                static_groups.setdefault(("local", int(head_config.window)), []).append(
                    head_index
                )
            elif isinstance(head_config, RetrievalAttentionConfig):
                group = retrieval_groups.setdefault(
                    head_config.group,
                    {"head_indices": [], "config": head_config.context_router},
                )
                if group["config"] != head_config.context_router:
                    raise ValueError(
                        f"retrieval group {head_config.group!r} must share one "
                        "ContextRouterConfig"
                    )
                group["head_indices"].append(head_index)
            else:
                raise TypeError(
                    f"unsupported attention head config: {type(head_config).__name__}"
                )

        self._static_head_groups = [
            {"kind": kind, "window": window, "head_indices": tuple(indices)}
            for (kind, window), indices in static_groups.items()
        ]

        self._retrieval_groups = []
        for group_name, group in retrieval_groups.items():
            head_indices = tuple(group["head_indices"])
            router_config = group["config"]
            if router_config.num_queries not in {1, len(head_indices)}:
                raise ValueError(
                    f"retrieval group {group_name!r} requires num_queries=1 or "
                    f"num_queries={len(head_indices)}"
                )
            module = ContextRetrievalAttention(
                d_model=self.d_model,
                config=router_config,
                rng=rng,
                input_std=input_std,
                name=f"{name}.retrieval.{group_name}",
                dtype=dtype,
            )
            self._retrieval_groups.append(
                {
                    "name": group_name,
                    "head_indices": head_indices,
                    "module": module,
                }
            )

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

    def _mixed_context_forward(self, x, q, k, v, return_cache):
        b, t, _, _ = q.shape
        context_dtype = xp.float32 if is_bfloat16_dtype(q.dtype) else q.dtype
        context = xp.zeros(q.shape, dtype=context_dtype)
        group_caches = []
        routing = {}

        for group in self._static_head_groups:
            head_indices = group["head_indices"]
            q_subset = q[:, :, head_indices, :]
            window = t if group["kind"] == "dense" else group["window"]
            plan = build_local_causal_plan(b, t, window)
            kv_head_indices = xp.asarray(
                head_indices, dtype=xp.int64
            ) // self.group_size
            if return_cache:
                subset, subset_cache = indexed_attention_forward(
                    q_subset,
                    k,
                    v,
                    plan,
                    kv_head_indices=kv_head_indices,
                    scale=self.scale,
                    return_cache=True,
                )
                group_caches.append(
                    {
                        "kind": "indexed",
                        "head_indices": head_indices,
                        "cache": subset_cache,
                    }
                )
            else:
                subset = indexed_attention_forward(
                    q_subset,
                    k,
                    v,
                    plan,
                    kv_head_indices=kv_head_indices,
                    scale=self.scale,
                    return_cache=False,
                )
            context[:, :, head_indices, :] = subset

        for group in self._retrieval_groups:
            head_indices = group["head_indices"]
            q_subset = q[:, :, head_indices, :]
            kv_head_indices = xp.asarray(
                head_indices, dtype=xp.int64
            ) // self.group_size
            if return_cache:
                subset, group_routing, subset_cache = group["module"].forward(
                    x,
                    q_subset,
                    k,
                    v,
                    kv_head_indices=kv_head_indices,
                    return_cache=True,
                )
                group_caches.append(
                    {
                        "kind": "retrieval",
                        "name": group["name"],
                        "head_indices": head_indices,
                        "module": group["module"],
                        "cache": subset_cache,
                    }
                )
            else:
                subset, group_routing = group["module"].forward(
                    x,
                    q_subset,
                    k,
                    v,
                    kv_head_indices=kv_head_indices,
                    return_cache=False,
                )
            context[:, :, head_indices, :] = subset
            routing[group["name"]] = group_routing

        return context, group_caches, routing

    def _mixed_context_backward(self, dcontext, cache):
        q, k, v = cache["q"], cache["k"], cache["v"]
        grad_dtype = xp.float32 if is_bfloat16_dtype(q.dtype) else q.dtype
        dq = xp.zeros(q.shape, dtype=grad_dtype)
        dk = xp.zeros(k.shape, dtype=grad_dtype)
        dv = xp.zeros(v.shape, dtype=grad_dtype)
        drouter_input = xp.zeros(cache["x"].shape, dtype=grad_dtype)

        for group_cache in cache["pattern_caches"]:
            head_indices = group_cache["head_indices"]
            dsubset = dcontext[:, :, head_indices, :]
            if group_cache["kind"] == "indexed":
                dq_sub, dk_sub, dv_sub, _ = indexed_attention_backward(
                    dsubset, group_cache["cache"]
                )
            else:
                drouter, dq_sub, dk_sub, dv_sub = group_cache["module"].backward(
                    dsubset, group_cache["cache"]
                )
                drouter_input += drouter.astype(grad_dtype, copy=False)

            dq[:, :, head_indices, :] += dq_sub.astype(grad_dtype, copy=False)
            dk += dk_sub.astype(grad_dtype, copy=False)
            dv += dv_sub.astype(grad_dtype, copy=False)

        return dq, dk, dv, drouter_input

    def _projection_backward(self, x, dq, dk, dv, cache):
        b, t, _ = x.shape
        dq_pre = rope_backward(dq, cache["q_rope_cache"])
        dk_pre = rope_backward(dk, cache["k_rope_cache"])

        x2 = x.reshape(-1, self.d_model)
        dq2 = dq_pre.reshape(-1, self.n_q_heads * self.d_head)
        dk2 = dk_pre.reshape(-1, self.n_kv_heads * self.d_head)
        dv2 = dv.reshape(-1, self.n_kv_heads * self.d_head)
        if is_bfloat16_dtype(x.dtype):
            dq2 = dq2.astype(x.dtype, copy=False)
            dk2 = dk2.astype(x.dtype, copy=False)
            dv2 = dv2.astype(x.dtype, copy=False)

        self.Wq.grad += x2.T @ dq2
        self.Wk.grad += x2.T @ dk2
        self.Wv.grad += x2.T @ dv2

        dx2 = (
            dq2 @ self.Wq.data.T
            + dk2 @ self.Wk.data.T
            + dv2 @ self.Wv.data.T
        )
        return dx2.reshape(b, t, self.d_model)

    def forward(self, x, return_cache=True):
        if x.ndim != 3:
            raise ValueError("attention input must have shape [B,T,D].")
        b, t, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError("input final dimension != d_model.")

        # Keep projection GEMMs strictly 2-D. CuPy 14 supports BF16 2-D GEMM
        # efficiently, but its generic N-D @ 2-D path currently fails on BF16.
        x2 = x.reshape(-1, self.d_model)
        q_pre = (x2 @ self.Wq.data).reshape(
            b, t, self.n_q_heads, self.d_head
        )
        k_pre = (x2 @ self.Wk.data).reshape(
            b, t, self.n_kv_heads, self.d_head
        )
        v = (x2 @ self.Wv.data).reshape(
            b, t, self.n_kv_heads, self.d_head
        )

        q, q_rope_cache = rope_forward(q_pre, self.rope_base)
        k, k_rope_cache = rope_forward(k_pre, self.rope_base)

        if self.attention_config is not None:
            context, pattern_caches, routing = self._mixed_context_forward(
                x, q, k, v, return_cache
            )
            merged = context.reshape(b, t, self.n_q_heads * self.d_head)
            merged_compute = (
                merged.astype(x.dtype, copy=False)
                if is_bfloat16_dtype(q.dtype) else merged
            )
            y = (
                merged_compute.reshape(-1, merged_compute.shape[-1]) @ self.Wo.data
            ).reshape(b, t, self.d_model)
            if not return_cache:
                return y
            return y, {
                "x": x,
                "q": q,
                "k": k,
                "v": v,
                "merged": merged,
                "q_rope_cache": q_rope_cache,
                "k_rope_cache": k_rope_cache,
                "pattern_caches": pattern_caches,
                "routing": routing,
                "mixed": True,
            }

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

        # CuPy 14's generic batched matmul does not currently accept BF16
        # (dtype code 'E'). For BF16 training, run only the comparatively small
        # attention batched products in FP32. The large projection/MoE GEMMs
        # remain BF16 tensor-core GEMMs, while this path is both robust and
        # numerically stronger.
        bf16_attention = is_bfloat16_dtype(q_grouped.dtype)
        if bf16_attention:
            q_scaled = q_grouped.astype("float32") * self.scale
            k_mat = k_heads.astype("float32")
        else:
            q_scaled = q_grouped * (self.scale * score_prescale)
            k_mat = k_heads

        scores_grouped = xp.matmul(
            q_scaled,
            k_mat[:, :, xp.newaxis, :, :].swapaxes(-2, -1),
        )                                               # [B,Hkv,G,T,T]
        scores = scores_grouped.reshape(b, self.n_q_heads, t, t)

        causal = self._get_causal_mask(t)
        scores_masked = xp.where(causal[None, None, :, :], -xp.inf, scores)
        probs = softmax_forward(
            scores_masked,
            axis=-1,
            logit_multiplier=(1.0 / score_prescale),
        )

        # Softmax is FP32 for low-precision inputs. Cast probabilities back to
        # the branch compute dtype only for the tensor-core P@V GEMM.
        if bf16_attention:
            probs_compute = probs
            v_mat = v_heads.astype("float32")
        else:
            probs_compute = (
                probs.astype(q_grouped.dtype, copy=False)
                if is_low_precision_dtype(q_grouped.dtype) else probs
            )
            v_mat = v_heads
        probs_grouped_compute = probs_compute.reshape(
            b, self.n_kv_heads, self.group_size, t, t
        )
        context_grouped = xp.matmul(
            probs_grouped_compute,
            v_mat[:, :, xp.newaxis, :, :],
        )                                               # [B,Hkv,G,T,Dh]
        context = self._ungroup_queries(context_grouped)
        merged = context.reshape(b, t, self.n_q_heads * self.d_head)
        merged_compute = (
            merged.astype(x.dtype, copy=False) if bf16_attention else merged
        )
        y = (
            merged_compute.reshape(-1, merged_compute.shape[-1]) @ self.Wo.data
        ).reshape(b, t, self.d_model)

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
        merged = cache["merged"]
        b, t, _ = x.shape

        self.Wo.grad += (
            merged.reshape(-1, merged.shape[-1]).T
            @ dy.reshape(-1, dy.shape[-1])
        )
        dy2 = dy.reshape(-1, dy.shape[-1])
        dmerged = (dy2 @ self.Wo.data.T).reshape(merged.shape)
        dcontext = dmerged.reshape(
            b, t, self.n_q_heads, self.d_head
        )

        if self.attention_config is not None:
            dq, dk, dv, drouter_input = self._mixed_context_backward(
                dcontext, cache
            )
            dx = self._projection_backward(x, dq, dk, dv, cache)
            return dx + drouter_input.astype(dx.dtype, copy=False)

        probs = cache["probs"]
        q_grouped = self._group_queries(q)              # [B,Hkv,G,T,D]
        dcontext_grouped = self._group_queries(dcontext)
        k_heads = k.transpose(0, 2, 1, 3)              # [B,Hkv,T,D]
        v_heads = v.transpose(0, 2, 1, 3)
        bf16_attention = is_bfloat16_dtype(q_grouped.dtype)
        if bf16_attention:
            probs_compute = probs
            dcontext_mat = dcontext_grouped.astype("float32")
            v_mat = v_heads.astype("float32")
        else:
            probs_compute = (
                probs.astype(q_grouped.dtype, copy=False)
                if is_low_precision_dtype(q_grouped.dtype) else probs
            )
            dcontext_mat = dcontext_grouped
            v_mat = v_heads
        probs_grouped_compute = probs_compute.reshape(
            b, self.n_kv_heads, self.group_size, t, t
        )

        dprobs_grouped = xp.matmul(
            dcontext_mat,
            v_mat[:, :, xp.newaxis, :, :].swapaxes(-2, -1),
        )                                               # [B,Hkv,G,T,T]

        # V is shared by all query heads in a group; sum group contributions.
        dv_heads = xp.matmul(
            probs_grouped_compute.swapaxes(-2, -1),
            dcontext_mat,
        ).sum(axis=2)                                   # [B,Hkv,T,D]

        dprobs = dprobs_grouped.reshape(
            b, self.n_q_heads, t, t
        )
        dscores = softmax_backward(dprobs, probs, axis=-1)
        dscores_grouped = dscores.reshape(
            b, self.n_kv_heads, self.group_size, t, t
        )
        if bf16_attention:
            dscores_compute = dscores_grouped.astype("float32", copy=False)
            k_grad_mat = k_heads.astype("float32")
            q_grad_mat = q_grouped.astype("float32")
        else:
            dscores_compute = (
                dscores_grouped.astype(q_grouped.dtype, copy=False)
                if is_low_precision_dtype(q_grouped.dtype) else dscores_grouped
            )
            k_grad_mat = k_heads
            q_grad_mat = q_grouped

        dq_grouped = xp.matmul(
            dscores_compute,
            k_grad_mat[:, :, xp.newaxis, :, :],
        ) * self.scale

        # K is also shared by every query head in a group.
        dk_heads = xp.matmul(
            dscores_compute.swapaxes(-2, -1),
            q_grad_mat * self.scale,
        ).sum(axis=2)

        dq = self._ungroup_queries(dq_grouped)
        dk = dk_heads.transpose(0, 2, 1, 3)
        dv = dv_heads.transpose(0, 2, 1, 3)

        return self._projection_backward(x, dq, dk, dv, cache)
