"""Mixture-of-Experts router with one differentiable top-k weighting rule.

For every number of experts ``E`` and active experts ``k``:

    p = softmax(logits over ALL E experts)
    chosen = argsort(-logits)[:k]
    weights = gather(p, chosen)

Only the selected experts execute. The chosen weights are deliberately **not**
renormalized: their sum is the probability mass assigned to selected experts.
This rule gives nonzero, exact (with fixed indices) gradients for top-1 and
higher-k without a model-specific routing flag. Discrete index selection has
no gradient; unselected expert *logits* still receive gradients through the
softmax denominator, but unselected expert *weights* do not execute/update.
"""

from ..backend import xp, resolve_dtype, is_low_precision_dtype
from ..performance_profiler import moe_detail_scope


def expert_topk_forward(logits, k, output_dtype=None, return_cache=True):
    """Full-expert softmax, followed by top-k selection without renormalization.

    Same function is used for both training and inference to enforce parity.
    """
    if logits.ndim < 1 or not (1 <= int(k) <= int(logits.shape[-1])):
        raise ValueError("expert router expects 1 <= k <= n_experts")
    work = (
        logits.astype(xp.float32, copy=False)
        if is_low_precision_dtype(logits.dtype) else logits
    )
    centered = work - xp.max(work, axis=-1, keepdims=True)
    unnormalized = xp.exp(centered)
    full_probs = unnormalized / xp.sum(unnormalized, axis=-1, keepdims=True)
    indices = xp.argsort(-work, axis=-1)[..., :int(k)]
    selected_probs = xp.take_along_axis(full_probs, indices, axis=-1)
    weights = (
        selected_probs.astype(output_dtype, copy=False)
        if output_dtype is not None and is_low_precision_dtype(output_dtype)
        else selected_probs
    )
    cache = (full_probs, indices) if return_cache else None
    return weights, indices, cache


def expert_topk_backward(dweights, cache):
    """Exact local softmax VJP while holding discrete selected indices fixed.

    For selected weights w_i = p_i, let a_j = dL/dw_j (selected only):

        dL/dz_i = p_i (a_i - sum_{j selected} p_j a_j)

    and a_i = 0 for unselected logits. Thus unselected router logits get a
    gradient, even for k=1. Works equally for k=1, k<E, and k=E.
    """
    full_probs, indices = cache
    dwork = (
        dweights.astype(xp.float32, copy=False)
        if is_low_precision_dtype(dweights.dtype) else dweights
    )
    if dwork.shape != indices.shape:
        raise ValueError("dweights shape does not match selected routing weights")
    selected_probs = xp.take_along_axis(full_probs, indices, axis=-1)
    weighted_grad = selected_probs * dwork
    correction = xp.sum(weighted_grad, axis=-1, keepdims=True)
    dlogits = -full_probs * correction
    flat = dlogits.reshape(-1, full_probs.shape[-1])
    flat_indices = indices.reshape(flat.shape[0], -1)
    flat[xp.arange(flat.shape[0])[:, None], flat_indices] += weighted_grad.reshape(flat_indices.shape)
    return dlogits


class Router:
    """Select k experts using probabilities normalized across all experts."""

    def __init__(self, d_model, n_experts, k, input_std, rng, name="router", dtype="float32"):
        self.d_model = d_model
        self.n_experts = n_experts
        if not (1 <= int(k) <= int(n_experts)):
            raise ValueError("expert router expects 1 <= k <= n_experts")
        self.k = int(k)
        # 0080AQ: optional read-only per-layer device statistics. Disabled by default.
        self.usage_collector = None
        self.usage_layer = None
        from ..parameter import Parameter
        W_router = xp.asarray(rng.normal((d_model, n_experts), std=input_std, dtype=dtype))
        b_router = xp.zeros((n_experts,), dtype=resolve_dtype(dtype))
        self.W_router_param = Parameter(W_router.copy(), name=f"{name}.W_router")
        self.b_router_param = Parameter(b_router.copy(), name=f"{name}.b_router")
        # Keep the legacy aliases: non-Parameter code may read them directly.
        self.W_router = W_router
        self.b_router = b_router

    def parameters(self):
        return [self.W_router_param, self.b_router_param]

    def zero_grad(self):
        self.W_router_param.zero_grad()
        self.b_router_param.zero_grad()

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_2d = x.reshape(-1, self.d_model)
        with moe_detail_scope("moe.router.logits"):
            logits = (x_2d @ self.W_router_param.data).reshape(
                batch_size, seq_len, self.n_experts
            ) + self.b_router_param.data
        with moe_detail_scope("moe.router.topk"):
            weights, indices, topk_cache = expert_topk_forward(
                logits, self.k, output_dtype=x.dtype,
            )
        collector = self.usage_collector
        if collector is not None:
            collector.observe(self.usage_layer, indices, topk_cache[0])
        return weights, indices, {"x": x, "topk_cache": topk_cache}

    def backward(self, dweights, cache):
        x = cache["x"]
        with moe_detail_scope("moe.router.topk_bwd"):
            dlogits = expert_topk_backward(dweights, cache["topk_cache"])
        dlogits_compute = (
            dlogits.astype(x.dtype, copy=False)
            if is_low_precision_dtype(x.dtype) else dlogits
        )
        dlogits_2d = dlogits_compute.reshape(-1, self.n_experts)
        with moe_detail_scope("moe.router.dx_gemm"):
            dx = (dlogits_2d @ self.W_router_param.data.T).reshape(x.shape)
        x_2d = x.reshape(-1, self.d_model)
        with moe_detail_scope("moe.router.wgrad_gemm"):
            self.W_router_param.grad += x_2d.T @ dlogits_2d
        with moe_detail_scope("moe.router.bgrad"):
            self.b_router_param.grad += xp.sum(dlogits, axis=(0, 1))
        return dx
