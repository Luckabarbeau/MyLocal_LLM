from ..backend import xp
from ..init import matrix_parameter
from .rope import rope_forward, rope_backward


def softmax_forward(x, axis=-1):
    x_max = xp.max(x, axis=axis, keepdims=True)
    e = xp.exp(x - x_max)
    return e / xp.sum(e, axis=axis, keepdims=True)


def softmax_backward(dp, p, axis=-1):
    dot = xp.sum(dp * p, axis=axis, keepdims=True)
    return p * (dp - dot)


class GQAAttention:
    """
    Explicit grouped-query causal self-attention.

    X       [B,T,D]
    Wq      [D,Hq*Dh]
    Wk/Wv   [D,Hkv*Dh]
    Wo      [Hq*Dh,D]
    Q       [B,T,Hq,Dh]
    K,V     [B,T,Hkv,Dh]
    scores  [B,Hq,T,T]
    """

    def __init__(
        self, d_model, n_q_heads, n_kv_heads, d_head,
        input_std, output_std, rng, rope_base=10_000.0,
        name="attention", dtype=None
    ):
        if n_q_heads * d_head != d_model:
            raise ValueError("n_q_heads * d_head must equal d_model.")
        if n_q_heads % n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads.")

        if dtype is None:
            dtype = "float32"

        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.group_size = n_q_heads // n_kv_heads
        self.scale = 1.0 / (d_head ** 0.5)
        self.rope_base = float(rope_base)

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

    def _expand_kv(self, x):
        return xp.repeat(x, self.group_size, axis=2)

    def _collapse_kv_grad(self, dx_exp):
        b, t, _, d = dx_exp.shape
        return dx_exp.reshape(
            b, t, self.n_kv_heads, self.group_size, d
        ).sum(axis=3)

    def forward(self, x, return_cache=True):
        if x.ndim != 3:
            raise ValueError("attention input must have shape [B,T,D].")
        b, t, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError("input final dimension != d_model.")

        q_pre = (x @ self.Wq.data).reshape(b, t, self.n_q_heads, self.d_head)
        k_pre = (x @ self.Wk.data).reshape(b, t, self.n_kv_heads, self.d_head)
        v = (x @ self.Wv.data).reshape(b, t, self.n_kv_heads, self.d_head)

        q, q_rope_cache = rope_forward(q_pre, self.rope_base)
        k, k_rope_cache = rope_forward(k_pre, self.rope_base)

        k_exp = self._expand_kv(k)
        v_exp = self._expand_kv(v)

        scores = xp.einsum("bthd,bshd->bhts", q, k_exp) * self.scale
        causal = xp.triu(xp.ones((t, t), dtype=bool), k=1)
        scores_masked = xp.where(causal[None, None, :, :], -xp.inf, scores)
        probs = softmax_forward(scores_masked, axis=-1)

        context = xp.einsum("bhts,bshd->bthd", probs, v_exp)
        merged = context.reshape(b, t, self.n_q_heads * self.d_head)
        y = merged @ self.Wo.data

        if not return_cache:
            return y

        return y, {
            "x": x,
            "q_pre": q_pre, "k_pre": k_pre, "v": v,
            "q": q, "k": k, "k_exp": k_exp, "v_exp": v_exp,
            "scores": scores, "scores_masked": scores_masked,
            "probs": probs, "context": context, "merged": merged,
            "q_rope_cache": q_rope_cache, "k_rope_cache": k_rope_cache,
        }

    def backward(self, dy, cache):
        x = cache["x"]
        q, k_exp, v_exp = cache["q"], cache["k_exp"], cache["v_exp"]
        probs, merged = cache["probs"], cache["merged"]
        b, t, _ = x.shape

        self.Wo.grad += merged.reshape(-1, merged.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        dmerged = dy @ self.Wo.data.T
        dcontext = dmerged.reshape(b, t, self.n_q_heads, self.d_head)

        dprobs = xp.einsum("bthd,bshd->bhts", dcontext, v_exp)
        dv_exp = xp.einsum("bhts,bthd->bshd", probs, dcontext)

        dscores = softmax_backward(dprobs, probs, axis=-1)

        dq = xp.einsum("bhts,bshd->bthd", dscores, k_exp) * self.scale
        dk_exp = xp.einsum("bhts,bthd->bshd", dscores, q) * self.scale

        dk = self._collapse_kv_grad(dk_exp)
        dv = self._collapse_kv_grad(dv_exp)

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
