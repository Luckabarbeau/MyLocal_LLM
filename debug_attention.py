"""Debug Attention backward pass."""

import numpy as np
from mini_llm.backend import xp, RandomStream
from mini_llm.ops.attention import GQAAttention

rng = RandomStream(10)
attn = GQAAttention(
    d_model=8, n_q_heads=2, n_kv_heads=1, d_head=4,
    input_std=0.08, output_std=0.04, rng=rng, dtype="float64"
)

x = xp.asarray(np.random.default_rng(12).normal(size=(1, 3, 8)), dtype="float64")
dy = xp.asarray(np.random.default_rng(13).normal(size=(1, 3, 8)), dtype="float64")

_, cache = attn.forward(x)
attn.zero_grad()
dx = attn.backward(dy, cache)

v = xp.asarray(np.random.default_rng(14).normal(size=x.shape), dtype="float64")
v /= xp.sqrt(xp.sum(v*v))
eps = 1e-6

def objective(z):
    out, _ = attn.forward(z)
    return float(xp.sum(out*dy))

fd = (objective(x+eps*v)-objective(x-eps*v))/(2*eps)
an = float(xp.sum(dx*v))
rel = abs(fd-an)/(abs(fd)+abs(an)+1e-12)

print(f"Attention backward check:")
print(f"FD: {fd}")
print(f"AN: {an}")
print(f"Relative error: {rel}")
