"""Debug RMSNorm backward pass."""

import numpy as np
from mini_llm.backend import xp
from mini_llm.ops.rmsnorm import RMSNorm

layer = RMSNorm(8, eps=1e-8, dtype="float64")
x = xp.asarray(np.random.default_rng(1).normal(size=(1, 3, 8)), dtype="float64")
dy = xp.asarray(np.random.default_rng(2).normal(size=x.shape), dtype="float64")

_, cache = layer.forward(x)
dx = layer.backward(dy, cache)

v = xp.asarray(np.random.default_rng(3).normal(size=x.shape), dtype="float64")
v /= xp.sqrt(xp.sum(v*v))
eps = 1e-6

def objective(z):
    out, _ = layer.forward(z)
    return float(xp.sum(out*dy))

fd = (objective(x+eps*v)-objective(x-eps*v))/(2*eps)
an = float(xp.sum(dx*v))
rel = abs(fd-an)/(abs(fd)+abs(an)+1e-12)

print(f"RMSNorm backward check:")
print(f"FD: {fd}")
print(f"AN: {an}")
print(f"Relative error: {rel}")
