"""Debug SwiGLU backward pass."""

import numpy as np
from mini_llm.backend import xp, RandomStream
from mini_llm.ops.swiglu import SwiGLU

rng = RandomStream(5)
layer = SwiGLU(8, 16, 0.1, 0.05, rng, dtype="float64")
x = xp.asarray(np.random.default_rng(6).normal(size=(1, 3, 8)), dtype="float64")
dy = xp.asarray(np.random.default_rng(7).normal(size=(1, 3, 8)), dtype="float64")

_, cache = layer.forward(x)
dx = layer.backward(dy, cache)

v = xp.asarray(np.random.default_rng(8).normal(size=x.shape), dtype="float64")
v /= xp.sqrt(xp.sum(v*v))
eps = 1e-6

def objective(z):
    out, _ = layer.forward(z)
    return float(xp.sum(out*dy))

fd = (objective(x+eps*v)-objective(x-eps*v))/(2*eps)
an = float(xp.sum(dx*v))
rel = abs(fd-an)/(abs(fd)+abs(an)+1e-12)

print(f"SwiGLU backward check:")
print(f"FD: {fd}")
print(f"AN: {an}")
print(f"Relative error: {rel}")
