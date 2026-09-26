"""Backend selection for NumPy (reference) or CuPy (GPU)."""

import os
import numpy as _np

_BACKEND = os.environ.get("MINI_LLM_BACKEND", "numpy").lower()

if _BACKEND == "cupy":
    try:
        import cupy as xp
    except ImportError as exc:
        raise RuntimeError(
            "MINI_LLM_BACKEND=cupy was requested but CuPy is not installed."
        ) from exc
    BACKEND_NAME = "cupy"
elif _BACKEND == "numpy":
    xp = _np
    BACKEND_NAME = "numpy"
else:
    raise ValueError("MINI_LLM_BACKEND must be 'numpy' or 'cupy'.")


def asnumpy(x):
    if BACKEND_NAME == "cupy":
        return x.get()
    return _np.asarray(x)


def scalar(x):
    if BACKEND_NAME == "cupy":
        return float(x.get())
    return float(x)


def synchronize():
    if BACKEND_NAME == "cupy":
        xp.cuda.Stream.null.synchronize()


class RandomStream:
    """Small backend-independent deterministic random stream."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.rng = xp.random.RandomState(self.seed)

    def normal(self, shape, std=1.0, dtype=None):
        x = self.rng.normal(0.0, float(std), size=shape)
        if dtype is not None:
            x = x.astype(dtype)
        return x
