"""Backend selection for NumPy (reference) or CuPy (GPU)."""

import os
import numpy as _np

try:
    import ml_dtypes as _ml_dtypes
except ImportError:
    _ml_dtypes = None

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
            x = x.astype(resolve_dtype(dtype))
        return x


def resolve_dtype(dtype):
    """Resolve public dtype names to backend-compatible dtype objects."""
    if dtype in ("bfloat16", "bf16"):
        if _ml_dtypes is None:
            raise RuntimeError(
                "bfloat16 requested but ml-dtypes is not installed. "
                "Install it with: python -m pip install ml-dtypes"
            )
        return _ml_dtypes.bfloat16
    return dtype


def dtype_name(dtype):
    """Return a stable lowercase dtype name for NumPy/CuPy/ml_dtypes dtypes."""
    return str(_np.dtype(dtype)).lower()


def is_low_precision_dtype(dtype):
    """True for FP16 and BF16 model/compute dtypes."""
    return dtype_name(dtype) in {"float16", "bfloat16"}


def is_bfloat16_dtype(dtype):
    return dtype_name(dtype) == "bfloat16"


def validate_bfloat16_backend():
    """Fail fast if the selected backend cannot execute a BF16 GEMM."""
    if _ml_dtypes is None:
        raise RuntimeError(
            "BF16 mixed precision requires ml-dtypes. "
            "Install it with: python -m pip install ml-dtypes"
        )
    bf16 = _ml_dtypes.bfloat16
    try:
        a = xp.ones((16, 16), dtype=bf16)
        b = xp.ones((16, 16), dtype=bf16)
        c = a @ b
        ok = xp.all(xp.isfinite(c))
        if hasattr(ok, "item"):
            ok = bool(ok.item())
        else:
            ok = bool(ok)
        if not ok:
            raise RuntimeError("BF16 GEMM returned non-finite values")
    except Exception as exc:
        raise RuntimeError(
            f"BF16 mixed precision is not usable with backend {BACKEND_NAME!r}: {exc}"
        ) from exc
    return bf16
