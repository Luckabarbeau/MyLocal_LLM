"""Process backend selection for NumPy, CuPy, or explicit PyTorch execution."""

import os
import numpy as _np

_torch_backend = None

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
elif _BACKEND in {"torch", "pytorch"}:
    try:
        import torch as _torch_backend
    except ImportError as exc:
        raise RuntimeError(
            "MINI_LLM_BACKEND=pytorch was requested but PyTorch is not installed."
        ) from exc
    # Keep the legacy xp surface NumPy-compatible during package import. Actual
    # PyTorch execution is owned by mini_llm.backends.pytorch and never routes
    # native handwritten NumPy/CuPy kernels through torch.
    xp = _np
    BACKEND_NAME = "pytorch"
else:
    raise ValueError("MINI_LLM_BACKEND must be 'numpy', 'cupy', or 'pytorch'.")


def asnumpy(x):
    """Return a host NumPy array from a NumPy *or* CuPy value.

    The GPU runtime also passes host-side token indices, causal context
    lengths, targets and metadata into shared utilities.  Only actual CuPy
    arrays support ``.get()``; assuming every value is device-resident makes
    context-conditioned fusion fail as soon as it receives host token arrays.
    """
    if BACKEND_NAME == "cupy" and isinstance(x, xp.ndarray):
        return x.get()
    if BACKEND_NAME == "pytorch" and isinstance(x, _torch_backend.Tensor):
        return (x.detach().cpu().float().numpy()
                if x.dtype == _torch_backend.bfloat16 else x.detach().cpu().numpy())
    return _np.asarray(x)


def scalar(x):
    if BACKEND_NAME == "cupy":
        return float(x.get())
    if BACKEND_NAME == "pytorch" and isinstance(x, _torch_backend.Tensor):
        return float(x.detach().item())
    return float(x)


def synchronize():
    if BACKEND_NAME == "cupy":
        xp.cuda.Stream.null.synchronize()
    elif BACKEND_NAME == "pytorch" and _torch_backend.cuda.is_available():
        _torch_backend.cuda.synchronize()


class RandomStream:
    """Small backend-independent deterministic random stream."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        if BACKEND_NAME == "pytorch":
            self.rng = _torch_backend.Generator(device="cpu").manual_seed(self.seed)
        else:
            self.rng = xp.random.RandomState(self.seed)

    def normal(self, shape, std=1.0, dtype=None):
        if BACKEND_NAME == "pytorch":
            return _torch_backend.randn(
                tuple(shape), generator=self.rng,
                dtype=resolve_dtype(dtype or "float32")
            ) * float(std)
        x = self.rng.normal(0.0, float(std), size=shape)
        if dtype is not None:
            x = x.astype(resolve_dtype(dtype))
        return x


def resolve_dtype(dtype):
    """Resolve public dtype names to backend-compatible dtype objects."""
    if BACKEND_NAME == "pytorch":
        aliases = {
            "bfloat16": _torch_backend.bfloat16, "bf16": _torch_backend.bfloat16,
            "float16": _torch_backend.float16, "float32": _torch_backend.float32,
        }
        return aliases.get(dtype, dtype)
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
    if BACKEND_NAME == "pytorch" and isinstance(dtype, _torch_backend.dtype):
        return str(dtype).removeprefix("torch.").lower()
    return str(_np.dtype(dtype)).lower()


def is_low_precision_dtype(dtype):
    """True for FP16 and BF16 model/compute dtypes."""
    return dtype_name(dtype) in {"float16", "bfloat16"}


def is_bfloat16_dtype(dtype):
    return dtype_name(dtype) == "bfloat16"


def validate_bfloat16_backend():
    """Fail fast if the selected backend cannot execute a BF16 GEMM."""
    if BACKEND_NAME == "pytorch":
        if not _torch_backend.cuda.is_available():
            raise RuntimeError("PyTorch BF16 training requires a CUDA device")
        try:
            a = _torch_backend.ones((16, 16), device="cuda", dtype=_torch_backend.bfloat16)
            b = _torch_backend.ones((16, 16), device="cuda", dtype=_torch_backend.bfloat16)
            c = a @ b
            if not bool(_torch_backend.isfinite(c).all().item()):
                raise RuntimeError("BF16 GEMM returned non-finite values")
        except Exception as exc:
            raise RuntimeError(f"BF16 mixed precision is not usable with PyTorch: {exc}") from exc
        return _torch_backend.bfloat16
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
