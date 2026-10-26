from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from .backend import xp, is_low_precision_dtype


# Opt-in construction mode for genuinely forward-only models. Scoped with a
# ContextVar so constructing a frozen foundation cannot change trainable model
# parameter allocation elsewhere in the process.
_NO_GRAD_ALLOCATION = ContextVar("mini_llm_no_grad_allocation", default=False)


@contextmanager
def no_parameter_grad_allocation():
    """Construct parameters without gradient buffers (frozen inference only).

    Existing training defaults are unchanged. Frozen parameters must never be
    passed to an optimizer or a module backward that accumulates their grads.
    """
    token = _NO_GRAD_ALLOCATION.set(True)
    try:
        yield
    finally:
        _NO_GRAD_ALLOCATION.reset(token)


@dataclass
class Parameter:
    data: object
    name: str = ""
    decay: bool = True

    def __post_init__(self):
        # Keep gradient accumulation in FP32 for both FP16 and BF16 model
        # parameters.  This avoids loss of small gradient contributions across
        # microbatches and gives AdamW a stable FP32 input.
        if _NO_GRAD_ALLOCATION.get():
            self.grad = None
            return
        grad_dtype = "float32" if is_low_precision_dtype(self.data.dtype) else self.data.dtype
        self.grad = xp.zeros(
            self.data.shape,
            dtype=grad_dtype,
        )

    def zero_grad(self):
        if self.grad is not None:
            self.grad[...] = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size
