from dataclasses import dataclass
from .backend import xp, is_low_precision_dtype


@dataclass
class Parameter:
    data: object
    name: str = ""
    decay: bool = True

    def __post_init__(self):
        # Keep gradient accumulation in FP32 for both FP16 and BF16 model
        # parameters.  This avoids loss of small gradient contributions across
        # microbatches and gives AdamW a stable FP32 input.
        grad_dtype = "float32" if is_low_precision_dtype(self.data.dtype) else self.data.dtype
        self.grad = xp.zeros(
            self.data.shape,
            dtype=grad_dtype,
        )

    def zero_grad(self):
        self.grad[...] = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size
