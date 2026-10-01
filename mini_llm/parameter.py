from dataclasses import dataclass
from .backend import xp


@dataclass
class Parameter:
    data: object
    name: str = ""
    decay: bool = True

    def __post_init__(self):
        # Use FP32 gradients for FP16 parameters to prevent overflow
        grad_dtype = (
            "float32"
            if self.data.dtype == xp.float16
            else self.data.dtype
        )
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
