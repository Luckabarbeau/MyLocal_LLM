from dataclasses import dataclass
from .backend import xp


@dataclass
class Parameter:
    data: object
    name: str = ""
    decay: bool = True

    def __post_init__(self):
        self.grad = xp.zeros_like(self.data)

    def zero_grad(self):
        self.grad[...] = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def default_dtype(self):
        return "float32"
