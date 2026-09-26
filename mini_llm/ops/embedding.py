from ..backend import xp
from ..init import matrix_parameter


class Embedding:
    def __init__(self, vocab_size, d_model, std, rng, name="embedding", dtype=None):
        self.W = matrix_parameter(
            (vocab_size, d_model), std, rng, f"{name}.W",
            dtype=dtype, decay=False,
        )

    def parameters(self):
        return [self.W]

    def forward(self, token_ids):
        return self.W.data[token_ids], {"token_ids": token_ids}

    def backward(self, dy, cache):
        ids = cache["token_ids"].reshape(-1)
        dy2 = dy.reshape(-1, dy.shape[-1])
        xp.add.at(self.W.grad, ids, dy2)
        return None
