from ..init import matrix_parameter


class Linear:
    """Bias-free explicit matrix multiplication Y = X W."""

    def __init__(self, d_in, d_out, std, rng, name="linear", dtype=None, decay=True):
        self.W = matrix_parameter(
            (d_in, d_out), std, rng, f"{name}.W", dtype=dtype, decay=decay
        )

    def parameters(self):
        return [self.W]

    def forward(self, x):
        y = x @ self.W.data
        return y, {"x": x}

    def backward(self, dy, cache):
        x = cache["x"]
        x2 = x.reshape(-1, x.shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])
        self.W.grad += x2.T @ dy2
        return dy @ self.W.data.T
