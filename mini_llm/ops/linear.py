from ..init import matrix_parameter


class Linear:
    """Bias-free explicit matrix multiplication Y = X W."""

    def __init__(self, d_in, d_out, std, rng, name="linear", dtype="float32", decay=True):
        self.W = matrix_parameter(
            (d_in, d_out), std, rng, f"{name}.W", dtype=dtype, decay=decay
        )

    def parameters(self):
        return [self.W]

    def forward(self, x):
        # CuPy 14 currently routes N-D @ 2-D BF16 operands through a generic
        # batched tensordot path that does not understand the BF16 dtype code.
        # Flatten all leading dimensions so projections always use the regular
        # 2-D GEMM path (which supports BF16 efficiently), then restore shape.
        x2 = x.reshape(-1, x.shape[-1])
        y2 = x2 @ self.W.data
        y = y2.reshape(*x.shape[:-1], self.W.data.shape[1])
        return y, {"x": x}

    def backward(self, dy, cache):
        x = cache["x"]
        x2 = x.reshape(-1, x.shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])
        self.W.grad += x2.T @ dy2
        dx2 = dy2 @ self.W.data.T
        return dx2.reshape(x.shape)

    def backward_input(self, dy, cache):
        """Backward only into the input when this projection is frozen.

        Router-only post-training needs the gradient through the large LM head
        but never updates its ~50M weights. Avoiding ``x.T @ dy`` saves one large
        gradient GEMM and a full output-head gradient write per microbatch.
        """
        x = cache["x"]
        dy2 = dy.reshape(-1, dy.shape[-1])
        dx2 = dy2 @ self.W.data.T
        return dx2.reshape(x.shape)
