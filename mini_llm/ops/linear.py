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


class _TransposedParameterView:
    """Non-owning output-head view of a registered embedding Parameter.

    Keep ``data`` and ``grad`` as fresh transpose views so optimizer/master
    weight updates to the underlying embedding cannot leave a stale copy.
    """

    def __init__(self, embedding_weight):
        self.embedding_weight = embedding_weight
        self.name = "output_proj.W(tied)"

    @property
    def data(self):
        return self.embedding_weight.data.T

    @property
    def grad(self):
        return self.embedding_weight.grad.T


class TiedOutputProjection(Linear):
    """Bias-free output head whose weights are exactly embedding.W.T.

    Inherited Linear forward/backward (and the chunked head in DecoderLM)
    add their output-head gradients into the SAME registered embedding grad.
    The model parameter list excludes this view: there is one optimizer state,
    one checkpoint tensor, and one weight decay update.
    """

    def __init__(self, embedding_weight):
        self.W = _TransposedParameterView(embedding_weight)

    def parameters(self):
        return []

    def backward(self, dy, cache):
        # Accumulate through the underlying *registered* embedding Parameter.
        # Avoid ``self.W.grad +=``: this is a read-only non-owning property.
        x = cache["x"]
        x2 = x.reshape(-1, x.shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])
        weight = self.W.embedding_weight
        weight.grad += (x2.T @ dy2).T
        return (dy2 @ weight.data).reshape(x.shape)
