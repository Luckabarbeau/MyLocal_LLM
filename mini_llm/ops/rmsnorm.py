from ..backend import xp
from ..init import ones_parameter


class RMSNorm:
    """
    y = gamma * x / sqrt(mean(x^2) + eps)
    """

    def __init__(self, d_model, eps=1e-6, name="rmsnorm", dtype="float32"):
        self.eps = float(eps)
        self.gamma = ones_parameter((d_model,), f"{name}.gamma", dtype=dtype, decay=False)

    def parameters(self):
        return [self.gamma]

    def forward(self, x):
        mean_sq = xp.mean(x * x, axis=-1, keepdims=True)
        inv_rms = 1.0 / xp.sqrt(mean_sq + self.eps)
        x_hat = x * inv_rms
        y = x_hat * self.gamma.data
        return y, {"x": x, "x_hat": x_hat, "inv_rms": inv_rms}

    def backward(self, dy, cache):
        x = cache["x"]
        x_hat = cache["x_hat"]
        inv_rms = cache["inv_rms"]
        d = x.shape[-1]

        reduce_axes = tuple(range(dy.ndim - 1))
        self.gamma.grad += xp.sum(dy * x_hat, axis=reduce_axes)

        dx_hat = dy * self.gamma.data
        projection = xp.sum(dx_hat * x, axis=-1, keepdims=True)
        return dx_hat * inv_rms - x * (inv_rms ** 3) * projection / float(d)
