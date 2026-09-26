from ..backend import xp
from ..init import matrix_parameter


def sigmoid(x):
    return 1.0 / (1.0 + xp.exp(-x))


def silu(x):
    return x * sigmoid(x)


def silu_prime(x):
    s = sigmoid(x)
    return s + x * s * (1.0 - s)


class SwiGLU:
    """
    G = X W_gate
    U = X W_up
    A = SiLU(G)
    H = A * U
    Y = H W_down
    """

    def __init__(
        self, d_model, d_ff, input_std, output_std, rng,
        name="swiglu", dtype=None
    ):
        if dtype is None:
            dtype = "float32"
            
        self.W_gate = matrix_parameter(
            (d_model, d_ff), input_std, rng, f"{name}.W_gate", dtype=dtype
        )
        self.W_up = matrix_parameter(
            (d_model, d_ff), input_std, rng, f"{name}.W_up", dtype=dtype
        )
        self.W_down = matrix_parameter(
            (d_ff, d_model), output_std, rng, f"{name}.W_down", dtype=dtype
        )

    def parameters(self):
        return [self.W_gate, self.W_up, self.W_down]

    def forward(self, x, return_cache=True):
        g = x @ self.W_gate.data
        u = x @ self.W_up.data
        a = silu(g)
        h = a * u
        y = h @ self.W_down.data
        
        if not return_cache:
            return y
            
        return y, {"x": x, "g": g, "u": u, "a": a, "h": h}

    def backward(self, dy, cache):
        x, g, u, a, h = (
            cache["x"], cache["g"], cache["u"], cache["a"], cache["h"]
        )

        x2 = x.reshape(-1, x.shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])
        h2 = h.reshape(-1, h.shape[-1])

        self.W_down.grad += h2.T @ dy2
        dh = dy @ self.W_down.data.T

        da = dh * u
        du = dh * a
        dg = da * silu_prime(g)

        self.W_gate.grad += x2.T @ dg.reshape(-1, dg.shape[-1])
        self.W_up.grad += x2.T @ du.reshape(-1, du.shape[-1])

        return dg @ self.W_gate.data.T + du @ self.W_up.data.T
