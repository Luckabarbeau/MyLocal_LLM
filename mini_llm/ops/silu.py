"""SiLU activation helpers shared by training and inference.

The implementation is branch-free and numerically stable.  The tanh form
avoids the boolean gather/scatter path that is especially expensive with CuPy.
"""

from mini_llm.backend import xp


def sigmoid_stable(x):
    """Stable sigmoid using tanh, preserving the input array dtype."""
    return 0.5 * (1.0 + xp.tanh(0.5 * x))


def silu(x):
    """SiLU(x) = x * sigmoid(x), without data-dependent indexing."""
    return x * sigmoid_stable(x)


def silu_prime(x):
    """Derivative of SiLU using the same stable sigmoid formulation."""
    s = sigmoid_stable(x)
    return s + x * s * (1.0 - s)
