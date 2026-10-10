"""SiLU activation helpers shared by training and inference.

The implementation is branch-free and numerically stable.  The tanh form
avoids the boolean gather/scatter path that is especially expensive with CuPy.
"""

from mini_llm.backend import xp


def sigmoid_stable(x):
    """Fast branch-free sigmoid for NumPy/CuPy hot paths.

    For very negative FP16 inputs ``exp(-x)`` may overflow to ``inf``; the
    reciprocal still evaluates to the correct limiting value 0.  This avoids
    boolean gather/scatter and lets us benchmark exp versus the current tanh
    formulation on the target GPU.
    """
    return 1.0 / (1.0 + xp.exp(-x))


def silu(x):
    """SiLU(x) = x * sigmoid(x), without data-dependent indexing."""
    return x * sigmoid_stable(x)


def silu_prime(x):
    """Derivative of SiLU using the same stable sigmoid formulation."""
    s = sigmoid_stable(x)
    return s + x * s * (1.0 - s)
