from .backend import xp, RandomStream
from .parameter import Parameter


def clipped_normal(shape, std, rng: RandomStream, cutoff=3.0, dtype="float32"):
    """
    Normal initialization followed by clipping at +/- cutoff * std.

    This is intentionally explicit. It is not rejection-sampled truncation.
    """
    x = rng.normal(shape, std=std, dtype=dtype)
    limit = float(cutoff) * float(std)
    return xp.clip(x, -limit, limit)


def matrix_parameter(
    shape,
    std,
    rng,
    name,
    cutoff=3.0,
    dtype="float32",
    decay=True,
):
    return Parameter(
        clipped_normal(shape, std, rng, cutoff=cutoff, dtype=dtype),
        name=name,
        decay=decay,
    )


def ones_parameter(shape, name, dtype="float32", decay=False):
    return Parameter(xp.ones(shape, dtype=dtype), name=name, decay=decay)
