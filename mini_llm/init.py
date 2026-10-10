from .backend import xp, RandomStream, resolve_dtype
from .parameter import Parameter


def clipped_normal(shape, std, rng: RandomStream, cutoff=3.0, dtype="float32"):
    """
    Normal initialization followed by clipping at +/- cutoff * std.

    This is intentionally explicit. It is not rejection-sampled truncation.
    """
    target_dtype = resolve_dtype(dtype)
    x = rng.normal(shape, std=std, dtype=target_dtype)
    limit = float(cutoff) * float(std)
    # Some backends promote BF16 when clipping against Python scalars. Cast
    # back explicitly so parameter storage always matches the requested dtype.
    return xp.clip(x, -limit, limit).astype(target_dtype, copy=False)


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
    return Parameter(xp.ones(shape, dtype=resolve_dtype(dtype)), name=name, decay=decay)
