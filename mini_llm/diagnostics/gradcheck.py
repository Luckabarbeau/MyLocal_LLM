from ..backend import xp, scalar


def directional_derivative_check(
    scalar_function, x, analytic_grad, direction=None, eps=1e-5, seed=123
):
    if direction is None:
        rng = xp.random.RandomState(seed)
        direction = rng.normal(0.0, 1.0, size=x.shape).astype(x.dtype)

    direction = direction / xp.sqrt(xp.sum(direction * direction))

    f_plus = scalar_function(x + eps * direction)
    f_minus = scalar_function(x - eps * direction)
    fd = (float(f_plus) - float(f_minus)) / (2.0 * eps)
    adjoint = scalar(xp.sum(analytic_grad * direction))
    rel = abs(fd - adjoint) / (abs(fd) + abs(adjoint) + 1e-12)

    return {
        "finite_difference": fd,
        "analytic": adjoint,
        "relative_error": rel,
    }
