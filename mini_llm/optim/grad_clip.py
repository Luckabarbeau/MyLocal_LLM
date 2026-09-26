from ..backend import xp, scalar


def global_grad_norm(parameters):
    total = xp.asarray(0.0, dtype="float64")
    for p in parameters:
        g = p.grad.astype("float64", copy=False)
        total = total + xp.sum(g * g)
    return scalar(xp.sqrt(total))


def clip_grad_global_norm(parameters, max_norm=1.0, eps=1e-12):
    norm = global_grad_norm(parameters)
    scale = min(1.0, float(max_norm) / (norm + float(eps)))
    if scale < 1.0:
        for p in parameters:
            p.grad *= scale
    return norm, scale
