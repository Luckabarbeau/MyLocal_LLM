from ..backend import xp, scalar


def global_grad_norm(parameters):
    """
    Compute global gradient norm in float32.
    
    Uses float32 accumulation for efficiency while maintaining
    sufficient precision for clipping decisions.
    """
    total = xp.asarray(0.0, dtype="float32")
    for p in parameters:
        g = p.grad.astype("float32", copy=False)
        total = total + xp.sum(g * g)
    return scalar(xp.sqrt(total))


def clip_grad_global_norm(parameters, max_norm=1.0, eps=1e-12):
    """
    Clip gradients by global norm in float32.
    
    Args:
        parameters: List of Parameter objects
        max_norm: Maximum allowed gradient norm
        eps: Small constant to avoid division by zero
        
    Returns:
        tuple: (global_norm, scale_factor)
    """
    norm = global_grad_norm(parameters)
    scale = min(1.0, float(max_norm) / (norm + float(eps)))
    if scale < 1.0:
        for p in parameters:
            p.grad *= scale
    return norm, scale
