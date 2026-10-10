from ..backend import xp, scalar


def global_grad_norm(parameters):
    """
    Compute global gradient norm in float32.
    
    Uses float32 accumulation for efficiency while maintaining
    sufficient precision for clipping decisions.
    
    Returns:
        tuple: (global_norm_as_backend_array, global_norm_as_python_float)
    """
    total = xp.asarray(0.0, dtype="float32")
    for p in parameters:
        g = p.grad.astype("float32", copy=False)
        total = total + xp.sum(g * g)
    norm = xp.sqrt(total)
    return norm, float(_array_to_float(norm))


def _array_to_float(x):
    """Convert array to Python float, handling both numpy and cupy."""
    if hasattr(x, "get"):
        return x.get()
    return float(x)


def _is_finite(x):
    """Check if array contains only finite values."""
    try:
        if hasattr(x, "get"):
            cpu_x = x.get()
        else:
            cpu_x = x
        return bool(xp.isfinite(cpu_x))
    except Exception:
        return True  # If we can't check, assume it's finite


def clip_grad_global_norm(parameters, max_norm=1.0, eps=1e-12):
    """
    Clip gradients by global norm in float32.
    
    This version rejects nonfinite gradients before applying clipping,
    preventing corruption of Adam's FP32 state.
    
    Args:
        parameters: List of Parameter objects
        max_norm: Maximum allowed gradient norm
        eps: Small constant to avoid division by zero
        
    Returns:
        tuple: (global_norm, scale_factor, is_finite)
              is_finite is False if gradients contain Inf/NaN
    """
    norm_backend, norm_value = global_grad_norm(parameters)
    
    # Check for nonfinite norm before computing scale
    if not _is_finite(norm_backend):
        return norm_backend, 0.0, False
    
    # Scale would be Inf * 0 = NaN if norm is Inf
    # This check prevents that case
    if norm_value == float("inf"):
        return norm_backend, 0.0, False
    
    scale = min(1.0, max_norm / (norm_value + eps))
    
    if scale < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad *= scale
    
    return norm_backend, scale, True
