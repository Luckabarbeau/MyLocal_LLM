import math

from ..backend import xp


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


def audit_gradients(parameters, top_k=8):
    """Read-only gradient health report using a scale-stable L2 norm.

    This intentionally runs only when explicitly requested after a rejected
    update. It distinguishes nonfinite gradient elements from overflow in the
    normal FP32 sum-of-squares reduction.
    """
    rows = []
    nonfinite = []
    stable_norm = 0.0
    parameter_count = 0
    element_count = 0

    for index, parameter in enumerate(parameters):
        gradient = parameter.grad
        if gradient is None:
            continue
        parameter_count += 1
        element_count += int(gradient.size)
        gradient32 = gradient.astype("float32", copy=False)
        finite_mask = xp.isfinite(gradient32)
        finite_count = int(_array_to_float(xp.sum(finite_mask)))
        nonfinite_count = int(gradient32.size) - finite_count
        name = parameter.name or f"parameter_{index}"

        if nonfinite_count:
            nan_count = int(_array_to_float(xp.sum(xp.isnan(gradient32))))
            inf_count = int(_array_to_float(xp.sum(xp.isinf(gradient32))))
            finite_abs = xp.where(finite_mask, xp.abs(gradient32), 0.0)
            max_finite_abs = float(_array_to_float(xp.max(finite_abs)))
            nonfinite.append({
                "name": name,
                "nonfinite": nonfinite_count,
                "nan": nan_count,
                "inf": inf_count,
                "max_finite_abs": max_finite_abs,
            })
            continue

        max_abs = float(_array_to_float(xp.max(xp.abs(gradient32))))
        if max_abs == 0.0:
            tensor_norm = 0.0
        else:
            scaled = gradient32 / max_abs
            scaled_sum_squares = float(_array_to_float(xp.sum(scaled * scaled)))
            tensor_norm = max_abs * math.sqrt(scaled_sum_squares)
        stable_norm = math.hypot(stable_norm, tensor_norm)
        rows.append({"name": name, "norm": tensor_norm, "max_abs": max_abs})

    rows.sort(key=lambda row: row["norm"], reverse=True)
    return {
        "parameter_count": parameter_count,
        "element_count": element_count,
        "all_finite": not nonfinite,
        "stable_norm": stable_norm if not nonfinite else None,
        "nonfinite": nonfinite,
        "top": rows[:max(0, int(top_k))],
    }


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
