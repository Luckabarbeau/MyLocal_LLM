"""0074B: CPU-side, read-only AdamW counterfactuals for sampled coordinates.

Never mutates the optimizer or production gradients. The returned *positive*
update direction u is the magnitude subtracted from FP32 master weights;
decoupled weight decay is deliberately excluded from direction comparisons.
"""

import math
import numpy as np


ADAMW_FIELDS = (
    "step", "accumulation_length", "reference_accumulation", "parameter_group",
    "parameter_name", "parameter_count", "valid_supervised_tokens",
    "global_gradient_norm", "global_clip_scale", "adamw_local_step",
    "adamw_update_norm", "adamw_update_cosine_to_reference",
    "adamw_relative_update_difference", "gradient_momentum_cosine",
    "innovation_norm", "innovation_over_update_norm",
    "heldout_gradient_cosine", "heldout_first_order_gain",
    "heldout_direction_cosine", "monitor_overhead_ms",
)


def _cosine(a, b):
    aa = float(np.dot(a.ravel().astype(np.float64), a.ravel().astype(np.float64)))
    bb = float(np.dot(b.ravel().astype(np.float64), b.ravel().astype(np.float64)))
    if not math.isfinite(aa * bb) or aa <= 0 or bb <= 0:
        return None
    dot = float(np.dot(a.ravel().astype(np.float64), b.ravel().astype(np.float64)))
    return dot / math.sqrt(aa * bb)


def counterfactual_direction(gradient, m0, v0, *, beta1, beta2, eps,
                             learning_rate, local_step, clip_scale=1.0):
    """Exact AdamW algebra for a selected FP32 parameter slice.

    Uses the repository's folded bias correction and evaluates the *next*
    parameter-local step. No weight decay, by design. Gradient clipping is
    applied before updating moments; m0/v0 are never changed.
    """
    if local_step <= 0:
        raise ValueError("AdamW parameter-local step must be positive")
    g = np.asarray(gradient, dtype=np.float32) * np.float32(clip_scale)
    m = np.asarray(m0, dtype=np.float32) * beta1 + (1.0 - beta1) * g
    v = np.asarray(v0, dtype=np.float32) * beta2 + (1.0 - beta2) * (g * g)
    c1 = 1.0 - beta1 ** local_step
    sqrt_c2 = math.sqrt(1.0 - beta2 ** local_step)
    step_size = learning_rate * sqrt_c2 / c1
    eps_scaled = eps * sqrt_c2
    return (m / (np.sqrt(v) + eps_scaled) * step_size).astype(np.float32)


def group_update_rows(*, step, label, name, points, snapshots, m0, v0,
                      optimizer, optimizer_index, clip_norms, grad_clip,
                      loss_scale, token_snapshots, learning_rate):
    """Compare prefix updates with max-point update + independent suffix.

    `snapshots` maps n -> *unnormalized* cumulative gradient G_n for a
    sampled parameter slice. `clip_norms` contains full-model ||G_n||, NOT
    slice norms; global clipping is thus reproduced exactly up to reduction
    rounding. Held-out suffixes are disjoint from their candidate prefixes.
    """
    reference_n = max(points)
    next_local_step = (int(optimizer.step_index) + 1
                       - int(optimizer.parameter_birth_steps[optimizer_index]))
    if next_local_step <= 0:
        raise ValueError("selected optimizer parameter not yet active")
    m0 = np.asarray(m0, dtype=np.float32)
    v0 = np.asarray(v0, dtype=np.float32)
    sums = {n: np.asarray(snapshots[n], dtype=np.float32) for n in points}
    unclipped = {n: sums[n] / np.float32(n * loss_scale) for n in points}
    scales = {}
    directions = {}
    for n in points:
        norm = clip_norms[n] / (n * loss_scale)
        # Identical eps=1e-12 and clip convention to clip_grad_global_norm.
        scales[n] = min(1.0, grad_clip / (norm + 1e-12))
        directions[n] = counterfactual_direction(
            unclipped[n], m0, v0, beta1=optimizer.beta1,
            beta2=optimizer.beta2, eps=optimizer.eps,
            learning_rate=learning_rate, local_step=next_local_step,
            clip_scale=scales[n])
    zero = np.zeros_like(m0)
    inherited = counterfactual_direction(
        zero, m0, v0, beta1=optimizer.beta1, beta2=optimizer.beta2,
        eps=optimizer.eps, learning_rate=learning_rate,
        local_step=next_local_step)
    reference = directions[reference_n]
    reference_norm = float(np.linalg.norm(reference.ravel().astype(np.float64)))
    rows = []
    for n in points:
        direction = directions[n]
        direction_norm = float(np.linalg.norm(direction.ravel().astype(np.float64)))
        diff = float(np.linalg.norm((direction - reference).ravel().astype(np.float64)))
        innovation = float(np.linalg.norm((direction - inherited).ravel().astype(np.float64)))
        # For n < reference_n, the suffix gradient is evaluated at identical
        # parameters but on microbatches disjoint from the candidate prefix.
        # At n=reference_n, no disjoint suffix exists: report unavailable.
        suffix = ((sums[reference_n] - sums[n]) /
                  np.float32((reference_n - n) * loss_scale)
                  if n < reference_n else None)
        heldout_cos = _cosine(direction, suffix) if suffix is not None else None
        gain = (float(np.dot(direction.ravel().astype(np.float64),
                             suffix.ravel().astype(np.float64)))
                if suffix is not None else None)
        rows.append({
            "step": step, "accumulation_length": n,
            "reference_accumulation": reference_n,
            "parameter_group": label, "parameter_name": name,
            "parameter_count": int(direction.size),
            "valid_supervised_tokens": token_snapshots[n],
            "global_gradient_norm": clip_norms[n] / (n * loss_scale),
            "global_clip_scale": scales[n],
            "adamw_local_step": next_local_step,
            "adamw_update_norm": direction_norm,
            "adamw_update_cosine_to_reference": _cosine(direction, reference),
            "adamw_relative_update_difference": (
                diff / reference_norm if reference_norm > 0 else None),
            "gradient_momentum_cosine": _cosine(unclipped[n], m0),
            "innovation_norm": innovation,
            "innovation_over_update_norm": (
                innovation / direction_norm if direction_norm > 0 else None),
            "heldout_gradient_cosine": _cosine(unclipped[n], suffix)
                if suffix is not None else None,
            "heldout_first_order_gain": gain,
            "heldout_direction_cosine": heldout_cos,
        })
    return rows
