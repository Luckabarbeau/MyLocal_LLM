"""0084E: observational within-accumulation gradient convergence.

This monitor is independent of the older 0074A disjoint-half estimator.  It
never writes model gradients, changes sampling, calls backward, or changes the
optimizer.  Only bounded slices of selected parameter gradients are copied.

Let delta_i denote the cumulative *scaled* gradient increment contributed by
microbatch i, and let w_i denote that microbatch's actual objective weight.
With W=sum(w_i), R=sum(delta_i), and Q=sum(||delta_i||²), the following
quantities are observable without individual backward replays:

    ||g_bar||² = ||R||² / (loss_scale W)²
    random_sign_reference² = Q / (loss_scale W)²
    signal_estimate² = (||R||² - Q) /
                       (loss_scale² (W² - sum(w_i²)))

The final expression is an unbiased estimator of ||E[g_i]||² when the
microbatch *mean gradients* are independent with the same expectation.  It
can be negative.  Tokens/documents are correlated in real pretraining; this
is an observational diagnostic, not a certified gradient-noise model.
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

from mini_llm.backend import xp
from mini_llm.diagnostics.gradient_noise import select_parameters, _asfloat


FIELDS = (
    "step", "accumulation_length", "parameter_group", "parameter_name",
    "parameter_count", "valid_supervised_tokens", "microbatch_weight_sum",
    "effective_microbatches", "gradient_mean_norm", "random_reference_norm",
    "estimated_signal_norm_squared", "estimated_signal_norm",
    "signal_positive", "estimated_noise_variance_of_mean",
    "noise_to_signal_squared", "actual_to_random_norm_ratio",
    "cosine_to_final_gradient", "random_cosine_to_final_reference",
    "monitor_overhead_ms",
)


def parse_convergence_points(raw, accum_steps):
    """Ascending unique positive accumulation counts (the end is automatic)."""
    if str(raw).strip().lower() == "auto":
        points = []
        k = 2
        while k <= accum_steps:
            points.append(k)
            k *= 2
        return tuple(points if points and points[-1] == accum_steps
                     else points + [accum_steps])
    if isinstance(raw, (list, tuple)):
        chunks = raw
    else:
        chunks = str(raw).split(",")
    try:
        pts = tuple(int(str(x).strip()) for x in chunks)
    except (TypeError, ValueError) as exc:
        raise ValueError("gradient convergence accumulation points must be integers") from exc
    if (not pts or any(n < 1 or n > accum_steps for n in pts)
            or len(set(pts)) != len(pts) or tuple(sorted(pts)) != pts):
        raise ValueError("gradient convergence points must be distinct ascending "
                         "integers from 1 to grad_accum_steps")
    return pts if pts[-1] == accum_steps else pts + (accum_steps,)


def convergence_statistics(raw_squared_norm, random_squared_norm,
                           weighted_individual_squared, weight_sum,
                           weight_square_sum, loss_scale):
    """Compute weighted diagnostics from sufficient statistics.

    raw_squared_norm = ||sum_i delta_i||²,
    random_squared_norm = sum_i ||delta_i||²,
    weighted_individual_squared = sum_i ||delta_i||² / w_i.
    No value is clipped except display-only sqrt and nonnegative variance.
    """
    if not (math.isfinite(weight_sum) and weight_sum > 0
            and math.isfinite(weight_square_sum) and weight_square_sum > 0
            and math.isfinite(loss_scale) and loss_scale > 0):
        raise ValueError("weights and loss scale must be finite and positive")
    r2 = float(raw_squared_norm)
    q2 = float(random_squared_norm)
    t2 = float(weighted_individual_squared)
    scale_sq = float(loss_scale) ** 2
    wsq = weight_sum * weight_sum
    norm_sq = max(0.0, r2 / (scale_sq * wsq))
    random_sq = max(0.0, q2 / (scale_sq * wsq))
    distinct_weight = wsq - weight_square_sum
    signal = ((r2 - q2) / (scale_sq * distinct_weight)
              if distinct_weight > 0 else None)
    # Weighted sample variance of *microbatch mean gradients*, using Bessel's
    # correction for unequal weights.  The mean's variance is sigma²/n_eff.
    degrees = weight_sum - weight_square_sum / weight_sum
    sigma2 = (max(0.0, (t2 - r2 / weight_sum) / (scale_sq * degrees))
              if degrees > 0 else None)
    noise_mean = (sigma2 * weight_square_sum / wsq if sigma2 is not None else None)
    positive = signal is not None and math.isfinite(signal) and signal > 0
    return {
        "gradient_mean_norm": math.sqrt(norm_sq),
        "random_reference_norm": math.sqrt(random_sq),
        "estimated_signal_norm_squared": signal,
        "estimated_signal_norm": math.sqrt(signal) if positive else None,
        "signal_positive": positive,
        "estimated_noise_variance_of_mean": noise_mean,
        "noise_to_signal_squared": (noise_mean / signal if positive else None),
        "actual_to_random_norm_ratio": (math.sqrt(norm_sq / random_sq)
                                        if random_sq > 0 else None),
        "effective_microbatches": wsq / weight_square_sum,
    }


class GradientConvergenceMonitor:
    """Observe sampled parameter-gradient slices during an existing update."""

    def __init__(self, parameters, *, accum_steps,
                 accum_points="auto", categories="attention,ffn",
                 interval=100, output="./logs/gradient_convergence.csv",
                 loss_scale=1.0):
        self.accum_steps = int(accum_steps)
        self.points = parse_convergence_points(accum_points, self.accum_steps)
        if interval < 1 or not math.isfinite(float(loss_scale)) or loss_scale <= 0:
            raise ValueError("interval and loss_scale must be positive")
        self.interval = int(interval)
        self.loss_scale = float(loss_scale)
        self.output = Path(output)
        self.categories = categories
        self.selected = select_parameters(parameters, categories)
        if not self.selected:
            raise ValueError("no matching trainable matrices for gradient convergence")
        self.step = None
        self._clear()

    def _clear(self):
        self.previous = []
        self.q = []
        self.weighted_q = []
        self.snapshots = {}
        self.weight_sum = 0.0
        self.weight_square_sum = 0.0
        self.valid_tokens = 0
        self.microbatches_seen = 0
        self.overhead_s = 0.0

    def refresh(self, parameters):
        """Rebind selected parameters when progressive growth replaces them."""
        self.selected = select_parameters(parameters, self.categories)
        if not self.selected:
            raise ValueError("no matching trainable matrices for gradient convergence")
        self._clear()

    def begin(self, completed_step):
        self.step = (completed_step + 1 if (completed_step + 1) % self.interval == 0
                     else None)
        self._clear()
        if self.step is not None:
            self.previous = [xp.zeros_like(p.grad[sel], dtype="float32")
                             for _, p, sel in self.selected]
            self.q = [xp.asarray(0.0, dtype="float64") for _ in self.selected]
            self.weighted_q = [xp.asarray(0.0, dtype="float64")
                               for _ in self.selected]
        return self.step is not None

    def capture(self, microbatches_completed, *, microbatch_weight, valid_tokens):
        """Call once after each backward and before final gradient normalization."""
        if self.step is None:
            return
        start = time.perf_counter()
        if microbatches_completed != self.microbatches_seen + 1:
            raise ValueError("gradient convergence captures must be consecutive")
        weight = float(microbatch_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("microbatch objective weight must be finite nonnegative")
        self.microbatches_seen = microbatches_completed
        self.weight_sum += weight
        self.weight_square_sum += weight * weight
        self.valid_tokens += int(valid_tokens)
        is_point = microbatches_completed in self.points
        snapshot = []
        for index, (_, p, sl) in enumerate(self.selected):
            # Read/copy only the current selected gradient slice.  The original
            # p.grad is never overwritten or normalized by this monitor.
            current = xp.array(p.grad[sl], dtype="float32", copy=True)
            delta = current - self.previous[index]
            squared = xp.sum(delta * delta, dtype="float64")
            self.q[index] = self.q[index] + squared
            if weight > 0:
                self.weighted_q[index] = self.weighted_q[index] + squared / weight
            self.previous[index] = current
            if is_point:
                snapshot.append((current, self.q[index], self.weighted_q[index]))
        if is_point:
            self.snapshots[microbatches_completed] = (
                snapshot, self.weight_sum, self.weight_square_sum,
                self.valid_tokens,
            )
        self.overhead_s += time.perf_counter() - start

    def finish(self):
        if self.step is None:
            return []
        if self.microbatches_seen != self.accum_steps:
            raise ValueError("incomplete gradient convergence accumulation")
        start = time.perf_counter()
        final = self.snapshots[self.accum_steps][0]
        # Transfer only scalar results to CPU after accumulation.  The
        # selected vectors stay on GPU; no extra model forward/backward.
        rows = []
        for point in self.points:
            snapshots, weight, weights_sq, tokens = self.snapshots[point]
            for idx, (label, p, _) in enumerate(self.selected):
                current, q, weighted_q = snapshots[idx]
                full, full_q, _ = final[idx]
                n2 = _asfloat(xp.sum(current * current, dtype="float64"))
                q2 = _asfloat(q)
                wq2 = _asfloat(weighted_q)
                if weight > 0:
                    stats = convergence_statistics(
                        n2, q2, wq2, weight, weights_sq, self.loss_scale,
                    )
                else:
                    # No supervised target has contributed yet.  A zero
                    # gradient is not evidence of a converged gradient.
                    stats = {
                        "gradient_mean_norm": None,
                        "random_reference_norm": None,
                        "estimated_signal_norm_squared": None,
                        "estimated_signal_norm": None,
                        "signal_positive": False,
                        "estimated_noise_variance_of_mean": None,
                        "noise_to_signal_squared": None,
                        "actual_to_random_norm_ratio": None,
                        "effective_microbatches": 0.0,
                    }
                denom = math.sqrt(max(0.0, n2) * max(
                    0.0, _asfloat(xp.sum(full * full, dtype="float64"))))
                cosine = (_asfloat(xp.sum(current * full, dtype="float64")) / denom
                          if denom > 0 else None)
                full_q_scalar = _asfloat(full_q)
                reference = (math.sqrt(max(0.0, q2 / full_q_scalar))
                             if full_q_scalar > 0 else None)
                rows.append({
                    "step": self.step, "accumulation_length": point,
                    "parameter_group": label, "parameter_name": p.name,
                    "parameter_count": int(current.size),
                    "valid_supervised_tokens": tokens,
                    "microbatch_weight_sum": weight,
                    **stats,
                    "cosine_to_final_gradient": cosine,
                    # Approximation for random independent high-dimensional
                    # increments with matched observed squared magnitudes.
                    "random_cosine_to_final_reference": reference,
                })
        self.overhead_s += time.perf_counter() - start
        for row in rows:
            row["monitor_overhead_ms"] = self.overhead_s * 1000
        self._clear()
        return rows

    def commit(self, rows):
        if not rows:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        need_header = not self.output.exists() or self.output.stat().st_size == 0
        if not need_header:
            with self.output.open(newline="") as f:
                old_header = next(csv.reader(f), [])
            if tuple(old_header) != FIELDS:
                raise ValueError("existing gradient convergence CSV header differs: "
                                 f"{self.output}")
        with self.output.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if need_header:
                writer.writeheader()
            writer.writerows(rows)
