"""0074A: observational disjoint-half gradient diagnostics.

Snapshots are read-only FP32 copies of small parameter *slices*, taken from
un-normalized cumulative gradients. The trainer's update is never touched.
"""

import csv
import math
import re
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, BACKEND_NAME
from mini_llm.diagnostics.adamw_update import ADAMW_FIELDS, group_update_rows


FIELDS = (
    "step", "accumulation_length", "parameter_group", "parameter_name",
    "parameter_count", "gradient_norm_a", "gradient_norm_b",
    "gradient_norm_full", "gradient_cosine", "gradient_difference_norm",
    "estimated_gradient_noise", "estimated_signal_norm_squared",
    "relative_gradient_uncertainty", "signal_positive", "valid_supervised_tokens",
    "nonzero_gradient_fraction_a", "nonzero_gradient_fraction_b",
    "monitor_overhead_ms",
)


def parse_accum_points(raw, accum_steps):
    """Validate nested, even-length comparison windows."""
    try:
        points = tuple(int(x.strip()) for x in str(raw).split(","))
    except ValueError as exc:
        raise ValueError("gradient-noise accumulation points must be integers") from exc
    if (not points or len(set(points)) != len(points)
            or tuple(sorted(points)) != points
            or any(n < 2 or n % 2 or n > accum_steps for n in points)):
        raise ValueError(
            "gradient-noise accumulation points must be distinct ascending "
            "even integers between 2 and grad_accum_steps"
        )
    return points


def _sample_slices(shape, limit=65536):
    """Contiguous-leading spatial subregion, not a random sketch."""
    extents = list(shape)
    count = 1
    for axis in range(len(shape) - 1, -1, -1):
        # Prefer complete trailing dimensions, then constrain leading rows.
        count *= int(shape[axis])
    if count <= limit:
        return tuple(slice(None) for _ in shape)
    # Keep a contiguous upper-left submatrix. Works with strided QKV views.
    if len(shape) == 1:
        return (slice(0, limit),)
    trailing = int(np.prod(shape[1:]))
    if trailing <= limit:
        return (slice(0, max(1, limit // trailing)),) + (slice(None),) * (len(shape) - 1)
    return (slice(0, 1), slice(0, min(limit, shape[1]))) + (slice(0, 1),) * (len(shape) - 2)


def select_parameters(parameters, categories="attention,moe,moe_router,memory,head"):
    """Choose one bounded slice per representative group from actual names."""
    requested = {item.strip() for item in categories.split(",") if item.strip()}
    allowed = {"attention", "moe", "moe_router", "memory", "head"}
    unknown = requested - allowed
    if unknown:
        raise ValueError(f"unknown gradient-noise groups: {sorted(unknown)}")
    parameters = list(parameters)
    named = {p.name: p for p in parameters if p.grad is not None}
    selected = []
    used = set()

    def pick(label, candidates, limit=65536):
        for name in candidates:
            if name in named and name not in used:
                p = named[name]
                selector = _sample_slices(p.grad.shape, limit)
                selected.append((label, p, selector))
                used.add(name)
                return

    if "attention" in requested:
        indexed = sorted(
            (int(m.group(1)), name) for name in named
            if (m := re.fullmatch(r"blocks\.(\d+)\.attention\.Wq", name))
        )
        if indexed:
            for label, pos in (("attention_early", 0),
                               ("attention_middle", len(indexed) // 2),
                               ("attention_final", len(indexed) - 1)):
                pick(label, [indexed[pos][1]])
    if "moe" in requested:
        pick("moe_expert", sorted(
            name for name in named if re.fullmatch(
                r"blocks\.\d+\.moe\.experts\.expert_0\.W_gate", name)))
    if "moe_router" in requested:
        pick("moe_router", sorted(
            name for name in named if re.fullmatch(
                r"blocks\.\d+\.moe\.router\.W_router", name)))
    if "memory" in requested:
        pick("memory_router", sorted(
            name for name in named if "memory_router.W_query" in name))
        pick("memory_reader", sorted(
            name for name in named if "memory_reader.W_q" in name))
    if "head" in requested:
        pick("lm_head", sorted(
            name for name in named if name.startswith("output_proj.")
            and len(named[name].grad.shape) >= 2), limit=32768)
    return selected


def _asfloat(v):
    return float(v.item()) if hasattr(v, "item") else float(v)


def half_statistics(first_sum, midpoint_sum, end_sum, n, *, loss_scale=1.0):
    """Disjoint half-means (equal-microbatch CE), with scale cancellation.

    Inputs G_{n/2}, G_n are *cumulative* gradients at fixed parameters.
    first_sum is the cumulative gradient at microbatch 0 (normally zero),
    permitting arbitrary adjacent windows in isolated synthetic tests.
    """
    if n < 2 or n % 2 or not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError("n must be positive even and loss_scale finite positive")
    factor = (2.0 / n) / float(loss_scale)
    a = (midpoint_sum - first_sum) * factor
    b = (end_sum - midpoint_sum) * factor
    aa = _asfloat(xp.sum(a * a, dtype="float64"))
    bb = _asfloat(xp.sum(b * b, dtype="float64"))
    dot = _asfloat(xp.sum(a * b, dtype="float64"))
    norm_a, norm_b = math.sqrt(max(aa, 0)), math.sqrt(max(bb, 0))
    # ||(a-b)/2||^2 is the full-mean estimation variance for IID halves.
    # Compute the difference directly: a+b cancellation loses the noise
    # estimate when two noisy half-means are nearly parallel.
    difference = a - b
    diff_sq = max(0.0, _asfloat(xp.sum(difference * difference, dtype="float64")))
    variance = diff_sq / 4.0
    full_sq = max(0.0, (aa + bb + 2.0 * dot) / 4.0)
    signal = full_sq - variance  # May be negative: never silently clip.
    finite = all(math.isfinite(x) for x in (aa, bb, dot, variance, signal))
    positive = finite and signal > 0
    return {
        "gradient_norm_a": norm_a, "gradient_norm_b": norm_b,
        "gradient_norm_full": math.sqrt(full_sq),
        "gradient_cosine": (dot / (norm_a * norm_b)
                            if norm_a > 0 and norm_b > 0 and finite else None),
        "gradient_difference_norm": math.sqrt(diff_sq),
        "estimated_gradient_noise": variance,
        "estimated_signal_norm_squared": signal,
        "relative_gradient_uncertainty": (math.sqrt(variance / signal)
                                          if positive else None),
        "signal_positive": positive,
        "nonzero_gradient_fraction_a": _asfloat(xp.mean(a != 0)),
        "nonzero_gradient_fraction_b": _asfloat(xp.mean(b != 0)),
    }


class GradientNoiseMonitor:
    """Collects only G_{n/2}, G_n at selected endpoints; never writes grads."""

    def __init__(self, parameters, *, accum_steps, accum_points="8,16,32",
                 categories="attention,moe,moe_router,memory,head", output="./logs/gradient_noise.csv",
                 interval=100, loss_scale=1.0,
                 adamw_optimizer=None, adamw_output=None, grad_clip=1.0):
        self.points = parse_accum_points(accum_points, accum_steps)
        self.accum_steps = int(accum_steps)
        if interval < 1:
            raise ValueError("gradient-noise interval must be >= 1")
        self.interval = int(interval)
        self.loss_scale = float(loss_scale)
        self.output = Path(output)
        self.adamw_optimizer = adamw_optimizer
        self.adamw_output = Path(adamw_output) if adamw_output is not None else None
        if self.adamw_output is not None and adamw_optimizer is None:
            raise ValueError("AdamW diagnostics require an optimizer")
        self.grad_clip = float(grad_clip)
        self.global_norms = {}
        self.adamw_rows = []
        self.categories = categories
        self.selected = select_parameters(parameters, categories)
        if not self.selected:
            raise ValueError("no matching trainable parameter groups for gradient-noise monitoring")
        self.boundaries = set(self.points)
        self.boundaries.update(n // 2 for n in self.points)
        self.snapshots = {}
        self.step = None
        self.valid_tokens = 0
        self.token_snapshots = {}
        self.overhead_s = 0.0

    def refresh(self, parameters):
        """Rebind after progressive-depth growth; called outside monitored cycle."""
        self.selected = select_parameters(parameters, self.categories)
        self.snapshots.clear()
        self.global_norms.clear()
        self.adamw_rows.clear()

    def begin(self, completed_step):
        # Completed step is 0-based; log the upcoming 1-based optimizer step.
        self.step = completed_step + 1 if (completed_step + 1) % self.interval == 0 else None
        self.snapshots.clear()
        self.global_norms.clear()
        self.adamw_rows.clear()
        self.token_snapshots.clear()
        self.valid_tokens = 0
        self.overhead_s = 0.0
        return self.step is not None

    def count_tokens(self, targets, loss_mask):
        if self.step is None:
            return
        start = time.perf_counter()
        if loss_mask is None:
            self.valid_tokens += int(targets.size)
        elif BACKEND_NAME == "numpy" or isinstance(loss_mask, np.ndarray):
            self.valid_tokens += int(np.count_nonzero(loss_mask))
        else:
            # Only routed or masked GPU paths reach this case; synchronize for
            # accurate counts. No RNG and no gradient/state mutation.
            self.valid_tokens += int(xp.count_nonzero(loss_mask).item())
        self.overhead_s += time.perf_counter() - start

    def capture(self, microbatches_completed):
        if self.step is None or microbatches_completed not in self.boundaries:
            return
        start = time.perf_counter()
        # Retain G_n for sampled slices. Final G_n can use live gradients in
        # finish(), so no extra copy is necessary at the last boundary.
        if microbatches_completed != self.accum_steps:
            self.snapshots[microbatches_completed] = [
                xp.array(p.grad[selector], dtype="float32", copy=True)
                for _, p, selector in self.selected
            ]
        self.token_snapshots[microbatches_completed] = self.valid_tokens
        if (self.adamw_output is not None
                and microbatches_completed in self.points):
            # Full-model gradient norm is required to reproduce clipping.
            # Only retain a scalar; never snapshot full parameter gradients.
            total = xp.asarray(0.0, dtype="float32")
            for param in self.adamw_optimizer.parameters:
                if param.grad is not None:
                    g = param.grad.astype("float32", copy=False)
                    total = total + xp.sum(g * g)
            self.global_norms[microbatches_completed] = xp.sqrt(total)
        self.overhead_s += time.perf_counter() - start

    def finish(self):
        if self.step is None:
            return []
        start = time.perf_counter()
        rows = []
        zero = None
        for n in self.points:
            mid = n // 2
            # G_n is available as live gradient only for the terminal point;
            # earlier endpoints are copied snapshots.
            for idx, (label, param, selector) in enumerate(self.selected):
                left = self.snapshots[mid][idx]
                if n == self.accum_steps:
                    right = param.grad[selector]
                else:
                    right = self.snapshots[n][idx]
                if zero is None or zero.shape != left.shape:
                    zero = xp.zeros_like(left, dtype="float32")
                metrics = half_statistics(zero, left, right, n, loss_scale=self.loss_scale)
                rows.append({
                    "step": self.step, "accumulation_length": n,
                    "parameter_group": label, "parameter_name": param.name,
                    "parameter_count": int(left.size),
                    "valid_supervised_tokens": self.token_snapshots[n],
                    **metrics,
                })
        if self.adamw_output is not None:
            self.adamw_rows = self._adamw_diagnostics()
        self.overhead_s += time.perf_counter() - start
        # Include instrumentation time; GPU kernel time is asynchronous and
        # is NOT fully represented by this host-side timer.
        for row in rows + self.adamw_rows:
            row["monitor_overhead_ms"] = 1000 * self.overhead_s
        self.snapshots.clear()
        return rows

    def _adamw_diagnostics(self):
        """Compare simulated optimizer updates before any real AdamW state change."""
        opt = self.adamw_optimizer
        indices = {id(p): i for i, p in enumerate(opt.parameters)}
        norms = {n: _asfloat(self.global_norms[n]) for n in self.points}
        rows = []
        for idx, (label, param, selector) in enumerate(self.selected):
            opt_idx = indices[id(param)]
            m = opt.m[opt_idx][selector]
            v = opt.v[opt_idx][selector]
            # Moments may live in pinned NumPy RAM or on CuPy GPU. Copy only
            # the bounded sampled coordinates to host and leave state intact.
            def host(value):
                return np.array(value.get() if hasattr(value, "get") else value,
                                dtype=np.float32, copy=True)
            sums = {
                n: host(param.grad[selector] if n == self.accum_steps
                        else self.snapshots[n][idx])
                for n in self.points
            }
            rows.extend(group_update_rows(
                step=self.step, label=label, name=param.name,
                points=self.points, snapshots=sums, m0=host(m), v0=host(v),
                optimizer=opt, optimizer_index=opt_idx, clip_norms=norms,
                grad_clip=self.grad_clip, loss_scale=self.loss_scale,
                token_snapshots=self.token_snapshots, learning_rate=opt.lr))
        return rows

    def commit(self, rows):
        """Emit only successful optimizer steps. Separate CSV from main log."""
        if not rows:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.output.exists() or self.output.stat().st_size == 0
        with self.output.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        if self.adamw_output is not None and self.adamw_rows:
            self.adamw_output.parent.mkdir(parents=True, exist_ok=True)
            header = (not self.adamw_output.exists()
                      or self.adamw_output.stat().st_size == 0)
            with self.adamw_output.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ADAMW_FIELDS)
                if header:
                    writer.writeheader()
                writer.writerows(self.adamw_rows)
        self.adamw_rows.clear()
