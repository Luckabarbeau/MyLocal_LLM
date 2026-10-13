"""0079L: observational gradient-noise diagnostics for token-weighted refinement.

Unlike 0074A's equal-microbatch estimator, refinement's gradients already
contain each microbatch's factor n_valid / N_total. Normalize disjoint halves
by *their own* valid-target count to avoid bias from short documents. Noise
estimates are descriptive IID-token proxies, not an assumption of independent
language tokens. Sample only bounded parameter slices and never mutate grads.
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, asnumpy
from mini_llm.diagnostics.gradient_noise import FIELDS, select_parameters, parse_accum_points
from mini_llm.diagnostics.adamw_update import (
    ADAMW_FIELDS, counterfactual_direction, _cosine,
)


def weighted_half_statistics(midpoint, final, valid_a, valid_b, *, total_valid):
    if valid_a <= 0 or valid_b <= 0 or valid_a + valid_b > total_valid:
        raise ValueError('disjoint halves need positive valid-token counts')
    # Inputs were globally token-normalized by total_valid before accumulating.
    first = midpoint.astype(xp.float32, copy=False) * (total_valid / valid_a)
    second = (final - midpoint).astype(xp.float32, copy=False) * (total_valid / valid_b)
    aa = float(xp.sum(first * first, dtype=xp.float64))
    bb = float(xp.sum(second * second, dtype=xp.float64))
    cross = float(xp.sum(first * second, dtype=xp.float64))
    delta = first - second
    dsq = max(0.0, float(xp.sum(delta * delta, dtype=xp.float64)))
    total = valid_a + valid_b
    # The variance of the token-weighted combined mean under IID sampling:
    # sigma²/n = ||a-b||²*(n_a*n_b)/(n_a+n_b)².
    variance = dsq * valid_a * valid_b / (total * total)
    full_sq = max(0.0, (valid_a**2 * aa + valid_b**2 * bb
                        + 2 * valid_a * valid_b * cross) / total**2)
    signal_sq = full_sq - variance
    positive = signal_sq > 0 and math.isfinite(signal_sq)
    norm_a, norm_b = math.sqrt(max(aa, 0)), math.sqrt(max(bb, 0))
    return dict(
        gradient_norm_a=norm_a, gradient_norm_b=norm_b,
        gradient_norm_full=math.sqrt(full_sq),
        gradient_cosine=(cross / (norm_a * norm_b)
                         if norm_a > 0 and norm_b > 0 else None),
        gradient_difference_norm=math.sqrt(dsq),
        estimated_gradient_noise=variance,
        estimated_signal_norm_squared=signal_sq,
        relative_gradient_uncertainty=(math.sqrt(variance/signal_sq) if positive else None),
        signal_positive=positive,
        nonzero_gradient_fraction_a=float(xp.mean(first != 0)),
        nonzero_gradient_fraction_b=float(xp.mean(second != 0)),
    )


class TokenWeightedRefinementGradientMonitor:
    def __init__(self, parameters, *, accum_steps, accum_points='auto',
                 categories='attention,moe,moe_router,head',
                 output='logs/refinement_gradient_noise.csv', interval=100,
                 optimizer=None, adamw_output=None, grad_clip=1.0):
        if accum_points == 'auto':
            accum_points = ','.join(str(n) for n in (2,4,8,16,32)
                                    if n <= accum_steps)
        self.points = parse_accum_points(accum_points, accum_steps)
        self.boundaries = set(self.points) | {n//2 for n in self.points}
        self.selected = select_parameters(parameters, categories)
        if not self.selected:
            raise ValueError('no matching gradient-noise parameter families')
        if interval < 1:
            raise ValueError('gradient-noise interval must be positive')
        self.interval = int(interval)
        self.output = Path(output)
        self.optimizer = optimizer
        self.adamw_output = Path(adamw_output) if adamw_output is not None else None
        if self.adamw_output is not None and optimizer is None:
            raise ValueError('AdamW update diagnostics require optimizer')
        self.grad_clip = float(grad_clip)
        self.adamw_rows = []
        self.global_norm_snapshots = {}
        self.snapshots = {}
        self.token_counts = {}
        self.active_step = None
        self.monitor_overhead = 0.0

    def begin(self, completed_step):
        self.active_step = (completed_step + 1
                            if (completed_step + 1) % self.interval == 0 else None)
        self.snapshots.clear()
        self.global_norm_snapshots.clear()
        self.adamw_rows.clear()
        self.token_counts.clear()
        self.monitor_overhead = 0.0
        return self.active_step is not None

    def capture(self, microbatches_completed, valid_tokens_cumulative):
        if self.active_step is None or microbatches_completed not in self.boundaries:
            return
        start = time.perf_counter()
        self.snapshots[microbatches_completed] = [
            xp.array(p.grad[sel], dtype=xp.float32, copy=True)
            for _, p, sel in self.selected
        ]
        self.token_counts[microbatches_completed] = int(valid_tokens_cumulative)
        if self.adamw_output is not None and microbatches_completed in self.points:
            total_sq = xp.asarray(0., dtype=xp.float64)
            for param in self.optimizer.parameters:
                if param.grad is not None:
                    g = param.grad.astype(xp.float32, copy=False)
                    total_sq += xp.sum(g * g, dtype=xp.float64)
            self.global_norm_snapshots[microbatches_completed] = float(xp.sqrt(total_sq))
        self.monitor_overhead += time.perf_counter() - start

    def finish(self, total_supervised_tokens):
        if self.active_step is None:
            return []
        started = time.perf_counter()
        rows = []
        for point in self.points:
            mid = point // 2
            count_a = self.token_counts[mid]
            count_b = self.token_counts[point] - count_a
            for idx, (label, param, _) in enumerate(self.selected):
                stats = weighted_half_statistics(
                    self.snapshots[mid][idx], self.snapshots[point][idx],
                    count_a, count_b, total_valid=total_supervised_tokens,
                )
                rows.append(dict(
                    step=self.active_step, accumulation_length=point,
                    parameter_group=label, parameter_name=param.name,
                    parameter_count=int(self.snapshots[point][idx].size),
                    valid_supervised_tokens=self.token_counts[point],
                    **stats,
                ))
        if self.adamw_output is not None:
            self.adamw_rows = self._adamw_diagnostics(total_supervised_tokens)
        self.monitor_overhead += time.perf_counter() - started
        for row in rows + self.adamw_rows:
            row['monitor_overhead_ms'] = self.monitor_overhead * 1000
        self.snapshots.clear()
        return rows

    def _adamw_diagnostics(self, total_supervised_tokens):
        """Exact token-normalized prefix AdamW counterfactuals (no state writes)."""
        optimizer = self.optimizer
        indices = {id(p): i for i, p in enumerate(optimizer.parameters)}
        reference_n = max(self.points)
        reference_tokens = self.token_counts[reference_n]
        reference_clip = None
        rows = []
        for idx, (label, param, selector) in enumerate(self.selected):
            opt_idx = indices[id(param)]
            m0 = np.asarray(asnumpy(optimizer.m[opt_idx][selector]), dtype=np.float32)
            v0 = np.asarray(asnumpy(optimizer.v[opt_idx][selector]), dtype=np.float32)
            local_step = (int(optimizer.step_index) + 1
                          - int(optimizer.parameter_birth_steps[opt_idx]))
            if local_step <= 0:
                raise ValueError('inactive optimizer parameter in refinement diagnostic')
            gradients = {}
            directions = {}
            clips = {}
            for n in self.points:
                valid = self.token_counts[n]
                if valid <= 0:
                    raise ValueError('no valid targets at monitored gradient prefix')
                prefix_gradient = (np.asarray(asnumpy(self.snapshots[n][idx]), dtype=np.float32)
                                   * np.float32(total_supervised_tokens / valid))
                gradients[n] = prefix_gradient
                full_norm = (self.global_norm_snapshots[n]
                             * total_supervised_tokens / valid)
                clips[n] = min(1., self.grad_clip/(full_norm+1e-12))
                directions[n] = counterfactual_direction(
                    prefix_gradient, m0, v0,
                    beta1=optimizer.beta1, beta2=optimizer.beta2, eps=optimizer.eps,
                    learning_rate=optimizer.lr, local_step=local_step,
                    clip_scale=clips[n],
                )
            zero = np.zeros_like(m0)
            inherited = counterfactual_direction(
                zero, m0, v0, beta1=optimizer.beta1, beta2=optimizer.beta2,
                eps=optimizer.eps, learning_rate=optimizer.lr, local_step=local_step)
            ref_g = gradients[reference_n]
            ref_dir = directions[reference_n]
            ref_norm = float(np.linalg.norm(ref_dir.astype(np.float64)))
            for n in self.points:
                d = directions[n]
                d_norm = float(np.linalg.norm(d.astype(np.float64)))
                delta = float(np.linalg.norm((d-ref_dir).astype(np.float64)))
                innovation = float(np.linalg.norm((d-inherited).astype(np.float64)))
                valid_prefix = self.token_counts[n]
                if reference_tokens > valid_prefix:
                    # Prefix and suffix expressed in exactly the same units:
                    # mean gradient per valid target, not mean per microbatch.
                    heldout = ((ref_g * reference_tokens
                                - gradients[n] * valid_prefix)
                               / (reference_tokens - valid_prefix))
                else:
                    heldout = None
                full_norm = (self.global_norm_snapshots[n]
                             * total_supervised_tokens / valid_prefix)
                rows.append(dict(
                    step=self.active_step, accumulation_length=n,
                    reference_accumulation=reference_n,
                    parameter_group=label, parameter_name=param.name,
                    parameter_count=int(d.size),
                    valid_supervised_tokens=valid_prefix,
                    global_gradient_norm=full_norm,
                    global_clip_scale=clips[n],
                    adamw_local_step=local_step,
                    adamw_update_norm=d_norm,
                    adamw_update_cosine_to_reference=_cosine(d,ref_dir),
                    adamw_relative_update_difference=(delta/ref_norm if ref_norm else None),
                    gradient_momentum_cosine=_cosine(gradients[n],m0),
                    innovation_norm=innovation,
                    innovation_over_update_norm=(innovation/d_norm if d_norm else None),
                    heldout_gradient_cosine=(_cosine(gradients[n],heldout)
                                             if heldout is not None else None),
                    heldout_first_order_gain=(float(np.dot(
                        d.ravel().astype(np.float64), heldout.ravel().astype(np.float64)))
                        if heldout is not None else None),
                    heldout_direction_cosine=(_cosine(d,heldout)
                                               if heldout is not None else None),
                ))
        return rows

    def commit(self, rows):
        if not rows:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        new = not self.output.exists() or self.output.stat().st_size == 0
        with self.output.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            writer.writerows(rows)
        if self.adamw_output is not None and self.adamw_rows:
            self.adamw_output.parent.mkdir(parents=True, exist_ok=True)
            fresh = not self.adamw_output.exists() or self.adamw_output.stat().st_size == 0
            with self.adamw_output.open('a', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=ADAMW_FIELDS)
                if fresh:
                    writer.writeheader()
                writer.writerows(self.adamw_rows)
        self.adamw_rows.clear()
