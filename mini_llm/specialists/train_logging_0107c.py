"""Source-weighted, foundation-style console logging for specialist training.

All losses and coverage refer to the SAME pre-update training microbatches.
The frozen foundation CE is optional because it adds a second vocabulary-head
forward pass, but never an extra foundation transformer pass.
"""
from __future__ import annotations

from mini_llm.diagnostics.source_metrics import empty_record, merge_record


class SpecialistIntervalLogger:
    """Accumulate source and CE metrics over `log_interval` optimizer updates."""

    def __init__(self, *, log_interval=1, total_steps=100000):
        if log_interval < 1:
            raise ValueError('log_interval must be positive')
        self.log_interval = int(log_interval)
        self.total_steps = int(total_steps)
        self.reset()

    def reset(self):
        self.steps = 0
        self.supervised = 0
        self.loss_nll = 0.0
        self.foundation_nll = 0.0
        self.elapsed = 0.0
        self.source_stats = {}
        self.last = None

    def observe(self, record):
        valid = int(record['supervised_tokens_step'])
        self.supervised += valid
        self.loss_nll += float(record['loss']) * valid
        if 'foundation_ce' in record:
            self.foundation_nll += float(record['foundation_ce']) * valid
        self.elapsed += float(record['step_time_s'])
        self.steps += 1
        self.last = record
        for name, source in record.get('per_source', {}).items():
            acc = self.source_stats.setdefault(name, empty_record())
            merge_record(acc, source)
            if 'foundation_nll_sum' in source:
                acc['foundation_nll_sum'] = (
                    acc.get('foundation_nll_sum', 0.0) + source['foundation_nll_sum'])
        return self.steps >= self.log_interval

    def flush(self):
        """Return formatted lines and clear the interval, or [] when empty."""
        if not self.steps:
            return []
        r = self.last
        avg_loss = self.loss_nll / self.supervised
        sec = max(float(r['step_time_s']), 1e-12)
        capacity = int(r['capacity_tokens_step'])
        line = (
            f"Step {r['step']}/{self.total_steps}: loss={avg_loss:.4f}, "
            f"lr={r['lr']:.6f}, grad_norm={r['grad_norm']:.4f}, "
            f"grad_weighting=supervised_tokens, "
            f"target_fill={100.0 * r['target_fill']:.1f}%, "
            f"supervised_total={r['supervised_tokens']:,}, "
            f"{self.steps / max(self.elapsed, 1e-12):.2f} steps/sec, "
            f"step_time={1000 * sec:.1f} ms, "
            f"supervised_tok/s={r['supervised_tokens_step'] / sec:,.0f}, "
            f"executed_tok/s={r['executed_tokens_step'] / sec:,.0f}, "
            f"capacity_tok/s={capacity / sec:,.0f}"
        )
        if 'foundation_ce' in r:
            f_ce = self.foundation_nll / self.supervised
            line += (f", foundation_train_ce={f_ce:.4f}, "
                     f"foundation_gain_ce={f_ce - avg_loss:+.4f}")
        lines = [line]
        if self.source_stats:
            lines.append('  Per-source train CE (valid-target weighted; this log interval):')
            for name in sorted(self.source_stats):
                stats = self.source_stats[name]
                supervised = int(stats['supervised_tokens'])
                weight = float(stats['supervised_token_weight'])
                capacity = int(stats['target_capacity_tokens'])
                docs = int(stats['sampled_document_spans'])
                text = (f"    {name}: CE={stats['loss_nll_sum'] / weight:.4f}, "
                        f"supervised={supervised:,}, "
                        f"target_fill={100.0 * weight / max(capacity, 1):.1f}%")
                if docs:
                    text += (f", sampled_doc_mean="
                             f"{stats['sampled_document_span_tokens_sum'] / docs:,.0f}")
                if 'foundation_nll_sum' in stats:
                    fce = stats['foundation_nll_sum'] / weight
                    text += (f", foundation_CE={fce:.4f}, "
                             f"foundation_gain={fce - stats['loss_nll_sum'] / weight:+.4f}")
                lines.append(text)
        self.reset()
        return lines
