"""0079L: observational training telemetry for predictive refinement.

Counters distinguish unique supervised targets, input/source capacity and the
repeated Transformer position evaluations. No source ID or diagnostic can enter
any model forward or change the optimizer update. The ordinary trainer's
0078A source-metrics CSV schema is reused for comparable fields.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.diagnostics.source_metrics import empty_record, merge_record


def empty_counters():
    return dict(
        microbatches=0, samples=0,
        source_capacity_tokens=0, valid_source_tokens=0,
        history_capacity_tokens=0, valid_history_tokens=0,
        target_capacity_tokens=0, valid_active_tokens=0,
        supervised_tokens=0,
        active_position_evaluations=0, skipped_empty_block_rows=0,
        scheduled_block_rows=0, refinement_full_model_passes=0,
        refinement_batches=0, causal_context_encodes=0, target_blocks=0,
        active_queries_per_pass_min=None, active_queries_per_pass_max=0,
        active_queries_per_pass_sum=0,
        context_keys_per_group_max=0,
        source_records={},
    )


def merge_counters(dst, src):
    for key in (
        'microbatches', 'samples', 'source_capacity_tokens', 'valid_source_tokens',
        'history_capacity_tokens', 'valid_history_tokens', 'target_capacity_tokens',
        'valid_active_tokens', 'supervised_tokens', 'active_position_evaluations',
        'refinement_full_model_passes', 'refinement_batches',
        'causal_context_encodes', 'target_blocks', 'active_queries_per_pass_sum',
        'skipped_empty_block_rows', 'scheduled_block_rows',
    ):
        dst[key] += src[key]
    if src['active_queries_per_pass_min'] is not None:
        if dst['active_queries_per_pass_min'] is None:
            dst['active_queries_per_pass_min'] = src['active_queries_per_pass_min']
        else:
            dst['active_queries_per_pass_min'] = min(
                dst['active_queries_per_pass_min'], src['active_queries_per_pass_min'])
    dst['context_keys_per_group_max'] = max(
        dst.get('context_keys_per_group_max', 0), src.get('context_keys_per_group_max', 0))
    dst['active_queries_per_pass_max'] = max(
        dst['active_queries_per_pass_max'], src['active_queries_per_pass_max'])
    for name, rec in src['source_records'].items():
        merge_record(dst['source_records'].setdefault(name, empty_record()), rec)
    return dst


def observe_microbatch(tokens, valid, prefix, result, *, configured_prefix,
                       window_blocks, block_size, observations=None,
                       source_counts=None, collect_sources=True):
    """Account for actual valid tokens, padded capacity and raw per-source CE.

    The model's normalizer does NOT enter source-level CE. These use raw NLL
    sums to avoid bias from uneven document lengths or gradient accumulation.
    """
    tokens = np.asarray(tokens)
    valid = np.asarray(valid, dtype=np.bool_)
    b, t = tokens.shape
    if valid.shape != tokens.shape or not 0 <= prefix < t:
        raise ValueError('invalid telemetry window/mask')
    valid_targets = valid[:, prefix:]
    supervised = int(valid_targets.sum())
    if int(result['supervised_tokens']) != supervised:
        raise AssertionError('model and telemetry supervised-token counts disagree')
    nominal = configured_prefix + window_blocks * block_size
    d = empty_counters()
    d.update(microbatches=1, samples=b,
             source_capacity_tokens=b * nominal,
             valid_source_tokens=int(valid.sum()),
             history_capacity_tokens=b * configured_prefix,
             valid_history_tokens=int(valid[:, :prefix].sum()),
             target_capacity_tokens=b * window_blocks * block_size,
             valid_active_tokens=supervised,
             supervised_tokens=supervised,
             active_position_evaluations=int(result['active_block_queries']),
             refinement_full_model_passes=int(result['refinement_passes']),
             refinement_batches=int(result['refinement_batches']),
             causal_context_encodes=int(result['causal_context_encodes']),
             target_blocks=int(result['target_blocks']),
             context_keys_per_group_max=int(result.get('context_keys_per_group_max', 0)),
             skipped_empty_block_rows=int(result.get('skipped_empty_block_rows', 0)),
             scheduled_block_rows=int(result['active_block_queries'] //
                                      ((block_size + 1) * block_size) +
                                      result.get('skipped_empty_block_rows', 0)))
    rows = result['executed_active_rows_per_pass']
    if rows:
        d['active_queries_per_pass_min'] = min(map(int, rows))
        d['active_queries_per_pass_max'] = max(map(int, rows))
        d['active_queries_per_pass_sum'] = sum(map(int, rows)) * (block_size + 1)
    if not collect_sources:
        return d
    if observations is not None and len(observations) != b:
        raise ValueError('source observations must contain one record per document')
    if observations is None:
        # A synthetic/dummy source is valid only when unambiguous; otherwise
        # explicitly attribute to unknown, never assign mixed losses randomly.
        name = (next(iter(source_counts)) if source_counts and len(source_counts) == 1
                else 'unattributed')
        observations = [{'source': name} for _ in range(b)]
    row_ce = (None if 'row_commitment_ce_sums' not in result else
              np.asarray(asnumpy(result['row_commitment_ce_sums']), dtype=np.float64))
    if row_ce is not None and row_ce.shape != (b,):
        raise ValueError('unexpected per-row CE shape')
    seen_sources = set()
    for i, obs in enumerate(observations):
        name = str(obs['source'])
        seen_sources.add(name)
        rec = d['source_records'].setdefault(name, empty_record())
        n = int(valid_targets[i].sum())
        rec['samples'] += 1
        rec['source_capacity_tokens'] += nominal
        rec['valid_source_tokens'] += int(valid[i].sum())
        rec['history_capacity_tokens'] += configured_prefix
        rec['valid_history_tokens'] += int(valid[i, :prefix].sum())
        rec['target_capacity_tokens'] += window_blocks * block_size
        rec['valid_active_tokens'] += n
        rec['supervised_tokens'] += n
        rec['supervised_token_weight'] += n
        if row_ce is not None:
            rec['loss_nll_sum'] += float(row_ce[i])
        else:
            rec['loss_nll_sum'] = float('nan')
        span = obs.get('sampled_document_span_tokens')
        if span is not None:
            span = int(span)
            rec['sampled_document_spans'] += 1
            rec['sampled_document_span_tokens_sum'] += span
            lo = rec['sampled_document_span_tokens_min']
            hi = rec['sampled_document_span_tokens_max']
            rec['sampled_document_span_tokens_min'] = span if lo is None else min(lo, span)
            rec['sampled_document_span_tokens_max'] = span if hi is None else max(hi, span)
    for name in seen_sources:
        d['source_records'][name]['microbatches'] += 1
    return d


def rates(stats, elapsed_seconds, *, step_seconds=None):
    """Precise per-step/interval rates. Active work is NOT unique supervision."""
    denom = max(float(elapsed_seconds), 1e-12)
    passes = stats['refinement_full_model_passes']
    mean_queries = (stats['active_queries_per_pass_sum'] / passes if passes else 0.0)
    return {
        'source_tok_per_second': stats['source_capacity_tokens'] / denom,
        'valid_source_tok_per_second': stats['valid_source_tokens'] / denom,
        'valid_history_tok_per_second': stats['valid_history_tokens'] / denom,
        'active_tok_per_second': stats['target_capacity_tokens'] / denom,
        'valid_active_tok_per_second': stats['valid_active_tokens'] / denom,
        'supervised_tok_per_second': stats['supervised_tokens'] / denom,
        'active_position_evals_per_second': stats['active_position_evaluations'] / denom,
        'target_fill': stats['supervised_tokens'] / max(stats['target_capacity_tokens'], 1),
        'active_block_row_fill': (
            1 - stats['skipped_empty_block_rows'] / max(stats['scheduled_block_rows'], 1)),
        'history_fill': stats['valid_history_tokens'] / max(stats['history_capacity_tokens'], 1),
        'active_queries_per_pass_mean': mean_queries,
        'active_queries_per_pass_min': stats['active_queries_per_pass_min'],
        'active_queries_per_pass_max': stats['active_queries_per_pass_max'],
        'context_keys_per_group_max': stats.get('context_keys_per_group_max', 0),
        'active_position_evals_per_unique_target': (
            stats['active_position_evaluations'] / max(stats['supervised_tokens'], 1)),
        'unique_targets_per_refinement_pass': stats['supervised_tokens'] / max(passes, 1),
        'model_refinement_passes_per_second': passes / denom,
        'steps_per_second': (stats['microbatches'] / denom if step_seconds is None
                             else 1.0 / max(step_seconds, 1e-12)),
    }


def source_summary(stats):
    out = {}
    for name, rec in sorted(stats['source_records'].items()):
        n = int(rec['supervised_tokens'])
        # CE is omitted, not set to zero, when exact row-level NLL is absent.
        out[name] = {
            'samples': int(rec['samples']),
            'supervised_tokens': n,
            'commitment_ce': (float(rec['loss_nll_sum']) / n
                              if n and math.isfinite(rec['loss_nll_sum']) else None),
            'target_fill': n / max(int(rec['target_capacity_tokens']), 1),
            'sampled_document_span_mean': (
                rec['sampled_document_span_tokens_sum'] / rec['sampled_document_spans']
                if rec['sampled_document_spans'] else None),
        }
    return out


def memory_snapshot():
    """Non-peak snapshot; all GPU allocation claims are explicitly labeled."""
    if BACKEND_NAME != 'cupy':
        return None
    try:
        pool = xp.get_default_memory_pool()
        free, total = xp.cuda.Device().mem_info
        return {
            'cupy_pool_live_gib': pool.used_bytes() / (1024 ** 3),
            'cupy_pool_reserved_gib': pool.total_bytes() / (1024 ** 3),
            'device_used_now_gib': (total - free) / (1024 ** 3),
            'device_free_now_gib': free / (1024 ** 3),
            'peak_measurement': False,
        }
    except Exception:
        return None


SCALAR_CSV_COLUMNS = (
    'step', 'tokens_processed', 'train_loss', 'lr', 'grad_norm',
    'val_loss', 'steps_per_sec', 'step_time_ms',
    'supervised_tok_per_second', 'source_tok_per_second',
    'valid_source_tok_per_second', 'active_queries_per_pass_mean',
    'active_position_evals_per_second', 'target_fill',
    'first_position_refinement_gain',
)


def write_scalar_csv(path, *, step, supervised_tokens=None, train_loss=None,
                     learning_rate=None, gradient_norm=None, val_loss=None,
                     steps_per_second=None, step_time_ms=None,
                     throughput=None, first_position_gain=None):
    """Legacy-compatible scalar training CSV plus refinement throughput columns.

    Validation is represented as its own sparse row, which avoids rewriting
    earlier training records or misrepresenting when validation was measured.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    fields = dict(
        step=int(step), tokens_processed=supervised_tokens,
        train_loss=train_loss, lr=learning_rate, grad_norm=gradient_norm,
        val_loss=val_loss, steps_per_sec=steps_per_second,
        step_time_ms=step_time_ms,
        first_position_refinement_gain=first_position_gain,
    )
    if throughput:
        for name in SCALAR_CSV_COLUMNS:
            if name in throughput:
                fields[name] = throughput[name]
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SCALAR_CSV_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({key: ('' if value is None else value)
                         for key, value in fields.items()})
