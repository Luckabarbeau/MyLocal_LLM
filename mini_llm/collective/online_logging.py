"""0084D: compatibility telemetry for the 0082F fusion streaming report format.

Keeps the same CSV header and console table shapes as the legacy cached
producer while measuring *online*, forward-only frozen teacher inference.
No examples/ dependency, no cached teacher logits, and no fabricated segments.
"""
from __future__ import annotations

from collections import Counter
import csv
import os
from pathlib import Path

import numpy as np


# Deliberately match _metrics_csv_fields in the existing 0082F example.
def metrics_csv_fields(source_names, model_ids):
    base = ['step', 'segment', 'members', 'batch_supervised_tokens',
            'interval_supervised_tokens', 'total_supervised_tokens',
            'teacher_input_tokens_total', 'teacher_scored_tokens_total',
            'teacher_producer_s_total', 'teacher_producer_segment_s',
            'teacher_input_tok_s_segment', 'teacher_scored_tok_s_segment',
            'fusion_train_interval_s', 'fusion_train_total_s', 'fusion_train_tok_s',
            'fusion_train_cumulative_tok_s', 'fusion_end_to_end_tok_s',
            'elapsed_active_s', 'ce', 'arithmetic_ce', 'geometric_ce',
            'weighted_ce', 'grad_norm', 'val_ce', 'val_arithmetic_ce',
            'val_geometric_ce', 'val_weighted_ce']
    base += [f'model_{mid}_ce' for mid in model_ids]
    for source in source_names:
        base += [f'{prefix}_{source}' for prefix in ('batch_source_tokens',
                  'interval_source_tokens', 'total_source_tokens',
                  'source_train_tok_s', 'source_mix_pct')]
    return base


def append_metrics_csv(path, fieldnames, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open('r', newline='', encoding='utf-8') as handle:
            header = next(csv.reader(handle), None)
        if header != fieldnames:
            raise ValueError(f'existing fusion CSV has incompatible columns: {path}; '
                             'use a new --log-file (the 0084D schema matches 0082F)')
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def reconcile_metrics_csv(path, fieldnames, saved_step):
    """Discard reported steps after the last durable online optimizer checkpoint."""
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fieldnames:
            raise ValueError(f'existing fusion CSV has incompatible columns: {path}; '
                             'use a new --log-file for the upgraded online logger')
        rows = list(reader)
    retained = [row for row in rows if int(row['step']) <= saved_step]
    if len(retained) != len(rows):
        tmp = path.with_name(path.name + '.tmp')
        with tmp.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(retained)
        os.replace(tmp, path)
        print(f'fusion_online_csv_reconciled saved_step={saved_step} '
              f'dropped_rows={len(rows)-len(retained)} path={path}', flush=True)


def source_table(names, *, batch, interval, total, total_rows, interval_seconds):
    width = max(15, *(len(name) for name in names))
    lines = [f'    Sources (train)     {"Source":<{width}} {"Batch":>7} '
             f'{"Interval":>10} {"Total":>12} {"Share":>8} {"tok/s":>9}']
    for name in names:
        n_total = int(total.get(name, 0))
        n_interval = int(interval.get(name, 0))
        lines.append(f'                        {name:<{width}} '
                     f'{int(batch.get(name, 0)):>7,} '
                     f'{n_interval:>10,} '
                     f'{n_total:>12,} '
                     f'{100 * n_total / max(1, total_rows):>7.2f}% '
                     f'{n_interval / max(1e-9, interval_seconds):>9,.1f}')
    return lines


def member_table(ids, ces, *, global_weights, context_means=None, context_stds=None):
    use_context = context_means is not None
    lines = ['    Models              ID        CE   Global weight'
             + ('   Context mean   Context std' if use_context else '')]
    for index, (mid, ce) in enumerate(zip(ids, ces)):
        row = (f'                        {int(mid):>2}  {float(ce):>10.5f} '
               f'{float(global_weights[index]):>14.4f}')
        if use_context:
            row += (f'   {float(context_means[index]):>12.4f}'
                    f'   {float(context_stds[index]):>11.4f}')
        lines.append(row)
    return lines


def comparison_lines(fusion, arithmetic, geometric, weighted, ids, member_ces):
    best_member = int(np.argmin(member_ces))
    best_id, best_ce = int(ids[best_member]), float(member_ces[best_member])
    best_reference = min(best_ce, arithmetic, geometric, weighted)
    return [
        f'    Gains               vs arithmetic={arithmetic-fusion:+.5f} '
        f'vs geometric={geometric-fusion:+.5f} vs weighted={weighted-fusion:+.5f}',
        f'                        vs best model {best_id}={best_ce-fusion:+.5f} '
        f'vs best reference={best_reference-fusion:+.5f}',
    ]


def print_step_block(step, total, phase, lines, *, members):
    # Segment is a disk-logit-cache construct; ONLINE has no segment number.
    label = f'Step {step:>{len(str(total))}}/{total} | Segment online | {phase} | Members {members}'
    print('\n'.join([label, *lines, '']), flush=True)


class OnlineFusionTelemetry:
    """Observed online teacher/combiner timing, supervised rows and provenance.

    Timing convention: teacher_s is measured on GPU using CUDA events if
    possible; train_s is wall-clock non-teacher time, including host dispatch.
    There is no separately exported segment in online mode. The legacy
    *_segment_* fields describe the last logging interval, not disk I/O.
    """
    def __init__(self, source_names, *, previous=None):
        previous = previous or {}
        self.names = tuple(sorted(source_names))
        self.total_rows = int(previous.get('total_rows', 0))
        self.source_rows = Counter({name: int(previous.get('source_rows', {}).get(name, 0))
                                    for name in self.names})
        self.teacher_input_tokens = int(previous.get('teacher_input_tokens', 0))
        self.teacher_scored_tokens = int(previous.get('teacher_scored_tokens', 0))
        self.teacher_seconds = float(previous.get('teacher_seconds', 0.0))
        self.fusion_seconds = float(previous.get('fusion_seconds', 0.0))
        self.active_seconds = float(previous.get('active_seconds', 0.0))
        self.interval_rows = 0
        self.interval_sources = Counter()
        self.interval_teacher_input = 0
        self.interval_teacher_scored = 0
        self.interval_teacher_s = 0.0
        self.interval_train_s = 0.0

    def record_step(self, *, by_source, members, context_length, scored, windows,
                    wall_seconds, teacher_seconds):
        rows = sum(by_source.values())
        self.total_rows += rows
        self.interval_rows += rows
        self.source_rows.update(by_source)
        self.interval_sources.update(by_source)
        inputs = int(windows * members * context_length)
        scored_tokens = int(windows * members * scored)
        self.teacher_input_tokens += inputs
        self.teacher_scored_tokens += scored_tokens
        self.interval_teacher_input += inputs
        self.interval_teacher_scored += scored_tokens
        teacher_seconds = max(0., float(teacher_seconds))
        self.teacher_seconds += teacher_seconds
        self.interval_teacher_s += teacher_seconds
        train_seconds = max(0., float(wall_seconds) - teacher_seconds)
        self.fusion_seconds += train_seconds
        self.interval_train_s += train_seconds
        self.active_seconds += float(wall_seconds)

    def rates(self):
        interval = max(1e-9, self.interval_train_s)
        teacher = max(1e-9, self.interval_teacher_s)
        return dict(fusion_train_interval_s=self.interval_train_s,
                    fusion_train_total_s=self.fusion_seconds,
                    fusion_train_tok_s=self.interval_rows / interval,
                    fusion_train_cumulative_tok_s=self.total_rows / max(1e-9, self.fusion_seconds),
                    fusion_end_to_end_tok_s=self.total_rows / max(1e-9, self.active_seconds),
                    teacher_producer_s_total=self.teacher_seconds,
                    teacher_producer_segment_s=self.interval_teacher_s,
                    teacher_input_tok_s_segment=self.interval_teacher_input / teacher,
                    teacher_scored_tok_s_segment=self.interval_teacher_scored / teacher,
                    elapsed_active_s=self.active_seconds)

    def finish_interval(self):
        self.interval_rows = 0
        self.interval_sources.clear()
        self.interval_teacher_input = 0
        self.interval_teacher_scored = 0
        self.interval_teacher_s = 0.0
        self.interval_train_s = 0.0

    def state(self):
        return dict(total_rows=self.total_rows, source_rows=dict(self.source_rows),
                    teacher_input_tokens=self.teacher_input_tokens,
                    teacher_scored_tokens=self.teacher_scored_tokens,
                    teacher_seconds=self.teacher_seconds, fusion_seconds=self.fusion_seconds,
                    active_seconds=self.active_seconds)
