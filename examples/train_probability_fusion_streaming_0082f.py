"""0082F: stream teacher logits from disjoint parent-held-out shards.

Rather than export an entire fixed 32K-vocabulary dataset, generate a bounded
segment of EOS-bounded token windows, evaluate frozen teacher checkpoints one
at a time, train on *all* aligned predicted-token distributions once, and
remove the temporary logits.  This is a staged online pipeline, NOT joint
backpropagation through the teachers; routed 64K-history evaluation remains
out of scope (plain causal contexts only).

Requires a pre-exported validation split from 0082D, held out from the
streaming *train* split.  The 0082B window metadata is mandatory so training
cannot accidentally sample from a differently partitioned validation split.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import gc
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import time

import numpy as np

from mini_llm.backend import asnumpy, scalar, xp
from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner
from mini_llm.collective.context_windows import AlignedContextWindows, checkpoint_fingerprint
from mini_llm.data.mixed_shards import load_weighted_shard_sources
from mini_llm.optim.adamw import AdamW
from examples.export_probability_fusion_logits_0082a import (
    _resolve_tokenizer, _scored_positions, main as export_main,
)
from examples.prepare_probability_fusion_windows_0082b import _partition_by_shard, _ShardSampler
from examples.train_probability_fusion_0082a import (
    _batch, _clip_grad_norm_, _evaluate, _open_members, _should_stop_early,
)


def _member_specs(specs):
    members = []
    for spec in specs:
        label, sep, path = spec.partition(":")
        if not sep:
            raise ValueError(f"expected MODEL_ID:CHECKPOINT, got {spec!r}")
        mid = int(label)
        directory = Path(path).expanduser().resolve()
        if mid < 0 or mid in [item[0] for item in members]:
            raise ValueError("model IDs must be unique, nonnegative integers")
        if not directory.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {directory}")
        members.append((mid, directory))
    if len(members) < 2:
        raise ValueError("streaming fusion expects two or more frozen members")
    return members


def _checkpoint_fingerprint(directory):
    # Backward-compatible name used by the 0082F streaming guard.
    return checkpoint_fingerprint(directory)


def _stream_samplers(root, manifest_path, partition_seed, context, rng):
    sources, manifest = load_weighted_shard_sources(root)
    parts = _partition_by_shard(sources, partition_seed)
    metadata = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if metadata.get('format') != 'fusion_eos_windows_v1':
        raise ValueError('expected 0082B windows_metadata.json for partition verification')
    if int(metadata['seed']) != partition_seed:
        raise ValueError('partition seed differs from existing fusion validation/test windows')
    if metadata.get('tokenizer_sha256') != manifest.get('tokenizer_sha256'):
        raise ValueError('fusion window tokenizer hash differs from mixed shard tokenizer')
    if Path(manifest_path).resolve().parent == Path(root).resolve():
        raise ValueError('split manifest should be a separate frozen 0082B windows manifest')
    for split in ('train', 'val', 'test'):
        recorded = metadata['splits'][split]['validation_shards']
        for source in sources:
            actual = {str(p.relative_to(Path(root).resolve())) for p in parts[split][source.name]}
            if set(recorded[source.name]) != actual:
                raise ValueError(f'{split}/{source.name} partition mismatch with 0082B manifest')
    used = set()
    for split in ('train', 'val', 'test'):
        paths = {p.resolve() for src in parts[split].values() for p in src}
        if used & paths:
            raise ValueError(f'shard leakage detected in {split} partition')
        used |= paths
    eos = int(manifest['eos_token_id'])
    dtype = np.dtype(manifest.get('token_dtype', 'uint16'))
    samplers = {s.name: _ShardSampler(parts['train'][s.name], dtype, eos, context+1, rng)
                for s in sources}
    weights = np.asarray([s.weight for s in sources], dtype=np.float64)
    weights /= weights.sum()
    return sources, manifest, samplers, weights


def _make_segment_windows(path, samplers, sources, weights, count, context, rng):
    """Draw fresh EOS-bounded contexts; record exact source file/offset trace."""
    rows = np.lib.format.open_memmap(path, mode='w+', dtype=np.int32,
                                      shape=(count, context+1))
    counts = Counter()
    trace = []
    for i in range(count):
        source = sources[int(rng.choice(len(sources), p=weights))]
        tokens, shard, offset = samplers[source.name].sample()
        rows[i, :] = tokens
        counts[source.name] += 1
        trace.append((source.name, str(shard), int(offset)))
    rows.flush()
    del rows
    return dict(counts), trace


def _segment_windows_for_budget(requested, *, members, vocab, scored, budget_gib):
    if requested < 1 or members < 1 or vocab < 1 or scored < 1 or budget_gib <= 0:
        raise ValueError('segment size, model count, vocabulary, score count and budget must be > 0')
    per_window = members * vocab * scored * np.dtype(np.float16).itemsize
    allowed = int(budget_gib * (1024 ** 3) // per_window)
    if allowed < 1:
        raise ValueError('max-cache-gib is too small even for one window of complete logits')
    return min(requested, allowed)


def _to_host(value):
    return np.asarray(asnumpy(value), dtype=np.float32)


def _batch_teacher_ces(logits, targets):
    """Geometric and per-member CE on the EXACT currently trained batch.

    Called only on logging steps to avoid extra full-vocabulary reductions on
    every optimizer update. Logits are [rows, selected_members, vocabulary].
    Neither these reference metrics nor their gradients update the teachers.
    """
    z = xp.asarray(logits, dtype=xp.float32)
    if z.ndim != 3:
        raise ValueError('expected [rows, members, vocabulary] teacher logits')
    y = np.asarray(targets, dtype=np.int64)
    if y.shape != (z.shape[0],):
        raise ValueError('targets must match the teacher logit rows')
    shift = z - xp.max(z, axis=-1, keepdims=True)
    logp = shift - xp.log(xp.sum(xp.exp(shift), axis=-1, keepdims=True))
    y_gpu = xp.asarray(y)
    selected = xp.take_along_axis(logp, y_gpu[:, None, None], axis=2)[:, :, 0]
    member_ce = -np.asarray(asnumpy(xp.mean(selected, axis=0)), dtype=np.float64)
    mean_logp = xp.mean(logp, axis=1)
    geom_shift = mean_logp - xp.max(mean_logp, axis=1, keepdims=True)
    geom_logp = geom_shift - xp.log(xp.sum(xp.exp(geom_shift), axis=1, keepdims=True))
    geometric_ce = -float(scalar(xp.mean(geom_logp[xp.arange(len(y)), y_gpu])))
    return geometric_ce, member_ce


def _comparison_fields(fusion_ce, arithmetic_ce, geometric_ce, model_ids, member_ces, *, weighted_ce=None):
    """Consistent, parseable model/baseline comparisons for train and val.

    Positive gains always indicate *lower* cross-entropy (better likelihood).
    Input model IDs refer to stable model identities, not arbitrary positions.
    """
    ids = np.asarray(model_ids, dtype=np.int64)
    ces = np.asarray(member_ces, dtype=np.float64)
    if not len(ids) or ids.shape != ces.shape or not np.all(np.isfinite(ces)):
        raise ValueError('member IDs and finite individual CEs are required')
    best = int(np.argmin(ces))
    best_id = int(ids[best])
    best_ce = float(ces[best])
    best_reference = min(best_ce, float(arithmetic_ce), float(geometric_ce),
                         *([float(weighted_ce)] if weighted_ce is not None else []))
    member_fields = ' '.join(f'model_{mid}_ce={ce:.5f}'
                             for mid, ce in sorted(zip(ids, ces)))
    weights_field = (f'gain_weighted_best_member={best_ce-weighted_ce:+.5f} '
                     if weighted_ce is not None else '')
    return (f'{member_fields} best_member_id={best_id} '
            f'best_member_ce={best_ce:.5f} '
            f'gain_arithmetic_best_member={best_ce-arithmetic_ce:+.5f} '
            f'gain_geometric_best_member={best_ce-geometric_ce:+.5f} '
            f'{weights_field}'
            f'gain_fusion_best_member={best_ce-fusion_ce:+.5f} '
            f'gain_fusion_best_reference={best_reference-fusion_ce:+.5f}')


def _validation_context_weights(model, context_source, ids, *, max_rows, batch_rows,
                                return_stats=False):
    """Held-out, row-weighted mean/std per contextual mixture member."""
    total = context_source.rows
    rows = (np.arange(total) if not max_rows or max_rows >= total
            else np.linspace(0, total - 1, num=max_rows, dtype=np.int64))
    sums = np.zeros(len(ids), dtype=np.float64)
    squares = np.zeros(len(ids), dtype=np.float64)
    for start in range(0, len(rows), batch_rows):
        tokens, lengths = context_source.batch(rows[start:start + batch_rows])
        w = _to_host(model.member_weights(ids, context_tokens=tokens, context_lengths=lengths))
        sums += np.sum(w, axis=0)
        squares += np.sum(w*w, axis=0)
    mean = sums / len(rows)
    std = np.sqrt(np.maximum(0.0, squares / len(rows) - mean**2))
    if return_stats:
        return mean, std
    return ' '.join(f'val_context_weight_model_{int(mid)}={mean[j]:.4f} '
                    f'val_context_weight_std_model_{int(mid)}={std[j]:.4f}'
                    for j, mid in enumerate(ids))


def _sync_fusion_backend():
    """Synchronize queued CuPy kernels only at timed/reporting boundaries."""
    if hasattr(xp, 'cuda'):
        xp.cuda.get_current_stream().synchronize()


def _source_rows_for_batch(row_ids, trace, scored):
    """Exporter is window-major; every scored position retains its window source."""
    if scored < 1:
        raise ValueError('scored positions must be positive')
    selected = Counter()
    for row in np.asarray(row_ids, dtype=np.int64):
        if row < 0 or row // scored >= len(trace):
            raise ValueError('fusion prediction row cannot be mapped to a source window')
        selected[trace[int(row // scored)][0]] += 1
    return selected


def _format_source_rows(prefix, counts, source_names):
    return ' '.join(f'{prefix}_{name}={int(counts.get(name, 0))}' for name in source_names)


class _FusionTelemetry:
    """Observed source contributions and non-overlapping producer/trainer clocks.

    A fusion row supervises ONE next-token label regardless of the vocabulary
    dimension or how many teachers supplied probabilities for that label.
    Teacher input/scored tokens are counted per member, not confused with
    fusion-supervised target tokens. All counters survive segment resumes.
    """
    def __init__(self, source_names, *, previous=None):
        self.names = tuple(source_names)
        previous = previous or {}
        self.total_rows = int(previous.get('total_rows', 0))
        self.source_rows = Counter({n: int(previous.get('source_rows', {}).get(n, 0))
                                    for n in self.names})
        self.teacher_input_tokens = int(previous.get('teacher_input_tokens', 0))
        self.teacher_scored_tokens = int(previous.get('teacher_scored_tokens', 0))
        self.teacher_seconds = float(previous.get('teacher_seconds', 0.0))
        self.fusion_seconds = float(previous.get('fusion_seconds', 0.0))
        self.interval_rows = 0
        self.interval_source_rows = Counter()
        self.interval_start_seconds = self.fusion_seconds
        self.interval_start_rows = self.total_rows

    def exported(self, n_windows, context_length, scored, n_members, seconds):
        self.teacher_input_tokens += int(n_windows * context_length * n_members)
        self.teacher_scored_tokens += int(n_windows * scored * n_members)
        self.teacher_seconds += float(seconds)

    def trained(self, by_source):
        count = sum(by_source.values())
        self.total_rows += count
        self.interval_rows += count
        self.source_rows.update(by_source)
        self.interval_source_rows.update(by_source)

    def train_seconds(self, seconds):
        self.fusion_seconds += float(seconds)

    def rates(self, active_wall_seconds):
        elapsed = max(1e-9, self.fusion_seconds-self.interval_start_seconds)
        return {
            'fusion_train_tok_s': self.interval_rows / elapsed,
            'fusion_train_cumulative_tok_s': self.total_rows / max(1e-9, self.fusion_seconds),
            'fusion_end_to_end_tok_s': self.total_rows / max(1e-9, active_wall_seconds),
            'fusion_train_interval_s': elapsed,
            'fusion_train_total_s': self.fusion_seconds,
            'teacher_producer_s_total': self.teacher_seconds,
        }

    def finish_interval(self):
        self.interval_rows = 0
        self.interval_source_rows.clear()
        self.interval_start_rows = self.total_rows
        self.interval_start_seconds = self.fusion_seconds

    def state(self):
        return {'total_rows': self.total_rows, 'source_rows': dict(self.source_rows),
                'teacher_input_tokens': self.teacher_input_tokens,
                'teacher_scored_tokens': self.teacher_scored_tokens,
                'teacher_seconds': self.teacher_seconds, 'fusion_seconds': self.fusion_seconds}


def _metrics_csv_fields(source_names, model_ids):
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


def _reconcile_metrics_csv(path, fieldnames, saved_step):
    """On resume, discard log rows after the most recent durable segment.

    The optimizer checkpoint is written at segment boundaries. An interrupted
    segment may have printed/CSV-logged steps that the optimizer will replay.
    """
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        return
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fieldnames:
            raise ValueError(f'existing metrics CSV has incompatible columns: {path}')
        rows = list(reader)
    retained = [row for row in rows if int(row['step']) <= saved_step]
    if len(retained) != len(rows):
        tmp = path.with_name(path.name + '.tmp')
        with tmp.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(retained)
        os.replace(tmp, path)
        print(f'fusion_stream_csv_reconciled saved_step={saved_step} '
              f'dropped_rows={len(rows)-len(retained)} path={path}', flush=True)


def _append_metrics_csv(path, fieldnames, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open('r', newline='', encoding='utf-8') as handle:
            header = next(csv.reader(handle), None)
        if header != fieldnames:
            raise ValueError(f'existing metrics CSV has incompatible column names: {path}')
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)



def _step_header(step, total, segment, *, phase, members=None):
    """One left-aligned progress header; every metric is indented under it."""
    label = f'Step {step:>{len(str(total))}}/{total} | Segment {segment} | {phase}'
    if members is not None:
        label += f' | Members {members}'
    return label


def _source_table(names, *, batch=None, interval=None, total=None,
                  total_rows=None, interval_seconds=None, window_counts=None,
                  available_counts=None, teacher_counts=None):
    """Compact, fixed-column, one-source-per-line console telemetry.

    All counts are observed values. Source share uses cumulative supervised
    positions, while rate uses the optimizer-only reporting interval.
    """
    width = max(15, *(len(name) for name in names))
    if window_counts is not None:
        lines = [f'    Sources (export)    {"Source":<{width}} {"Windows":>9} '
                 f'{"Available":>11} {"Teacher input":>14}']
        for name in names:
            lines.append(f'                        {name:<{width}} '
                         f'{int(window_counts.get(name, 0)):>9,} '
                         f'{int(available_counts.get(name, 0)):>11,} '
                         f'{int(teacher_counts.get(name, 0)):>14,}')
    else:
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


def _member_table(ids, ces, *, global_weights, context_means=None, context_stds=None):
    """Per-identity reliability/CE on the same evaluated prediction rows."""
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


def _comparison_lines(fusion, arithmetic, geometric, weighted, ids, member_ces):
    best_member = int(np.argmin(member_ces))
    best_id, best_ce = int(ids[best_member]), float(member_ces[best_member])
    best_reference = min(best_ce, arithmetic, geometric, weighted)
    return [
        f'    Gains               vs arithmetic={arithmetic-fusion:+.5f} '
        f'vs geometric={geometric-fusion:+.5f} vs weighted={weighted-fusion:+.5f}',
        f'                        vs best model {best_id}={best_ce-fusion:+.5f} '
        f'vs best reference={best_reference-fusion:+.5f}',
    ]


def _print_step_block(step, total, segment, phase, lines, *, members=None):
    print('\n'.join([_step_header(step, total, segment, phase=phase, members=members),
                     *lines, '']), flush=True)


def _write_last_state(path, model, optimizer, rng, *, state):
    """Atomic segment-boundary checkpoint with AdamW state and sampler RNG."""
    payload = {'__meta_json__': np.asarray(json.dumps({**state, 'rng_state': rng.bit_generator.state})),
               '__config_json__': np.asarray(json.dumps(model.config.__dict__))}
    for name, parameter in model.parameters_by_name.items():
        payload['weight/' + name] = _to_host(parameter.data)
    for index, name in enumerate(model.parameters_by_name):
        payload['moment1/' + name] = _to_host(optimizer.m[index])
        payload['moment2/' + name] = _to_host(optimizer.v[index])
        payload['master/' + name] = _to_host(optimizer.master_weights[index])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        np.savez(stream, **payload)
    os.replace(tmp, path)


def _read_last_state(path, model, optimizer, *, expected):
    with np.load(path, allow_pickle=False) as saved:
        info = json.loads(str(saved['__meta_json__']))
        config = json.loads(str(saved['__config_json__']))
        if FusionConfig(**config).__dict__ != model.config.__dict__:
            raise ValueError('fusion model configuration differs from streaming checkpoint')
        for key, required in expected.items():
            if info.get(key) != required:
                raise ValueError(f'cannot resume: {key} changed ({info.get(key)!r} != {required!r})')
        if 'weight/reference_logits' not in saved:
            raise ValueError('cannot resume an 0082A-H optimizer checkpoint with the new '
                             '0082I reference parameter; start a fresh stream run '
                             'with a new --output path')
        model.load_state_dict({name: saved['weight/' + name]
                               for name in model.parameters_by_name})
        for index, name in enumerate(model.parameters_by_name):
            if not optimizer.restore_moments(index, saved['moment1/' + name],
                                              saved['moment2/' + name]):
                raise ValueError(f'invalid saved moments for {name}')
            if not optimizer.restore_master_weight(index, saved['master/' + name]):
                raise ValueError(f'invalid saved master weights for {name}')
        optimizer.step_index = int(info['step'])
    return info


def _run(args):
    members = _member_specs(args.member)
    if len(members) > args.max_models or max(m[0] for m in members) >= args.max_models:
        raise ValueError('model IDs exceed --max-models')
    if args.min_members > len(members):
        raise ValueError('--min-members exceeds available models')
    root = Path(args.mixed_shard_root).expanduser().resolve()
    rng = np.random.default_rng(args.seed)
    sources, manifest, samplers, weights = _stream_samplers(
        root, args.windows_metadata, args.partition_seed, args.context_length, rng)
    vocab = int(manifest['vocab_size'])
    ids = [member[0] for member in members]
    val = _open_members(args.val_member, args.val_targets)
    if val[0].tolist() != ids or val[1][0].shape[1] != vocab:
        raise ValueError('validation member IDs/order/vocabulary differ from checkpoint members')
    tokenizer_hash = manifest.get('tokenizer_sha256')
    teacher_fingerprints = {}
    for mid, checkpoint in members:
        digest = hashlib.sha256(_resolve_tokenizer(checkpoint).read_bytes()).hexdigest()
        if digest != tokenizer_hash:
            raise ValueError(f'model {mid} tokenizer does not match mixed shard tokenizer')
        teacher_fingerprints[str(mid)] = _checkpoint_fingerprint(checkpoint)
    positions = _scored_positions(args.context_length, score_last_k=args.score_last_k,
                                  score_stride=args.score_stride)
    scored = len(positions)
    windows_per_segment = _segment_windows_for_budget(
        args.segment_windows, members=len(members), vocab=vocab,
        scored=scored, budget_gib=args.max_cache_gib)
    per_segment_gib = (len(members) * vocab * scored * windows_per_segment * 2 / 1024**3)
    print(f'0082F streaming: members={ids} vocab={vocab} scored/window={scored} '
          f'windows/segment={windows_per_segment} temp_logits={per_segment_gib:.3f}GiB '
          f'train_source=parent_HELDOUT_validation_shards', flush=True)
    config = FusionConfig(vocab_size=vocab, max_models=args.max_models,
                          embed_dim=args.embed_dim, member_width=args.member_width,
                          decoder_width=args.decoder_width, decoder_layers=args.decoder_layers,
                          context_heads=args.context_heads, context_head_dim=args.context_head_dim,
                          context_hidden=args.context_hidden, seed=args.seed)
    model = ProbabilityResidualCombiner(config)
    if args.initialize_from_fusion:
        previous = ProbabilityResidualCombiner.load(args.initialize_from_fusion)
        prior = previous.config
        for field in ('vocab_size', 'max_models', 'embed_dim', 'member_width',
                      'decoder_width', 'decoder_layers'):
            if getattr(prior, field) != getattr(config, field):
                raise ValueError(f'pretrained fusion checkpoint has mismatched {field}')
        if prior.context_heads:
            raise ValueError('initialize-from-fusion expects a previous context-disabled combiner')
        model.load_state_dict(previous.state_dict())
        print(f'fusion context attention initialized from {args.initialize_from_fusion}', flush=True)
    val_context = None
    if args.context_heads:
        # These windows must be exactly those used to export --val-member.
        val_context = AlignedContextWindows(args.val_context_windows,
                                            args.val_context_metadata, val[2],
                                            tokenizer_sha256=tokenizer_hash)
        metadata = json.loads(Path(args.val_context_metadata).read_text(encoding='utf-8'))
        provenance = {int(entry['id']): entry for entry in metadata.get('models', [])}
        for mid, checkpoint in members:
            recorded = provenance.get(mid, {})
            if (not recorded.get('checkpoint_fingerprint') or
                    recorded['checkpoint_fingerprint'] != teacher_fingerprints[str(mid)] or
                    Path(recorded.get('checkpoint', '')).resolve() != checkpoint):
                raise ValueError(f'validation model {mid} was not exported from the same '
                                 'frozen checkpoint; regenerate validation logits')
    print(f'fusion architecture: weighted_reference=trainable_by_model_id (equal_initially) '
          f'embed_dim={args.embed_dim} member_width={args.member_width} '
          f'decoder_width={args.decoder_width} decoder_layers={args.decoder_layers} '
          f'context_heads={args.context_heads} context_head_dim={args.context_head_dim} '
          f'parameters={sum(int(p.data.size) for p in model.parameters()):,}; '
          f'input=[prediction_rows,models,{vocab}] (no context axis); '
          f'fusion_microbatch={args.batch_rows} grad_accum_steps={args.grad_accum_steps} '
          f'effective_batch_rows={args.batch_rows*args.grad_accum_steps}', flush=True)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_path = Path(args.output)
    last_path = Path(args.last_state) if args.last_state else best_path.with_name(best_path.stem + '_last.npz')
    run_dir = Path(args.work_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_expected = {'root': str(root), 'members': [[int(mid), str(ckpt)] for mid, ckpt in members],
                     'teacher_fingerprints': teacher_fingerprints,
                     'validation_paths': [str(Path(s.split(':', 1)[1]).resolve()) for s in args.val_member] +
                                         [str(Path(args.val_targets).resolve())],
                     'partition_seed': args.partition_seed, 'context_length': args.context_length,
                     'score_last_k': args.score_last_k, 'score_stride': args.score_stride,
                     'batch_rows': args.batch_rows, 'seed': args.seed}
    if args.context_heads:
        meta_expected.update({'context_heads': args.context_heads, 'context_head_dim': args.context_head_dim,
                              'context_hidden': args.context_hidden,
                              'val_context_windows': str(Path(args.val_context_windows).resolve()),
                              'val_context_metadata': str(Path(args.val_context_metadata).resolve())})
    best = float('inf')
    stale = 0
    step = 0
    segment = 0
    if args.resume:
        info = _read_last_state(last_path, model, opt, expected=meta_expected)
        rng.bit_generator.state = info['rng_state']
        best = float(info['best_val_ce'])
        stale = int(info['stale_evals'])
        step = int(info['step'])
        segment = int(info['segment'])
        print(f'resumed step={step} segment={segment} best_val_ce={best:.6f}', flush=True)
    else:
        if last_path.exists():
            raise FileExistsError(f'{last_path} exists: use --resume or choose new --output')
    # Older checkpoint files may lack telemetry counters; in that case the
    # counters begin at zero on this invocation (never infer them from steps).
    prior_telemetry = info.get('telemetry') if args.resume else None
    if args.resume and prior_telemetry is None:
        print('fusion_stream_telemetry reset=1 reason=legacy_checkpoint '
              'counts_start_at_resume', flush=True)
    meter = _FusionTelemetry([source.name for source in sources], previous=prior_telemetry)
    names = meter.names
    if len({re.sub(r'[^A-Za-z0-9_]', '_', n) for n in names}) != len(names):
        raise ValueError('source names conflict after CSV column normalization')
    csv_fields = _metrics_csv_fields(names, ids)
    if args.log_file:
        if args.resume:
            _reconcile_metrics_csv(args.log_file, csv_fields, step)
        elif Path(args.log_file).is_file() and Path(args.log_file).stat().st_size:
            raise FileExistsError(f'{args.log_file} exists: choose a new log file or --resume')
    previous_active_wall_s = float(info.get('telemetry_active_wall_s', 0.0)) if args.resume else 0.0
    active = run_dir / 'active_segment'
    provenance_path = run_dir / 'sample_trace.jsonl'
    start_time = time.perf_counter()
    early_stop = False
    while step < args.steps and not early_stop:
        # Never silently mix model snapshots in a multi-segment run.
        for mid, ckpt in members:
            if _checkpoint_fingerprint(ckpt) != teacher_fingerprints[str(mid)]:
                raise RuntimeError(f'parent checkpoint changed during fusion streaming: model {mid}')
        if active.exists():
            shutil.rmtree(active)  # stale cache after interrupted segment
        active.mkdir(parents=True)
        try:
            producer_start = time.perf_counter()
            windows_file = active / 'windows.npy'
            counts, trace = _make_segment_windows(windows_file, samplers, sources, weights,
                                                    windows_per_segment, args.context_length, rng)
            out_dir = active / 'logits'
            cmd = []
            for mid, ckpt in members:
                cmd.extend(['--member', f'{mid}:{ckpt}'])
            cmd.extend(['--windows', str(windows_file), '--output-dir', str(out_dir),
                        '--context-length', str(args.context_length),
                        '--score-last-k', str(args.score_last_k),
                        '--score-stride', str(args.score_stride),
                        '--lm-head-rows', str(args.lm_head_rows),
                        '--batch-windows', str(args.batch_windows)])
            export_main(cmd)
            with (out_dir / 'metadata.json').open(encoding='utf-8') as stream:
                exported = json.load(stream)
            if exported['tokenizer_sha256'] != tokenizer_hash or exported['rows'] != windows_per_segment*scored:
                raise RuntimeError('exported teacher logits do not match expected tokenizer/rows')
            specs = [f'{mid}:{out_dir / f"model_{mid}_logits.npy"}' for mid, _ in members]
            src = _open_members(specs, out_dir / 'targets.npy')
            train_context = (AlignedContextWindows(windows_file, out_dir / 'metadata.json',
                                                   src[2], tokenizer_sha256=tokenizer_hash)
                             if args.context_heads else None)
            if src[0].tolist() != ids:
                raise RuntimeError('export member identity mismatch')
            if len(trace) * scored != len(src[2]):
                raise RuntimeError('source window/teacher prediction alignment mismatch')
            _sync_fusion_backend()
            segment_producer_s = time.perf_counter() - producer_start
            meter.exported(windows_per_segment, args.context_length, scored,
                           len(members), segment_producer_s)
            segment_input = windows_per_segment * args.context_length * len(members)
            segment_scored = windows_per_segment * scored * len(members)
            _print_step_block(step, args.steps, segment, 'TEACHER EXPORT', [
                f'    Produced            windows={windows_per_segment:,} '
                f'teachers={len(members)} scored/window={scored:,}',
                f'    Token positions     teacher input={segment_input:,} '
                f'teacher scored={segment_scored:,} fusion available={len(src[2]):,}',
                f'    Producer speed      input={segment_input/max(1e-9,segment_producer_s):,.1f} tok/s '
                f'scored={segment_scored/max(1e-9,segment_producer_s):,.1f} tok/s '
                f'elapsed={segment_producer_s:.3f}s',
                *_source_table(
                    names, window_counts=counts,
                    available_counts={k: v*scored for k, v in counts.items()},
                    teacher_counts={k: v*args.context_length*len(members)
                                    for k, v in counts.items()}),
            ])
            order = rng.permutation(len(src[2]))
            train_clock = time.perf_counter()
            # ONE AdamW step per accumulation group.  The selected teacher
            # subset is fixed across microbatches within this optimizer step.
            # Partial groups at segment boundaries are trained rather than
            # silently dropped; all gradient contributions are row-weighted.
            effective_rows = args.batch_rows * args.grad_accum_steps
            for start in range(0, len(order), effective_rows):
                if step >= args.steps:
                    break
                group_ids = order[start:start + effective_rows]
                if not len(group_ids):
                    continue
                group_count = len(group_ids)
                micro_count = (group_count + args.batch_rows - 1) // args.batch_rows
                count = int(rng.integers(args.min_members, len(ids)+1))
                member_idx = rng.choice(len(ids), size=count, replace=False)
                batch_ids = src[0][member_idx]
                opt.zero_grad()
                ce = baseline = weighted_ce = None
                for micro_start in range(0, group_count, args.batch_rows):
                    row_ids = group_ids[micro_start:micro_start + args.batch_rows]
                    batch_logits = _batch(src[1], row_ids, member_idx,
                                          validate_finite=True)
                    batch_targets = src[2][row_ids]
                    context_kwargs = {}
                    if train_context is not None:
                        tokens, lengths = train_context.batch(row_ids)
                        context_kwargs = {'context_tokens': tokens, 'context_lengths': lengths}
                    share = len(row_ids) / group_count
                    micro_ce, micro_baseline, micro_weighted = model.loss_and_backward(
                        batch_logits, batch_ids, batch_targets, return_weighted=True,
                        defer_metrics=True, check_finite=False,
                        gradient_scale=share, **context_kwargs)
                    if ce is None:
                        ce = micro_ce * share
                        baseline = micro_baseline * share
                        weighted_ce = micro_weighted * share
                    else:
                        ce += micro_ce * share
                        baseline += micro_baseline * share
                        weighted_ce += micro_weighted * share
                grad_norm = _clip_grad_norm_(model.parameters(), args.grad_clip,
                                            device_only=True)
                opt.step()
                step += 1
                batch_sources = _source_rows_for_batch(group_ids, trace, scored)
                meter.trained(batch_sources)
                csv_row = None
                report_lines = []
                if step == 1 or step % args.log_interval == 0:
                    # GPU work is asynchronous: synchronize at logging boundaries,
                    # NOT every optimizer step, for meaningful wall-clock rates.
                    _sync_fusion_backend()
                    meter.train_seconds(time.perf_counter() - train_clock)
                    active_wall_s = (previous_active_wall_s +
                                     time.perf_counter() - start_time)
                    rates = meter.rates(active_wall_s)
                    # Resolve device-resident metrics only at log boundaries.
                    # For nonfinite norm, fail explicitly before checkpointing.
                    ce, baseline, weighted_ce = (float(scalar(metric)) for metric in
                                                 (ce, baseline, weighted_ce))
                    if not all(math.isfinite(v) for v in (ce, baseline, weighted_ce)):
                        raise FloatingPointError('nonfinite fusion CE at log boundary')
                    if grad_norm is not None:
                        grad_norm = float(scalar(grad_norm))
                        if not math.isfinite(grad_norm):
                            raise FloatingPointError('fusion gradient norm is nonfinite')
                    # Evaluate per-member reference metrics across ALL
                    # microbatches of this optimizer step.  Do not concatenate
                    # large [effective_rows,N,V] temporary GPU arrays.
                    batch_geom = 0.0
                    batch_member_ces = np.zeros(len(batch_ids), dtype=np.float64)
                    dyn_sum = dyn_squares = None
                    if args.context_heads:
                        dyn_sum = np.zeros(len(batch_ids), dtype=np.float64)
                        dyn_squares = np.zeros(len(batch_ids), dtype=np.float64)
                    for micro_start in range(0, group_count, args.batch_rows):
                        row_ids = group_ids[micro_start:micro_start + args.batch_rows]
                        current_logits = _batch(src[1], row_ids, member_idx,
                                                validate_finite=True)
                        current_targets = src[2][row_ids]
                        micro_geom, micro_ces = _batch_teacher_ces(current_logits, current_targets)
                        share = len(row_ids) / group_count
                        batch_geom += micro_geom * share
                        batch_member_ces += micro_ces * share
                        if dyn_sum is not None:
                            tokens, lengths = train_context.batch(row_ids)
                            dynamic = _to_host(model.member_weights(
                                batch_ids, context_tokens=tokens, context_lengths=lengths))
                            dyn_sum += dynamic.sum(axis=0)
                            dyn_squares += (dynamic**2).sum(axis=0)
                    if args.context_heads:
                        context_means = dyn_sum / group_count
                        context_stds = np.sqrt(np.maximum(
                            0.0, dyn_squares/group_count - context_means**2))
                        global_scores = _to_host(model.reference_logits.data[batch_ids])
                        global_w = np.exp(global_scores - np.max(global_scores))
                        global_w /= global_w.sum()
                    else:
                        context_means = context_stds = None
                        global_w = _to_host(model.member_weights(batch_ids))
                    report_lines.extend([
                        f'    Train CE            fusion={ce:.5f} weighted={weighted_ce:.5f} '
                        f'arithmetic={baseline:.5f} geometric={batch_geom:.5f}',
                        *_member_table(batch_ids, batch_member_ces,
                                       global_weights=global_w,
                                       context_means=context_means, context_stds=context_stds),
                        *_comparison_lines(ce, baseline, batch_geom, weighted_ce,
                                           batch_ids, batch_member_ces),
                        f'    Optimizer           grad_norm={grad_norm if grad_norm is not None else "disabled"} '
                        f'microbatches={micro_count} accum_target={args.grad_accum_steps}',
                        f'    Supervised tokens   batch={group_count:,} '
                        f'interval={meter.interval_rows:,} total={meter.total_rows:,}',
                        f'    Fusion speed        interval={rates["fusion_train_tok_s"]:,.1f} tok/s '
                        f'cumulative={rates["fusion_train_cumulative_tok_s"]:,.1f} tok/s '
                        f'end-to-end={rates["fusion_end_to_end_tok_s"]:,.1f} tok/s',
                        f'    Teacher totals      input={meter.teacher_input_tokens:,} '
                        f'scored={meter.teacher_scored_tokens:,} '
                        f'producer segment={segment_producer_s:.3f}s '
                        f'producer total={meter.teacher_seconds:.3f}s',
                        f'    Elapsed             train interval={rates["fusion_train_interval_s"]:.3f}s '
                        f'active={active_wall_s:.3f}s',
                        *_source_table(names, batch=batch_sources,
                                       interval=meter.interval_source_rows,
                                       total=meter.source_rows,
                                       total_rows=meter.total_rows,
                                       interval_seconds=rates['fusion_train_interval_s']),
                    ])
                    csv_row = {
                        'step': step, 'segment': segment, 'members': count,
                        'batch_supervised_tokens': group_count,
                        'interval_supervised_tokens': meter.interval_rows,
                        'total_supervised_tokens': meter.total_rows,
                        'teacher_input_tokens_total': meter.teacher_input_tokens,
                        'teacher_scored_tokens_total': meter.teacher_scored_tokens,
                        'teacher_producer_s_total': meter.teacher_seconds,
                        'teacher_producer_segment_s': segment_producer_s,
                        'teacher_input_tok_s_segment': segment_input/max(1e-9,segment_producer_s),
                        'teacher_scored_tok_s_segment': segment_scored/max(1e-9,segment_producer_s),
                        'elapsed_active_s': active_wall_s, 'ce': ce,
                        'arithmetic_ce': baseline, 'geometric_ce': batch_geom,
                        'weighted_ce': weighted_ce, 'grad_norm': grad_norm,
                        **rates}
                    for mid, val_ce in zip(batch_ids, batch_member_ces):
                        csv_row[f'model_{int(mid)}_ce'] = float(val_ce)
                    for src_name in names:
                        csv_row[f'batch_source_tokens_{src_name}'] = batch_sources[src_name]
                        csv_row[f'interval_source_tokens_{src_name}'] = meter.interval_source_rows[src_name]
                        csv_row[f'total_source_tokens_{src_name}'] = meter.source_rows[src_name]
                        csv_row[f'source_train_tok_s_{src_name}'] = (
                            meter.interval_source_rows[src_name]/max(1e-9,rates['fusion_train_interval_s']))
                        csv_row[f'source_mix_pct_{src_name}'] = (
                            100*meter.source_rows[src_name]/max(1,meter.total_rows))
                    meter.finish_interval()
                if step == 1 or step % args.eval_interval == 0 or step == args.steps:
                    if csv_row is None:
                        # An evaluation can occur between training log intervals;
                        # charge the queued optimizer work before starting it.
                        _sync_fusion_backend()
                        meter.train_seconds(time.perf_counter() - train_clock)
                    score, arith, geom, members_ce, weighted_val = _evaluate(
                        model, val, args.batch_rows, max_rows=args.val_max_rows or None,
                        return_weighted=True, context_source=val_context)
                    if args.context_heads:
                        global_scores = _to_host(model.reference_logits.data[val[0]])
                        global_w = np.exp(global_scores - np.max(global_scores))
                        global_w /= global_w.sum()
                        mean_w, std_w = _validation_context_weights(
                            model, val_context, val[0], max_rows=args.val_max_rows,
                            batch_rows=args.batch_rows, return_stats=True)
                    else:
                        global_w = _to_host(model.member_weights(val[0]))
                        mean_w = std_w = None
                    report_lines.extend([
                        f'    Validation CE       fusion={score:.5f} weighted={weighted_val:.5f} '
                        f'arithmetic={arith:.5f} geometric={geom:.5f}',
                        *_member_table(val[0], members_ce, global_weights=global_w,
                                       context_means=mean_w, context_stds=std_w),
                        *_comparison_lines(score, arith, geom, weighted_val,
                                           val[0], members_ce),
                    ])
                    if csv_row is not None:
                        csv_row.update(val_ce=score, val_arithmetic_ce=arith,
                                       val_geometric_ce=geom, val_weighted_ce=weighted_val)
                    if score < best:
                        best = score
                        stale = 0
                        model.save(best_path)
                        report_lines.append(f'    Best checkpoint     saved={best_path} (val CE={score:.5f})')
                    else:
                        stale += 1
                    if _should_stop_early(stale, args.early_stop_patience):
                        report_lines.append(f'    Early stop          best val CE={best:.5f}')
                        early_stop = True
                if report_lines:
                    ran_validation = step == 1 or step % args.eval_interval == 0 or step == args.steps
                    phase = ('TRAIN + VALIDATION' if csv_row is not None and ran_validation
                             else ('TRAIN' if csv_row is not None else 'VALIDATION'))
                    _print_step_block(step, args.steps, segment, phase,
                                      report_lines, members=count)
                if csv_row is not None and args.log_file:
                    _append_metrics_csv(args.log_file, csv_fields, csv_row)
                if (csv_row is not None or step == 1 or
                        step % args.eval_interval == 0 or step == args.steps):
                    # Metrics and validation are excluded from the optimizer rate.
                    train_clock = time.perf_counter()
                if early_stop:
                    break
            # Charge unreported optimizer time before checkpointing the segment.
            _sync_fusion_backend()
            meter.train_seconds(time.perf_counter() - train_clock)
            # Keep a reproducible segment trace; no full-vocab logits retained.
            with provenance_path.open('a', encoding='utf-8') as stream:
                record = {'segment': segment, 'step_end': step, 'counts': counts,
                          'sample_locations': trace,
                          'export_sha256': exported['token_windows_sha256'],
                          'context': args.context_length, 'positions': positions.tolist(),
                          'teacher_fingerprints': teacher_fingerprints}
                stream.write(json.dumps(record) + '\n')
            segment += 1
            _write_last_state(last_path, model, opt, rng, state={
                **meta_expected, 'grad_accum_steps': args.grad_accum_steps,
                'step': step, 'segment': segment,
                'best_val_ce': best, 'stale_evals': stale,
                'telemetry': meter.state(),
                'telemetry_active_wall_s': previous_active_wall_s +
                      time.perf_counter() - start_time})
        finally:
            # Memory-mapped views must be closed before deleting the files;
            # explicitly drop source references from the completed segment.
            if 'src' in locals():
                del src
            if 'train_context' in locals():
                del train_context
            gc.collect()
            if not args.keep_segments:
                shutil.rmtree(active, ignore_errors=True)
    print(f'fusion_stream_done step={step} segments={segment} '
          f'total_supervised_tokens={meter.total_rows} '
          f'teacher_input_tokens_total={meter.teacher_input_tokens} '
          f'teacher_scored_tokens_total={meter.teacher_scored_tokens} '
          f'{_format_source_rows("total_source_tokens", meter.source_rows, names)} '
          f'best_val_ce={best:.5f} elapsed_s={time.perf_counter()-start_time:.1f} '
          f'best={best_path} resumable={last_path}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--member', action='append', required=True, help='MODEL_ID:checkpoint directory')
    parser.add_argument('--mixed-shard-root', required=True)
    parser.add_argument('--windows-metadata', required=True,
                        help='0082B windows_metadata.json proving fusion train/val/test shard partitions')
    parser.add_argument('--partition-seed', type=int, default=82)
    parser.add_argument('--val-member', action='append', required=True, help='MODEL_ID:validation logits.npy')
    parser.add_argument('--val-targets', required=True)
    parser.add_argument('--output', required=True, help='best fusion checkpoint')
    parser.add_argument('--last-state', help='resumable state (default OUTPUT_STEM_last.npz)')
    parser.add_argument('--work-dir', default='./features/fusion_0082f_stream_cache')
    parser.add_argument('--keep-segments', action='store_true', help='retain last generated segment for debugging')
    parser.add_argument('--resume', action='store_true', help='resume at last finished segment')
    parser.add_argument('--teacher-context-length', '--context-length', dest='context_length',
                        type=int, default=256, help='ONLY the frozen teacher input window; not a fusion-model parameter')
    parser.add_argument('--score-last-k', type=int, default=256)
    parser.add_argument('--score-stride', type=int, default=1)
    parser.add_argument('--producer-segment-windows', '--segment-windows', dest='segment_windows', type=int, default=32,
                        help='bounded teacher-inference scratch segment; never a fusion context length')
    parser.add_argument('--max-cache-gib', type=float, default=2.0)
    parser.add_argument('--teacher-batch-windows', '--batch-windows', dest='batch_windows', type=int, default=4,
                        help='teacher forward microbatch (windows, not fusion prediction rows)')
    parser.add_argument('--lm-head-rows', type=int, default=64)
    parser.add_argument('--fusion-batch-rows', '--batch-rows', dest='batch_rows', type=int, default=32,
                        help='combiner optimizer batch of aligned probability distributions')
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                        help='number of fusion microbatches per AdamW update (effective rows = batch-rows * grad-accum-steps)')
    parser.add_argument('--max-models', type=int, default=10)
    parser.add_argument('--min-members', type=int, default=2)
    parser.add_argument('--fusion-embed-dim', '--embed-dim', dest='embed_dim', type=int, default=16)
    parser.add_argument('--fusion-member-width', '--member-width', dest='member_width', type=int, default=8)
    parser.add_argument('--fusion-decoder-width', '--decoder-width', dest='decoder_width', type=int, default=24)
    parser.add_argument('--fusion-decoder-layers', '--decoder-layers', dest='decoder_layers', type=int, default=1)
    parser.add_argument('--context-heads', type=int, choices=(0, 2), default=0,
                        help='0: existing 0082I; 2: model-query context attention')
    parser.add_argument('--context-head-dim', type=int, default=16)
    parser.add_argument('--context-hidden', type=int, default=32)
    parser.add_argument('--val-context-windows',
                        help='EXACT token windows used for frozen validation logit export')
    parser.add_argument('--val-context-metadata',
                        help='metadata.json of same validation logits export; checked against window SHA256 and targets')
    parser.add_argument('--initialize-from-fusion',
                        help='load trained 0082I combiner before adding zero-initialized attention (new optimizer)')
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--val-max-rows', type=int, default=2048)
    parser.add_argument('--eval-interval', type=int, default=200)
    parser.add_argument('--early-stop-patience', type=int, default=12)
    parser.add_argument('--log-interval', type=int, default=25)
    parser.add_argument('--log-file', help='optional per-log-step CSV (batch/source counts, throughput, CE)')
    parser.add_argument('--seed', type=int, default=82)
    args = parser.parse_args(argv)
    for name in ('context_length', 'score_last_k', 'score_stride', 'segment_windows',
                 'batch_windows', 'lm_head_rows', 'batch_rows', 'grad_accum_steps', 'max_models',
                 'embed_dim', 'member_width', 'decoder_width', 'decoder_layers',
                 'min_members', 'steps', 'eval_interval', 'log_interval',
                 'context_head_dim', 'context_hidden'):
        if getattr(args, name) < 1:
            parser.error(f'{name.replace("_", "-")} must be >=1')
    if args.score_last_k > args.context_length or args.early_stop_patience < 0 or args.val_max_rows < 0:
        parser.error('invalid score-last-k / early-stop-patience / val-max-rows')
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in (args.lr, args.max_cache_gib)):
        parser.error('lr and max-cache-gib must be finite and >0')
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in (args.weight_decay, args.grad_clip)):
        parser.error('weight-decay and grad-clip must be finite and >=0')
    if args.context_heads and (not args.val_context_windows or not args.val_context_metadata):
        parser.error('context attention requires --val-context-windows and --val-context-metadata')
    if args.resume and args.initialize_from_fusion:
        parser.error('--resume cannot be combined with --initialize-from-fusion')
    _run(args)


if __name__ == '__main__':
    main()
