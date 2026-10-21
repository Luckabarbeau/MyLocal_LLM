"""Core online frozen-teacher probability-fusion trainer (0084B).

No examples/ dependency and NO disk cache of full vocabulary teacher logits.
The teachers execute forward-only, sequentially, on the same causal windows;
only ProbabilityResidualCombiner receives optimizer updates.  The current
first-class mode uses short-context/plain teacher inference, not routed 64K
historical evaluation.  Its splits use whole teacher-validation shards.
"""
from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from ..backend import xp, asnumpy, scalar, BACKEND_NAME
from ..checkpoint import load_checkpoint
from ..config import ModelConfig
from ..model.decoder_lm import DecoderLanguageModel
from ..optim.adamw import AdamW
from ..runtime_defaults import apply_runtime_defaults
from .context_windows import checkpoint_fingerprint
from .online_data import make_online_samplers
from .probability_fusion import FusionConfig, ProbabilityResidualCombiner, _log_softmax
from .online_logging import (OnlineFusionTelemetry, metrics_csv_fields, append_metrics_csv,
                             reconcile_metrics_csv, source_table, member_table,
                             comparison_lines, print_step_block)


def parse_member_specs(specs):
    out = []
    for spec in specs:
        label, separator, directory = spec.partition(':')
        if not separator:
            raise ValueError(f'expected --member ID:CHECKPOINT, got {spec!r}')
        idx = int(label)
        if idx < 0 or idx in [m for m, _ in out]:
            raise ValueError('teacher model IDs must be distinct nonnegative integers')
        out.append((idx, Path(directory).expanduser().resolve()))
    if len(out) < 2:
        raise ValueError('online fusion requires >=2 frozen teachers')
    return out


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def _tokenizer_in(directory):
    for path in (directory / 'tokenizer.json', directory.parent / 'tokenizer.json'):
        if path.exists():
            return path
    raise FileNotFoundError(f'no teacher tokenizer.json in {directory} or its parent')


def validate_teacher_tokenizers(members, mixed_root, manifest):
    expected = _sha(Path(mixed_root) / 'tokenizer.json')
    if manifest.get('tokenizer_sha256') not in (None, expected):
        raise ValueError('mixture tokenizer SHA256 mismatch')
    fingerprints = {}
    for model_id, directory in members:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        if _sha(_tokenizer_in(directory)) != expected:
            raise ValueError(f'tokenizer SHA256 mismatch: teacher {model_id} at {directory}')
        fingerprints[str(model_id)] = checkpoint_fingerprint(directory)
    return expected, fingerprints


def load_frozen_teacher(directory, context_length):
    """Load a checkpoint without training machinery or 64K history buffers."""
    with (directory / 'config.json').open(encoding='utf-8') as f:
        config = ModelConfig(**json.load(f))
    memory = getattr(config, 'memory_context', None)
    if memory is not None:
        config = dataclasses.replace(config, memory_context=dataclasses.replace(memory, enabled=False))
    config = dataclasses.replace(config, context_length=int(context_length))
    teacher = DecoderLanguageModel(config, rng_seed=42, dtype=config.dtype)
    names = [p.name for p in teacher.parameters()]
    state, _, _ = load_checkpoint(directory, param_names=names,
                                  skip_optimizer=True, skip_training=True)
    for parameter in teacher.parameters():
        if parameter.name not in state:
            raise RuntimeError(f'frozen teacher checkpoint missing parameter {parameter.name}')
        parameter.data[...] = state[parameter.name]
    return teacher, config


def prediction_positions(context_length, last_k, stride):
    n = int(context_length)
    k = n if last_k is None else int(last_k)
    s = int(stride)
    if n < 2 or k < 1 or k > n or s < 1:
        raise ValueError('invalid context-length, score-last-k or score-stride')
    return np.arange(n-1, n-k-1, -s, dtype=np.int64)[::-1].copy()


def aligned_training_rows(windows, positions):
    """Return target and strictly preceding token contexts per selected row."""
    rows = int(len(windows)) * len(positions)
    indices = np.repeat(np.arange(len(windows)), len(positions))
    lengths = np.tile(positions + 1, len(windows))
    tokens = np.asarray(windows[indices, :-1], dtype=np.int64)
    targets = np.asarray(windows[:, positions+1].reshape(rows), dtype=np.int64)
    return targets, tokens, lengths


def frozen_teacher_logits(teachers, windows, positions, *, head_rows=64, timing_events=None):
    """Compute exact complete-vocabulary aligned logits, without backward tape.

    Each teacher's temporary activations are released before the next teacher
    runs; only its bounded logits are retained. All teachers see identical
    token windows and prediction positions. The final stack lives on the GPU.
    """
    if int(head_rows) < 1:
        raise ValueError('lm-head-rows must be positive')
    batch = xp.asarray(windows[:, :-1], dtype=xp.int64)
    rows = int(len(windows) * len(positions))
    arrays = []
    # CUDA events timestamp *device* work without synchronizing every teacher
    # tile; both teachers remain forward-only and run on the same stream.
    started = None
    if timing_events is not None:
        if BACKEND_NAME == 'cupy':
            started = xp.cuda.Event()
            started.record()
        else:
            started = time.perf_counter()
    for teacher in teachers:
        hidden = teacher.forward_body(batch, return_cache=False)
        chosen = hidden[:, positions, :].reshape(rows, hidden.shape[-1])
        tiles = []
        for first in range(0, rows, head_rows):
            # Contiguous BF16 projection inputs preserve the grouped GEMM path.
            tile = xp.array(chosen[first:first+head_rows, None, :], copy=True, order='C')
            logits, _ = teacher.output_proj.forward(tile)
            tiles.append(logits[:, 0, :].astype(xp.float32))
            del tile, logits
        arrays.append(xp.concatenate(tiles, axis=0))
        del hidden, chosen, tiles
    result = xp.stack(arrays, axis=1)
    if timing_events is not None:
        if BACKEND_NAME == 'cupy':
            ended = xp.cuda.Event()
            ended.record()
            timing_events.append((started, ended))
        else:
            timing_events.append(time.perf_counter() - started)
    return result


def measured_teacher_seconds(events):
    """Read queued CUDA event durations only at an optimizer log boundary."""
    if BACKEND_NAME == 'cupy':
        return sum(xp.cuda.get_elapsed_time(first, last) * 0.001
                   for first, last in events)
    return sum(float(seconds) for seconds in events)


def _sync_backend():
    if BACKEND_NAME == 'cupy':
        xp.cuda.get_current_stream().synchronize()


def _reference_diagnostics(logits, targets):
    """Full-vocabulary teacher and geometric CE on exactly the scored rows."""
    lp = _log_softmax(xp.asarray(logits, dtype=xp.float32))
    y = xp.asarray(targets, dtype=xp.int64)
    idx = xp.arange(len(targets))
    individual = -xp.sum(xp.take_along_axis(lp, y[:, None, None], axis=2)[:, :, 0], axis=0)
    geo = _log_softmax(xp.mean(lp, axis=1))
    geometric = -xp.sum(geo[idx, y])
    return individual, geometric


def _global_weights(model, ids):
    x = np.asarray(asnumpy(model.reference_logits.data[np.asarray(ids, dtype=np.int64)]), dtype=np.float64)
    z = np.exp(x - np.max(x))
    return z / z.sum()


def _clip_grad(parameters, cap):
    if cap <= 0:
        return None
    square = xp.asarray(0., dtype=xp.float32)
    for param in parameters:
        square += xp.sum(xp.asarray(param.grad, dtype=xp.float32)**2)
    norm = math.sqrt(float(scalar(square)))
    if not math.isfinite(norm):
        raise FloatingPointError(f'non-finite fusion gradient norm: {norm}')
    if norm > cap:
        scale = cap / (norm + 1e-12)
        for param in parameters:
            param.grad *= scale
    return norm


def train_online_step(model, optimizer, teachers, sampler, positions, *,
                      rows_per_update, teacher_batch_windows, batch_rows,
                      lm_head_rows, grad_clip, member_ids, diagnostics=False):
    """Accumulate EXACT supervised-row-weighted CE, with optional 0082F telemetry.

    Diagnostic reductions reuse the *same* online teacher logits and scored
    targets. They never trigger a second teacher forward or retain logits on
    disk or across optimizer updates.
    """
    if min(rows_per_update, teacher_batch_windows, batch_rows, lm_head_rows) < 1:
        raise ValueError('online batch and tile sizes must be positive')
    model.zero_grad()
    total = baseline = weighted = 0.0
    seen = window_count = micro_count = 0
    source_rows = Counter()
    teacher_events = []
    refs = xp.zeros(len(member_ids), dtype=xp.float64) if diagnostics else None
    geom_sum = xp.asarray(0., dtype=xp.float64) if diagnostics else None
    dyn_sum = xp.zeros(len(member_ids), dtype=xp.float64) if diagnostics and model.config.context_heads else None
    dyn_square = xp.zeros(len(member_ids), dtype=xp.float64) if dyn_sum is not None else None
    scored = int(len(positions))
    while seen < rows_per_update:
        windows_needed = min(teacher_batch_windows,
                             math.ceil((rows_per_update - seen) / scored))
        windows, names = sampler.batch(windows_needed)
        window_count += len(windows)
        logits = frozen_teacher_logits(teachers, windows, positions, head_rows=lm_head_rows,
                                       timing_events=teacher_events)
        targets, contexts, lengths = aligned_training_rows(windows, positions)
        remaining = rows_per_update - seen
        limit = min(len(targets), remaining)
        for j, name in enumerate(names):
            source_rows[name] += max(0, min(scored, limit - j*scored))
        for start in range(0, limit, batch_rows):
            stop = min(start + batch_rows, limit)
            count = stop - start
            context_kwargs = ({'context_tokens': contexts[start:stop],
                               'context_lengths': lengths[start:stop]}
                              if model.config.context_heads else {})
            ce, ar, we = model.loss_and_backward(
                logits[start:stop], member_ids, targets[start:stop],
                return_weighted=True, gradient_scale=count / rows_per_update,
                check_finite=False, **context_kwargs)
            total += ce * count
            baseline += ar * count
            weighted += we * count
            micro_count += 1
            if diagnostics:
                member_sum, geometric = _reference_diagnostics(
                    logits[start:stop], targets[start:stop])
                refs += member_sum
                geom_sum += geometric
                if dyn_sum is not None:
                    dynamic = model.member_weights(member_ids, **context_kwargs)
                    dyn_sum += xp.sum(dynamic, axis=0)
                    dyn_square += xp.sum(dynamic * dynamic, axis=0)
        seen += limit
        del logits
    norm = _clip_grad(model.parameters(), grad_clip)
    optimizer.step()
    result = {'fusion_ce': total / seen, 'arithmetic_ce': baseline / seen,
              'weighted_ce': weighted / seen, 'supervised_rows': seen, 'grad_norm': norm,
              'source_rows': source_rows, 'teacher_windows': window_count,
              'microbatches': micro_count, 'teacher_events': teacher_events}
    if diagnostics:
        result['geometric_ce'] = float(scalar(geom_sum)) / seen
        result['member_ce'] = (np.asarray(asnumpy(refs), dtype=np.float64) / seen).tolist()
        if dyn_sum is not None:
            means = np.asarray(asnumpy(dyn_sum), dtype=np.float64) / seen
            squares = np.asarray(asnumpy(dyn_square), dtype=np.float64) / seen
            result['context_means'] = means
            result['context_stds'] = np.sqrt(np.maximum(0., squares - means**2))
    return result


def evaluate_online(model, teachers, windows, positions, *, member_ids,
                    teacher_batch_windows, batch_rows, lm_head_rows):
    """Held-out evaluation of exactly aligned member, geometric and fused CE."""
    ce_sum = ar_sum = weighted_sum = geom_sum = 0.0
    teacher_sum = np.zeros(len(member_ids), dtype=np.float64)
    dynamic_sum = np.zeros(len(member_ids), dtype=np.float64)
    dynamic_squares = np.zeros(len(member_ids), dtype=np.float64)
    total = 0
    for wstart in range(0, len(windows), teacher_batch_windows):
        batch = windows[wstart:wstart+teacher_batch_windows]
        logits = frozen_teacher_logits(teachers, batch, positions, head_rows=lm_head_rows)
        targets, context, lengths = aligned_training_rows(batch, positions)
        for first in range(0, len(targets), batch_rows):
            last = min(first + batch_rows, len(targets))
            z = logits[first:last]
            y = xp.asarray(targets[first:last])
            kw = ({'context_tokens': context[first:last],
                   'context_lengths': lengths[first:last]}
                  if model.config.context_heads else {})
            fused = model.forward(z, member_ids, check_finite=False, **kw)
            logp = _log_softmax(z)
            pa = xp.mean(xp.exp(logp), axis=1)
            if model.config.context_heads:
                weights = model.member_weights(member_ids, **kw)
                dynamic = np.asarray(asnumpy(weights), dtype=np.float64)
                dynamic_sum += dynamic.sum(axis=0)
                dynamic_squares += (dynamic**2).sum(axis=0)
            else:
                weights = model.member_weights(member_ids)[None, :]
            pw = xp.sum(xp.exp(logp) * weights[:, :, None], axis=1)
            rows = last - first
            idx = xp.arange(rows)
            ce_sum += -float(scalar(xp.sum(xp.log(xp.maximum(fused[idx, y], 1e-37)))))
            ar_sum += -float(scalar(xp.sum(xp.log(xp.maximum(pa[idx, y], 1e-37)))))
            weighted_sum += -float(scalar(xp.sum(xp.log(xp.maximum(pw[idx, y], 1e-37)))))
            geo = _log_softmax(xp.mean(logp, axis=1))
            geom_sum += -float(scalar(xp.sum(geo[idx, y])))
            teacher_sum += np.asarray(asnumpy(-xp.sum(
                xp.take_along_axis(logp, y[:, None, None], axis=2)[:, :, 0], axis=0)),
                dtype=np.float64)
            total += rows
        del logits
    result = {'fusion_ce': ce_sum / total, 'arithmetic_ce': ar_sum / total,
              'weighted_ce': weighted_sum / total, 'geometric_ce': geom_sum / total,
              'member_ce': (teacher_sum / total).tolist(), 'rows': total}
    if model.config.context_heads:
        means = dynamic_sum / total
        result['context_means'] = means
        result['context_stds'] = np.sqrt(np.maximum(0., dynamic_squares / total - means**2))
    return result


def _atomic_save_npz(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **payload)
    os.replace(tmp, path)


def save_last_state(path, model, optimizer, rng, metadata):
    payload = {'__meta_json__': np.asarray(json.dumps({**metadata,
                'rng_state': rng.bit_generator.state})),
               '__config_json__': np.asarray(json.dumps(dataclasses.asdict(model.config)))}
    for i, (name, parameter) in enumerate(model.parameters_by_name.items()):
        payload[f'weight/{name}'] = np.asarray(asnumpy(parameter.data))
        payload[f'moment1/{name}'] = np.asarray(asnumpy(optimizer.m[i]))
        payload[f'moment2/{name}'] = np.asarray(asnumpy(optimizer.v[i]))
        payload[f'master/{name}'] = np.asarray(asnumpy(optimizer.master_weights[i]))
    _atomic_save_npz(path, payload)


def restore_last_state(path, model, optimizer, expected):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['__meta_json__']))
        saved_config = FusionConfig(**json.loads(str(data['__config_json__'])))
        if saved_config != model.config:
            raise ValueError('fusion configuration changed since saved optimizer checkpoint')
        for key, val in expected.items():
            if meta.get(key) != val:
                raise ValueError(f'cannot resume: {key} changed')
        model.load_state_dict({name: data[f'weight/{name}'] for name in model.parameters_by_name})
        for i, name in enumerate(model.parameters_by_name):
            if not optimizer.restore_moments(i, data[f'moment1/{name}'], data[f'moment2/{name}']):
                raise ValueError(f'invalid moments: {name}')
            if not optimizer.restore_master_weight(i, data[f'master/{name}']):
                raise ValueError(f'invalid master weights: {name}')
        optimizer.step_index = int(meta['step'])
    return meta


def build_parser():
    p = argparse.ArgumentParser(description='Core online frozen-teacher probability-fusion training')
    p.add_argument('--member', action='append', required=True, help='ID:CHECKPOINT_DIRECTORY')
    p.add_argument('--mixed-shard-root', required=True)
    p.add_argument('--output', required=True, help='Portable best fusion .npz checkpoint')
    p.add_argument('--resume', action='store_true', help='Resume from output.last.npz (AdamW moments included)')
    p.add_argument('--context-length', type=int, default=512)
    p.add_argument('--score-last-k', type=int, default=512)
    p.add_argument('--score-stride', type=int, default=1)
    p.add_argument('--batch-windows', type=int, default=1, help='Teacher inference windows per call')
    p.add_argument('--lm-head-rows', type=int, default=64)
    p.add_argument('--batch-rows', type=int, default=128, help='Combiner gradient row tile')
    p.add_argument('--grad-accum-steps', type=int, default=128, help='Combiner row tiles per optimizer step')
    p.add_argument('--max-models', type=int, default=10)
    p.add_argument('--min-members', type=int, default=2)
    p.add_argument('--context-heads', type=int, choices=(0, 2), default=2)
    p.add_argument('--context-head-dim', type=int, default=16)
    p.add_argument('--context-hidden', type=int, default=32)
    p.add_argument('--embed-dim', type=int, default=16)
    p.add_argument('--member-width', type=int, default=8)
    p.add_argument('--decoder-width', type=int, default=24)
    p.add_argument('--decoder-layers', type=int, default=1)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--val-windows', type=int, default=8)
    p.add_argument('--eval-interval', type=int, default=200)
    p.add_argument('--save-interval', type=int, default=200)
    p.add_argument('--log-interval', type=int, default=10)
    p.add_argument('--early-stop-patience', type=int, default=12)
    p.add_argument('--partition-seed', type=int, default=82)
    p.add_argument('--seed', type=int, default=82)
    p.add_argument('--runtime-profile', choices=('auto', 'gpu-fast', 'reference', 'consumer-gpu'), default='gpu-fast')
    p.add_argument('--log-file')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.steps < 1 or args.val_windows < 1 or args.grad_accum_steps < 1:
        raise ValueError('steps/val-windows/grad-accum-steps must be positive')
    if not (2 <= args.min_members <= len(args.member) <= args.max_models):
        raise ValueError('online initial implementation trains the full listed teacher set; min-members must be <= count')
    members = parse_member_specs(args.member)
    positions = prediction_positions(args.context_length, args.score_last_k, args.score_stride)
    samplers, manifest, paths = make_online_samplers(args.mixed_shard_root,
                      args.context_length, args.partition_seed, args.seed)
    tokenizer_sha, teacher_fingerprints = validate_teacher_tokenizers(
        members, args.mixed_shard_root, manifest)
    apply_runtime_defaults(args.runtime_profile, backend_name=BACKEND_NAME, training=True, precision="float32")
    np.random.seed(args.seed)
    xp.random.seed(args.seed)
    teachers = []
    vocab = None
    for model_id, directory in members:
        print(f'loading frozen teacher {model_id}: {directory}', flush=True)
        teacher, cfg = load_frozen_teacher(directory, args.context_length)
        if vocab is None:
            vocab = int(cfg.vocab_size)
        elif int(cfg.vocab_size) != vocab:
            raise ValueError('teacher vocabulary sizes differ')
        teachers.append(teacher)
    config = FusionConfig(vocab_size=vocab, max_models=args.max_models,
             embed_dim=args.embed_dim, member_width=args.member_width,
             decoder_width=args.decoder_width, decoder_layers=args.decoder_layers,
             context_heads=args.context_heads, context_head_dim=args.context_head_dim,
             context_hidden=args.context_hidden, seed=args.seed)
    model = ProbabilityResidualCombiner(config)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # The validation windows are chosen once from a disjoint file partition.
    val_windows, _ = samplers['val'].batch(args.val_windows)
    rng = samplers['train'].rng
    member_ids = [item[0] for item in members]
    output = Path(args.output)
    last = output.with_name(output.stem + '.last.npz')
    expected = {'teachers': [[i, str(p)] for i, p in members],
                'teacher_fingerprints': teacher_fingerprints,
                'tokenizer_sha256': tokenizer_sha, 'partition_seed': args.partition_seed,
                'context_length': args.context_length, 'prediction_positions': positions.tolist(),
                'batch_rows': args.batch_rows, 'grad_accum_steps': args.grad_accum_steps,
                'batch_windows': args.batch_windows,
                'val_windows_sha256': hashlib.sha256(val_windows.tobytes()).hexdigest()}
    best = float('inf'); stale = 0; start_step = 0
    if args.resume:
        data = restore_last_state(last, model, optimizer, expected)
        start_step = int(data['step']); best = float(data['best_ce']); stale = int(data['stale'])
        rng.bit_generator.state = data['rng_state']
        print(f'fusion_online_resume step={start_step} best_ce={best:.6f}', flush=True)
    elif output.exists() or last.exists():
        raise FileExistsError('online fusion output exists: use --resume or a new --output')
    print(f'fusion_online members={member_ids} params={sum(p.size for p in model.parameters())} '
          f'context={args.context_length} scored={len(positions)} '
          f'rows/update={args.batch_rows*args.grad_accum_steps} '
          f'val_windows={len(val_windows)} disk_logit_cache=disabled '
          'teacher_memory=plain, frozen=true', flush=True)
    source_names = sorted(samplers['train'].names) if hasattr(samplers['train'], 'names') else sorted(
        getattr(s, 'name', '') for s in getattr(samplers['train'], 'sources', []))
    # Tiny externally provided samplers may not publish source names. In real
    # mixed-shard sampling this list is known before training begins.
    if not source_names:
        source_names = sorted(manifest.get('sources', {})) or ['src']
    meter = OnlineFusionTelemetry(source_names,
            previous=data.get('telemetry') if args.resume else None)
    csv_fields = metrics_csv_fields(source_names, member_ids)
    log_path = Path(args.log_file) if args.log_file else None
    if log_path:
        if args.resume:
            reconcile_metrics_csv(log_path, csv_fields, start_step)
        elif log_path.exists() and log_path.stat().st_size:
            raise FileExistsError(f'fusion metrics CSV already exists: {log_path}; '
                                  'use a new --log-file or --resume')
    print('fusion_online_logging=0082f_compatible '
          'segment=online teacher_producer_segment=last_log_interval '
          'fusion_train_clock=nonteacher_wall estimate '
          'teacher_forward_timing=cuda_events_or_cpu_wall', flush=True)
    for step in range(start_step+1, args.steps+1):
        start = time.perf_counter()
        count = int(rng.integers(args.min_members, len(teachers) + 1))
        chosen = np.sort(rng.choice(len(teachers), size=count, replace=False))
        active_teachers = [teachers[int(j)] for j in chosen]
        active_ids = [member_ids[int(j)] for j in chosen]
        ran_validation = step == 1 or step % args.eval_interval == 0 or step == args.steps
        write_report = step == 1 or step % args.log_interval == 0 or ran_validation
        metrics = train_online_step(model, optimizer, active_teachers, samplers['train'], positions,
                rows_per_update=args.batch_rows*args.grad_accum_steps,
                teacher_batch_windows=args.batch_windows, batch_rows=args.batch_rows,
                lm_head_rows=args.lm_head_rows, grad_clip=args.grad_clip, member_ids=active_ids,
                diagnostics=write_report)
        # Work is queued asynchronously on CuPy; synchronize once per
        # optimizer update before recording the end-to-end timer.
        _sync_backend()
        elapsed = time.perf_counter() - start
        teacher_seconds = measured_teacher_seconds(metrics['teacher_events'])
        meter.record_step(by_source=metrics['source_rows'], members=len(active_ids),
                          context_length=args.context_length, scored=len(positions),
                          windows=metrics['teacher_windows'], wall_seconds=elapsed,
                          teacher_seconds=teacher_seconds)
        val = None
        if ran_validation:
            val_start = time.perf_counter()
            val = evaluate_online(model, teachers, val_windows, positions,
                    member_ids=member_ids, teacher_batch_windows=args.batch_windows,
                    batch_rows=args.batch_rows, lm_head_rows=args.lm_head_rows)
            _sync_backend()
            meter.active_seconds += time.perf_counter() - val_start
        if write_report:
            rates = meter.rates()
            global_weights = _global_weights(model, active_ids)
            train_ces = np.asarray(metrics['member_ce'])
            train_geom = metrics['geometric_ce']
            report = [
                f'    Train CE            fusion={float(metrics["fusion_ce"]):.5f} '
                f'weighted={float(metrics["weighted_ce"]):.5f} '
                f'arithmetic={float(metrics["arithmetic_ce"]):.5f} '
                f'geometric={train_geom:.5f}',
                *member_table(active_ids, train_ces, global_weights=global_weights,
                              context_means=metrics.get('context_means'),
                              context_stds=metrics.get('context_stds')),
                *comparison_lines(float(metrics['fusion_ce']), float(metrics['arithmetic_ce']),
                                  train_geom, float(metrics['weighted_ce']),
                                  active_ids, train_ces),
                f'    Optimizer           grad_norm={metrics["grad_norm"] if metrics["grad_norm"] is not None else "disabled"} '
                f'microbatches={metrics["microbatches"]} accum_target={args.grad_accum_steps}',
                f'    Supervised tokens   batch={metrics["supervised_rows"]:,} '
                f'interval={meter.interval_rows:,} total={meter.total_rows:,}',
                f'    Fusion speed        interval={rates["fusion_train_tok_s"]:,.1f} tok/s '
                f'cumulative={rates["fusion_train_cumulative_tok_s"]:,.1f} tok/s '
                f'end-to-end={rates["fusion_end_to_end_tok_s"]:,.1f} tok/s',
                f'    Teacher totals      input={meter.teacher_input_tokens:,} '
                f'scored={meter.teacher_scored_tokens:,} '
                f'online interval={rates["teacher_producer_segment_s"]:.3f}s '
                f'online total={meter.teacher_seconds:.3f}s',
                f'    Elapsed             train interval={rates["fusion_train_interval_s"]:.3f}s '
                f'active={rates["elapsed_active_s"]:.3f}s',
                *source_table(source_names, batch=metrics['source_rows'],
                              interval=meter.interval_sources, total=meter.source_rows,
                              total_rows=meter.total_rows,
                              interval_seconds=rates['fusion_train_interval_s']),
            ]
            csv_row = {
                'step': step, 'segment': 0, 'members': len(active_ids),
                'batch_supervised_tokens': metrics['supervised_rows'],
                'interval_supervised_tokens': meter.interval_rows,
                'total_supervised_tokens': meter.total_rows,
                'teacher_input_tokens_total': meter.teacher_input_tokens,
                'teacher_scored_tokens_total': meter.teacher_scored_tokens,
                'elapsed_active_s': rates['elapsed_active_s'],
                'ce': float(metrics['fusion_ce']),
                'arithmetic_ce': float(metrics['arithmetic_ce']),
                'geometric_ce': train_geom,
                'weighted_ce': float(metrics['weighted_ce']),
                'grad_norm': metrics['grad_norm'],
                **rates,
            }
            for member_id, value in zip(active_ids, train_ces):
                csv_row[f'model_{member_id}_ce'] = float(value)
            for name in source_names:
                csv_row[f'batch_source_tokens_{name}'] = metrics['source_rows'].get(name, 0)
                csv_row[f'interval_source_tokens_{name}'] = meter.interval_sources[name]
                csv_row[f'total_source_tokens_{name}'] = meter.source_rows[name]
                csv_row[f'source_train_tok_s_{name}'] = (
                    meter.interval_sources[name] / max(1e-9, rates['fusion_train_interval_s']))
                csv_row[f'source_mix_pct_{name}'] = (
                    100 * meter.source_rows[name] / max(1, meter.total_rows))
            if val is not None:
                report.extend([
                    f'    Validation CE       fusion={val["fusion_ce"]:.5f} '
                    f'weighted={val["weighted_ce"]:.5f} '
                    f'arithmetic={val["arithmetic_ce"]:.5f} '
                    f'geometric={val["geometric_ce"]:.5f}',
                    *member_table(member_ids, val['member_ce'],
                                  global_weights=_global_weights(model, member_ids),
                                  context_means=val.get('context_means'),
                                  context_stds=val.get('context_stds')),
                    *comparison_lines(val['fusion_ce'], val['arithmetic_ce'],
                                      val['geometric_ce'], val['weighted_ce'],
                                      member_ids, val['member_ce']),
                ])
                csv_row.update(val_ce=val['fusion_ce'],
                               val_arithmetic_ce=val['arithmetic_ce'],
                               val_geometric_ce=val['geometric_ce'],
                               val_weighted_ce=val['weighted_ce'])
                if val['fusion_ce'] < best:
                    best = val['fusion_ce']; stale = 0
                    model.save(output)
                    report.append(f'    Best checkpoint     saved={output} (val CE={best:.5f})')
                else:
                    stale += 1
            print_step_block(step, args.steps,
                             'TRAIN + VALIDATION' if val is not None else 'TRAIN',
                             report, members=len(active_ids))
            if log_path:
                append_metrics_csv(log_path, csv_fields, csv_row)
            meter.finish_interval()
        if step % args.save_interval == 0 or step == args.steps or (args.early_stop_patience and stale >= args.early_stop_patience):
            save_last_state(last, model, optimizer, rng,
                            {**expected, 'step': step, 'best_ce': best, 'stale': stale,
                             'telemetry': meter.state()})
        if args.early_stop_patience and stale >= args.early_stop_patience:
            print(f'fusion_online_early_stop step={step} best_ce={best:.6f}', flush=True)
            break
    print(f'fusion_online_complete saved_best={output} saved_resume={last}', flush=True)


if __name__ == '__main__':
    main()
