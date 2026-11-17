"""Isolated production-checkpoint GPU diagnostic; never saves a trainer checkpoint.

Launch via .venv/bin/python -B -u scripts/probe_0107z_gpu_checkpoint_ablation.py.
Native train_model.main performs the complete production resume, including the
saved environment BEFORE backend imports. A terminal train() interception runs
this experiment instead. Production source, weights, state and logs are read-only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOW = 7.104382866412835e-5
HIGH = 2.841753146565134e-4


def digest_array(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    h = hashlib.sha256(str((array.shape, str(array.dtype))).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def digest_tree(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return digest_array(value)
    if isinstance(value, dict):
        return {str(k): digest_tree(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [digest_tree(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def square_sum(value):
    import numpy as np
    flat = np.asarray(value).reshape(-1)
    return float(np.einsum('i,i->', flat, flat, dtype=np.float64))


def investigate(trainer, args, output):
    import cupy as cp
    import numpy as np
    import ml_dtypes
    import mini_llm.train_extended as training
    from mini_llm.heterogeneous_production import sampler_state, restore_sampler
    from mini_llm.diagnostics.component_updates import ComponentUpdateMonitor, component_name

    started = time.monotonic()
    checkpoint = args.checkpoint.resolve()
    saved = json.loads((checkpoint / 'training_state.json').read_text())
    optimizer = trainer.optimizer
    parameters = list(optimizer.parameters)
    locations = {p.name: (group, i) for group in (optimizer.adam, optimizer.muon)
                 for i, p in enumerate(group.parameters)}
    muon_names = set(optimizer.muon_names)
    adam_names = {p.name for p in optimizer.adam.parameters}
    original_step = optimizer.step
    original_batch = trainer.get_train_batch
    original_clip = training.clip_grad_global_norm
    original_index = optimizer.step_index
    initial_sampler = sampler_state(trainer)
    records = []
    current = {}
    captured_gradients = {}
    high_deltas = {}

    def refresh():
        trainer.model.refresh_compute_buffers()
        cp.cuda.get_current_stream().synchronize()

    def host_state(subdirectory):
        return {p.stem: np.load(p, mmap_mode='r') for p in
                (checkpoint / 'optimizer_state' / subdirectory).glob('*.npy')}

    anchor = {key: host_state(key) for key in ('master_weights', 'm', 'v', 'muon_m')}
    anchor['step'] = original_index

    def param_bits(name):
        return np.load(checkpoint / 'model_params' / (name + '.npy'),
                       mmap_mode='r').view(np.uint16)

    def restore():
        mappings = {k: dict(v) if isinstance(v, dict) else v for k, v in anchor.items()}
        counts = optimizer.restore_checkpoint(mappings, from_hybrid=True)
        assert counts == (302, 82, 220), counts
        optimizer.restore_parameter_birth_steps(saved['optimizer_parameter_birth_steps'])
        for p in parameters:
            assert str(p.data.dtype) == 'bfloat16', (p.name, p.data.dtype)
            p.data.view(cp.uint16).set(np.ascontiguousarray(param_bits(p.name)))
        optimizer.zero_grad()
        refresh()

    def parity():
        """Exact equality, not a loose allclose: compute == BF16(FP32 master)."""
        bad = {}
        for p in parameters:
            group, index = locations[p.name]
            expected = cp.asarray(group.master_weights[index]).astype(p.data.dtype)
            count = int(cp.count_nonzero(p.data.view(cp.uint16) != expected.view(cp.uint16)))
            if count:
                bad[p.name] = count
        cp.cuda.get_current_stream().synchronize()
        return bad

    def gradients(keep=False):
        by_component, by_layer, per_parameter = {}, {}, {}
        groups = {'muon': 0., 'adamw': 0.}
        full_hash = hashlib.sha256()
        for p in parameters:
            g = cp.asnumpy(p.grad)
            n2 = square_sum(g)
            algorithm = 'muon' if p.name in muon_names else 'adamw'
            groups[algorithm] += n2
            component = component_name(p.name)
            by_component[component] = by_component.get(component, 0.) + n2
            layer = p.name.split('.')[1] if p.name.startswith('blocks.') else 'head'
            by_layer[layer] = by_layer.get(layer, 0.) + n2
            sha = digest_array(g)
            full_hash.update((p.name + sha).encode())
            per_parameter[p.name] = {'norm': math.sqrt(n2),
                                     'rms': math.sqrt(n2 / g.size), 'sha256': sha}
            if keep:
                captured_gradients[p.name] = g
        return {'global_norm': math.sqrt(sum(groups.values())),
                'groups': {k: math.sqrt(v) for k, v in groups.items()},
                'components': {k: math.sqrt(v) for k, v in by_component.items()},
                'layers': {k: math.sqrt(v) for k, v in by_layer.items()},
                'parameters': per_parameter, 'sha256': full_hash.hexdigest()}

    def observed_clip(params, *pos, **kw):
        # This observes the raw token-mean gradient BEFORE the actual production clip.
        current['gradients'] = gradients(keep=current.get('keep', False))
        result = original_clip(params, *pos, **kw)
        current['clip_scale'] = float(result[1])
        current['clip_finite'] = bool(result[2])
        return result

    def observed_batch(*pos, **kw):
        batch = original_batch(*pos, **kw)
        inputs, targets, metadata = batch
        identity = {'source': trainer._last_train_batch_source,
                    'inputs': digest_array(inputs), 'targets': digest_array(targets),
                    'metadata': digest_tree(metadata)}
        current.setdefault('batches', []).append(identity)
        count = len(current['batches'])
        if count % 256 == 0:
            print(f"  {current['label']}: microbatch {count}/{trainer.grad_accum_steps}", flush=True)
        return batch

    def no_update(lr=None, active_parameter_names=None):
        current['scheduled_lr'] = float(lr)
        current['active_names'] = sorted(active_parameter_names) if active_parameter_names is not None else None
        # Return normally: train_step clears gradients and completes bookkeeping,
        # but no optimizer tensors/counters are advanced and no weight is changed.

    def evaluate(label, sampler, keep=False, scheduler_lr=None):
        nonlocal current
        current = {'label': label, 'keep': keep}
        trainer.step = saved['step'] if keep else saved['step'] + 1
        restore_sampler(trainer, sampler)
        optimizer.zero_grad()
        trainer.optimizer.step = no_update
        schedule = trainer.scheduler
        if scheduler_lr is not None:
            trainer.scheduler = lambda step: scheduler_lr
        t0 = time.monotonic()
        try:
            loss, norm = trainer.train_step()
        finally:
            trainer.scheduler = schedule
        current.update(loss=float(loss), reported_grad_norm=float(norm),
                       supervised_tokens=float(trainer._last_step_supervised_tokens),
                       executed_tokens=int(trainer._last_step_executed_tokens),
                       seconds=time.monotonic() - t0)
        current['batch_sha256'] = hashlib.sha256(json.dumps(current['batches'], sort_keys=True).encode()).hexdigest()
        print(f"{label}: loss={loss:.9f} gradient={norm:.9f} "
              f"tokens={current['supervised_tokens']:.0f} clip={current['clip_scale']:.9f} "
              f"[{current['seconds']:.1f}s]", flush=True)
        record = dict(current)
        record.pop('keep')
        records.append(record)
        save_report()
        return record

    def save_report():
        report['evaluations'] = records
        report['elapsed_seconds'] = time.monotonic() - started
        (output / 'report.json').write_text(json.dumps(report, indent=2))

    report = {'checkpoint': str(checkpoint), 'repo': str(ROOT),
              'python': sys.executable, 'cupy': cp.__version__,
              'device': str(cp.cuda.runtime.getDeviceProperties(0)['name']),
              'trainer_step': trainer.step, 'optimizer_step': original_index,
              'parameter_count': sum(p.data.size for p in parameters),
              'adam_names': sorted(adam_names), 'muon_names': sorted(muon_names),
              'runtime_environment': {k: v for k, v in os.environ.items() if k.startswith('MINI_LLM_')},
              'resume_contract': trainer.training_resume_contract(),
              'saved_resume_contract': saved['training_resume_contract'],
              'branches': {}}
    assert report['parameter_count'] == 348261376
    assert optimizer.adam.offload_mode == optimizer.muon.offload_mode == 'full'
    assert trainer.grad_accum_steps == 1024 and trainer.seq_length == 8192
    assert trainer.gradient_accumulation_weighting == 'supervised_tokens'
    assert not getattr(trainer, 'accumulation_peer', None)
    report['initial_compute_master_mismatches'] = parity()
    assert not report['initial_compute_master_mismatches']
    print('REAL CHECKPOINT READY: BF16, full CPU offload, context 8192, accumulation 1024', flush=True)

    # Disable sidecar writes on cache misses, without changing the sampler math.
    loader = training.load_or_build_packed_document_index
    def readonly_index(*pos, **kw):
        kw['write_sidecar'] = False
        return loader(*pos, **kw)

    monitor = ComponentUpdateMonitor(parameters, optimizer, interval=1,
                                     output=output / 'component_monitor.csv')

    def update_branch(label, muon_lr, adam_lr, embedding_lr=None):
        restore()
        for p in parameters:
            p.grad[...] = cp.asarray(captured_gradients[p.name])
        # All observed production gradients are below the clip limit; no extra
        # normalization or clipping is performed when replaying this same tensor.
        monitor.begin(saved['step'])
        monitor.capture_before(parameters, learning_rate=max(muon_lr, adam_lr), clip_scale=1.)
        if embedding_lr is not None:
            optimizer.adam.unembedding_lr_ratio = embedding_lr / adam_lr if adam_lr else 0.
        else:
            optimizer.adam.unembedding_lr_ratio = 1.
        if muon_lr == adam_lr and embedding_lr is None:
            original_step(lr=muon_lr)
        else:
            optimizer.adam.step(lr=adam_lr)
            optimizer.muon.step(eta=muon_lr)
            optimizer._step_index += 1
        cp.cuda.get_current_stream().synchronize()
        sampled = monitor.finish()
        for row in sampled:
            row['learning_rate'] = muon_lr if row['optimizer'] == 'muon' else adam_lr
            row['effective_learning_rate'] = (embedding_lr if embedding_lr is not None and
                row['parameter_name'] == 'embedding.W' else row['learning_rate'])
        monitor.output = output / (label + '_monitor.csv')
        monitor.commit(sampled)
        mismatches = parity()
        assert not mismatches, mismatches
        refresh()
        rows = []
        for p in parameters:
            group, index = locations[p.name]
            before = anchor['master_weights'][p.name]
            after = group.master_weights[index]
            delta = after - before
            if label == 'both_high':
                high_deltas[p.name] = delta.copy()
            param_before = param_bits(p.name).view(ml_dtypes.bfloat16).astype(np.float32)
            param_after = cp.asnumpy(p.data.astype(cp.float32))
            def rms(x):
                return math.sqrt(square_sum(x) / x.size)
            row = {'name': p.name, 'component': component_name(p.name),
                   'optimizer': 'muon' if p.name in muon_names else 'adamw',
                   'elements': p.data.size, 'master_rms_before': rms(before),
                   'master_rms_after': rms(after), 'master_delta_rms': rms(delta),
                   'compute_rms_before': rms(param_before), 'compute_rms_after': rms(param_after),
                   'compute_delta_rms': rms(param_after - param_before),
                   'gradient_rms': rms(captured_gradients[p.name]),
                   'relative_master_update': rms(delta) / rms(before),
                   'master_delta_norm': math.sqrt(square_sum(delta))}
            eta = (muon_lr if p.name in muon_names else
                   embedding_lr if embedding_lr is not None and p.name == 'embedding.W' else adam_lr)
            row['effective_lr'] = eta
            if high_deltas and eta:
                ref = high_deltas[p.name]
                expected = ref * (eta / HIGH)
                denominator = math.sqrt(square_sum(ref) * square_sum(delta))
                row['cosine_vs_high_update'] = float(np.einsum('i,i->', ref.ravel(), delta.ravel(), dtype=np.float64) / denominator) if denominator else None
                row['relative_error_vs_scaled_high_delta'] = math.sqrt(square_sum(delta - expected) / max(square_sum(expected), 1e-300))
            rows.append(row)
        with (output / (label + '_updates.csv')).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader()
            writer.writerows(rows)
        report['branches'][label] = {'muon_lr': muon_lr, 'adam_lr': adam_lr,
            'embedding_lr': embedding_lr, 'compute_master_mismatches': mismatches,
            'delta_norm': math.sqrt(sum(r['master_delta_norm'] ** 2 for r in rows))}
        optimizer.zero_grad()
        save_report()

    with patch.object(training, 'clip_grad_global_norm', observed_clip), \
         patch.object(training, 'load_or_build_packed_document_index', readonly_index), \
         patch.object(trainer, 'get_train_batch', observed_batch):
        first = evaluate('initial_update_gradient', initial_sampler, keep=True, scheduler_lr=HIGH)
        assert first['clip_scale'] == 1., first['clip_scale']
        next_sampler = sampler_state(trainer)
        report['initial_matches_logged_tokens'] = first['supervised_tokens'] == 3400191
        report['initial_loss_error_vs_log'] = first['loss'] - 2.2153923511505127
        report['initial_grad_error_vs_log'] = first['reported_grad_norm'] - .30366477370262146
        probe = evaluate('no_update_next_batch', next_sampler)
        for label, muon_lr, adam_lr, embedding_lr in [
            ('both_high', HIGH, HIGH, None),
            ('both_low', LOW, LOW, None),
            ('muon_high_adam_low', HIGH, LOW, None),
            ('muon_low_adam_high', LOW, HIGH, None),
            ('both_high_embedding_low', HIGH, HIGH, LOW),
            ('both_high_embedding_frozen', HIGH, HIGH, 0.),
        ]:
            if args.branches and label not in args.branches.split(','):
                continue
            update_branch(label, muon_lr, adam_lr, embedding_lr)
            result = evaluate(label + '_next_batch', next_sampler)
            assert result['batch_sha256'] == probe['batch_sha256'], 'fixed probe data mismatch'
        # Exactly the original first batch and weights, but a different scheduler LR.
        if args.repeat_initial:
            restore()
            repeated = evaluate('initial_gradient_low_lr', initial_sampler, scheduler_lr=LOW)
            report['initial_preupdate_gradient_hash_equal'] = repeated['gradients']['sha256'] == first['gradients']['sha256']
            assert repeated['batch_sha256'] == first['batch_sha256']
        save_report()
    optimizer.step = original_step
    print('DIAGNOSTIC COMPLETE:', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/model_b_batch1_8k_norecompute_long_16200')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--branches', default='')
    parser.add_argument('--repeat-initial', action='store_true')
    args = parser.parse_args()
    output = args.output or ROOT / 'logs' / ('lr_gpu_16200_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    # The CLI's import-time bootstrap restores the checkpoint runtime environment.
    sys.argv = ['train_model.py', '--backend', 'cupy',
        '--model', 'dual-resolution-matrix-pool-sparse-1024-20l', '--objective', 'ar',
        '--precision', 'bf16-mixed', '--runtime-profile', 'consumer-gpu',
        '--mixed-shard-root', str(ROOT / 'token_shards_ensemble_32k_mixed'),
        '--resume-from', str(args.checkpoint), '--context-length', '8192',
        '--batch-size', '1', '--grad-accum-steps', '1024',
        '--gradient-accumulation-weighting', 'supervised_tokens',
        '--optimizer', 'hybrid-muon', '--muon-momentum', '.95', '--muon-ns-steps', '5',
        '--muon-matmul-precision', 'bf16', '--unembedding-lr-ratio', '1',
        '--weight-decay', '.1', '--grad-clip', '1', '--total-steps', '100000',
        '--warmup-steps', '750', '--peak-lr', '.00015', '--seed', '42',
        '--val-interval', '100', '--val-steps', '4', '--val-per-source-steps', '1',
        '--val-fixed-mixed-steps', '4', '--val-fixed-mixed-seed', '78078',
        '--save-interval', '50', '--log-interval', '1', '--max-run-steps', '1']
    import train_model
    class Finished(BaseException):
        pass
    def intercept(trainer, **kw):
        investigate(trainer, args, output)
        raise Finished()
    with patch.object(train_model.ExtendedTrainer, 'train', intercept):
        try:
            train_model.main()
        except Finished:
            return
    raise RuntimeError('native resume did not reach diagnostic interception')


if __name__ == '__main__':
    main()
