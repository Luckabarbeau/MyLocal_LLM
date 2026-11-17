"""Follow-up to the verified full Model B LR ablation.

Real trained checkpoint/state/first accumulated gradient:
* compare production full-CPU-offload optimizer with all-device optimizer;
* export real block-entry activations for independent smooth gradchecks;
* measure fixed-next-batch loss/gradients along ONE fixed realised update.
Uses the native resume harness without changing production sources or snapshots.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

import probe_0107z_gpu_checkpoint_ablation as harness


def investigate(trainer, args, output):
    import cupy as cp
    import numpy as np
    import mini_llm.train_extended as training
    from mini_llm.parameter import Parameter
    from mini_llm.optim.hybrid_muon import HybridMuonAdamW
    from mini_llm.heterogeneous_production import sampler_state, restore_sampler

    started = time.monotonic()
    opt = trainer.optimizer
    params = list(opt.parameters)
    ckpt = args.checkpoint.resolve()
    state_json = json.loads((ckpt / 'training_state.json').read_text())
    anchor = {sub: {p.stem: np.load(p, mmap_mode='r') for p in
                   (ckpt / 'optimizer_state' / sub).glob('*.npy')}
              for sub in ('master_weights', 'm', 'v', 'muon_m')}
    anchor['step'] = opt.step_index
    original_step = opt.step
    original_batch = trainer.get_train_batch
    locations = {p.name: (group, i) for group in (opt.adam, opt.muon)
                 for i, p in enumerate(group.parameters)}
    muon_names = set(opt.muon_names)
    gradients = {}
    direction = {}
    current = {}
    activations = {}
    export_allowed = False
    report = {'checkpoint': str(ckpt), 'device': str(cp.cuda.runtime.getDeviceProperties(0)['name']),
              'precision': 'production BF16, FP32 master/gradient',
              'trainer_step': trainer.step, 'optimizer_step': opt.step_index,
              'curve': [], 'parity': {}, 'component_ablations': []}

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        (output / 'validation_report.json').write_text(json.dumps(report, indent=2))

    def fresh_state():
        return {k: dict(v) if isinstance(v, dict) else v for k, v in anchor.items()}

    def batch(*pos, **kw):
        nonlocal export_allowed
        result = original_batch(*pos, **kw)
        x, y, meta = result
        current.setdefault('batches', []).append({
            'source': trainer._last_train_batch_source,
            'x': harness.digest_array(x), 'y': harness.digest_array(y),
            'metadata': harness.digest_tree(meta)})
        valid = meta.get('target_valid_lengths')
        export_allowed = (current.get('capture', False) and x.shape[1] >= 256 and
                          (valid is None or int(np.min(valid)) >= 256))
        count = len(current['batches'])
        if count % 256 == 0:
            print(f"  {current['label']} {count}/1024", flush=True)
        return result

    def block_forward(original, index):
        def forward(x, *pos, **kw):
            key = f'layer_{index}_input'
            if export_allowed and key not in activations:
                activations[key] = cp.asnumpy(x[:, :256].astype(cp.float32))
                positions = kw.get('position_ids')
                activations['position_ids'] = (np.arange(256)[None, :] if positions is None
                    else cp.asnumpy(cp.asarray(positions)[:, :256]))
                if all(f'layer_{i}_input' in activations for i in (0, 10, 19)):
                    np.savez(output / 'block_inputs.npz', **activations)
                    print('Exported real trained block-entry activations', flush=True)
            return original(x, *pos, **kw)
        return forward

    def inspect_gradient(lr=None, active_parameter_names=None):
        assert active_parameter_names is None, 'unexpected freeze in production anchor'
        if current.get('capture'):
            for p in params:
                gradients[p.name] = cp.asnumpy(p.grad)
        elif direction:
            derivative = 0.
            groups = {'muon': 0., 'adamw': 0.}
            for p in params:
                g = cp.asnumpy(p.grad)
                derivative += float(np.einsum('i,i->', g.ravel(), direction[p.name].ravel(), dtype=np.float64))
                groups['muon' if p.name in muon_names else 'adamw'] += harness.square_sum(g)
            current['analytic_dloss_dalpha'] = derivative
            current['gradient_groups'] = {k: math.sqrt(v) for k, v in groups.items()}
        # Deliberately no optimizer/state update; train_step returns loss and clears grads.

    def evaluate(label, sampler, capture=False):
        nonlocal current
        current = {'label': label, 'capture': capture}
        restore_sampler(trainer, sampler)
        trainer.step = state_json['step'] if capture else state_json['step'] + 1
        opt.zero_grad()
        loss, norm = trainer.train_step()
        current.update(loss=float(loss), gradient_norm=float(norm),
                       supervised_tokens=float(trainer._last_step_supervised_tokens))
        assert norm < 1., 'directional projection would need preclip gradient capture'
        current['batch_sha256'] = hashlib.sha256(json.dumps(current.pop('batches'), sort_keys=True).encode()).hexdigest()
        print(f"{label}: loss={loss:.9f} |gradient|={norm:.9f} "
              f"slope={current.get('analytic_dloss_dalpha')}", flush=True)
        return dict(current)

    loader = training.load_or_build_packed_document_index
    def readonly_index(*pos, **kw):
        kw['write_sidecar'] = False
        return loader(*pos, **kw)

    with ExitStack() as stack:
        stack.enter_context(patch.object(trainer, 'get_train_batch', batch))
        stack.enter_context(patch.object(opt, 'step', inspect_gradient))
        stack.enter_context(patch.object(training, 'load_or_build_packed_document_index', readonly_index))
        for i in (0, 10, 19):
            block = trainer.model.blocks[i]
            stack.enter_context(patch.object(block, 'forward', block_forward(block.forward, i)))
        report['first'] = evaluate('initial_real_gradient', sampler_state(trainer), capture=True)
        next_sampler = sampler_state(trainer)
        save()
        cp.get_default_memory_pool().free_all_blocks()

        # All-device clone: same trained BF16 weights, FP32 gradients and restored
        # masters/moments/counters. Only placement/streaming implementation differs.
        clones = [Parameter(p.data.copy(), name=p.name, decay=p.decay) for p in params]
        with patch.dict('os.environ', {'MINI_LLM_OPTIMIZER_OFFLOAD': 'none'}):
            device = HybridMuonAdamW(clones, lr=harness.HIGH,
                beta1=.9, beta2=.95, eps=1e-8, weight_decay=.1,
                muon_momentum=.95, muon_ns_steps=5, muon_matmul_precision='bf16',
                unembedding_lr_ratio=1., unembedding_parameter_name='embedding.W')
        counts = device.restore_checkpoint(fresh_state(), from_hybrid=True)
        assert counts == (302, 82, 220)
        device.restore_parameter_birth_steps(state_json['optimizer_parameter_birth_steps'])
        for original, clone in zip(params, clones):
            original.grad[...] = cp.asarray(gradients[original.name])
            clone.grad[...] = cp.asarray(gradients[original.name])
        original_step(lr=harness.HIGH)
        device.step(lr=harness.HIGH)
        cp.cuda.get_current_stream().synchronize()
        parity_rows = []
        for source_group, device_group in ((opt.adam, device.adam), (opt.muon, device.muon)):
            for index, (p, q) in enumerate(zip(source_group.parameters, device_group.parameters)):
                assert p.name == q.name
                master = source_group.master_weights[index]
                reference = cp.asnumpy(device_group.master_weights[index])
                direction[p.name] = (master - anchor['master_weights'][p.name]).copy()
                mismatches = int(cp.count_nonzero(p.data.view(cp.uint16) != q.data.view(cp.uint16)))
                own_parity = int(cp.count_nonzero(p.data.view(cp.uint16) !=
                    cp.asarray(master).astype(p.data.dtype).view(cp.uint16)))
                delta_error = master - reference
                row = {'parameter': p.name, 'compute_bit_mismatches': mismatches,
                       'master_to_compute_mismatches': own_parity,
                       'master_max_absolute_error': float(np.max(np.abs(delta_error))),
                       'master_rms_error': math.sqrt(harness.square_sum(delta_error) / master.size)}
                for moment in ('m', 'v'):
                    if hasattr(source_group, moment):
                        a = getattr(source_group, moment)[index]
                        b = cp.asnumpy(getattr(device_group, moment)[index])
                        row[moment + '_max_absolute_error'] = float(np.max(np.abs(a - b)))
                parity_rows.append(row)
        report['parity'] = {'parameters': parity_rows,
            'compute_bit_mismatches': sum(r['compute_bit_mismatches'] for r in parity_rows),
            'master_to_compute_mismatches': sum(r['master_to_compute_mismatches'] for r in parity_rows),
            'master_max_absolute_error': max(r['master_max_absolute_error'] for r in parity_rows),
            'moment_max_absolute_error': max(r.get('m_max_absolute_error', 0.) for r in parity_rows),
            'v_max_absolute_error': max(r.get('v_max_absolute_error', 0.) for r in parity_rows)}
        print('FULL TRAINED OPTIMIZER OFFLOAD/DEVICE PARITY:',
              {k: v for k, v in report['parity'].items() if k != 'parameters'}, flush=True)
        save()
        # Exact deterministic placement parity is expected from identical operations.
        assert report['parity']['compute_bit_mismatches'] == 0
        assert report['parity']['master_to_compute_mismatches'] == 0
        assert report['parity']['master_max_absolute_error'] == 0.
        assert report['parity']['moment_max_absolute_error'] == 0.
        assert report['parity']['v_max_absolute_error'] == 0.
        del device, clones, q, p, source_group, device_group
        opt.zero_grad()
        cp.get_default_memory_pool().free_all_blocks()

        def set_displacement(alpha, component=None):
            # No optimizer reevaluation: use ONE fixed actual high-LR FP32 delta.
            # Cast to BF16 exactly as production does, then refresh packed buffers.
            for p in params:
                base = anchor['master_weights'][p.name]
                scale = alpha
                if component is not None:
                    name = p.name
                    selected = ((component == 'local_attention' and '.attention.' in name and
                                  '.global.' not in name and '.matrix_pool.' not in name) or
                                (component == 'compressed_global' and '.attention.global.' in name) or
                                (component == 'ffn' and '.dense_ffn.' in name))
                    scale = 1. if selected else .25
                weights = base + float(scale) * direction[p.name]
                group, index = locations[p.name]
                np.copyto(group.master_weights[index], weights)
                p.data[...] = cp.asarray(weights).astype(p.data.dtype)
            trainer.model.refresh_compute_buffers()
            cp.cuda.get_current_stream().synchronize()

        reference_hash = None
        for alpha in (0., .125, .25, .5, .75, 1.):
            set_displacement(alpha)
            row = evaluate(f'fixed_direction_alpha_{alpha:g}', next_sampler)
            row['alpha'] = alpha
            reference_hash = reference_hash or row['batch_sha256']
            assert row['batch_sha256'] == reference_hash
            report['curve'].append(row)
            save()
        for component in ('local_attention', 'compressed_global', 'ffn'):
            set_displacement(.25, component=component)
            row = evaluate('low_except_high_' + component, next_sampler)
            row['high_muon_component'] = component
            assert row['batch_sha256'] == reference_hash
            report['component_ablations'].append(row)
            save()
    print('UPDATE VALIDATION COMPLETE:', output, flush=True)


if __name__ == '__main__':
    harness.investigate = investigate
    harness.main()
