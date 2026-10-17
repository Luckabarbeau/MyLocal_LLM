#!/usr/bin/env python3
"""0080AC: audit checkpointed decoder parameters versus historical packed QKV.

No training, optimizer steps, parameter changes, or checkpoint writes. The
second evaluation temporarily substitutes *only* each block-decoder attention
layer's non-serialized packed-QKV compute buffer with its deterministic initial
value. This tests the 0080AB stale-packed-QKV hypothesis directly.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, asnumpy, synchronize
from mini_llm.block_proposal.decoder import BlockPrefixDecoder
from mini_llm.block_proposal.objective import BlockProposalObjective
from benchmarks.evaluate_block_proposal_production import (
    load_production, sample_heldout, seed_for_source,
)


def _disk_array(path):
    value = np.load(path, allow_pickle=False)
    if value.dtype.kind == 'V' and value.dtype.itemsize == 2:
        import ml_dtypes
        value = value.view(ml_dtypes.bfloat16)
    return value


def check_decoder_saved_weights(checkpoint, net):
    """Check *actual* saved parameter contents, not just tensor filenames."""
    root = Path(checkpoint)
    params = [p for p in net.parameters() if p.name.startswith(
        ('block_decoder.', 'block_proposal.'))]
    if not params:
        raise ValueError('No proposal/decoder parameters found in loaded architecture')
    mismatches = []
    master_mismatches = []
    master_max_abs_difference = 0.0
    n_masters = 0
    for p in params:
        name = p.name
        saved = _disk_array(root / 'model_params' / f'{name}.npy')
        live = np.asarray(asnumpy(p.data))
        # Byte-exact comparison covers BF16, FP16 and FP32 without rounding.
        if (saved.shape != live.shape or saved.dtype.itemsize != live.dtype.itemsize
                or saved.tobytes() != live.tobytes()):
            mismatches.append(name)
        master_path = root / 'optimizer_state' / 'master_weights' / f'{name}.npy'
        if master_path.is_file():
            n_masters += 1
            master = np.load(master_path, mmap_mode='r', allow_pickle=False)
            if master.shape != saved.shape:
                master_mismatches.append(name)
                continue
            # An optimizer's FP32 master is *expected* to differ slightly
            # from its BF16 rounded model parameter. Verify rounded equality.
            roundtrip = master.astype(saved.dtype)
            if roundtrip.tobytes() != saved.tobytes():
                master_mismatches.append(name)
            master_max_abs_difference = max(
                master_max_abs_difference,
                float(np.max(np.abs(master.astype(np.float32) - saved.astype(np.float32))))
                if saved.size else 0.0,
            )
    return {
        'decoder_and_proposal_tensors': len(params),
        'checkpoint_to_loaded_byte_mismatches': mismatches,
        'optimizer_master_tensors': n_masters,
        'fp32_masters_not_equal_to_saved_after_rounding': master_mismatches,
        'max_abs_fp32_master_minus_saved_weight': master_max_abs_difference,
    }


def fresh_initial_decoder(reference_decoder):
    model = reference_decoder.model
    layer = reference_decoder.layers[0]
    return BlockPrefixDecoder(
        model, block_size=reference_decoder.block_size,
        width=reference_decoder.width, n_layers=len(reference_decoder.layers),
        n_heads=layer.attn.n_q_heads,
        ff_multiplier=int(layer.ffn.W_gate.data.shape[1]) // reference_decoder.width,
        rng_seed=81,
    )


def _packed_from_parameters(attn):
    return xp.concatenate([attn.Wq.data, attn.Wk.data, attn.Wv.data], axis=1)


@contextmanager
def temporarily_replace_decoder_packs_with_initial(decoder, initial_decoder):
    """Override derived buffers only; restore original references even on error."""
    if len(decoder.layers) != len(initial_decoder.layers):
        raise ValueError('initial and loaded decoder depths differ')
    originals = []
    try:
        for trained, initial in zip(decoder.layers, initial_decoder.layers):
            attn = trained.attn
            originals.append((attn, attn._packed_qkv_weight, attn._packed_qkv_valid))
            attn._packed_qkv_weight = _packed_from_parameters(initial.attn)
            attn._packed_qkv_valid = True
        yield
    finally:
        for attn, packed, valid in originals:
            attn._packed_qkv_weight = packed
            attn._packed_qkv_valid = valid


def score_batch(objective, inputs, targets, valid, metadata):
    metrics = objective.evaluate_batch(inputs, targets, valid, metadata)
    counts = np.asarray(asnumpy(metrics['position_counts']), dtype=np.float64)
    p = np.asarray(asnumpy(metrics['proposal_position_ce_sums']), dtype=np.float64) / counts
    f = np.asarray(asnumpy(metrics['final_position_ce_sums']), dtype=np.float64) / counts
    return {'counts': counts.astype(int).tolist(), 'proposal_ce': p.tolist(),
            'final_ce': f.tolist(), 'gain': (p-f).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mixed-shard-root', required=True)
    parser.add_argument('--source', default='fineweb')
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--group-size', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=78078)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    net, _, eos, state, sources, _ = load_production(
        args.checkpoint, mixture_root=args.mixed_shard_root)
    selected = [s for s in sources if s.name == args.source]
    if len(selected) != 1:
        raise ValueError(f'Unknown source {args.source}: {[s.name for s in sources]}')
    src = selected[0]
    paths = src.val_paths if args.split == 'val' else src.train_paths
    memory = net.proposal_head.model.config.memory_context
    inputs, targets, valid, meta = next(sample_heldout(
        paths, eos=eos, memory=memory, batches=1, batch_size=args.batch_size,
        seed=seed_for_source(args.seed, args.source),
        sampling_policy='document_first', retrieval_policy='production',
        checkpoint_step=state.get('step', 0)))
    digest = hashlib.sha256()
    for array in (inputs, targets, valid, meta['history_valid_starts'], meta['target_valid_lengths']):
        digest.update(np.ascontiguousarray(array).tobytes())
    obj = BlockProposalObjective(net, block_group_size=args.group_size)
    disk = check_decoder_saved_weights(args.checkpoint, net)
    original = score_batch(obj, inputs, targets, valid, meta)
    fresh = fresh_initial_decoder(net.decoder)
    change_rms = []
    for trained, init in zip(net.decoder.layers, fresh.layers):
        left = np.asarray(asnumpy(_packed_from_parameters(trained.attn)), dtype=np.float32)
        right = np.asarray(asnumpy(_packed_from_parameters(init.attn)), dtype=np.float32)
        change_rms.append(float(np.sqrt(np.mean((left-right)**2))))
    with temporarily_replace_decoder_packs_with_initial(net.decoder, fresh):
        initialized_packs = score_batch(obj, inputs, targets, valid, meta)
    del fresh
    synchronize()
    restored = score_batch(obj, inputs, targets, valid, meta)
    maximum_restore_delta = max(abs(a-b) for label in ('proposal_ce','final_ce')
                                for a,b in zip(original[label],restored[label]))
    if maximum_restore_delta > 0.02:
        raise RuntimeError(f'Packed-QKV restore was not reversible: {maximum_restore_delta:.6f} CE')
    result = {
        'checkpoint_step': state.get('step'), 'source': args.source,
        'split': args.split, 'sample_sha256': digest.hexdigest(),
        'target_valid_lengths': np.asarray(meta['target_valid_lengths']).tolist(),
        'trained_vs_initialized_qkv_rms_by_decoder_layer': change_rms,
        'disk_weight_integrity': disk, 'checkpoint_computed_packs': original,
        'initialization_era_packs': initialized_packs,
        'recovered_checkpoint_packs': restored,
        'max_restore_ce_difference': maximum_restore_delta,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print('Checkpoint step:', result['checkpoint_step'],
          'source:', args.source, 'split:', args.split,
          'sample:', result['sample_sha256'][:16])
    print('Saved decoder/proposal tensors:', disk['decoder_and_proposal_tensors'],
          'byte mismatches:', len(disk['checkpoint_to_loaded_byte_mismatches']),
          'master rounding mismatches:', len(disk['fp32_masters_not_equal_to_saved_after_rounding']))
    print('Decoder QKV RMS distance from initialization:',
          ', '.join(f'{v:.5f}' for v in change_rms))
    for title, record in (('checkpoint/current', original),
                          ('initialization-era packed QKV', initialized_packs)):
        print(title)
        for i, (p, f, g) in enumerate(zip(record['proposal_ce'], record['final_ce'], record['gain']), 1):
            print(f'  h{i}: proposal={p:.4f} final={f:.4f} gain={g:+.4f}')
    print('Restore reversibility max |ΔCE|:', f'{maximum_restore_delta:.6f}')
    print('Report:', args.output)


if __name__ == '__main__':
    main()
