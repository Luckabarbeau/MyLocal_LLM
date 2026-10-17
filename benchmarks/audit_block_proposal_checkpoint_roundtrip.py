#!/usr/bin/env python3
"""0080AD: strict 500M checkpoint save/load parity in two sequential GPU models.

Read-only with respect to the input production checkpoint. Loads a model,
evaluates a fixed held-out batch, writes a transient copy of its trainable
weights via the PRODUCTION serializer, frees the first GPU model, cold-loads
another model, restores that transient copy and compares intermediate outputs.

The checkpoint must ALREADY have been produced by actual training optimizer
steps. The separate CPU test tests the actual live optimizer update -> save.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.block_proposal.checkpoint import (
    load_two_stage_checkpoint, save_two_stage_checkpoint,
)
from mini_llm.diagnostics.block_proposal_roundtrip import (
    compare_fingerprints, decoder_pack_integrity, fixed_batch_fingerprint,
)
from benchmarks.evaluate_block_proposal_production import (
    load_production, sample_heldout, seed_for_source,
)


def _stage_hashes(net):
    names = ('block_decoder.', 'block_proposal.')
    return {
        p.name: hashlib.sha256(np.ascontiguousarray(asnumpy(p.data)).tobytes()).hexdigest()
        for p in net.parameters() if p.name.startswith(names)
    }


def _sample_hash(inputs, labels, valid, metadata):
    sha = hashlib.sha256()
    for tensor in (inputs, labels, valid, metadata['target_valid_lengths'],
                   metadata['history_valid_starts']):
        a = np.ascontiguousarray(tensor)
        sha.update(str(a.shape).encode('ascii'))
        sha.update(a.tobytes())
    return sha.hexdigest()[:16]


def _unload():
    gc.collect()
    if BACKEND_NAME == 'cupy':
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()


def run(args):
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    net, tokenizer, eos, state, sources, tokenizer_path = load_production(
        checkpoint, mixture_root=args.mixed_shard_root)
    source = next((item for item in sources if item.name == args.source), None)
    if source is None:
        raise ValueError(f'source {args.source!r} not found')
    step = int(state['step'])
    if args.expect_step is not None and step != args.expect_step:
        raise ValueError(f'checkpoint step {step} != expected {args.expect_step}')
    if step < 1:
        raise ValueError('roundtrip audit requires a checkpoint after an actual optimizer update')
    windows = sample_heldout(
        source.val_paths, eos=eos,
        memory=net.proposal_head.model.config.memory_context, batches=1,
        batch_size=args.batch_size, seed=seed_for_source(args.seed, source.name),
        sampling_policy='document_first', retrieval_policy='production',
        stochastic=False, checkpoint_step=step)
    inputs, labels, valid, metadata = next(windows)
    batch_hash = _sample_hash(inputs, labels, valid, metadata)
    print(f'Checkpoint step={step} source={args.source} sample={batch_hash}')
    pre_packs = decoder_pack_integrity(net)
    before = fixed_batch_fingerprint(net, inputs, labels, valid, metadata,
                                     group_size=args.group_size)
    hashes = _stage_hashes(net)
    print('Before save: gain=' + ', '.join(f'{v:+.4f}' for v in before['decoder_gain']))

    if args.temporary_root is not None:
        Path(args.temporary_root).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='0080ad_roundtrip_', dir=args.temporary_root) as scratch:
        transient = Path(scratch) / 'saved_model'
        # This is the strict same serializer used by the production 0080 model;
        # original checkpoint and optimizer are never changed.
        save_two_stage_checkpoint(transient, net, training_step=step,
                                  supervised_tokens=int(state.get('supervised_tokens', 0)))
        del net, tokenizer
        _unload()
        # Construct a wholly new model, then load the saved weights into it.
        # Sequential GPU ownership prevents the 16-GB RTX 5080 from needing
        # room for TWO 500M models at once.
        fresh, _, _, fresh_state, _, _ = load_production(
            checkpoint, mixture_root=args.mixed_shard_root)
        if int(fresh_state['step']) != step:
            raise ValueError('input checkpoint changed while audit was running')
        load_two_stage_checkpoint(transient, fresh)
        post_packs = decoder_pack_integrity(fresh)
        if pre_packs != post_packs:
            raise AssertionError('decoder packed-QKV metadata differs after reload')
        restored_hashes = _stage_hashes(fresh)
        bad = [name for name, sha in hashes.items() if restored_hashes.get(name) != sha]
        if bad or hashes.keys() != restored_hashes.keys():
            raise AssertionError('decoder/proposal checkpoint weight mismatch: ' + str(bad[:5]))
        after = fixed_batch_fingerprint(fresh, inputs, labels, valid, metadata,
                                        group_size=args.group_size)
        del fresh
        _unload()
    parity = compare_fingerprints(before, after, ce_atol=args.ce_tolerance,
                                  logits_atol=args.logit_tolerance)
    report = {
        'checkpoint': str(checkpoint), 'step': step, 'source': args.source,
        'sample_fingerprint': batch_hash, 'batch_size': args.batch_size,
        'group_size': args.group_size,
        'decoder_proposal_tensors_verified': len(hashes),
        'decoder_packed_qkv_verified': len(pre_packs),
        'supervised_tokens_by_horizon': before['counts'].tolist(),
        'before': {
            'proposal_ce': before['proposal_ce'].tolist(),
            'final_ce': before['final_ce'].tolist(),
            'decoder_gain': before['decoder_gain'].tolist(),
        },
        'after': {
            'proposal_ce': after['proposal_ce'].tolist(),
            'final_ce': after['final_ce'].tolist(),
            'decoder_gain': after['decoder_gain'].tolist(),
        },
        'parity': parity,
        'passed': True,
    }
    print('After reload: gain=' + ', '.join(f'{v:+.4f}' for v in after['decoder_gain']))
    print('Max |CE difference|:', f'{parity["max_ce_error"]:.8g}')
    for stage, value in parity['stage_max_abs_errors'].items():
        print(f'  {stage}: max |difference|={value:.8g}')
    print(f'PASS: {len(hashes)} proposal/decoder weight tensors, {len(pre_packs)} QKV packs')
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
        print('Report:', output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mixed-shard-root', required=True)
    parser.add_argument('--source', default='fineweb')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--group-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=78078)
    parser.add_argument('--expect-step', type=int, default=None)
    parser.add_argument('--ce-tolerance', type=float, default=0.01)
    parser.add_argument('--logit-tolerance', type=float, default=0.04)
    parser.add_argument('--temporary-root', default=None,
                        help='disk with room for a temporary full-model checkpoint')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or args.group_size < 1 or args.ce_tolerance < 0 or args.logit_tolerance < 0:
        parser.error('batch/group sizes must be positive and tolerances must be nonnegative')
    run(args)


if __name__ == '__main__':
    main()
