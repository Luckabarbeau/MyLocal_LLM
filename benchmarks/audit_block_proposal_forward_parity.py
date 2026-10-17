"""Read-only 0080 production same-batch forward parity audit.

Compare the trained objective's ordinary inference-style forward path with
its cache-retaining training forward path, on the same held-out token IDs,
validity mask, checkpoint weights, routing metadata and decoder inputs.

A large difference identifies a production kernel/cache-mode mismatch.  If
these paths agree but train logs differ from held-out, investigate checkpoint
step, source sampling distribution, target compaction, and gradients instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from mini_llm.backend import asnumpy, synchronize, xp
from mini_llm.block_proposal.objective import BlockProposalObjective
from benchmarks.evaluate_block_proposal_production import (
    load_production, sample_heldout, seed_for_source,
)


def check_next_token_alignment(inputs, targets, valid, history_length):
    """Check x[t+1] == y[t], without assuming anything about padded positions."""
    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    valid = np.asarray(valid, dtype=np.bool_)
    b, t = targets.shape
    if valid.shape != (b, t) or inputs.shape[0] != b or inputs.shape[1] < history_length + t:
        raise ValueError('source/target shape mismatch')
    lengths = valid.sum(axis=1).astype(int)
    for row, n in enumerate(lengths):
        if not np.all(valid[row, :n]) or np.any(valid[row, n:]):
            raise ValueError('non-prefix target mask')
        if n > 1 and not np.array_equal(
                inputs[row, history_length + 1:history_length + n],
                targets[row, :n - 1]):
            raise AssertionError(f'next-token labels misaligned in sample row {row}')
    return lengths.tolist()


def summarize(metrics, k):
    n = int(metrics['unique_supervised_tokens'])
    counts = np.asarray(asnumpy(metrics['position_counts']), dtype=np.float64)
    proposal = np.asarray(asnumpy(metrics['proposal_position_ce_sums']), dtype=np.float64)
    final = np.asarray(asnumpy(metrics['final_position_ce_sums']), dtype=np.float64)
    if n != int(counts.sum()):
        raise AssertionError('horizon target counts do not add up')
    return {
        'supervised_tokens': n,
        'counts': counts.astype(int).tolist(),
        'proposal': (proposal / counts).tolist(),
        'final': (final / counts).tolist(),
        'gain': ((proposal - final) / counts).tolist(),
        'proposal_ce': float(proposal.sum() / n),
        'final_ce': float(final.sum() / n),
    }


def compare_scores(reference, challenger):
    if reference['counts'] != challenger['counts']:
        raise AssertionError('same-batch horizon counts changed')
    return {
        key: [float(b - a) for a, b in zip(reference[key], challenger[key])]
        for key in ('proposal', 'final', 'gain')
    }


def eval_same_batch(net, inputs, targets, mask, metadata, group_size, *, cached_forward):
    objective = BlockProposalObjective(net, block_group_size=group_size)
    model = net.proposal_head.model
    original_forward = model.forward_body

    if cached_forward:
        def emulate_train_forward(*args, **kwargs):
            if kwargs.get('return_cache') is not False:
                raise AssertionError('parity audit expected inference-style forward request')
            # ONLY replace the trunk forward caching mode. Do not compute gradients,
            # mutate parameters, alter target labels, or change router metadata.
            options = dict(kwargs)
            options['return_cache'] = True
            options['activation_checkpoint'] = False
            hidden, tape = original_forward(*args, **options)
            del tape
            return hidden
        model.forward_body = emulate_train_forward
    try:
        metrics = objective.evaluate_batch(inputs, targets, mask, metadata)
        synchronize()
        return summarize(metrics, int(net.proposal_head.block_size))
    finally:
        if cached_forward:
            del model.forward_body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mixed-shard-root', required=True)
    parser.add_argument('--source', default='fineweb')
    parser.add_argument('--split', choices=('train', 'val'), default='val',
                        help='compare fresh windows from training shards to held-out windows')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--group-size', type=int, default=16)
    parser.add_argument('--compare-softmax', action='store_true',
                        help='compare fused vs reference soft-token PLAN expectation on the same batch')
    parser.add_argument('--compare-group-size', type=int, default=0,
                        help='optional extra inference evaluation with a different decoder group size')
    parser.add_argument('--sampling-policy', default='document_first',
                        choices=('document_first','legacy'))
    parser.add_argument('--seed', type=int, default=78078)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or args.group_size < 1 or args.compare_group_size < 0:
        parser.error('batch and group sizes must be positive, optional comparison nonnegative')

    net, tokenizer, eos, state, sources, _ = load_production(
        args.checkpoint, mixture_root=args.mixed_shard_root)
    source = next((s for s in sources if s.name == args.source), None)
    if source is None:
        parser.error(f'source {args.source!r} not found; available: {[s.name for s in sources]}')
    memory = net.proposal_head.model.config.memory_context
    paths = source.train_paths if args.split == 'train' else source.val_paths
    sample = next(sample_heldout(
        paths, eos=eos, memory=memory, batches=1,
        batch_size=args.batch_size, seed=seed_for_source(args.seed, source.name),
        sampling_policy=args.sampling_policy, retrieval_policy='production',
        stochastic=False, checkpoint_step=int(state.get('step', 0))))
    inputs, targets, mask, metadata = sample
    lengths = check_next_token_alignment(inputs, targets, mask, int(memory.distant_memory_length))
    digest = hashlib.sha256()
    for x in (inputs, targets, mask, metadata['history_valid_starts'], metadata['target_valid_lengths']):
        x = np.ascontiguousarray(x)
        digest.update(str(x.shape).encode())
        digest.update(x.tobytes())

    print('Checkpoint step:', state.get('step', '<missing>'))
    print('Source:', source.name, 'split:', args.split, 'target_valid_lengths:', lengths,
          'router_retrieval:', metadata['use_retrieval'])
    print('Sample:', digest.hexdigest()[:16])
    # Both calls use exactly the same target tokens, mask, router selection and
    # checkpoint parameters. The only change is return_cache=True/False.
    reference = eval_same_batch(net, inputs, targets, mask, metadata, args.group_size,
                                cached_forward=False)
    cached = eval_same_batch(net, inputs, targets, mask, metadata, args.group_size,
                             cached_forward=True)
    diffs = compare_scores(reference, cached)
    print('\nCache parity (cached - inference) CE by horizon:')
    for i, (dproposal, dfinal) in enumerate(zip(diffs['proposal'], diffs['final']), 1):
        print(f'  h{i}: proposal={dproposal:+.6f}, final={dfinal:+.6f}')
    result = {'checkpoint_step': state.get('step'), 'source': source.name, 'split': args.split,
              'sample_sha256': digest.hexdigest(), 'target_valid_lengths': lengths,
              'retrieval_used': bool(metadata['use_retrieval']),
              'inference': reference, 'cached_forward': cached,
              'cached_minus_inference': diffs}
    if args.compare_group_size:
        other = eval_same_batch(net, inputs, targets, mask, metadata,
                                args.compare_group_size, cached_forward=False)
        group_diffs = compare_scores(reference, other)
        result['other_group_size'] = args.compare_group_size
        result['other_group'] = other
        result['group_minus_baseline'] = group_diffs
        print('\nGroup parity (new group - original) CE by horizon:')
        for i, (p, f) in enumerate(zip(group_diffs['proposal'], group_diffs['final']), 1):
            print(f'  h{i}: proposal={p:+.6f}, final={f:+.6f}')
    if args.compare_softmax:
        old = os.environ.get('MINI_LLM_FUSED_SOFTMAX_FORWARD')
        try:
            os.environ['MINI_LLM_FUSED_SOFTMAX_FORWARD'] = '0'
            ordinary = eval_same_batch(net, inputs, targets, mask, metadata,
                                       args.group_size, cached_forward=False)
            os.environ['MINI_LLM_FUSED_SOFTMAX_FORWARD'] = '1'
            fused = eval_same_batch(net, inputs, targets, mask, metadata,
                                    args.group_size, cached_forward=False)
        finally:
            if old is None:
                os.environ.pop('MINI_LLM_FUSED_SOFTMAX_FORWARD', None)
            else:
                os.environ['MINI_LLM_FUSED_SOFTMAX_FORWARD'] = old
        softmax_diffs = compare_scores(ordinary, fused)
        result['softmax_reference'] = ordinary
        result['softmax_fused'] = fused
        result['softmax_fused_minus_reference'] = softmax_diffs
        print('\nSoft-token PLAN softmax parity (fused - reference) CE:')
        for i, (p, f) in enumerate(zip(softmax_diffs['proposal'], softmax_diffs['final']), 1):
            print(f'  h{i}: proposal={p:+.6f}, final={f:+.6f}')
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + '\n')
        print('Report:', path)
    large = max(abs(v) for k in ('proposal','final') for v in diffs[k])
    if large > 0.05:
        print(f'WARNING: large forward-path discrepancy: max |ΔCE|={large:.4f}')
    else:
        print(f'Forward path parity within 0.05 CE: max |ΔCE|={large:.6f}')


if __name__ == '__main__':
    main()
