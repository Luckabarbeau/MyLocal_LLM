#!/usr/bin/env python3
"""0080Z: compare actual production forward+backward against frozen evaluation.

No optimizer or parameter updates. Run independent train/val samples from the
same checkpoint. The critical control tests training-only compact-target mode,
not merely return_cache=True in an evaluation forward.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, asnumpy, synchronize
from mini_llm.block_proposal.objective import BlockProposalObjective
from benchmarks.audit_block_proposal_forward_parity import (
    check_next_token_alignment, compare_scores, summarize,
)
from benchmarks.evaluate_block_proposal_production import (
    load_production, sample_heldout, seed_for_source,
)


@contextmanager
def compact_target_mode(enabled):
    key = 'MINI_LLM_BLOCK_PROPOSAL_COMPACT_TARGET'
    old = os.environ.get(key)
    os.environ[key] = '1' if enabled else '0'
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def score_same_batch(net, inputs, targets, valid, metadata, *, group_size,
                     mode='eval'):
    """No optimizer update. `train_*` calls the ACTUAL full backward path.

    A zero `grad_scale` exercises training execution without accumulating a
    meaningful gradient update; the forward losses are identical to the real
    production training objective for the same tokens and metadata.
    """
    if mode not in ('eval', 'train_full', 'train_compact'):
        raise ValueError(f'unknown audit mode: {mode}')
    objective = BlockProposalObjective(net, block_group_size=group_size)
    is_train = mode.startswith('train_')
    with compact_target_mode(mode == 'train_compact'):
        if is_train:
            net.zero_grad()
            try:
                metrics = objective.forward_backward(
                    inputs, targets, valid, metadata, grad_scale=0.0)
                synchronize()
                result = summarize(metrics, int(net.proposal_head.block_size))
                result['trunk_working_tokens_per_row'] = int(metrics['trunk_working_tokens_per_row'])
                result['trunk_padding_tokens_saved_per_row'] = int(metrics['trunk_padding_tokens_saved_per_row'])
            finally:
                net.zero_grad()
        else:
            metrics = objective.evaluate_batch(inputs, targets, valid, metadata)
            synchronize()
            result = summarize(metrics, int(net.proposal_head.block_size))
            result['trunk_working_tokens_per_row'] = int(metrics['trunk_working_tokens_per_row'])
            result['trunk_padding_tokens_saved_per_row'] = int(metrics['trunk_padding_tokens_saved_per_row'])
    return result


def print_score(name, result):
    print(f'  {name}: targets={result["supervised_tokens"]} '
          f'working={result["trunk_working_tokens_per_row"]} '
          f'proposal={result["proposal_ce"]:.4f} final={result["final_ce"]:.4f}')
    print('    gain:', ', '.join(f'h{j}={v:+.4f}' for j, v in enumerate(result['gain'], 1)))


def accumulate_summaries(results):
    """Token-weighted, then horizon-weighted totals over fixed sampled windows."""
    if not results:
        raise ValueError('need at least one scored batch')
    counts = np.sum([r['counts'] for r in results], axis=0, dtype=np.int64)
    prop = np.sum([np.asarray(r['proposal']) * np.asarray(r['counts'])
                   for r in results], axis=0)
    final = np.sum([np.asarray(r['final']) * np.asarray(r['counts'])
                    for r in results], axis=0)
    if np.any(counts == 0):
        raise ValueError('no targets at one or more horizons')
    total = int(counts.sum())
    return {
        'supervised_tokens': total, 'counts': counts.tolist(),
        'proposal': (prop / counts).tolist(),
        'final': (final / counts).tolist(),
        'gain': ((prop - final) / counts).tolist(),
        'proposal_ce': float(prop.sum() / total),
        'final_ce': float(final.sum() / total),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--mixed-shard-root', required=True)
    p.add_argument('--source', default='fineweb')
    p.add_argument('--split', choices=('train', 'val', 'both'), default='both')
    p.add_argument('--batches-per-split', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--group-size', type=int, default=16)
    p.add_argument('--sampling-policy', choices=('document_first', 'legacy'),
                   default='document_first')
    p.add_argument('--seed', type=int, default=78078)
    p.add_argument('--eval-only', action='store_true',
                   help='skip the expensive real backward parity comparison')
    p.add_argument('--output', default=None)
    args = p.parse_args()
    if args.batches_per_split < 1 or args.batch_size < 1 or args.group_size < 1:
        p.error('all batch/group sizes must be positive')
    net, _, eos, state, sources, _ = load_production(
        args.checkpoint, mixture_root=args.mixed_shard_root)
    source = next((s for s in sources if s.name == args.source), None)
    if source is None:
        p.error(f'{args.source} not found; sources={[s.name for s in sources]}')
    memory = net.proposal_head.model.config.memory_context
    report = {'checkpoint_step': state.get('step'), 'source': source.name,
              'sampling_policy': args.sampling_policy, 'splits': {}}
    splits = ('train', 'val') if args.split == 'both' else (args.split,)
    for split in splits:
        paths = source.train_paths if split == 'train' else source.val_paths
        rows = []
        hashes = hashlib.sha256()
        for batch_number, (inputs, targets, mask, metadata) in enumerate(sample_heldout(
                paths, eos=eos, memory=memory, batches=args.batches_per_split,
                batch_size=args.batch_size, seed=seed_for_source(args.seed, source.name),
                sampling_policy=args.sampling_policy, retrieval_policy='production',
                stochastic=False, checkpoint_step=int(state.get('step', 0)))):
            lengths = check_next_token_alignment(
                inputs, targets, mask, int(memory.distant_memory_length))
            for a in (inputs, targets, mask, metadata['history_valid_starts'],
                      metadata['target_valid_lengths']):
                arr = np.ascontiguousarray(a)
                hashes.update(str(arr.shape).encode())
                hashes.update(arr.tobytes())
            print(f'\n{split} batch {batch_number}: lengths={lengths} retrieval={metadata["use_retrieval"]}')
            scores = {'eval': score_same_batch(net, inputs, targets, mask, metadata,
                                               group_size=args.group_size)}
            print_score('inference', scores['eval'])
            if not args.eval_only:
                for mode in ('train_full', 'train_compact'):
                    scores[mode] = score_same_batch(
                        net, inputs, targets, mask, metadata,
                        group_size=args.group_size, mode=mode)
                    print_score(mode, scores[mode])
                    difference = compare_scores(scores['eval'], scores[mode])
                    mx = max(abs(z) for k in ('proposal','final') for z in difference[k])
                    print(f'    maximum |CE - eval CE| = {mx:.6f}')
            rows.append(scores)
        summary = {mode: accumulate_summaries([r[mode] for r in rows])
                   for mode in rows[0]}
        print(f'\n{split} aggregated over {len(rows)} batch(es):')
        for mode, score in summary.items():
            print_score(mode, {**score, 'trunk_working_tokens_per_row': '-',
                               'trunk_padding_tokens_saved_per_row': '-'})
        report['splits'][split] = {
            'sample_sha256': hashes.hexdigest(),
            'batches': rows, 'aggregate': summary,
        }
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + '\n')
        print('\nReport:', target)
    if args.split == 'both':
        train = report['splits']['train']['aggregate']['eval']
        val = report['splits']['val']['aggregate']['eval']
        print('\nHeld-out minus fresh-training-window CE (same evaluator):')
        print('  proposal:', ', '.join(f'{b-a:+.4f}' for a,b in zip(train['proposal'],val['proposal'])))
        print('  final:   ', ', '.join(f'{b-a:+.4f}' for a,b in zip(train['final'],val['final'])))
        print('  gain:    ', ', '.join(f'{b-a:+.4f}' for a,b in zip(train['gain'],val['gain'])))


if __name__ == '__main__':
    main()
