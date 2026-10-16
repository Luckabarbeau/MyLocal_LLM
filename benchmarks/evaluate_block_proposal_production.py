#!/usr/bin/env python3
"""Frozen production block-proposal checkpoint benchmark for the main model.

Modes:
  heldout     Fixed document-aware validation windows, by source; optional cross-position ablations.
  wikitext    Teacher-forced, non-overlapping next-token WikiText CE/PPL for the
              trained *block proposal + prefix decoder* objective (NOT AR head).
  generation  Production two-stage sequential-commit, causal generation.

Never loads optimizer tensors, changes weights, or touches training sampler RNG.
All evaluation windows are fixed for a given source/seed/manifest (independent of checkpoint).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp, synchronize
from mini_llm.block_proposal.checkpoint import ARCHITECTURE, architecture_metadata
from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
from mini_llm.block_proposal.diagnostics import SCOPES, score_window
from mini_llm.block_proposal.objective import BlockProposalObjective, BlockValidationAccumulator
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.checkpoint import initialize_matching_parameters
from mini_llm.config import ModelConfig
from mini_llm.data.mixed_shards import load_weighted_shard_sources
from mini_llm.data.token_shards import (
    create_hierarchical_memory_minibatch,
    load_or_build_packed_document_index,
    map_token_shard,
)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from benchmarks.evaluate_wikitext import load_tokenizer, split_wikitext_articles


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_production(checkpoint, mixture_root=None):
    checkpoint = Path(checkpoint).expanduser().resolve()
    config = ModelConfig(**json.loads((checkpoint / 'config.json').read_text()))
    state = json.loads((checkpoint / 'training_state.json').read_text())
    arch = state.get('architecture', {})
    if state.get('prediction_objective') != 'block-proposal' or arch.get('type') != ARCHITECTURE:
        raise ValueError('not a production 0080 block-proposal checkpoint; refusing AR fallback')
    tokenizer, tokenizer_path = load_tokenizer(checkpoint)
    token_hash = sha256(tokenizer_path)
    expected_hash = (state.get('tokenizer_hash') or
                     (state.get('data_provenance') or {}).get('tokenizer_sha256'))
    if expected_hash and expected_hash != token_hash:
        raise ValueError('checkpoint tokenizer SHA256 mismatch')
    if len(tokenizer) != config.vocab_size:
        raise ValueError('checkpoint tokenizer vocabulary differs from model config')
    if mixture_root is not None:
        sources, manifest = load_weighted_shard_sources(mixture_root)
        if manifest.get('vocab_size') != config.vocab_size:
            raise ValueError('mixture vocabulary differs from checkpoint')
        if sha256(Path(mixture_root) / 'tokenizer.json') != token_hash:
            raise ValueError('mixture tokenizer differs from checkpoint tokenizer')
        saved_manifest = (state.get('data_provenance') or {}).get('local_mixture_manifest_sha256')
        if saved_manifest and sha256(Path(mixture_root) / 'mixture_manifest.json') != saved_manifest:
            raise ValueError('mixture manifest differs from checkpoint provenance')
    else:
        sources = []
    model = DecoderLanguageModel(config, rng_seed=42, dtype=config.dtype)
    head = HorizonProposalHead(model, int(arch['block_size']),
                               hidden_width=int(arch['proposal_hidden_width']),
                               residual_scale=float(arch['proposal_residual_scale']))
    width = int(arch['decoder_width'])
    ff_width = int(arch['decoder_ff_width'])
    if ff_width % width:
        raise ValueError('non-integral decoder FF multiplier')
    decoder = BlockPrefixDecoder(model, int(arch['block_size']), width=width,
                                 n_layers=int(arch['decoder_layers']),
                                 n_heads=int(arch['decoder_heads']),
                                 ff_multiplier=ff_width // width)
    net = TwoStageBlockModel(head, decoder, lambda_proposal=float(arch['lambda_proposal']))
    if architecture_metadata(net) != arch:
        raise ValueError('reconstructed production architecture differs from checkpoint')
    params = net.parameters()
    names = {p.name for p in params}
    if len(names) != len(params):
        raise ValueError('duplicate production parameter names')
    # Strict complete checkpoint loading, streamed one tensor at a time: avoid
    # a second 26M parameter copy on small-VRAM GPUs such as the Quadro P1000.
    saved_names = {p.stem for p in (checkpoint / 'model_params').glob('*.npy')}
    if saved_names != names:
        raise ValueError('missing production weights: ' + str(sorted(names - saved_names)[:5])
                         + '; unexpected weights: ' + str(sorted(saved_names - names)[:5]))
    loaded_names, missing, _ = initialize_matching_parameters(checkpoint, params)
    if missing or set(loaded_names) != names:
        raise ValueError('incomplete production checkpoint: ' + str(missing[:5]))
    if BACKEND_NAME == 'cupy':
        xp.get_default_memory_pool().free_all_blocks()
    for block in model.blocks:
        if hasattr(block, 'refresh_compute_buffers'):
            block.refresh_compute_buffers()
    for layer in decoder.layers:
        if hasattr(layer.attn, 'refresh_compute_buffers'):
            layer.attn.refresh_compute_buffers()
    eos = (tokenizer._tokenizer.token_to_id(tokenizer.eos_token)
           if hasattr(tokenizer, '_tokenizer') else tokenizer.token_to_id[tokenizer.eos_token])
    if eos is None:
        raise ValueError('tokenizer does not have an EOS token')
    return net, tokenizer, int(eos), state, sources, tokenizer_path


def seed_for_source(seed, source):
    material = f'0080V|{int(seed)}|{source}'.encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], 'little')


def eval_metadata(valid_lengths, history_valid_starts, *, memory=None,
                  retrieval_policy='production', stochastic=False,
                  gumbel_seed=None, checkpoint_step=0):
    """Match ExtendedTrainer's validation retrieval-eligibility contract.

    Production validation only requests a routed prefix when *all* rows have
    genuine history and *all* target windows are full. The earlier benchmark
    forced retrieval for partial-document windows, changing model inputs.
    ``force`` is provided solely to reproduce/audit the older 0080W results.
    """
    lengths = np.asarray(valid_lengths, dtype=np.int64)
    starts = np.asarray(history_valid_starts, dtype=np.int64)
    if retrieval_policy not in ('production', 'force'):
        raise ValueError('unknown retrieval policy')
    if memory is None and retrieval_policy == 'production':
        raise ValueError('memory configuration required for production retrieval policy')
    eligible = (bool(np.all(lengths == int(memory.target_length)))
                and bool(np.all(int(memory.distant_memory_length) - starts > 0))) if memory else True
    metadata = dict(history_valid_starts=starts, target_valid_lengths=lengths,
                    causal_prefix_query=True, causal_masked_target_refinement=True,
                    router_stochastic=bool(stochastic),
                    use_retrieval=(eligible if retrieval_policy == 'production' else True))
    if stochastic:
        if memory is None or gumbel_seed is None:
            raise ValueError('stochastic comparison requires a memory config and seed')
        anneal = max(1, int(memory.router_temperature_anneal_steps))
        progress = min(max(float(checkpoint_step) / anneal, 0.), 1.)
        metadata['router_temperature'] = (float(memory.router_temperature) + progress *
                                         (float(memory.router_temperature_min) -
                                          float(memory.router_temperature)))
        metadata['router_gumbel_seed'] = int(gumbel_seed)
    elif memory is not None:
        metadata['router_temperature'] = float(memory.router_temperature_min)
    return metadata


def sample_heldout(paths, *, eos, memory, batches, batch_size, seed,
                   sampling_policy='document_first', retrieval_policy='production',
                   stochastic=False, checkpoint_step=0):
    rng = np.random.default_rng(seed)
    for batch_index in range(batches):
        path = paths[batch_index % len(paths)]
        tokens = map_token_shard(str(path))
        index = load_or_build_packed_document_index(path, tokens, eos, write_sidecar=False)
        source, target, meta = create_hierarchical_memory_minibatch(
            tokens, batch_size, int(memory.distant_memory_length),
            int(memory.target_length), rng, eos_token_id=eos,
            document_index=index, pad_token_id=eos, document_aware=True,
            return_metadata=True, min_history_tokens=(int(memory.top_k_blocks) + 1)
            * int(memory.block_size) - 1, sampling_policy=sampling_policy)
        valid = np.arange(memory.target_length)[None, :] < meta['target_valid_lengths'][:, None]
        # Gumbel seeding must not consume the document sampler's RNG stream:
        # the deterministic and stochastic evaluations must score IDENTICAL samples.
        noise_seed = seed_for_source(seed, f'gumbel-{batch_index}') % (2**31 - 1)
        yield source, target, valid, eval_metadata(
            meta['target_valid_lengths'], meta['history_valid_starts'],
            memory=memory, retrieval_policy=retrieval_policy,
            stochastic=stochastic, gumbel_seed=noise_seed,
            checkpoint_step=checkpoint_step)


def summarize_ablation(counts, proposal_sum, scope_sums, elapsed):
    counts = np.asarray(counts, dtype=np.int64)
    if not np.all(counts > 0):
        raise ValueError('not all horizons have validation examples')
    reference = scope_sums['all'] / counts
    results = {}
    total = int(counts.sum())
    for scope in scope_sums:
        results[scope] = {
            'proposal_ce': float(proposal_sum.sum() / total),
            'final_ce': float(scope_sums[scope].sum() / total),
            'proposal_ce_by_horizon': (proposal_sum / counts).tolist(),
            'final_ce_by_horizon': (scope_sums[scope] / counts).tolist(),
            'decoder_gain_by_horizon': ((proposal_sum - scope_sums[scope]) / counts).tolist(),
            'ce_increase_vs_all_by_horizon': (scope_sums[scope] / counts - reference).tolist(),
            'ce_increase_vs_all': float((scope_sums[scope] - scope_sums['all']).sum() / total),
        }
    results['all']['proposal_perplexity'] = math.exp(results['all']['proposal_ce'])
    results['all']['final_perplexity'] = math.exp(results['all']['final_ce'])
    report = {'count_by_horizon': counts.tolist(), 'supervised_tokens': total,
              'seconds': elapsed, 'conditions': results}
    if 'own_only' in results:
        report['cross_position_gain_all_vs_own_only'] = results['own_only']['ce_increase_vs_all']
    if 'past_and_current' in results:
        report['future_plan_gain_all_vs_past_current'] = results['past_and_current']['ce_increase_vs_all']
    return report


def heldout_benchmark(net, sources, eos, *, batches, batch_size, seed, group_size,
                      ablations=False, sampling_policy='document_first',
                      retrieval_policy='production', stochastic=False,
                      checkpoint_step=0):
    memory = net.proposal_head.model.config.memory_context
    k = int(net.proposal_head.block_size)
    report = {}
    for source in sources:
        counts = np.zeros(k, dtype=np.int64)
        proposal_sum = np.zeros(k, dtype=np.float64)
        final = {scope: np.zeros(k, dtype=np.float64) for scope in
                 (SCOPES if ablations else ('all',))}
        sample_hash = hashlib.sha256()
        t0 = time.perf_counter()
        objective = BlockProposalObjective(net, block_group_size=group_size) if not ablations else None
        trainer = net_trainer(net) if ablations else None
        for inputs, targets, valid, meta in sample_heldout(
                source.val_paths, eos=eos, memory=memory, batches=batches,
                batch_size=batch_size, seed=seed_for_source(seed, source.name),
                sampling_policy=sampling_policy, retrieval_policy=retrieval_policy,
                stochastic=stochastic, checkpoint_step=checkpoint_step):
            # Fingerprint the *actual* sampled tensors and document-validity metadata.
            for values in (inputs, targets, valid, meta['history_valid_starts'],
                           meta['target_valid_lengths']):
                arr = np.ascontiguousarray(values)
                sample_hash.update(str(arr.shape).encode())
                sample_hash.update(arr.tobytes())
            if ablations:
                scored = score_window(trainer, xp.asarray(inputs, dtype=xp.int64),
                                      xp.asarray(targets, dtype=xp.int64),
                                      xp.asarray(valid, dtype=xp.bool_), meta,
                                      group_size=group_size)
                counts += scored['counts']
                proposal_sum += scored['proposal_sums']
                for scope in SCOPES:
                    final[scope] += scored['final_sums'][scope]
            else:
                metrics = objective.evaluate_batch(inputs, targets, valid, meta)
                from mini_llm.backend import asnumpy
                counts += np.asarray(asnumpy(metrics['position_counts']), dtype=np.int64)
                proposal_sum += np.asarray(asnumpy(metrics['proposal_position_ce_sums']),
                                           dtype=np.float64)
                final['all'] += np.asarray(asnumpy(metrics['final_position_ce_sums']),
                                           dtype=np.float64)
        synchronize()
        result = summarize_ablation(counts, proposal_sum, final, time.perf_counter() - t0)
        result['validation_sample_sha256'] = sample_hash.hexdigest()
        report[source.name] = result
        print(f'\n{source.name}: {result["supervised_tokens"]} held-out targets; sample={sample_hash.hexdigest()[:16]}')
        for j, (p, f, gain) in enumerate(zip(result['conditions']['all']['proposal_ce_by_horizon'],
                                           result['conditions']['all']['final_ce_by_horizon'],
                                           result['conditions']['all']['decoder_gain_by_horizon']), 1):
            print(f'  h{j}: proposal={p:.4f} final={f:.4f} gain={gain:+.4f}')
        for name in SCOPES[1:] if ablations else ():
            print(f'  ablation {name}: ΔCE vs all={result["conditions"][name]["ce_increase_vs_all"]:+.4f}')
    return report


def net_trainer(net):
    # Minimal view expected by the existing validated 0080K diagnostic.
    from mini_llm.block_proposal.training import SinglePassBlockTrainer
    return SinglePassBlockTrainer(net)


def text_windows(tokens, *, memory, eos):
    """One next-token target for every scoreable position; never cross document."""
    m = int(memory.distant_memory_length)
    t = int(memory.target_length)
    if len(tokens) <= 1:
        return
    for start in range(0, len(tokens) - 1, t):
        valid = min(t, len(tokens) - 1 - start)
        x = np.full((1, m+t), eos, dtype=np.int64)
        y = np.full((1, t), eos, dtype=np.int64)
        previous = tokens[max(0, start-m):start]
        x[0, m-len(previous):m] = previous
        x[0, m:m+valid] = tokens[start:start+valid]
        y[0, :valid] = tokens[start+1:start+valid+1]
        mask = np.arange(t)[None, :] < valid
        yield x, y, mask, eval_metadata([valid], [m-len(previous)], memory=memory)


def wikitext_benchmark(net, tokenizer, eos, path, *, max_tokens, document_mode, group_size):
    raw = Path(path).read_text(encoding='utf-8')
    texts = ([raw] if document_mode == 'continuous' else split_wikitext_articles(raw))
    remaining = None if max_tokens is None else int(max_tokens)
    acc = BlockValidationAccumulator(net.proposal_head.block_size)
    objective = BlockProposalObjective(net, block_group_size=group_size)
    consumed = 0
    count_documents = 0
    start_clock = time.perf_counter()
    for text in texts:
        if remaining is not None and remaining < 2:
            break
        ids = tokenizer.encode(text)
        if remaining is not None:
            ids = ids[:remaining]
        if len(ids) < 2:
            continue
        count_documents += 1
        consumed += len(ids)
        if remaining is not None:
            remaining -= len(ids)
        for inputs, targets, valid, meta in text_windows(
                ids, memory=net.proposal_head.model.config.memory_context, eos=eos):
            metrics = objective.evaluate_batch(inputs, targets, valid, meta)
            acc.add(metrics)
    synchronize()
    summary = acc.summary()
    summary['proposal_perplexity'] = math.exp(summary['proposal_ce_nats_per_target'])
    summary['final_perplexity'] = math.exp(summary['final_ce_nats_per_target'])
    summary['documents'] = count_documents
    summary['tokenized_tokens'] = consumed
    summary['seconds'] = time.perf_counter() - start_clock
    summary['partial_corpus'] = max_tokens is not None
    summary['document_mode'] = document_mode
    print(f'\nWikiText: {summary["block_unique_supervised_tokens"]} targets / '
          f'{count_documents} documents; final CE={summary["final_ce_nats_per_target"]:.4f}; '
          f'final PPL={summary["final_perplexity"]:.2f}')
    for h in range(1, net.proposal_head.block_size+1):
        print(f'  h{h}: count={summary[f"h{h}_count"]} proposal={summary[f"h{h}_proposal_ce"]:.4f} '
              f'final={summary[f"h{h}_final_ce"]:.4f} gain={summary[f"h{h}_decoder_gain"]:+.4f}')
    return summary


def generation_benchmark(net, tokenizer, *, prompt, max_new_tokens, commit_stride):
    """Use production InferenceModel and sequential causal two-stage commit.

    No research checkpoint adapter, example import, or AR-head replacement.
    """
    from mini_llm.inference_model import InferenceModel
    from mini_llm.block_proposal.generation import SequentialCommitBlockGenerator

    model = net.proposal_head.model
    config = model.config
    max_context = (config.memory_context.memory_length if model.hierarchical_memory_enabled
                   else config.context_length)
    k = int(net.proposal_head.block_size)
    if not 1 <= commit_stride <= k:
        raise ValueError(f'commit stride must be in [1, K={k}]')
    ids = list(tokenizer.encode(prompt))
    if not ids or len(ids) >= max_context:
        raise ValueError('prompt is empty or too long for the checkpoint context')
    eos_token = tokenizer.eos_token
    eos_id = (tokenizer._tokenizer.token_to_id(eos_token)
              if hasattr(tokenizer, '_tokenizer') else tokenizer.token_to_id[eos_token])
    if eos_id is None:
        raise ValueError('tokenizer has no EOS token')
    infer = InferenceModel(config, dtype=model.dtype)
    infer.set_weights(model)
    capacity = min(max_context, len(ids) + int(max_new_tokens))
    state = infer.create_generation_state(batch_size=1, max_length=capacity)
    t0 = time.perf_counter()
    infer.prefill(xp.asarray([ids], dtype=xp.int32), state)
    synchronize()
    prefill_seconds = time.perf_counter() - t0
    gen = SequentialCommitBlockGenerator(
        infer, net.proposal_head, net.decoder,
        chooser=lambda logits: int(xp.argmax(logits)),
        eos_token_id=int(eos_id), commit_stride=commit_stride)
    generated = gen.generate(state, max_new_tokens)
    synchronize()
    total_seconds = time.perf_counter() - t0
    token_ids = [int(i) for i in generated['tokens']]
    decoded = tokenizer.decode(token_ids)
    print('\nPrompt:', prompt)
    print('Generated:', decoded)
    print(f'Generation: {len(token_ids)} tokens / {total_seconds:.2f} s')
    return {'prompt': prompt, 'generated_text': decoded,
            'generated_token_ids': token_ids,
            'tokens_per_second': len(token_ids)/total_seconds if total_seconds > 0 else None,
            'total_seconds': total_seconds, 'prefill_seconds': prefill_seconds,
            'generation': generated,
            'warning': 'Sequential main-KV updates: correctness reference, not K-fold speedup'}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--mode', choices=['heldout', 'wikitext', 'generation', 'all'],
                        default='heldout')
    parser.add_argument('--mixed-shard-root', type=Path)
    parser.add_argument('--wiki-test', type=Path, help='WikiText-2 wiki.test.raw')
    parser.add_argument('--document-mode', choices=['article', 'continuous'], default='article')
    parser.add_argument('--max-wiki-tokens', type=int, default=1024,
                        help='default smoke test; use 0 for complete WikiText')
    parser.add_argument('--batches-per-source', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1,
                        help='held-out benchmark batch size; default 1 to minimize VRAM')
    parser.add_argument('--group-size', type=int, default=1)
    parser.add_argument('--ablations', action='store_true',
                        help='run expensive 5-condition per-horizon future-plan ablations')
    parser.add_argument('--sampling-policy', choices=('document_first', 'legacy'),
                        default='document_first',
                        help='document_first matches representative validation/training; '
                             'legacy matches fixed per-source validation')
    parser.add_argument('--retrieval-policy', choices=('production', 'force'),
                        default='production',
                        help='production honors full-target/history eligibility; '
                             'force reproduces old 0080W always-retrieve behavior')
    parser.add_argument('--router-stochastic', action='store_true',
                        help='diagnostic: use seeded Gumbel router at checkpoint-step temperature')
    parser.add_argument('--seed', type=int, default=78078)
    parser.add_argument('--prompt', default='The main advantage of a numerical simulation is')
    parser.add_argument('--new-tokens', type=int, default=32)
    parser.add_argument('--commit-stride', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batches_per_source < 1 or args.batch_size < 1 or args.group_size < 1:
        parser.error('batches-per-source, batch-size and group-size must be positive')
    if args.max_wiki_tokens < 0:
        parser.error('max-wiki-tokens must be nonnegative')
    if args.mode in ('all', 'heldout') and args.mixed_shard_root is None:
        parser.error('heldout mode requires --mixed-shard-root')
    if args.mode in ('all', 'wikitext') and args.wiki_test is None:
        parser.error('wikitext mode requires --wiki-test')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite benchmark report: {args.output}')
    net, tokenizer, eos, state, sources, tokenizer_path = load_production(
        args.checkpoint, args.mixed_shard_root if args.mode in ('all','heldout') else None)
    report = {'version': 'production-block-benchmark-v1', 'backend': BACKEND_NAME,
              'checkpoint': str(args.checkpoint.resolve()), 'step': state.get('step'),
              'architecture': state['architecture'], 'tokenizer_sha256':
              sha256(tokenizer_path), 'seed': args.seed,
              'sampling_policy': args.sampling_policy,
              'retrieval_policy': args.retrieval_policy,
              'router_stochastic': bool(args.router_stochastic)}
    if args.mode in ('all', 'heldout'):
        report['heldout'] = heldout_benchmark(net, sources, eos, batches=args.batches_per_source,
                                             batch_size=args.batch_size, seed=args.seed,
                                             group_size=args.group_size,
                                             ablations=args.ablations,
                                             sampling_policy=args.sampling_policy,
                                             retrieval_policy=args.retrieval_policy,
                                             stochastic=args.router_stochastic,
                                             checkpoint_step=int(state.get('step', 0)))
    if args.mode in ('all', 'wikitext'):
        report['wikitext'] = wikitext_benchmark(net, tokenizer, eos, args.wiki_test,
                         max_tokens=(args.max_wiki_tokens or None),
                         document_mode=args.document_mode, group_size=args.group_size)
    if args.mode in ('all', 'generation'):
        report['generation'] = generation_benchmark(net, tokenizer, prompt=args.prompt,
                          max_new_tokens=args.new_tokens, commit_stride=args.commit_stride)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nReport:', args.output)
    return report


if __name__ == '__main__':
    main()
