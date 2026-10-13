"""0080H: paired CPU training-speed experiment: routed AR vs K-proposal decoder.

Both use identical main-model architecture, initial trunk seed, optimizer,
preprocessing, document-first samples, and unique next-token supervision.
0080 additionally has proposal and small-decoder parameters/loss, which is
precisely the additional cost this system comparison is intended to measure.

Run from the repository root after preparing the two-source 0080G mixture:
  MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m \\
      examples.compare_block_proposal_cpu_0080h \\
      --mixed-shard-root data/block_proposal_cpu_two_sources \\
      --steps 6 --warmup-steps 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
from mini_llm.backend import BACKEND_NAME, xp, asnumpy, synchronize
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.block_proposal.baseline import MatchedAutoregressiveTrainer
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
from mini_llm.block_proposal.training import SinglePassBlockTrainer
from mini_llm.block_proposal.real_data import PackedRoutedBlockProposalDocuments
from mini_llm.predictive_refinement.real_data import sha256
from examples.train_predictive_refinement_text import _load_tokenizer, _clip_grads
from examples.train_block_proposal_routed_text import _model_config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mixed-shard-root', type=Path, required=True)
    p.add_argument('--model-preset', choices=['cpu_tiny', 'cpu_medium'], default='cpu_medium')
    p.add_argument('--dtype', choices=['float32'], default='float32')
    p.add_argument('--architecture', choices=['both', 'standard_ar', 'two_stage'], default='both')
    p.add_argument('--steps', type=int, default=6, help='optimizer steps per architecture, including warmup')
    p.add_argument('--warmup-steps', type=int, default=1, help='exclude early steps from throughput aggregate')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--block-size', type=int, choices=(1, 4, 8, 16), default=8)
    p.add_argument('--block-group-size', type=int, default=2)
    p.add_argument('--proposal-width', type=int, default=96)
    p.add_argument('--decoder-width', type=int, default=128)
    p.add_argument('--decoder-layers', type=int, default=2)
    p.add_argument('--decoder-heads', type=int, default=4)
    p.add_argument('--lambda-proposal', type=float, default=0.5)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--clip-grad-norm', type=float, default=1.)
    p.add_argument('--val-batches', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, default=Path('benchmarks/results/0080h_cpu_comparison'))
    p.add_argument('--order', choices=('ar-first', 'two-stage-first'), default='ar-first')
    a = p.parse_args(argv)
    if BACKEND_NAME != 'numpy':
        p.error('0080H is a native-CPU experiment; set MINI_LLM_BACKEND=numpy')
    if a.steps <= 0 or not 0 <= a.warmup_steps < a.steps:
        p.error('need steps > 0 and 0 <= warmup-steps < steps')
    for k in ('batch_size', 'grad_accum', 'block_group_size', 'proposal_width',
              'decoder_width', 'decoder_layers', 'decoder_heads', 'val_batches'):
        if getattr(a, k) <= 0:
            p.error(f'{k} must be positive')
    if a.decoder_width % a.decoder_heads:
        p.error('decoder-width must be divisible by decoder-heads')
    if a.learning_rate <= 0 or a.lambda_proposal <= 0:
        p.error('learning rate and lambda-proposal must be positive')
    return a


def _metadata(m):
    return {k: xp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in m.items()}


def _run_window(trainer, sample, a, *, train=False, weight=1.0, two_stage=False):
    src, y, mask, metadata, _ = sample
    kw = dict(valid_target_mask=xp.asarray(mask, dtype=xp.bool_),
              memory_metadata=_metadata(metadata), compute_gradients=train,
              grad_scale=weight)
    if two_stage:
        kw['block_group_size'] = a.block_group_size
    return trainer.train_window(xp.asarray(src, dtype=xp.int64),
                                xp.asarray(y, dtype=xp.int64), **kw)


def _count_parameters(params):
    return int(sum(p.data.size for p in params))


def _initial_model_digest(model):
    digest=hashlib.sha256()
    for p in model.parameters():
        digest.update(p.name.encode('utf8')+b'\0')
        digest.update(np.ascontiguousarray(asnumpy(p.data)).tobytes())
    return digest.hexdigest()


def _sample_digest(samples):
    """Verify paired training uses identical actual examples, not just counts."""
    digest=hashlib.sha256()
    for source, labels, valid, metadata, observations in samples:
        for array in (source,labels,valid,metadata['history_valid_starts'],
                      metadata['target_valid_lengths']):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(json.dumps(observations,sort_keys=True).encode('utf8'))
    return digest.hexdigest()


def _run_architecture(name, a, cfg, train_data, fixed_validation):
    is_two_stage = name == 'two_stage'
    model = DecoderLanguageModel(cfg, rng_seed=a.seed, dtype='float32')
    main_params = _count_parameters(model.parameters())
    main_digest = _initial_model_digest(model)
    if is_two_stage:
        proposal = HorizonProposalHead(model, a.block_size,
                                       hidden_width=a.proposal_width, rng_seed=a.seed+1)
        decoder = BlockPrefixDecoder(model, a.block_size, width=a.decoder_width,
                 n_layers=a.decoder_layers, n_heads=a.decoder_heads, rng_seed=a.seed+2)
        net = TwoStageBlockModel(proposal, decoder,
                                 lambda_proposal=a.lambda_proposal)
        trainer = SinglePassBlockTrainer(net)
    else:
        trainer = MatchedAutoregressiveTrainer(model)
    all_params = trainer.parameters()
    total_params = _count_parameters(all_params)
    optimizer = AdamW(all_params, lr=a.learning_rate,
                      weight_decay=a.weight_decay)
    a.output.mkdir(parents=True, exist_ok=True)
    out = a.output / f'{name}.jsonl'
    if out.exists():
        raise FileExistsError(f'refusing to overwrite benchmark results: {out}')
    rng = np.random.default_rng(a.seed+3)
    count_reported = []
    sampled_digests = []
    measured = []
    with out.open('w', encoding='utf8') as f:
        for step in range(1, a.steps+1):
            start = time.perf_counter()
            samples = [train_data.sample_causal(rng, batch_size=a.batch_size,
                memory_length=cfg.memory_context.distant_memory_length,
                target_length=cfg.memory_context.target_length,
                min_history_tokens=(cfg.memory_context.top_k_blocks+1)
                                   * cfg.memory_context.block_size-1)
                for _ in range(a.grad_accum)]
            unique = sum(int(s[2].sum()) for s in samples)
            sampled_digest = _sample_digest(samples)
            sampled_at = time.perf_counter()
            if not unique:
                raise RuntimeError('sampled step has no valid supervised tokens')
            trainer.zero_grad()
            ce = 0.; prop_ce = 0.
            for sample in samples:
                n = int(sample[2].sum())
                if not n:
                    continue
                m = _run_window(trainer, sample, a, train=True,
                                weight=n/unique, two_stage=is_two_stage)
                ce += m['final_ce'] * n/unique
                if is_two_stage:
                    prop_ce += m['proposal_ce'] * n/unique
            synchronize()
            trained_at = time.perf_counter()
            grad_norm = float(_clip_grads(trainer.parameters(), a.clip_grad_norm))
            optimizer.step()
            model.refresh_compute_buffers()
            if is_two_stage:
                for layer in decoder.layers:
                    if hasattr(layer.attn, 'refresh_compute_buffers'):
                        layer.attn.refresh_compute_buffers()
            synchronize()
            end = time.perf_counter()
            rec = dict(architecture=name, step=step, measured=step>a.warmup_steps,
                       valid_tokens=unique, sample_digest=sampled_digest, target_slots=a.batch_size*a.grad_accum
                       *cfg.memory_context.target_length,
                       final_ce=float(ce), grad_norm=grad_norm,
                       step_seconds=end-start, data_seconds=sampled_at-start,
                       forward_backward_seconds=trained_at-sampled_at,
                       optimizer_seconds=end-trained_at,
                       supervised_tok_s=unique/max(end-start, 1e-9),
                       optimizer_step=optimizer.step_index)
            if is_two_stage:
                rec['proposal_ce'] = float(prop_ce)
                rec['decoder_gain'] = float(prop_ce-ce)
            f.write(json.dumps(rec, sort_keys=True)+'\n')
            f.flush()
            count_reported.append(unique)
            sampled_digests.append(sampled_digest)
            if step>a.warmup_steps:
                measured.append(rec)
            print(f'{name} {step}/{a.steps}: CE={ce:.4f} valid={unique} '
                  f'time={end-start:.3f}s tok/s={rec["supervised_tok_s"]:.1f}'
                  + (f' proposal={prop_ce:.4f}' if is_two_stage else ''), flush=True)
    # Fixed held-out examples are exactly the same for both architectures.
    val_count=0; val_nll=0.
    for sample in fixed_validation:
        n=int(sample[2].sum())
        if n:
            metric=_run_window(trainer, sample, a, train=False, two_stage=is_two_stage)
            val_count += n
            val_nll += metric['final_ce']*n
    result = dict(architecture=name, model_preset=a.model_preset,
                  model_parameters=main_params, trainable_parameters=total_params,
                  added_parameters=total_params-main_params,
                  mean_step_seconds=statistics.mean(r['step_seconds'] for r in measured),
                  median_step_seconds=statistics.median(r['step_seconds'] for r in measured),
                  total_measured_seconds=sum(r['step_seconds'] for r in measured),
                  total_measured_unique_tokens=sum(r['valid_tokens'] for r in measured),
                  weighted_supervised_tokens_per_sec=(sum(r['valid_tokens'] for r in measured)/
                         sum(r['step_seconds'] for r in measured)),
                  target_fill=(sum(r['valid_tokens'] for r in measured)/
                        sum(r['target_slots'] for r in measured)),
                  mean_data_seconds=statistics.mean(r['data_seconds'] for r in measured),
                  mean_forward_backward_seconds=statistics.mean(r['forward_backward_seconds'] for r in measured),
                  mean_optimizer_seconds=statistics.mean(r['optimizer_seconds'] for r in measured),
                  validation_final_ce=(val_nll/val_count if val_count else None),
                  validation_unique_tokens=val_count,
                  step_supervised_counts=count_reported,
                  step_sample_digests=sampled_digests,
                  main_initial_weights_sha256=main_digest,
                  warmup_steps=a.warmup_steps, total_steps=a.steps)
    return result


def main(argv=None):
    a=parse_args(argv)
    root=a.mixed_shard_root.expanduser().resolve()
    tok,eos=_load_tokenizer(root/'tokenizer.json')
    cfg=_model_config(a, len(tok))
    kwargs=dict(mixed_root=root, eos_token_id=eos, token_dtype='uint16')
    train_data=PackedRoutedBlockProposalDocuments(**kwargs, split='train')
    val_data=PackedRoutedBlockProposalDocuments(**kwargs, split='val')
    val_rng=np.random.default_rng(a.seed+4)
    fixed_validation=[val_data.sample_causal(val_rng,batch_size=a.batch_size,
        memory_length=cfg.memory_context.distant_memory_length,
        target_length=cfg.memory_context.target_length,
        min_history_tokens=(cfg.memory_context.top_k_blocks+1)
                           *cfg.memory_context.block_size-1)
        for _ in range(a.val_batches)]
    order=(['standard_ar', 'two_stage'] if a.order=='ar-first'
           else ['two_stage', 'standard_ar'])
    if a.architecture!='both':
        order=[a.architecture]
    a.output.mkdir(parents=True,exist_ok=True)
    print(f'Running {order} on {BACKEND_NAME}: '
          f'{cfg.n_layers} layers x {cfg.d_model}, '
          f'{cfg.memory_context.target_length}-token supervised window, '
          f'{cfg.memory_context.distant_memory_length}-token history, '
          f'vocab={len(tok)}, K={a.block_size}', flush=True)
    results={'experiment':dict(tokenizer_sha256=sha256(root/'tokenizer.json'),
        train_provenance=train_data.provenance(),val_provenance=val_data.provenance(),
        seed=a.seed,model_preset=a.model_preset,block_size=a.block_size,
        block_group_size=a.block_group_size,decoder_width=a.decoder_width,
        decoder_layers=a.decoder_layers,decoder_heads=a.decoder_heads,
        proposal_width=a.proposal_width,batch_size=a.batch_size,
        grad_accum=a.grad_accum,learning_rate=a.learning_rate,
        steps=a.steps,warmup_steps=a.warmup_steps)}
    for name in order:
        results[name]=_run_architecture(name,a,cfg,train_data,fixed_validation)
    if 'standard_ar' in results and 'two_stage' in results:
        ar=results['standard_ar']; tw=results['two_stage']
        if ar['step_sample_digests'] != tw['step_sample_digests']:
            raise AssertionError('paired runs did not use exactly identical samples')
        if ar['main_initial_weights_sha256'] != tw['main_initial_weights_sha256']:
            raise AssertionError('paired runs did not start from identical main weights')
        results['comparison']={
            'two_stage_over_ar_step_time':tw['mean_step_seconds']/ar['mean_step_seconds'],
            'two_stage_over_ar_tok_s':tw['weighted_supervised_tokens_per_sec'] /
                                      ar['weighted_supervised_tokens_per_sec'],
            'ar_over_two_stage_tok_s':ar['weighted_supervised_tokens_per_sec'] /
                                      tw['weighted_supervised_tokens_per_sec'],
            'interpretation':'CPU float32 training wall-clock only, not GPU inference or quality',
        }
    path=a.output/'summary.json'
    if path.exists():
        raise FileExistsError(f'refusing to overwrite comparison: {path}')
    path.write_text(json.dumps(results,indent=2),encoding='utf8')
    print(json.dumps(results,indent=2),flush=True)
    return results


if __name__ == '__main__':
    main()
