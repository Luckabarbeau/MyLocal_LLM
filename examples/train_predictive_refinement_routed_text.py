"""0079Q4: true packed real-text training of 64K/2K+4K K=8 refinement.

Run a small CUDA preflight *first*. This still performs the complete 6144
position causal trunk. Optimizer states are CPU-offloaded when the environment
sets MINI_LLM_OPTIMIZER_OFFLOAD=full (recommended on 16-GiB devices).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp, asnumpy, synchronize, validate_bfloat16_backend
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.predictive_refinement.training_reference import RefinementReferenceConfig
from mini_llm.predictive_refinement.window_training import WindowRefinementTrainer
from mini_llm.predictive_refinement.routed_real_data import PackedRoutedRefinementDocuments
from mini_llm.predictive_refinement.real_data import sha256
from mini_llm.predictive_refinement.checkpoint import save_refinement_checkpoint, load_refinement_checkpoint
from examples.train_predictive_refinement_text import _load_tokenizer, _clip_grads


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument('--mixed-shard-root', type=Path)
    sources.add_argument('--shard-dir', type=Path)
    p.add_argument('--tokenizer-path', type=Path)
    p.add_argument('--token-dtype', choices=('uint16', 'uint32'), default='uint16')
    p.add_argument('--dtype', choices=('bfloat16','float32'), default='bfloat16')
    p.add_argument('--block-size', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--refinement-group-size', type=int, default=1)
    p.add_argument('--max-active-queries', type=int, default=8)
    p.add_argument('--max-history-keys', type=int, default=6144)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--max-hours', type=float, default=0.,
                   help='optional wall-clock limit; finish current optimizer step and save before exiting')
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--clip-grad-norm', type=float, default=1.0)
    p.add_argument('--initial-loss-weight', type=float, default=0.1)
    p.add_argument('--feedback-mode', choices=('full', 'unknown'), default='full')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log-every', type=int, default=1)
    p.add_argument('--val-every', type=int, default=50)
    p.add_argument('--val-batches', type=int, default=2)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints/refinement-500m-routed-k8-q4'))
    p.add_argument('--resume', type=Path)
    p.add_argument('--no-save', action='store_true')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--preflight-only', action='store_true', help='verify dataset, tokenizer, and configuration without allocating a 500M model')
    p.add_argument('--max-supervised-blocks', type=int, default=None,
                   help='DIAGNOSTIC ONLY; requires --smoke, truncates loss after N blocks')
    p.add_argument('--smoke', action='store_true', help='allow a deliberately truncated diagnostic step (never for serious training)')
    args = p.parse_args(argv)
    if min(args.block_size, args.batch_size, args.grad_accum, args.refinement_group_size,
           args.max_active_queries, args.max_history_keys, args.val_batches,
           args.log_every, args.save_every, args.val_every) < 1:
        p.error('batch, block, gradient, query and log parameters must be positive')
    if args.refinement_group_size > 64 or args.block_size > 16:
        p.error('block-size must be <= 16 and refinement-group-size <= 64')
    if args.max_active_queries < args.batch_size * args.refinement_group_size * args.block_size:
        p.error('active-query safety limit is too small for this physical batch')
    if args.max_supervised_blocks is not None and (not args.smoke or args.max_supervised_blocks < 1):
        p.error('--max-supervised-blocks requires --smoke and a positive count')
    if args.smoke and args.steps != 1:
        p.error('--smoke requires --steps 1')
    if args.eval_only and not args.resume:
        p.error('--eval-only requires --resume')
    if args.steps < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or (
        not math.isfinite(args.max_hours) or args.max_hours < 0):
        p.error('invalid training steps or learning rate')
    return args


def _serialize_rng(rng):
    return rng.bit_generator.state


def _json_append(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf8') as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')


def _provenance(args, path, train, val):
    return dict(experiment='0079Q4_true_routed_refinement_v1',
                tokenizer_sha256=sha256(path), train_data=train.provenance(),
                val_data=val.provenance(), seed=args.seed,
                model='wide-500m-memory-64k', block_size=args.block_size,
                batch_size=args.batch_size, grad_accum=args.grad_accum,
                refinement_group_size=args.refinement_group_size,
                min_history_policy='production_top_k_times_block_plus_block_minus_one',
                mask_policy='tail_only_causal', target_alignment='next_token_shifted_one',
                dtype=args.dtype, initial_loss_weight=args.initial_loss_weight,
                feedback_mode=args.feedback_mode, max_history_keys=args.max_history_keys)


def _sample(data, rng, args, cfg):
    mc = cfg.memory_context
    return data.sample_routed(rng, batch_size=args.batch_size,
                              memory_length=mc.distant_memory_length,
                              target_length=mc.target_length,
                              min_history_tokens=mc.top_k_blocks*mc.block_size+mc.block_size-1)


def _run_window(ref, sample, args, *, normalization, compute_gradients, record_row_ce=True):
    source, valid, metadata, observations = sample
    if int(valid.sum()) == 0:
        return None
    result = ref.train_routed_window(
        xp.asarray(source, dtype=xp.int64),
        memory_metadata={k:(xp.asarray(v) if isinstance(v,np.ndarray) else v) for k,v in metadata.items()},
        valid_token_mask=xp.asarray(valid), group_size=args.refinement_group_size,
        loss_normalization_tokens=int(normalization),
        compact_active_blocks=True, compact_history=True,
        resident_history=True, max_window_tokens=ref.model.config.memory_context.target_length,
        max_active_queries=args.max_active_queries, max_history_keys=args.max_history_keys,
        max_supervised_blocks=args.max_supervised_blocks if args.smoke else None,
        compute_gradients=compute_gradients, record_row_ce=record_row_ce,
    )
    return result


def _fixed_validation(val_data, args, cfg):
    vrng = np.random.default_rng(args.seed + 0x1000)
    return [_sample(val_data, vrng, args, cfg) for _ in range(args.val_batches)]


def _evaluate(ref, samples, args):
    total = sum(int(x[1].sum()) for x in samples)
    if args.smoke and args.max_supervised_blocks:
        total = sum(int(x[1][:,:args.max_supervised_blocks*args.block_size].sum()) for x in samples)
    if total == 0:
        return None
    ce, initial, observed = 0., 0., 0
    pos_sum = np.zeros(args.block_size,dtype=np.float64)
    pos_count = np.zeros(args.block_size,dtype=np.int64)
    by_source = {}
    for sample in samples:
        if not sample[1].any():
            continue
        result = _run_window(ref, sample, args, normalization=total, compute_gradients=False)
        ce += float(result['commitment_ce'])
        initial += float(result['initial_ce'])
        pos_sum += np.asarray(asnumpy(result['commitment_position_ce_sums']),dtype=np.float64)
        pos_count += np.asarray(asnumpy(result['position_counts']),dtype=np.int64)
        observed += int(result['supervised_tokens'])
        nll = np.asarray(asnumpy(result['row_commitment_ce_sums']),dtype=np.float64)
        for i,ob in enumerate(sample[3]):
            name=ob['source']
            row=by_source.setdefault(name, {'nll_sum':0.,'tokens':0})
            row['nll_sum'] += float(nll[i]); row['tokens'] += int(sample[1][i].sum())
    out=dict(commitment_ce=ce, initial_ce=initial, supervised_tokens=observed,
             position_ce=(pos_sum/np.maximum(pos_count,1)).tolist(),
             position_counts=pos_count.tolist(),
             by_source={k:dict(tokens=v['tokens'], ce=v['nll_sum']/v['tokens'] if v['tokens'] else None)
                        for k,v in by_source.items()})
    return out


def _save(args, ref, optimizer, step, total, provenance, train_rng):
    # Write step-specific checkpoints first. Never overwrite an existing
    # finalized checkpoint during failed/incomplete saving.
    path = args.checkpoint_dir / f'step_{step:08d}'
    if path.exists():
        raise FileExistsError(f'checkpoint already exists: {path}')
    save_refinement_checkpoint(path, ref, optimizer=optimizer,
                               training_step=step, supervised_tokens=total,
                               tokenizer_fingerprint=provenance['tokenizer_sha256'],
                               data_provenance=provenance)
    (path/'routed_training_rng.json').write_text(json.dumps(dict(
        step=step, train_rng=_serialize_rng(train_rng), provenance=provenance)), encoding='utf8')
    print(f'checkpoint={path}',flush=True)


def main(argv=None):
    args=parse_args(argv)
    root=args.mixed_shard_root or args.shard_dir
    path=(args.tokenizer_path or root/'tokenizer.json').expanduser().resolve()
    tok,eos=_load_tokenizer(path)
    if len(tok)!=65280:
        raise ValueError(f'production routed model needs existing 65280-vocab tokenizer, got {len(tok)}')
    kw=dict(mixed_root=args.mixed_shard_root,shard_dir=args.shard_dir,
            eos_token_id=eos,token_dtype=args.token_dtype)
    train=PackedRoutedRefinementDocuments(**kw,split='train')
    val=PackedRoutedRefinementDocuments(**kw,split='val')
    provenance=_provenance(args,path,train,val)
    config=replace(ModelConfig.wide_500m_memory_64k(),dtype=args.dtype)
    mc=config.memory_context
    if mc.target_length % args.block_size or mc.retrieved_length % args.block_size:
        raise ValueError('refinement block must divide target and routed prefix')
    print(f'Q4 config: model=wide-500m-memory-64k source={mc.source_input_length} '
          f'working={mc.target_length} routed_max={mc.retrieved_length} K={args.block_size} '
          f'passes={args.block_size+1} batch={args.batch_size} accum={args.grad_accum} '
          f'groups={args.refinement_group_size} backend={BACKEND_NAME} '
          f'optimizer_offload=environment',flush=True)
    if args.preflight_only:
        sample=_sample(train,np.random.default_rng(args.seed+2),args,config)
        print(f'preflight sampled: source={sample[0].shape} valid_targets={int(sample[1].sum())} '
              f'history_valid_starts={sample[2]["history_valid_starts"].tolist()} '
              f'sources={[x["source"] for x in sample[3]]}',flush=True)
        return
    if args.dtype=='bfloat16':
        validate_bfloat16_backend()
    ref=WindowRefinementTrainer(
        DecoderLanguageModel(config,rng_seed=args.seed,dtype=args.dtype),
        RefinementReferenceConfig(block_size=args.block_size,
                                  initial_loss_weight=args.initial_loss_weight,
                                  feedback_mode=args.feedback_mode,
                                  enable_shared_context_training=True,
                                  compact_recurrent_tape=True),rng_seed=args.seed+1)
    optimizer=AdamW(ref.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    train_rng=np.random.default_rng(args.seed+2)
    fixed_val=_fixed_validation(val,args,config)
    start=0;total_tokens=0
    if args.resume:
        state=load_refinement_checkpoint(args.resume,ref,optimizer=optimizer)
        if state['architecture'].get('data_provenance')!=provenance:
            raise ValueError('dataset/experiment provenance mismatch on resume')
        start=int(state['step']); total_tokens=int(state['supervised_tokens'])
        saved=json.loads((args.resume/'routed_training_rng.json').read_text())
        if int(saved['step'])!=start:
            raise ValueError('routed RNG step mismatch')
        train_rng.bit_generator.state=saved['train_rng']
    if args.eval_only:
        evaluation=_evaluate(ref,fixed_val,args)
        print('fixed_validation='+json.dumps(evaluation,sort_keys=True),flush=True)
        return
    log_path=args.checkpoint_dir/'routed_training.jsonl'
    last_saved=None
    run_start=time.perf_counter()
    final_step=start
    for step in range(start+1,args.steps+1):
        final_step=step
        t0=time.perf_counter()
        micro=[_sample(train,train_rng,args,config) for _ in range(args.grad_accum)]
        # Preserve original sample distribution. Empty one-token documents
        # contribute 0 supervision, never trigger a length-conditioned retry.
        expected=sum(int(s[1][:,:args.max_supervised_blocks*args.block_size].sum())
                     if args.smoke and args.max_supervised_blocks else int(s[1].sum())
                     for s in micro)
        ref.zero_grad()
        loss=initial=0.; actual=0; modes={}
        sources={}
        if expected:
            for sample in micro:
                if sample[1].any():
                    result=_run_window(ref,sample,args,normalization=expected,compute_gradients=True)
                    loss+=float(result['commitment_ce']);initial+=float(result['initial_ce'])
                    actual+=int(result['supervised_tokens'])
                    mode=str(result['routed_prefix_mode'])
                    modes[mode]=modes.get(mode,0)+1
                    row_nll=np.asarray(asnumpy(result['row_commitment_ce_sums']),dtype=np.float64)
                    for i,ob in enumerate(sample[3]):
                        name=ob['source']
                        count=int(sample[1][i].sum())
                        if args.smoke and args.max_supervised_blocks:
                            count=int(sample[1][i,:args.max_supervised_blocks*args.block_size].sum())
                        row=sources.setdefault(name,dict(tokens=0,nll_sum=0.,samples=0,
                                                        document_span_sum=0))
                        row['tokens']+=count
                        row['nll_sum']+=float(row_nll[i])
                        row['samples']+=1
                        row['document_span_sum']+=int(ob['sampled_document_span_tokens'])
        if actual!=expected:
            raise AssertionError(f'loss normalization mismatch: {actual} != {expected}')
        grad_norm=0.
        if actual:
            grad_norm=_clip_grads(ref.parameters(),args.clip_grad_norm)
            optimizer.step()
        synchronize()
        elapsed=time.perf_counter()-t0
        total_tokens+=actual
        record=dict(kind='train',step=step,supervised_tokens=actual,
                    cumulative_supervised_tokens=total_tokens,
                    commitment_ce=loss if actual else None,
                    initial_ce=initial if actual else None,
                    grad_norm=grad_norm,step_ms=elapsed*1000.,
                    supervised_tok_s=actual/max(elapsed,1e-9),
                    routed_prefix_modes=modes,
                    optimizer_step=int(optimizer.step_index),
                    by_source={name:dict(tokens=row['tokens'],
                        commitment_ce=(row['nll_sum']/row['tokens'] if row['tokens'] else None),
                        sampled_document_span_mean=row['document_span_sum']/row['samples'])
                        for name,row in sources.items()},
                    diagnostic_truncated=bool(args.smoke and args.max_supervised_blocks))
        _json_append(log_path,record)
        if step==start+1 or step%args.log_every==0:
            print('step={step} tokens={supervised_tokens} total={cumulative_supervised_tokens} '
                  'CE={commitment_ce} grad={grad_norm:.4g} tok/s={supervised_tok_s:.1f} '
                  'seconds={seconds:.2f} route={routed_prefix_modes}'.format(
                      seconds=elapsed,**record),flush=True)
        if step%args.val_every==0:
            v0=time.perf_counter()
            val_metrics=_evaluate(ref,fixed_val,args)
            synchronize()
            print(f'validation step={step} {json.dumps(val_metrics,sort_keys=True)} '
                  f'elapsed_s={time.perf_counter()-v0:.1f}',flush=True)
            _json_append(log_path,dict(kind='validation',step=step,**(val_metrics or {})))
        if not args.no_save and step%args.save_every==0:
            _save(args,ref,optimizer,step,total_tokens,provenance,train_rng)
            last_saved=step
        if args.max_hours and (time.perf_counter()-run_start)/3600 >= args.max_hours:
            print(f'wall-clock budget reached after step {step}; stopping cleanly',flush=True)
            break
    if not args.no_save and final_step>start and last_saved!=final_step:
        _save(args,ref,optimizer,final_step,total_tokens,provenance,train_rng)


if __name__=='__main__':
    main()
