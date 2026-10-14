"""0080I: matched, checkpointed CPU learning experiment on existing packed sources.

Compares identical routed AR trunks and training examples to two-stage 0080.
Records fixed held-out CE trajectories (and proposal/horizon gains), saves full
AdamW+sampler RNG checkpoints, and supports deterministic recovery of each run.

A validation CE decrease demonstrates next-token generalization but is NOT
free-running generation. The production routed-memory generator is still pending.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp, asnumpy, synchronize
from mini_llm.checkpoint import save_checkpoint, load_checkpoint
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.block_proposal.baseline import MatchedAutoregressiveTrainer
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
from mini_llm.block_proposal.training import SinglePassBlockTrainer
from mini_llm.block_proposal.checkpoint import save_two_stage_checkpoint, load_two_stage_checkpoint
from mini_llm.block_proposal.real_data import PackedRoutedBlockProposalDocuments
from mini_llm.predictive_refinement.real_data import sha256
from examples.train_predictive_refinement_text import _load_tokenizer, _clip_grads
from examples.train_block_proposal_routed_text import _model_config
from examples.compare_block_proposal_cpu_0080h import (
    _run_window, _sample_digest, _count_parameters, _initial_model_digest)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mixed-shard-root', type=Path, required=True)
    p.add_argument('--model-preset', choices=('cpu_tiny', 'cpu_medium'), default='cpu_medium')
    p.add_argument('--dtype', choices=('float32',), default='float32')
    p.add_argument('--architecture', choices=('both', 'standard_ar', 'two_stage'), default='both')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--warmup-steps', type=int, default=10)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--val-batches', type=int, default=8)
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
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, default=Path('benchmarks/results/0080i_cpu_learning'))
    p.add_argument('--order', choices=('ar-first', 'two-stage-first'), default='ar-first')
    p.add_argument('--resume', action='store_true', help='restore the newest checkpoint per architecture in --output')
    p.add_argument('--warm-start-checkpoint', type=Path, default=None,
                   help='initialize a NEW two_stage run from this 0080I checkpoint, retaining model weights only')
    a = p.parse_args(argv)
    if BACKEND_NAME != 'numpy':
        p.error('0080I is a NumPy CPU experiment; set MINI_LLM_BACKEND=numpy')
    for k in ('steps', 'eval_every', 'save_every', 'log_every', 'val_batches', 'batch_size',
              'grad_accum', 'block_size', 'block_group_size', 'proposal_width',
              'decoder_width', 'decoder_layers', 'decoder_heads'):
        if getattr(a, k) < 1:
            p.error(f'--{k.replace("_", "-")} must be >= 1')
    if a.warm_start_checkpoint is not None and a.architecture != 'two_stage':
        p.error('--warm-start-checkpoint currently supports --architecture two_stage only')
    if a.warmup_steps < 0:
        p.error('--warmup-steps must be nonnegative')
    if a.decoder_width % a.decoder_heads:
        p.error('--decoder-width must be divisible by --decoder-heads')
    if a.learning_rate <= 0 or a.lambda_proposal <= 0 or a.clip_grad_norm <= 0:
        p.error('positive learning rate, proposal weight and gradient clipping required')
    return a


def _build(name, a, cfg):
    model = DecoderLanguageModel(cfg, rng_seed=a.seed, dtype='float32')
    main_digest = _initial_model_digest(model)
    if name == 'two_stage':
        head = HorizonProposalHead(model, a.block_size, hidden_width=a.proposal_width,
                                   rng_seed=a.seed + 1)
        dec = BlockPrefixDecoder(model, a.block_size, width=a.decoder_width,
                                 n_layers=a.decoder_layers, n_heads=a.decoder_heads,
                                 rng_seed=a.seed + 2)
        net = TwoStageBlockModel(head, dec, lambda_proposal=a.lambda_proposal)
        trainer = SinglePassBlockTrainer(net)
    else:
        dec = net = None
        trainer = MatchedAutoregressiveTrainer(model)
    opt = AdamW(trainer.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    return model, net, dec, trainer, opt, main_digest


def _signature(a, cfg, root, tok_path, train, val, fixed_validation):
    # Do not include --steps / output / log/validation/checkpoint intervals so
    # a saved short run can be extended without changing the stochastic path.
    return dict(version='0080I-v1', tokenizer_sha256=sha256(tok_path),
        mixture_root=str(root), train_provenance=train.provenance(),
        val_provenance=val.provenance(),
        validation_digests=[_sample_digest([v]) for v in fixed_validation],
        model_config=repr(cfg), seed=a.seed, batch_size=a.batch_size,
        grad_accum=a.grad_accum, block_size=a.block_size,
        block_group_size=a.block_group_size, proposal_width=a.proposal_width,
        decoder_width=a.decoder_width, decoder_layers=a.decoder_layers,
        decoder_heads=a.decoder_heads, lambda_proposal=a.lambda_proposal,
        learning_rate=a.learning_rate, weight_decay=a.weight_decay,
        clip_grad_norm=a.clip_grad_norm)


def _evaluate(name, trainer, records, a, step):
    two_stage = name == 'two_stage'
    n = 0
    fsum = psum = 0.
    horizon_count = np.zeros(a.block_size, dtype=np.int64)
    horizon_proposal = np.zeros(a.block_size, dtype=np.float64)
    horizon_final = np.zeros(a.block_size, dtype=np.float64)
    source_scores = {}
    for sample in records:
        count = int(sample[2].sum())
        if not count:
            continue
        m = _run_window(trainer, sample, a, train=False, two_stage=two_stage)
        n += count
        fsum += m['final_ce'] * count
        if two_stage:
            psum += m['proposal_ce'] * count
            horizon_count += np.asarray(asnumpy(m['position_counts']), dtype=np.int64)
            horizon_proposal += np.asarray(asnumpy(m['proposal_position_ce_sums']), dtype=np.float64)
            horizon_final += np.asarray(asnumpy(m['final_position_ce_sums']), dtype=np.float64)
            per_row = np.asarray(asnumpy(m['row_final_ce_sums']), dtype=np.float64)
            for j, row in enumerate(sample[4]):
                entry = source_scores.setdefault(row['source'], [0., 0])
                entry[0] += float(per_row[j])
                entry[1] += int(sample[2][j].sum())
    if not n:
        raise RuntimeError('fixed validation set has no supervised tokens')
    final = fsum / n
    result = dict(event='validation', architecture=name, step=step, valid_tokens=n,
                  final_ce=final, final_ppl=float(np.exp(min(final, 80.))))
    if two_stage:
        proposal = psum / n
        p = horizon_proposal / np.maximum(horizon_count, 1)
        f = horizon_final / np.maximum(horizon_count, 1)
        result.update(proposal_ce=proposal, decoder_gain=proposal - final,
            horizon_counts=horizon_count.tolist(),
            proposal_ce_by_horizon=p.tolist(), final_ce_by_horizon=f.tolist(),
            decoder_gain_by_horizon=(p-f).tolist(),
            by_source={s:dict(valid_tokens=int(v[1]), final_ce=v[0]/v[1])
                       for s, v in source_scores.items() if v[1]})
    return result


def _save_checkpoint(path, name, model, net, trainer, opt, signature, step, tokens, rng):
    if path.exists():
        raise FileExistsError(f'refusing to overwrite checkpoint: {path}')
    if name == 'two_stage':
        save_two_stage_checkpoint(path, net, optimizer=opt, training_step=step,
            supervised_tokens=tokens, tokenizer_hash=signature['tokenizer_sha256'],
            data_provenance=signature, rng_state=rng.bit_generator.state)
    else:
        params = trainer.parameters()
        save_checkpoint(path, {p.name:p.data for p in params},
            optimizer_state=dict(step=int(opt.step_index),
                m={p.name:v for p,v in zip(params,opt.m)},
                v={p.name:v for p,v in zip(params,opt.v)},
                master_weights={p.name:v for p,v in zip(params,opt.master_weights)}),
            training_state=dict(architecture='matched_ar_0080i_v1',
                signature=signature, step=step, supervised_tokens=tokens,
                rng_state=rng.bit_generator.state,
                optimizer_options={k:float(getattr(opt,k)) for k in
                                   ('lr','beta1','beta2','eps','weight_decay')}))


def _restore_checkpoint(path, name, model, net, trainer, opt, signature, rng):
    if name == 'two_stage':
        state = load_two_stage_checkpoint(path, net, optimizer=opt,
            expected_tokenizer_hash=signature['tokenizer_sha256'],
            expected_data_provenance=signature)
    else:
        saved, saved_opt, state = load_checkpoint(path)
        if state is None or state.get('architecture') != 'matched_ar_0080i_v1' or state.get('signature') != signature:
            raise ValueError('incompatible matched-AR checkpoint architecture/configuration/dataset')
        params = trainer.parameters()
        if set(saved) != {p.name for p in params}:
            raise ValueError('missing/extra AR checkpoint parameters')
        if saved_opt is None or any(set(saved_opt.get(k,{})) != {p.name for p in params}
                                   for k in ('m','v','master_weights')):
            raise ValueError('missing AR AdamW state')
        if state.get('optimizer_options') != {k:float(getattr(opt,k)) for k in
                    ('lr','beta1','beta2','eps','weight_decay')}:
            raise ValueError('AR optimizer configuration mismatch')
        for p in params:
            if saved[p.name].shape != p.data.shape:
                raise ValueError(f'AR parameter shape mismatch: {p.name}')
        for p in params:
            p.data[...] = saved[p.name]
        for block in model.blocks:
            if hasattr(block,'refresh_compute_buffers'):
                block.refresh_compute_buffers()
        for i,p in enumerate(params):
            if not opt.restore_master_weight(i,saved_opt['master_weights'][p.name]):
                raise RuntimeError(f'cannot restore AR master weight {p.name}')
            if not opt.restore_moments(i,saved_opt['m'][p.name],saved_opt['v'][p.name]):
                raise RuntimeError(f'cannot restore AR moments {p.name}')
        opt.step_index = int(saved_opt['step'])
    if state.get('rng_state') is None or opt.step_index != int(state['step']):
        raise ValueError('checkpoint optimizer step or sampling RNG missing/inconsistent')
    rng.bit_generator.state = state['rng_state']
    return int(state['step']), int(state['supervised_tokens'])


def _read_records(path):
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines() if x.strip()]


def _write_records(path, records):
    path.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in records), encoding='utf8')


def _run_arch(name, a, cfg, train, fixed_validation, signature):
    two_stage = name == 'two_stage'
    model, net, decoder, trainer, opt, main_digest = _build(name, a, cfg)
    main_params = _count_parameters(model.parameters())
    total_params = _count_parameters(trainer.parameters())
    root = a.output
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / f'{name}_learning.jsonl'
    val_path = root / f'{name}_validation.jsonl'
    summary_path = root / f'{name}_learning_summary.json'
    ckpt_dir = root / 'checkpoints' / name
    rng = np.random.default_rng(a.seed+3)
    start = 0
    tokens = 0
    if a.resume:
        checkpoints = sorted(p for p in ckpt_dir.glob('step_*') if (p/'training_state.json').is_file())
        if not checkpoints:
            raise FileNotFoundError(f'no checkpoint available for {name} under {ckpt_dir}')
        start,tokens = _restore_checkpoint(checkpoints[-1],name,model,net,trainer,opt,signature,rng)
        if start > a.steps:
            raise ValueError(f'{name} checkpoint step {start} exceeds requested --steps={a.steps}')
        records = [r for r in _read_records(log_path) if r['step'] <= start]
        validations = [r for r in _read_records(val_path) if r['step'] <= start]
        if len(records) != start or not validations or max(r['step'] for r in records) != start:
            raise RuntimeError(f'{name}: training/validation logs incomplete at checkpoint step {start}')
        if sum(r['valid_tokens'] for r in records) != tokens:
            raise ValueError(f'{name}: checkpoint supervised token accounting mismatch')
        _write_records(log_path,records)
        _write_records(val_path,validations)
        print(f'{name}: resumed at step {start}, {tokens} supervised tokens',flush=True)
    else:
        if any(p.exists() for p in (log_path,val_path,summary_path)) or ckpt_dir.exists():
            raise FileExistsError(f'{name}: output already exists; choose new --output or use --resume')
        if a.warm_start_checkpoint is not None:
            origin = load_two_stage_checkpoint(a.warm_start_checkpoint, net, optimizer=None,
                expected_tokenizer_hash=signature['tokenizer_sha256'])
            # Construct fresh AdamW masters AFTER loading the trained weights.
            # Reusing the optimizer built against random initialization would
            # silently update from stale master weights at the first step.
            opt = AdamW(trainer.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
            print(f'{name}: weight-only warm start from {a.warm_start_checkpoint} '
                  f'(source step {origin["step"]}, fresh AdamW and sampling RNG)', flush=True)
        records=[]
        validations=[_evaluate(name,trainer,fixed_validation,a,step=0)]
        validations[0]['cumulative_training_seconds']=0.
        _write_records(val_path,validations)
        print(f'{name}: validation step 0 CE={validations[-1]["final_ce"]:.4f}',flush=True)
    elapsed_training = sum(r['step_seconds'] for r in records)
    if start == a.steps:
        print(f'{name}: already finished {start} steps',flush=True)
    for step in range(start+1,a.steps+1):
        begin=time.perf_counter()
        samples=[train.sample_causal(rng,batch_size=a.batch_size,
            memory_length=cfg.memory_context.distant_memory_length,
            target_length=cfg.memory_context.target_length,
            min_history_tokens=(cfg.memory_context.top_k_blocks+1)*cfg.memory_context.block_size-1)
                 for _ in range(a.grad_accum)]
        unique=sum(int(s[2].sum()) for s in samples)
        digest=_sample_digest(samples)
        loaded=time.perf_counter()
        if not unique:
            raise RuntimeError(f'{name}: no valid supervised tokens at step {step}')
        trainer.zero_grad()
        ce=prop_ce=0.
        for sample in samples:
            n=int(sample[2].sum())
            if not n: continue
            metric=_run_window(trainer,sample,a,train=True,weight=n/unique,two_stage=two_stage)
            ce+=metric['final_ce']*n/unique
            if two_stage:
                prop_ce+=metric['proposal_ce']*n/unique
        synchronize()
        back=time.perf_counter()
        grad_norm=float(_clip_grads(trainer.parameters(),a.clip_grad_norm))
        if not np.isfinite(grad_norm) or not np.isfinite(ce):
            raise FloatingPointError(f'{name}: nonfinite gradient/loss at step {step}')
        opt.step()
        model.refresh_compute_buffers()
        if decoder is not None:
            for layer in decoder.layers:
                if hasattr(layer.attn,'refresh_compute_buffers'):
                    layer.attn.refresh_compute_buffers()
        synchronize()
        end=time.perf_counter()
        tokens+=unique
        rec=dict(event='train',architecture=name,step=step,valid_tokens=unique,
            total_supervised_tokens=tokens,sample_digest=digest,final_ce=float(ce),
            grad_norm=grad_norm, step_seconds=end-begin, data_seconds=loaded-begin,
            forward_backward_seconds=back-loaded,optimizer_seconds=end-back,
            supervised_tok_s=unique/max(end-begin,1e-10), optimizer_step=opt.step_index)
        if two_stage:
            rec.update(proposal_ce=float(prop_ce),decoder_gain=float(prop_ce-ce))
        with log_path.open('a', encoding='utf8') as f:
            f.write(json.dumps(rec,sort_keys=True)+'\n')
        records.append(rec)
        elapsed_training += rec['step_seconds']
        if step==1 or step%a.log_every==0 or step==a.steps:
            print(f'{name} step {step}/{a.steps}: CE={ce:.4f} '
                  f'valid={unique} tok/s={rec["supervised_tok_s"]:.1f}'
                  + (f' proposal={prop_ce:.4f} gain={prop_ce-ce:+.4f}' if two_stage else ''),
                  flush=True)
        if step%a.eval_every==0 or step==a.steps:
            v=_evaluate(name,trainer,fixed_validation,a,step)
            v['cumulative_training_seconds']=elapsed_training
            validations.append(v)
            with val_path.open('a',encoding='utf8') as f:
                f.write(json.dumps(v,sort_keys=True)+'\n')
            print(f'{name} VALIDATION step {step}: CE={v["final_ce"]:.4f}'
                  + (f' proposal={v["proposal_ce"]:.4f} gain={v["decoder_gain"]:+.4f}'
                     if two_stage else ''),flush=True)
        if step%a.save_every==0 or step==a.steps:
            _save_checkpoint(ckpt_dir/f'step_{step:08d}',name,model,net,trainer,opt,
                             signature,step,tokens,rng)
    measured=[r for r in records if r['step']>a.warmup_steps]
    if not measured:
        raise ValueError('no measured steps: reduce --warmup-steps')
    final=validations[-1]
    result=dict(architecture=name,main_initial_weights_sha256=main_digest,
        model_parameters=main_params,trainable_parameters=total_params,
        added_parameters=total_params-main_params,steps=a.steps,
        supervised_tokens=tokens,warmup_steps=a.warmup_steps,
        mean_step_seconds=statistics.mean(r['step_seconds'] for r in measured),
        weighted_supervised_tokens_per_sec=sum(r['valid_tokens'] for r in measured)/sum(r['step_seconds'] for r in measured),
        validation_initial_ce=validations[0]['final_ce'],validation_final_ce=final['final_ce'],
        validation_improvement=validations[0]['final_ce']-final['final_ce'],
        validation_records=validations,step_sample_digests=[r['sample_digest'] for r in records],
        measured_supervised_tokens=sum(r['valid_tokens'] for r in measured))
    if two_stage:
        result.update(validation_proposal_ce=final['proposal_ce'],
                      validation_decoder_gain=final['decoder_gain'],
                      decoder_gain_by_horizon=final['decoder_gain_by_horizon'])
    summary_path.write_text(json.dumps(result,indent=2),encoding='utf8')
    return result


def main(argv=None):
    a=parse_args(argv)
    root=a.mixed_shard_root.expanduser().resolve()
    tokenizer_path=root/'tokenizer.json'
    tok,eos=_load_tokenizer(tokenizer_path)
    cfg=_model_config(a,len(tok))
    kwargs=dict(mixed_root=root,eos_token_id=eos,token_dtype='uint16')
    train=PackedRoutedBlockProposalDocuments(**kwargs,split='train')
    val=PackedRoutedBlockProposalDocuments(**kwargs,split='val')
    val_rng=np.random.default_rng(a.seed+4)
    fixed_validation=[val.sample_causal(val_rng,batch_size=a.batch_size,
         memory_length=cfg.memory_context.distant_memory_length,
         target_length=cfg.memory_context.target_length,
         min_history_tokens=(cfg.memory_context.top_k_blocks+1)*cfg.memory_context.block_size-1)
         for _ in range(a.val_batches)]
    sig=_signature(a,cfg,root,tokenizer_path,train,val,fixed_validation)
    if a.warm_start_checkpoint is not None:
        source = a.warm_start_checkpoint.expanduser().resolve()
        state_file = source / 'training_state.json'
        if not state_file.is_file():
            raise FileNotFoundError(f'missing warm-start checkpoint metadata: {state_file}')
        source_state = json.loads(state_file.read_text(encoding='utf8'))
        sig['warm_start'] = dict(source_checkpoint=str(source),
            source_metadata_sha256=sha256(state_file),
            source_step=int(source_state['step']),
            source_supervised_tokens=int(source_state['supervised_tokens']))
    order=(['standard_ar','two_stage'] if a.order=='ar-first'
           else ['two_stage','standard_ar']) if a.architecture=='both' else [a.architecture]
    print(f'0080I paired learning on {BACKEND_NAME}: {order}, '
          f'{cfg.n_layers} layers x {cfg.d_model}, K={a.block_size}, '
          f'target={cfg.memory_context.target_length}, '
          f'validation={sum(int(s[2].sum()) for s in fixed_validation)} tokens, '
          f'up to {a.steps} steps',flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    results=dict(experiment=dict(signature=sig,steps=a.steps,eval_every=a.eval_every,
        save_every=a.save_every,log_every=a.log_every,warmup_steps=a.warmup_steps))
    for name in order:
        if a.resume and (a.output/f'{name}_learning_summary.json').is_file():
            prior=json.loads((a.output/f'{name}_learning_summary.json').read_text(encoding='utf8'))
            if prior['steps']==a.steps and (a.output/'checkpoints'/name/f'step_{a.steps:08d}'/'training_state.json').is_file():
                # Only skip a completed architecture when its checkpoint metadata
                # and fully reproduced sample path can be verified.
                trainer_model,net,_,trainer,opt,dig = _build(name,a,cfg)
                _restore_checkpoint(a.output/'checkpoints'/name/f'step_{a.steps:08d}',
                                    name,trainer_model,net,trainer,opt,sig,np.random.default_rng())
                if dig != prior['main_initial_weights_sha256']:
                    raise ValueError('completed architecture initial parameter hash mismatch')
                results[name]=prior
                print(f'{name}: validated completed run, skipping',flush=True)
                continue
        results[name]=_run_arch(name,a,cfg,train,fixed_validation,sig)
    if 'standard_ar' in results and 'two_stage' in results:
        ar,tw=results['standard_ar'],results['two_stage']
        if ar['main_initial_weights_sha256'] != tw['main_initial_weights_sha256']:
            raise AssertionError('main initialization differs between matched architectures')
        if ar['step_sample_digests'] != tw['step_sample_digests']:
            raise AssertionError('actual sampled training windows differ')
        results['comparison']=dict(two_stage_over_ar_step_time=(tw['mean_step_seconds']/ar['mean_step_seconds']),
            two_stage_over_ar_tok_s=(tw['weighted_supervised_tokens_per_sec']/ar['weighted_supervised_tokens_per_sec']),
            final_ce_difference_two_stage_minus_ar=(tw['validation_final_ce']-ar['validation_final_ce']),
            interpretation='Matched CPU train + fixed teacher-forced heldout validation; not free-running generation')
    summary=a.output/'learning_summary.json'
    summary.write_text(json.dumps(results,indent=2),encoding='utf8')
    print('0080I final summary: '+json.dumps({k:{'validation_final_ce':v['validation_final_ce'],
        'weighted_supervised_tokens_per_sec':v['weighted_supervised_tokens_per_sec']}
        for k,v in results.items() if k in ('standard_ar','two_stage')}),flush=True)
    return results


if __name__=='__main__':
    main()
