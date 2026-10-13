"""0080J: frozen-checkpoint plan/prefix ablation on fixed 0080I validation windows.

All six conditions use identical main-transformer weights, prior logits,
true targets, EOS boundaries and validation samples. Only decoder inputs vary.
No optimizer steps are taken, no checkpoint is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
from mini_llm.block_proposal.training import SinglePassBlockTrainer
from mini_llm.block_proposal.real_data import PackedRoutedBlockProposalDocuments
from mini_llm.block_proposal.checkpoint import load_two_stage_checkpoint
from mini_llm.predictive_refinement.real_data import sha256
from examples.train_predictive_refinement_text import _load_tokenizer
from examples.train_block_proposal_routed_text import _model_config
from examples.compare_block_proposal_cpu_0080h import _metadata, _sample_digest
from examples.train_block_proposal_cpu_learning_0080i import _signature

CONDITIONS = (
    ('normal', 'normal'),
    ('shuffled', 'normal'),
    ('neutral', 'normal'),
    ('normal', 'masked'),
    ('shuffled', 'masked'),
    ('neutral', 'masked'),
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-output', type=Path, required=True,
                   help='unchanged 0080I output directory containing checkpoints')
    p.add_argument('--mixed-shard-root', type=Path, required=True,
                   help='the SAME tokenized corpus used to train the checkpoint')
    p.add_argument('--checkpoint', type=Path,
                   help='explicit checkpoint directory; default latest two_stage step')
    p.add_argument('--output', type=Path,
                   help='JSON result; default training-output/ablations_0080j.json')
    return p.parse_args(argv)


def _checkpoint(args):
    if args.checkpoint is not None:
        return args.checkpoint.expanduser().resolve()
    folder = args.training_output.expanduser().resolve()/'checkpoints'/'two_stage'
    candidates = sorted(p for p in folder.glob('step_*') if (p/'training_state.json').is_file())
    if not candidates:
        raise FileNotFoundError(f'no two-stage checkpoint in {folder}')
    return candidates[-1]


def _reconstruct(signature, tokenizer_len):
    a = SimpleNamespace(seed=signature['seed'], dtype='float32',
        block_size=signature['block_size'],
        block_group_size=signature['block_group_size'],
        proposal_width=signature['proposal_width'],
        decoder_width=signature['decoder_width'],
        decoder_layers=signature['decoder_layers'],
        decoder_heads=signature['decoder_heads'],
        lambda_proposal=signature['lambda_proposal'],
        learning_rate=signature['learning_rate'],
        weight_decay=signature['weight_decay'],
        clip_grad_norm=signature['clip_grad_norm'],
        batch_size=signature['batch_size'],grad_accum=signature['grad_accum'])
    for preset in ('cpu_tiny','cpu_medium'):
        a.model_preset=preset
        cfg=_model_config(a,tokenizer_len)
        if repr(cfg)==signature['model_config']:
            return a,cfg
    raise ValueError('checkpoint model configuration does not match a supported CPU preset')


def evaluate(trainer, samples, k, group_size):
    results={}
    # The controls deliberately leave proposal_logits in the final residual
    # unchanged. Proposal CE must thus be numerically invariant in all modes.
    for plan,prefix in CONDITIONS:
        total=0
        count=np.zeros(k,dtype=np.int64)
        psum=np.zeros(k,dtype=np.float64)
        fsum=np.zeros(k,dtype=np.float64)
        for sample in samples:
            src,y,mask,metadata,_=sample
            metrics=trainer.train_window(xp.asarray(src,dtype=xp.int64),
                xp.asarray(y,dtype=xp.int64),
                valid_target_mask=xp.asarray(mask,dtype=xp.bool_),
                memory_metadata=_metadata(metadata),
                block_group_size=group_size,compute_gradients=False,
                plan_condition=plan,prefix_condition=prefix)
            n=int(metrics['unique_supervised_tokens'])
            total+=n
            count+=np.asarray(asnumpy(metrics['position_counts']),dtype=np.int64)
            psum+=np.asarray(asnumpy(metrics['proposal_position_ce_sums']),dtype=np.float64)
            fsum+=np.asarray(asnumpy(metrics['final_position_ce_sums']),dtype=np.float64)
        if total!=sum(int(s[2].sum()) for s in samples):
            raise RuntimeError('ablation changed the number of valid supervised tokens')
        if not np.all(count>0) or int(count.sum())!=total:
            raise RuntimeError('missing horizon in fixed validation set')
        result=dict(plan=plan,prefix=prefix,valid_tokens=total,
            proposal_ce=float(psum.sum()/total),final_ce=float(fsum.sum()/total),
            decoder_gain=float((psum.sum()-fsum.sum())/total),
            horizon_counts=count.tolist(),
            proposal_ce_by_horizon=(psum/count).tolist(),
            final_ce_by_horizon=(fsum/count).tolist(),
            decoder_gain_by_horizon=((psum-fsum)/count).tolist())
        name=f'{plan}_plan__{prefix}_prefix'
        results[name]=result
        print(f'{name}: final={result["final_ce"]:.4f} '
              f'proposal={result["proposal_ce"]:.4f} '
              f'gain={result["decoder_gain"]:+.4f}',flush=True)
    base=results['normal_plan__normal_prefix']
    for key,value in results.items():
        if not np.allclose(value['proposal_ce_by_horizon'],
                           base['proposal_ce_by_horizon'],rtol=0,atol=1e-9):
            raise RuntimeError(f'ablation accidentally modified the direct proposal prior: {key}')
        value['final_ce_increase_vs_normal']=value['final_ce']-base['final_ce']
        value['horizon_final_ce_increase_vs_normal']=(
            np.asarray(value['final_ce_by_horizon'])-
            np.asarray(base['final_ce_by_horizon'])).tolist()
    return results


def main(argv=None):
    if BACKEND_NAME!='numpy':
        raise RuntimeError('0080J CPU checkpoint evaluator requires MINI_LLM_BACKEND=numpy')
    args=parse_args(argv)
    ckpt=_checkpoint(args)
    state=json.loads((ckpt/'training_state.json').read_text(encoding='utf8'))
    signature=state['data_provenance']
    root=args.mixed_shard_root.expanduser().resolve()
    tok_path=root/'tokenizer.json'
    tokenizer,eos=_load_tokenizer(tok_path)
    a,cfg=_reconstruct(signature,len(tokenizer))
    kwargs=dict(mixed_root=root,eos_token_id=eos,token_dtype='uint16')
    train=PackedRoutedBlockProposalDocuments(**kwargs,split='train')
    val=PackedRoutedBlockProposalDocuments(**kwargs,split='val')
    val_rng=np.random.default_rng(a.seed+4)
    validation=[val.sample_causal(val_rng,batch_size=a.batch_size,
        memory_length=cfg.memory_context.distant_memory_length,
        target_length=cfg.memory_context.target_length,
        min_history_tokens=(cfg.memory_context.top_k_blocks+1)*cfg.memory_context.block_size-1)
        for _ in signature['validation_digests']]
    reconstructed=_signature(a,cfg,root,tok_path,train,val,validation)
    if reconstructed!=signature:
        raise ValueError('tokenizer/data, checkpoint hyperparameters or fixed validation changed; '
                         'refusing to compare mismatched batches')
    model=DecoderLanguageModel(cfg,rng_seed=a.seed,dtype=a.dtype)
    head=HorizonProposalHead(model,a.block_size,hidden_width=a.proposal_width,rng_seed=a.seed+1)
    decoder=BlockPrefixDecoder(model,a.block_size,width=a.decoder_width,
        n_layers=a.decoder_layers,n_heads=a.decoder_heads,rng_seed=a.seed+2)
    net=TwoStageBlockModel(head,decoder,lambda_proposal=a.lambda_proposal)
    load_two_stage_checkpoint(ckpt,net,optimizer=None,
        expected_tokenizer_hash=sha256(tok_path),expected_data_provenance=signature)
    print(f'0080J frozen checkpoint: {ckpt} (step={state["step"]}), '
          f'validation batches={len(validation)}, batch size={a.batch_size}',flush=True)
    results=evaluate(SinglePassBlockTrainer(net),validation,a.block_size,a.block_group_size)
    report=dict(architecture='0080J_frozen_plan_prefix_ablation',
                checkpoint=str(ckpt),training_step=int(state['step']),
                tokenizer_sha256=sha256(tok_path),
                manifest_sha256=train.manifest_hash,
                validation_digests=signature['validation_digests'],
                conditions=results,
                notes='Fixed held-out teacher forcing; direct proposal-logit prior and targets unchanged. '
                      'The shuffled control swaps compact plans across different sampled rows; sampling with replacement may rarely repeat a document. '
                      'Masked prefix replaces committed-token content with BOS, keeping position embeddings. '
                      'Ablations are out-of-distribution controls, not retrained model comparisons.')
    output=(args.output or args.training_output/'ablations_0080j.json').expanduser().resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite existing ablation results: {output}')
    output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(f'Wrote {output}',flush=True)
    return report


if __name__=='__main__':
    main()
