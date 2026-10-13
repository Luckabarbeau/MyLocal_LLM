"""0080F research trainer: one-pass 64K routed K-proposal + prefix decoder.

BEFORE expensive GPU training, use --preflight-only to inspect the real
production mixture. GPU/BF16 correctness, peak VRAM and throughput MUST be
measured on the target RTX 5080; none are inferred from CPU tests.

Example (choose your actual shard root, no implicit dataset paths):
  MINI_LLM_BACKEND=cupy MINI_LLM_OPTIMIZER_OFFLOAD=full \\
    python -m examples.train_block_proposal_routed_text \\
      --mixed-shard-root YOUR_EXISTING_MIXED_SHARDS --preflight-only

Experimental initial single-step run:
  MINI_LLM_BACKEND=cupy MINI_LLM_OPTIMIZER_OFFLOAD=full \\
    python -m examples.train_block_proposal_routed_text \\
      --mixed-shard-root YOUR_EXISTING_MIXED_SHARDS \\
      --steps 1 --batch-size 1 --grad-accum 1 --block-group-size 1
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from mini_llm.backend import BACKEND_NAME,xp,asnumpy,synchronize,validate_bfloat16_backend
from mini_llm.config import (ModelConfig, MemoryContextConfig, AttentionLayerConfig,
    LocalAttentionConfig, DilatedAttentionConfig, GlobalSparseAttentionConfig,
    RetrievalAttentionConfig, ContextRouterConfig)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.predictive_refinement.real_data import sha256
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.decoder import BlockPrefixDecoder,TwoStageBlockModel
from mini_llm.block_proposal.training import SinglePassBlockTrainer
from mini_llm.block_proposal.real_data import PackedRoutedBlockProposalDocuments
from mini_llm.block_proposal.checkpoint import save_two_stage_checkpoint,load_two_stage_checkpoint
from examples.train_predictive_refinement_text import _load_tokenizer,_clip_grads


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sources=parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--mixed-shard-root',type=Path)
    sources.add_argument('--shard-dir',type=Path)
    parser.add_argument('--tokenizer-path',type=Path)
    parser.add_argument('--token-dtype',choices=['uint16','uint32'],default='uint16')
    parser.add_argument('--dtype',choices=['bfloat16','float32'],default='bfloat16')
    parser.add_argument('--model-preset', choices=['wide_500m','cpu_tiny','cpu_medium'], default='wide_500m',
        help='cpu_tiny uses small learned-routing architecture and actual tokenizer vocabulary')
    parser.add_argument('--block-size',type=int,default=8)
    parser.add_argument('--block-group-size',type=int,default=1)
    parser.add_argument('--proposal-width',type=int,default=384)
    parser.add_argument('--decoder-width',type=int,default=384)
    parser.add_argument('--decoder-layers',type=int,default=3)
    parser.add_argument('--decoder-heads',type=int,default=6)
    parser.add_argument('--lambda-proposal',type=float,default=0.5)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--grad-accum',type=int,default=1)
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--weight-decay',type=float,default=.1)
    parser.add_argument('--clip-grad-norm',type=float,default=1.)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--val-batches',type=int,default=2)
    parser.add_argument('--val-every',type=int,default=50)
    parser.add_argument('--save-every',type=int,default=50)
    parser.add_argument('--log-every',type=int,default=1)
    parser.add_argument('--checkpoint-dir',type=Path,default=Path('checkpoints/block-proposal-0080'))
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--eval-only',action='store_true')
    parser.add_argument('--no-save',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args(argv)
    if min(args.block_size,args.block_group_size,args.proposal_width,args.decoder_width,
           args.decoder_layers,args.decoder_heads,args.batch_size,args.grad_accum,
           args.val_batches,args.val_every,args.save_every,args.log_every)<1:
        parser.error('all shape, batch and reporting frequencies must be positive')
    if args.block_size not in (1,4,8,16):
        parser.error('baseline supported block sizes are 1,4,8,16')
    if args.decoder_width%args.decoder_heads:
        parser.error('decoder width must be divisible by decoder heads')
    if args.steps<0 or args.learning_rate<=0 or args.lambda_proposal<=0:
        parser.error('steps >=0 and positive loss/learning-rate required')
    if args.eval_only and not args.resume:
        parser.error('--eval-only requires --resume')
    return args


def _sample(data,rng,args,cfg):
    mc=cfg.memory_context
    return data.sample_causal(rng,batch_size=args.batch_size,
         memory_length=mc.distant_memory_length,target_length=mc.target_length,
         min_history_tokens=mc.top_k_blocks*mc.block_size+mc.block_size-1)


def _provenance(args,tokenizer_path,train,val):
    return dict(experiment='0080_two_stage_block_v1',
        tokenizer_sha256=sha256(tokenizer_path),train_data=train.provenance(),
        val_data=val.provenance(),sampling='original_unshifted_next_token_document_first',
        seed=args.seed,block_size=args.block_size,decoder_width=args.decoder_width,
        decoder_layers=args.decoder_layers,decoder_heads=args.decoder_heads,
        block_group_size=args.block_group_size,proposal_width=args.proposal_width,
        lambda_proposal=args.lambda_proposal,dtype=args.dtype,model_preset=args.model_preset)


def _window(trainer,sample,args,*,grad,scale=1.):
    source,targets,valid,metadata,observations=sample
    if not valid.any():
        return None
    return trainer.train_window(xp.asarray(source,dtype=xp.int64),
        xp.asarray(targets,dtype=xp.int64),
        valid_target_mask=xp.asarray(valid,dtype=xp.bool_),
        memory_metadata={key:(xp.asarray(v) if isinstance(v,np.ndarray) else v)
                         for key,v in metadata.items()},
        block_group_size=args.block_group_size,
        compute_gradients=grad,grad_scale=scale)


def _evaluate(trainer,records,args):
    n=sum(int(x[2].sum()) for x in records)
    if n==0:
        return None
    sums=dict(proposal_ce=0.,final_ce=0.,total_loss=0.)
    position_count=np.zeros(args.block_size,dtype=np.int64)
    proposal_position=np.zeros(args.block_size,dtype=np.float64)
    final_position=np.zeros(args.block_size,dtype=np.float64)
    by_source={}
    observed=0
    for sample in records:
        if not sample[2].any():
            continue
        metric=_window(trainer,sample,args,grad=False)
        weight=metric['unique_supervised_tokens']/n
        observed+=metric['unique_supervised_tokens']
        for key in sums:
            sums[key]+=metric[key]*weight
        position_count+=np.asarray(asnumpy(metric['position_counts']),dtype=np.int64)
        proposal_position+=np.asarray(asnumpy(metric['proposal_position_ce_sums']),dtype=np.float64)
        final_position+=np.asarray(asnumpy(metric['final_position_ce_sums']),dtype=np.float64)
        per_row=np.asarray(asnumpy(metric['row_final_ce_sums']),dtype=np.float64)
        for row,ob in enumerate(sample[4]):
            source=ob['source']
            info=by_source.setdefault(source,dict(tokens=0,final_nll=0.))
            info['tokens']+=int(sample[2][row].sum())
            info['final_nll']+=float(per_row[row])
    p=proposal_position/np.maximum(position_count,1)
    f=final_position/np.maximum(position_count,1)
    return dict(sums,unique_supervised_tokens=observed,
                position_counts=position_count.tolist(),
                proposal_ce_per_horizon=p.tolist(),final_ce_per_horizon=f.tolist(),
                decoder_gain_per_horizon=(p-f).tolist(),
                by_source={name:dict(tokens=v['tokens'],final_ce=v['final_nll']/v['tokens'])
                           for name,v in by_source.items() if v['tokens']},
                proposal_perplexity=float(np.exp(min(sums['proposal_ce'],80))),
                final_perplexity=float(np.exp(min(sums['final_ce'],80))))


def _model_config(args, vocab_size):
    if args.model_preset == 'wide_500m':
        if vocab_size != 65_280:
            raise ValueError('wide_500m requires its original 65,280-token vocabulary')
        return replace(ModelConfig.wide_500m_memory_64k(), dtype=args.dtype)
    if args.model_preset == 'cpu_medium':
        # CPU benchmark: identical 6-layer routed + MoE trunk for both models.
        mc = MemoryContextConfig(enabled=True, memory_length=256,
            recent_length=64, target_length=64, block_size=16,
            top_k_blocks=2, router_query_length=64, router_dim=32,
            integration_mode='routed_prefix', memory_training='joint',
            router_gumbel_noise=False, read_heads=1, read_kv_heads=1)
        router = ContextRouterConfig(history_block_size=16, routing_stride=16,
            query_window=16, router_dim=16, top_k_blocks=2,
            exclude_recent_tokens=8, query_pooling='mean',
            history_pooling='mean', num_queries=1)
        attention = AttentionLayerConfig(heads=(
            LocalAttentionConfig(window=64), LocalAttentionConfig(window=64),
            DilatedAttentionConfig(window=96, dilation=4),
            GlobalSparseAttentionConfig(stride=16),
            RetrievalAttentionConfig(router), RetrievalAttentionConfig(router)))
        return ModelConfig(tokenizer_vocab_size=vocab_size,
            context_length=mc.active_length, n_layers=6, d_model=192,
            n_q_heads=6, n_kv_heads=3, d_head=32, d_ff=384,
            n_experts=4, top_k=2, memory_context=mc,
            attention_layers=(attention,) * 6, dtype=args.dtype)
    # CPU correctness/learning test. Preserve the production ROUTED-PREFIX
    # topology and a genuine per-layer learned-retrieval head, but use a short
    # bounded history and small model to make native NumPy feasible.
    mc = MemoryContextConfig(enabled=True, memory_length=128,
        recent_length=32, target_length=32, block_size=8,
        top_k_blocks=2, router_query_length=32, router_dim=16,
        integration_mode='routed_prefix', memory_training='joint',
        router_gumbel_noise=False, read_heads=1, read_kv_heads=1)
    router = ContextRouterConfig(history_block_size=8, routing_stride=8,
        query_window=8, router_dim=8, top_k_blocks=2,
        exclude_recent_tokens=4, query_pooling='mean', history_pooling='mean',
        num_queries=1)
    attention = AttentionLayerConfig(heads=(
        LocalAttentionConfig(window=32), RetrievalAttentionConfig(router)))
    return ModelConfig(tokenizer_vocab_size=vocab_size,context_length=mc.active_length,
        n_layers=2,d_model=64,n_q_heads=2,n_kv_heads=1,d_head=32,
        d_ff=128,n_experts=2,top_k=1,memory_context=mc,
        attention_layers=(attention,attention),dtype=args.dtype)


def main(argv=None):
    args=parse_args(argv)
    root=args.mixed_shard_root or args.shard_dir
    tok_path=(args.tokenizer_path or root/'tokenizer.json').expanduser().resolve()
    tokenizer,eos=_load_tokenizer(tok_path)
    data_kwargs=dict(mixed_root=args.mixed_shard_root,shard_dir=args.shard_dir,
                     eos_token_id=eos,token_dtype=args.token_dtype)
    train=PackedRoutedBlockProposalDocuments(**data_kwargs,split='train')
    val=PackedRoutedBlockProposalDocuments(**data_kwargs,split='val')
    provenance=_provenance(args,tok_path,train,val)
    config=_model_config(args,len(tokenizer))
    mc=config.memory_context
    print(f'0080 architecture={config.n_layers} layers, d={config.d_model}, '
          f'full-context={mc.memory_length},active={mc.active_length},'
          f'block-K={args.block_size},decoder={args.decoder_layers}x{args.decoder_width},'
          f'group={args.block_group_size},backend={BACKEND_NAME}',flush=True)
    if args.preflight_only:
        s=_sample(train,np.random.default_rng(args.seed+2),args,config)
        print(json.dumps(dict(event='preflight',source_shape=s[0].shape,
            labels_shape=s[1].shape,supervised_tokens=int(s[2].sum()),
            valid_starts=s[3]['history_valid_starts'].tolist(),
            sources=[x['source'] for x in s[4]],provenance=provenance),
            sort_keys=True),flush=True)
        return
    if args.dtype=='bfloat16':
        validate_bfloat16_backend()
    model=DecoderLanguageModel(config,rng_seed=args.seed,dtype=args.dtype)
    head=HorizonProposalHead(model,args.block_size,
          hidden_width=args.proposal_width,rng_seed=args.seed+1)
    decoder=BlockPrefixDecoder(model,args.block_size,width=args.decoder_width,
          n_layers=args.decoder_layers,n_heads=args.decoder_heads,rng_seed=args.seed+2)
    net=TwoStageBlockModel(head,decoder,lambda_proposal=args.lambda_proposal)
    trainer=SinglePassBlockTrainer(net)
    optimizer=AdamW(net.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    rng=np.random.default_rng(args.seed+3)
    val_rng=np.random.default_rng(args.seed+4)
    fixed_validation=[_sample(val,val_rng,args,config) for _ in range(args.val_batches)]
    total=0;start=0
    if args.resume:
        resumed=load_two_stage_checkpoint(args.resume,net,optimizer=optimizer,
                expected_tokenizer_hash=provenance['tokenizer_sha256'],
                expected_data_provenance=provenance)
        if resumed['rng_state'] is None:
            raise ValueError('checkpoint missing dataset train sampling RNG')
        rng.bit_generator.state=resumed['rng_state']
        total=int(resumed['supervised_tokens']);start=int(resumed['step'])
    if args.eval_only:
        print(json.dumps(dict(event='validation',step=start,
                              **(_evaluate(trainer,fixed_validation,args) or {}))),flush=True)
        return
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    for step in range(start+1,args.steps+1):
        t0=time.perf_counter()
        micros=[_sample(train,rng,args,config) for _ in range(args.grad_accum)]
        unique=sum(int(s[2].sum()) for s in micros)
        trainer.zero_grad()
        vals=dict(proposal_ce=0.,final_ce=0.,total_loss=0.)
        position_count=np.zeros(args.block_size,dtype=np.int64)
        proposal_position=np.zeros(args.block_size,dtype=np.float64)
        final_position=np.zeros(args.block_size,dtype=np.float64)
        if unique:
            for sample in micros:
                count=int(sample[2].sum())
                if not count:
                    continue
                m=_window(trainer,sample,args,grad=True,scale=count/unique)
                for name in vals:
                    vals[name]+=m[name]*count/unique
                position_count+=np.asarray(asnumpy(m['position_counts']),dtype=np.int64)
                proposal_position+=np.asarray(asnumpy(m['proposal_position_ce_sums']),dtype=np.float64)
                final_position+=np.asarray(asnumpy(m['final_position_ce_sums']),dtype=np.float64)
            grad_norm=_clip_grads(trainer.parameters(),args.clip_grad_norm)
            optimizer.step()
            model.refresh_compute_buffers()
            for layer in decoder.layers:
                if hasattr(layer.attn,'refresh_compute_buffers'):
                    layer.attn.refresh_compute_buffers()
        else:
            grad_norm=0.
        synchronize()
        secs=time.perf_counter()-t0
        total+=unique
        position_p=proposal_position/np.maximum(position_count,1)
        position_f=final_position/np.maximum(position_count,1)
        record=dict(event='train',step=step,**vals,unique_supervised_tokens=unique,
                    cumulative_supervised_tokens=total,step_seconds=secs,
                    supervised_tok_s=unique/max(secs,1e-9),grad_norm=float(grad_norm),
                    position_counts=position_count.tolist(),
                    proposal_ce_per_horizon=position_p.tolist(),
                    final_ce_per_horizon=position_f.tolist(),
                    decoder_gain_per_horizon=(position_p-position_f).tolist(),
                    optimizer_step=optimizer.step_index)
        with (args.checkpoint_dir/'metrics.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
        if step==start+1 or step%args.log_every==0:
            print(json.dumps(record,sort_keys=True),flush=True)
        if step%args.val_every==0:
            valrec=dict(event='validation',step=step,**(_evaluate(trainer,fixed_validation,args) or {}))
            print(json.dumps(valrec,sort_keys=True),flush=True)
        if not args.no_save and step%args.save_every==0:
            path=args.checkpoint_dir/f'step_{step:08d}'
            if path.exists():
                raise FileExistsError(f'checkpoint already exists: {path}')
            save_two_stage_checkpoint(path,net,optimizer=optimizer,training_step=step,
                 supervised_tokens=total,tokenizer_hash=provenance['tokenizer_sha256'],
                 data_provenance=provenance,rng_state=rng.bit_generator.state)
    if not args.no_save and args.steps>start and args.steps%args.save_every:
        path=args.checkpoint_dir/f'step_{args.steps:08d}'
        if path.exists():
            raise FileExistsError(f'checkpoint already exists: {path}')
        save_two_stage_checkpoint(path,net,optimizer=optimizer,training_step=args.steps,
             supervised_tokens=total,tokenizer_hash=provenance['tokenizer_sha256'],
             data_provenance=provenance,rng_state=rng.bit_generator.state)


if __name__=='__main__':
    main()
