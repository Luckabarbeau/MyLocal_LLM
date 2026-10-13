"""0080K: does the decoder use *other* future-position proposals?

Frozen checkpoint, deterministic fixed held-out examples, main forward once
per validation window, small decoder replicated only within bounded groups.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
from mini_llm.block_proposal.training import SinglePassBlockTrainer
from mini_llm.block_proposal.diagnostics import SCOPES, score_window
from mini_llm.block_proposal.real_data import PackedRoutedBlockProposalDocuments
from mini_llm.block_proposal.checkpoint import load_two_stage_checkpoint
from mini_llm.predictive_refinement.real_data import sha256
from examples.train_predictive_refinement_text import _load_tokenizer
from examples.compare_block_proposal_cpu_0080h import _metadata
from examples.train_block_proposal_cpu_learning_0080i import _signature
from examples.evaluate_block_proposal_ablations_0080j import _checkpoint, _reconstruct


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-output', type=Path, required=True)
    p.add_argument('--mixed-shard-root', type=Path, required=True)
    p.add_argument('--checkpoint',type=Path, help='default: latest saved two_stage checkpoint')
    p.add_argument('--output',type=Path, help='default: training-output/ablations_0080k_cross_position.json')
    return p.parse_args(argv)


def evaluate(trainer, samples, k, group_size):
    count=np.zeros(k,dtype=np.int64)
    proposal=np.zeros(k,dtype=np.float64)
    final={name:np.zeros(k,dtype=np.float64) for name in SCOPES}
    main_forward_passes=0
    for sample in samples:
        source, y, mask, meta, _ = sample
        result=score_window(trainer, xp.asarray(source,dtype=xp.int64),
            xp.asarray(y,dtype=xp.int64), xp.asarray(mask,dtype=xp.bool_),
            _metadata(meta),group_size=group_size)
        count+=result['counts']
        proposal+=result['proposal_sums']
        for name in SCOPES:
            final[name]+=result['final_sums'][name]
        main_forward_passes+=result['main_forward_passes']
    total=int(sum(count))
    if total!=sum(int(sample[2].sum()) for sample in samples):
        raise RuntimeError('validation supervised token count changed')
    reference=final['all']/count
    results={}
    for name in SCOPES:
        nll=final[name]/count
        results[name]=dict(valid_tokens=total,horizon_counts=count.tolist(),
            proposal_ce=float(sum(proposal)/total),
            final_ce=float(sum(final[name])/total),
            proposal_ce_by_horizon=(proposal/count).tolist(),
            final_ce_by_horizon=nll.tolist(),
            horizon_ce_increase_vs_all=(nll-reference).tolist(),
            final_ce_increase_vs_all=float((sum(final[name])-sum(final['all']))/total))
        print(f'{name:>19}: final={results[name]["final_ce"]:.4f} '
              f'delta={results[name]["final_ce_increase_vs_all"]:+.4f}',flush=True)
    return dict(conditions=results,total_supervised_tokens=total,
                main_forward_passes=main_forward_passes,
                contrast_nats_per_token={
                    'cross_position_all_vs_own_only':results['own_only']['final_ce_increase_vs_all'],
                    'future_plans_all_vs_past_and_current':results['past_and_current']['final_ce_increase_vs_all'],
                    'current_plan_all_vs_all_except_current':results['all_except_current']['final_ce_increase_vs_all'],
                    'all_vs_neutral':results['neutral']['final_ce_increase_vs_all']})


def main(argv=None):
    if BACKEND_NAME!='numpy':
        raise RuntimeError('0080K frozen diagnostic requires MINI_LLM_BACKEND=numpy')
    args=parse_args(argv)
    checkpoint=_checkpoint(args)
    state=json.loads((checkpoint/'training_state.json').read_text(encoding='utf8'))
    signature=state['data_provenance']
    root=args.mixed_shard_root.expanduser().resolve()
    tok=root/'tokenizer.json'
    tokenizer,eos=_load_tokenizer(tok)
    a,cfg=_reconstruct(signature,len(tokenizer))
    kwargs=dict(mixed_root=root,eos_token_id=eos,token_dtype='uint16')
    train=PackedRoutedBlockProposalDocuments(**kwargs,split='train')
    val=PackedRoutedBlockProposalDocuments(**kwargs,split='val')
    rng=np.random.default_rng(a.seed+4)
    records=[val.sample_causal(rng,batch_size=a.batch_size,
        memory_length=cfg.memory_context.distant_memory_length,
        target_length=cfg.memory_context.target_length,
        min_history_tokens=(cfg.memory_context.top_k_blocks+1)*cfg.memory_context.block_size-1)
        for _ in signature['validation_digests']]
    if _signature(a,cfg,root,tok,train,val,records)!=signature:
        raise ValueError('validation samples/tokenizer/mixture do not match checkpoint provenance')
    model=DecoderLanguageModel(cfg,rng_seed=a.seed,dtype=a.dtype)
    head=HorizonProposalHead(model,a.block_size,hidden_width=a.proposal_width,rng_seed=a.seed+1)
    decoder=BlockPrefixDecoder(model,a.block_size,width=a.decoder_width,n_layers=a.decoder_layers,
                               n_heads=a.decoder_heads,rng_seed=a.seed+2)
    net=TwoStageBlockModel(head,decoder,lambda_proposal=a.lambda_proposal)
    load_two_stage_checkpoint(checkpoint,net,optimizer=None,
        expected_tokenizer_hash=sha256(tok),expected_data_provenance=signature)
    print(f'0080K frozen checkpoint step {state["step"]}: '
          f'{len(records)} fixed validation batches of {a.batch_size}',flush=True)
    score=evaluate(SinglePassBlockTrainer(net),records,a.block_size,a.block_group_size)
    report=dict(architecture='0080K_frozen_cross_position_plan_ablation',
        checkpoint=str(checkpoint),training_step=int(state['step']),
        tokenizer_sha256=sha256(tok),manifest_sha256=train.manifest_hash,
        validation_digests=signature['validation_digests'],
        **score,notes=(
            'Teacher-forced fixed validation. Conditions mask only compact plan embeddings '
            '(P@E), not original proposal logits, prior, targets or committed prefixes. '
            'Masks are query-horizon-specific; small decoder work is replicated K-fold, '
            'but main Transformer only once per validation sample. Frozen-model plan '
            'masking is out of distribution and gives a sensitivity measurement, '
            'not a fully retrained architectural comparison.'))
    output=(args.output or args.training_output/'ablations_0080k_cross_position.json').expanduser().resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite existing diagnostic: {output}')
    output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(f'Wrote {output}',flush=True)
    return report


if __name__=='__main__':
    main()
