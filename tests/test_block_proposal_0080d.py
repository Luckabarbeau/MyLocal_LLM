"""0080D: exactly one original causal main traversal across all blocks."""
import numpy as np
import pytest
from mini_llm.backend import xp,asnumpy
from mini_llm.config import (ModelConfig,MemoryContextConfig,AttentionLayerConfig,
  LocalAttentionConfig,RetrievalAttentionConfig,ContextRouterConfig)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.decoder import BlockPrefixDecoder,TwoStageBlockModel
from mini_llm.block_proposal.training import SinglePassBlockTrainer


def _build(learned=False):
    m=MemoryContextConfig(enabled=True,memory_length=48,recent_length=16,target_length=16,
        block_size=4,top_k_blocks=2,router_query_length=8,router_dim=4,
        query_pooling='mean',history_pooling='mean',integration_mode='routed_prefix',
        memory_training='joint',router_gumbel_noise=False,read_heads=1,read_kv_heads=1)
    layer=None
    if learned:
        router=ContextRouterConfig(history_block_size=2,routing_stride=8,
            query_window=4,router_dim=2,top_k_blocks=2,exclude_recent_tokens=2,
            query_pooling='mean',history_pooling='mean',num_queries=1)
        layer=AttentionLayerConfig(heads=(LocalAttentionConfig(window=8), RetrievalAttentionConfig(router)))
    cfg=ModelConfig(tokenizer_vocab_size=67,context_length=m.active_length,n_layers=2,
        d_model=16,n_q_heads=2,n_kv_heads=1,d_head=8,d_ff=32,
        n_experts=2,top_k=1,memory_context=m,dtype='float64',
        attention_layers=None if layer is None else (layer,layer))
    large=DecoderLanguageModel(cfg,rng_seed=98,dtype='float64')
    head=HorizonProposalHead(large,block_size=4,hidden_width=12,rng_seed=70)
    decoder=BlockPrefixDecoder(large,block_size=4,width=12,n_layers=2,n_heads=3,rng_seed=71)
    return SinglePassBlockTrainer(TwoStageBlockModel(head,decoder,lambda_proposal=0.5))


def _sample(batch=2):
    # Production aligned: source target window contains x0..x15; labels
    # contain x1..x16. Each row is one independent document.
    src=np.mod(np.arange(batch*48).reshape(batch,48)*7+3,67)
    next_ids=np.mod(src[:,32:]+1,67)
    valid=np.ones((batch,16),dtype=bool)
    valid[1,7:]=False
    src[1,32+7:]=0
    next_ids[1,7:]=-100
    meta=dict(history_valid_starts=np.array([0,20]),target_valid_lengths=np.array([16,7]),
        router_stochastic=False,causal_prefix_query=True,use_retrieval=True)
    return xp.asarray(src),xp.asarray(next_ids),xp.asarray(valid),meta


@pytest.mark.parametrize('learned',[False,True])
def test_group_sizes_match_objective_every_gradient_and_optimizer_state(learned):
    model=_build(learned)
    src,labels,valid,meta=_sample()
    def run(group):
        model.zero_grad()
        stats=model.train_window(src,labels,valid_target_mask=valid,
                                memory_metadata=meta,block_group_size=group)
        grads=[(p.name,asnumpy(p.grad).copy()) for p in model.parameters()]
        return stats,grads
    a,ga=run(1)
    b,gb=run(16)
    assert a['main_forward_passes']==b['main_forward_passes']==1
    assert a['main_backward_passes']==b['main_backward_passes']==1
    assert a['unique_supervised_tokens']==b['unique_supervised_tokens']==23
    assert a['active_blocks']==b['active_blocks']==6
    for key in ['proposal_ce','final_ce','total_loss']:
        np.testing.assert_allclose(a[key],b[key],atol=1e-9,rtol=1e-9)
    assert len(ga)==len(gb)
    for (name,x),(name2,y) in zip(ga,gb):
        assert name==name2
        np.testing.assert_allclose(x,y,rtol=2e-6,atol=2e-7,err_msg=name)
    assert np.linalg.norm(asnumpy(model.model.embedding.W.grad))>0
    assert np.linalg.norm(asnumpy(model.model.output_proj.W.grad))>0
    if learned:
        assert model.model.blocks[0].attention._retrieval_groups


def test_document_tail_future_padding_does_not_change_loss_or_grads():
    model=_build(learned=True)
    src,labels,valid,meta=_sample()
    def get(source,targets):
        model.zero_grad()
        metrics=model.train_window(source,targets,valid_target_mask=valid,memory_metadata=meta,
                                   block_group_size=2)
        return metrics,[(p.name,asnumpy(p.grad).copy()) for p in model.parameters()]
    a,ga=get(src,labels)
    other=src.copy();other[1,32+7:]=23
    changed=labels.copy();changed[1,7:]=42
    b,gb=get(other,changed)
    np.testing.assert_allclose(a['total_loss'],b['total_loss'],rtol=0,atol=1e-12)
    for (name,x),(name2,y) in zip(ga,gb):
        assert name==name2
        np.testing.assert_allclose(x,y,rtol=1e-7,atol=1e-9,err_msg=name)


def test_noncausal_router_configuration_is_rejected():
    model=_build()
    src,labels,valid,meta=_sample()
    meta['causal_prefix_query']=False
    with pytest.raises(ValueError,match='noncausal'):
        model.train_window(src,labels,valid_target_mask=valid,memory_metadata=meta)
