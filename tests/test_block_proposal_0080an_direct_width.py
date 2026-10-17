"""0080AN: 768-to-768 direct probability decoder, manual gradients, parity, v4."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
from mini_llm.block_proposal.decoder import BlockPrefixDecoder,TwoStageBlockModel
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.checkpoint import (ARCHITECTURE_V4,
    architecture_metadata,save_two_stage_checkpoint,load_two_stage_checkpoint)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from test_block_proposal_0080l_baselines import _configuration

pytestmark=pytest.mark.skipif(BACKEND_NAME!='numpy',reason='CPU reference')


def _make(k=4, layers=2):
    model=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='float64')
    d=int(model.config.d_model)
    decoder=BlockPrefixDecoder(model,k,width=d,n_layers=layers,n_heads=4,
        ff_multiplier=2,rng_seed=8082,
        soft_interface_config=SoftInterfaceConfig(
            mode='probability_embedding_direct', probability_embedding_dim=d),
        experimental_cpu_full_vocab=True)
    return model,decoder


def _samples(k):
    rng=np.random.default_rng(1199)
    return (rng.normal(size=(2,k,101)),rng.integers(0,101,size=(2,k),dtype=np.int64),
            np.ones((2,k),dtype=bool))


def _fd(array,idx,func,eps=1e-5):
    old=array[idx].copy()
    try:
        array[idx]=old+eps;a=func()
        array[idx]=old-eps;b=func()
    finally:
        array[idx]=old
    return (a-b)/(2*eps)


def test_same_width_all_through_no_redundant_projections():
    model,dec=_make()
    assert dec.width==model.config.d_model==16
    assert dec.probability_embedding.shape==(101,16)
    assert dec.probability_adapter is None
    assert dec.plan_projection is None
    assert dec.token_projection is None
    assert dec.up is None
    assert dec.vocab_read is None
    assert dec.probability_plan_norm is not None
    assert len({p.name for p in dec.parameters()})==len(dec.parameters())
    z,tape,adapter=dec._distribution_plan(_samples(4)[0])
    assert adapter is None
    assert z.shape==(2,4,16)
    assert tape.probs.shape==(2,4,101)
    p=tape.probs
    np.testing.assert_allclose(z,p@dec.probability_embedding.data,atol=1e-15,rtol=1e-14)


def test_all_manual_gradients_without_projection_matrices():
    model,dec=_make()
    logits,targets,valid=_samples(4)
    upstream=np.random.default_rng(17).normal(scale=.1,size=logits.shape)
    dec.zero_grad();model.embedding.W.zero_grad();model.output_proj.W.zero_grad()
    output,tape=dec.forward(logits,targets,valid)
    dlogits=dec.backward(upstream,tape)
    scalar=lambda:np.sum(dec.forward(logits,targets,valid)[0]*upstream)
    for idx in ((0,0,0),(1,2,37),(0,3,99)):
        np.testing.assert_allclose(dlogits[idx],_fd(logits,idx,scalar),atol=1e-8,rtol=2e-4)
    for parameter,indices in ((dec.probability_embedding,[(0,0),(37,9),(100,15)]),
                              (dec.layers[0].attn.Wq,[(0,0)]),
                              (dec.probability_plan_norm.gamma,[(0,)]),
                              (model.embedding.W,[(13,3)]),
                              (model.output_proj.W,[(1,3)])):
        for idx in indices:
            np.testing.assert_allclose(parameter.grad[idx],_fd(parameter.data,idx,scalar),
                                       atol=1e-8,rtol=3e-4,err_msg=parameter.name)
    assert np.any(dec.probability_embedding.grad!=0)


@pytest.mark.parametrize('k',[1,4])
@pytest.mark.parametrize('layers',[1,2])
def test_teacher_forcing_matches_cached_and_preserves_causality(k,layers):
    _,dec=_make(k,layers)
    logits,targets,valid=_samples(k)
    reference,_=dec.forward(logits,targets,valid)
    state=dec.prepare_plan_cache(logits)
    for j in range(k):
        logits_cached=dec.cached_step(state,None if j==0 else targets[:,j-1])
        np.testing.assert_allclose(logits_cached,reference[:,j,:],atol=3e-12,rtol=4e-11)
        changed=targets.copy()
        changed[:,j:]=(changed[:,j:]+19)%101
        altered,_=dec.forward(logits,changed,valid)
        np.testing.assert_array_equal(altered[:,:j+1,:],reference[:,:j+1,:])


def test_joint_loss_checkpoint_and_strict_v3_rejection(tmp_path):
    model,dec=_make()
    head=HorizonProposalHead(model,4,hidden_width=12,rng_seed=8081)
    net=TwoStageBlockModel(head,dec,lambda_proposal=.5)
    state=architecture_metadata(net)
    assert state['type']==ARCHITECTURE_V4
    assert state['decoder_width']==state['model_config']['d_model']==16
    assert state['probability_embedding']['adapter']=='none'
    assert not state['direct_decoder']['input_projections']
    opt=AdamW(net.parameters(),lr=.001,weight_decay=.1)
    boundaries=np.random.default_rng(4).normal(size=(2,16))
    _,targets,valid=_samples(4)
    net.zero_grad()
    metrics,tape=net.forward(boundaries,targets,valid)
    assert metrics['total_loss']==metrics['final_ce']+.5*metrics['proposal_ce']
    net.backward(tape);opt.step()
    after,_=net.forward(boundaries,targets,valid)
    save_two_stage_checkpoint(tmp_path/'v4',net,optimizer=opt,training_step=1,tokenizer_hash='same')
    main2,decoder2=_make()
    net2=TwoStageBlockModel(HorizonProposalHead(main2,4,hidden_width=12,rng_seed=8081),decoder2)
    opt2=AdamW(net2.parameters(),lr=.001,weight_decay=.1)
    assert load_two_stage_checkpoint(tmp_path/'v4',net2,optimizer=opt2,
                                     expected_tokenizer_hash='same')['step']==1
    for p,q in zip(net.parameters(),net2.parameters()):
        np.testing.assert_array_equal(p.data,q.data)
    again,_=net2.forward(boundaries,targets,valid)
    assert again['final_ce']==after['final_ce']
    other=BlockPrefixDecoder(main2,4,width=12,n_layers=2,n_heads=3,
                             soft_interface_config=SoftInterfaceConfig(
                                 mode='probability_embedding',probability_embedding_dim=16),
                             experimental_cpu_full_vocab=True)
    with pytest.raises(ValueError,match='architecture'):
        load_two_stage_checkpoint(tmp_path/'v4',TwoStageBlockModel(
            HorizonProposalHead(main2,4,hidden_width=12),other),expected_tokenizer_hash='same')


def test_invalid_width_and_embedding_dimension_rejected():
    model,_=_make()
    with pytest.raises(ValueError,match='direct probability embedding'):
        SoftInterfaceConfig(mode='probability_embedding_direct',probability_embedding_dim=8).validate(
            model_width=16,decoder_layers=2,block_size=4)
    with pytest.raises(ValueError,match='hidden width'):
        BlockPrefixDecoder(model,4,width=12,n_layers=2,n_heads=3,
            soft_interface_config=SoftInterfaceConfig(
                mode='probability_embedding_direct',probability_embedding_dim=16),
            experimental_cpu_full_vocab=True)


def test_cli_builds_fresh_and_resumes_from_v4():
    from types import SimpleNamespace
    from train_model import build_training_objective
    model=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='float64')
    args=SimpleNamespace(objective='block-proposal',block_proposal_group_size=2,
        block_proposal_k=4,block_proposal_decoder_width=16,
        block_proposal_decoder_layers=2,block_proposal_decoder_heads=4,
        block_proposal_decoder_ff_multiplier=2,
        block_proposal_soft_interface='probability_embedding_direct',
        block_proposal_lambda=.5)
    objective=build_training_objective(model,args)
    metadata=objective.checkpoint_metadata()
    assert metadata['type']==ARCHITECTURE_V4
    assert metadata['soft_interface']['probability_embedding_dim']==16
    saved={'architecture':metadata,'prediction_objective':'block-proposal'}
    assert build_training_objective(model,args,checkpoint_state=saved).checkpoint_metadata()==metadata
