"""0080AM: full-distribution learned PLAN embedding, manual VJP, v3 parity."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.checkpoint import (
    architecture_metadata, ARCHITECTURE, ARCHITECTURE_V2, ARCHITECTURE_V3,
    save_two_stage_checkpoint, load_two_stage_checkpoint, warmstart_v2_from_v1,
)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.block_proposal.training import SinglePassBlockTrainer
from test_block_proposal_0080l_baselines import _configuration, _window

pytestmark=pytest.mark.skipif(BACKEND_NAME != 'numpy',reason='CPU mathematical reference')


def _model(k=4, dim=8, boundary=False):
    model=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='float64')
    decoder=BlockPrefixDecoder(
        model,k,width=12,n_layers=2,n_heads=3,rng_seed=8082,
        soft_interface_config=SoftInterfaceConfig(
            mode='probability_embedding',probability_embedding_dim=dim,
            boundary_condition='dec_residual' if boundary else 'off'),
        experimental_cpu_full_vocab=True)
    return model,decoder


def _inputs(k=4):
    r=np.random.default_rng(512)
    return (r.normal(size=(2,k,101)),
            r.integers(0,101,size=(2,k),dtype=np.int64),
            np.ones((2,k),dtype=bool))


def _fd(arr,index,fn,eps=1e-5):
    old=arr[index].copy()
    try:
        arr[index]=old+eps; a=fn()
        arr[index]=old-eps; b=fn()
    finally:
        arr[index]=old
    return (a-b)/(2*eps)


@pytest.mark.parametrize('dim',[4,8,16,32])
def test_probability_embedding_is_full_distribution_and_dimension_is_configurable(dim):
    model,dec=_model(dim=dim)
    assert dec.probability_embedding.shape==(101,dim)
    assert dec.probability_adapter.W.shape==(dim,16)
    logits,targets,valid=_inputs()
    z,tape,adapter=dec._distribution_plan(logits)
    p=np.exp(logits-logits.max(axis=-1,keepdims=True))
    p/=p.sum(axis=-1,keepdims=True)
    np.testing.assert_allclose(z,(p@dec.probability_embedding.data)@dec.probability_adapter.W.data,
                               atol=2e-15,rtol=2e-14)
    assert tape.probs.shape==logits.shape
    assert np.count_nonzero(tape.probs)==logits.size  # no top-k truncation
    assert adapter is not None
    out,_=dec.forward(logits,targets,valid)
    assert out.shape==logits.shape


def test_correct_manual_gradients_include_probabilities_and_new_parameters():
    model,dec=_model(dim=8)
    logits,targets,valid=_inputs()
    upstream=np.random.default_rng(70).normal(scale=.1,size=logits.shape)
    dec.zero_grad()
    model.embedding.W.zero_grad()
    model.output_proj.W.zero_grad()
    output,tape=dec.forward(logits,targets,valid)
    d_logits=dec.backward(upstream,tape)
    def scalar():
        result,_=dec.forward(logits,targets,valid)
        return float(np.sum(result*upstream))
    for ix in [(0,0,2),(1,2,8),(0,3,100)]:
        np.testing.assert_allclose(d_logits[ix],_fd(logits,ix,scalar),atol=3e-8,rtol=3e-5)
    for par,inds in [
        (dec.probability_embedding,[(0,0),(55,4),(99,7)]),
        (dec.probability_adapter.W,[(0,1),(6,15)]),
        (dec.plan_projection.W,[(0,0),(15,9)]),
    ]:
        for ix in inds:
            numerical=_fd(par.data,ix,scalar)
            np.testing.assert_allclose(par.grad[ix],numerical,atol=5e-8,rtol=5e-4,
                                       err_msg=f'{par.name} {ix}')
    assert np.max(np.abs(dec.probability_embedding.grad))>0
    assert np.max(np.abs(dec.probability_adapter.W.grad))>0
    assert dec.vocab_read is None


@pytest.mark.parametrize('k',[1,4])
@pytest.mark.parametrize('dim',[4,16])
def test_cached_generation_parity_and_future_label_isolation(k,dim):
    _,dec=_model(k,dim)
    logits,targets,valid=_inputs(k)
    full,_=dec.forward(logits,targets,valid)
    state=dec.prepare_plan_cache(logits)
    for j in range(k):
        previous=None if j==0 else targets[:,j-1]
        inferred=dec.cached_step(state,previous)
        np.testing.assert_allclose(inferred,full[:,j,:],atol=3e-12,rtol=3e-11)
        changed=targets.copy()
        changed[:,j:]=(changed[:,j:]+17)%101
        recomputed,_=dec.forward(logits,changed,valid)
        np.testing.assert_array_equal(recomputed[:,:j+1,:],full[:,:j+1,:])


def test_objective_lambda_and_single_main_backward():
    model,dec=_model(k=4,dim=16)
    head=HorizonProposalHead(model,4,hidden_width=12,rng_seed=8081)
    net=TwoStageBlockModel(head,dec,lambda_proposal=.5)
    boundaries=np.random.default_rng(89).normal(size=(2,16))
    _,targets,valid=_inputs()
    metrics,tape=net.forward(boundaries,targets,valid)
    assert np.isclose(metrics['total_loss'],metrics['final_ce']+.5*metrics['proposal_ce'],atol=0,rtol=0)
    net.zero_grad()
    result=net.backward(tape)
    assert result.shape==boundaries.shape
    assert np.max(np.abs(dec.probability_embedding.grad))>0
    assert np.max(np.abs(head.W_context.grad))>0
    trainer=SinglePassBlockTrainer(net)
    x,y,m,metadata=_window()
    report=trainer.train_window(x,y,valid_target_mask=m,memory_metadata=metadata,block_group_size=2)
    assert report['main_forward_passes']==report['main_backward_passes']==1


def test_v1_v2_unchanged_and_v3_strict_roundtrip(tmp_path):
    model,dec=_model(dim=12)
    head=HorizonProposalHead(model,4,hidden_width=12,rng_seed=8081)
    net=TwoStageBlockModel(head,dec)
    expected=architecture_metadata(net)
    assert expected['type']==ARCHITECTURE_V3
    assert expected['probability_embedding']['dimension']==12
    assert expected['soft_interface']['probability_embedding_dim']==12
    assert len({p.name for p in net.parameters()})==len(net.parameters())
    opt=AdamW(net.parameters(),lr=.001,weight_decay=.1)
    boundaries=np.random.default_rng(12).normal(size=(2,16))
    _,targets,valid=_inputs()
    net.zero_grad(); _,tape=net.forward(boundaries,targets,valid);net.backward(tape);opt.step()
    f,_=net.forward(boundaries,targets,valid)
    save_two_stage_checkpoint(tmp_path/'v3',net,optimizer=opt,training_step=1,tokenizer_hash='same')
    new_main,new_decoder=_model(dim=12)
    new_net=TwoStageBlockModel(HorizonProposalHead(new_main,4,hidden_width=12,rng_seed=8081),new_decoder)
    new_opt=AdamW(new_net.parameters(),lr=.001,weight_decay=.1)
    restored=load_two_stage_checkpoint(tmp_path/'v3',new_net,optimizer=new_opt,expected_tokenizer_hash='same')
    assert restored['step']==1
    for a,b in zip(net.parameters(),new_net.parameters()):
        np.testing.assert_array_equal(a.data,b.data)
    assert new_opt.step_index==1
    g,_=new_net.forward(boundaries,targets,valid)
    assert g['final_ce']==f['final_ce']
    other_main,other_dec=_model(dim=8)
    other=TwoStageBlockModel(HorizonProposalHead(other_main,4,hidden_width=12),other_dec)
    with pytest.raises(ValueError,match='architecture'):
        load_two_stage_checkpoint(tmp_path/'v3',other,expected_tokenizer_hash='same')
    # The original v1/v2 metadata must not gain the new field.
    old=BlockPrefixDecoder(model,4,width=12,n_layers=2,n_heads=3,rng_seed=8082)
    v1=TwoStageBlockModel(head,old)
    assert architecture_metadata(v1)['type']==ARCHITECTURE
    assert 'soft_interface' not in architecture_metadata(v1)
    v2_dec=BlockPrefixDecoder(model,4,width=12,n_layers=2,n_heads=3,rng_seed=8082,
        soft_interface_config=SoftInterfaceConfig(mode='full_vocab'),experimental_cpu_full_vocab=True)
    v2=TwoStageBlockModel(head,v2_dec)
    assert architecture_metadata(v2)['type']==ARCHITECTURE_V2
    assert 'probability_embedding_dim' not in architecture_metadata(v2)['soft_interface']


def test_v1_warmstart_transfers_legacy_tensors_only(tmp_path):
    model,dec=_model(dim=8)
    head=HorizonProposalHead(model,4,hidden_width=12,rng_seed=8081)
    v3=TwoStageBlockModel(head,dec)
    base_main=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='float64')
    base=TwoStageBlockModel(HorizonProposalHead(base_main,4,hidden_width=12,rng_seed=8081),
        BlockPrefixDecoder(base_main,4,width=12,n_layers=2,n_heads=3,rng_seed=8082))
    save_two_stage_checkpoint(tmp_path/'v1',base,tokenizer_hash='same')
    result=warmstart_v2_from_v1(tmp_path/'v1',v3,expected_tokenizer_hash='same')
    assert result['initialized_v2']==sorted([
        'block_decoder.probability_embedding','block_decoder.probability_adapter.W'])
    old={p.name:p.data for p in base.parameters()}
    for par in v3.parameters():
        if par.name in old:
            np.testing.assert_array_equal(par.data,old[par.name])


def test_invalid_configuration_rejected_before_parameter_allocation():
    for dim in (0,-1,4097,8.5):
        with pytest.raises(ValueError,match='probability embedding'):
            SoftInterfaceConfig(mode='probability_embedding',probability_embedding_dim=dim).validate(
                model_width=16,decoder_layers=2,block_size=4)
    with pytest.raises(ValueError,match='probability embedding'):
        SoftInterfaceConfig(mode='full_vocab',probability_embedding_dim=16).validate(
            model_width=16,decoder_layers=2,block_size=4)


def test_cli_builder_v3_and_strict_dimension_resume():
    from types import SimpleNamespace
    from train_model import build_training_objective
    model=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='float64')
    args=SimpleNamespace(objective='block-proposal',block_proposal_group_size=2,
        block_proposal_k=4,block_proposal_decoder_width=12,
        block_proposal_decoder_layers=2,block_proposal_decoder_heads=3,
        block_proposal_soft_interface='probability_embedding',
        block_proposal_prob_embedding_dim=16,
        block_proposal_lambda=.5)
    obj=build_training_objective(model,args)
    metadata=obj.checkpoint_metadata()
    assert metadata['type']==ARCHITECTURE_V3
    assert metadata['soft_interface']['probability_embedding_dim']==16
    assert 'vocab_read_execution' not in metadata
    saved={'architecture':metadata,'prediction_objective':'block-proposal'}
    assert build_training_objective(model,args,checkpoint_state=saved).checkpoint_metadata()==metadata
    args.block_proposal_prob_embedding_dim=32
    with pytest.raises(ValueError,match='resume mismatch'):
        build_training_objective(model,args,checkpoint_state=saved)


def test_weighted_loss_head_and_probability_gradients_match_finite_difference():
    model,dec=_model(dim=8)
    head=HorizonProposalHead(model,4,hidden_width=12,rng_seed=8081)
    net=TwoStageBlockModel(head,dec,lambda_proposal=.5)
    boundaries=np.random.default_rng(101).normal(size=(2,16))
    _,targets,valid=_inputs()
    net.zero_grad()
    metrics,tape=net.forward(boundaries,targets,valid)
    net.backward(tape)
    def objective():
        m,_=net.forward(boundaries,targets,valid)
        return float(m['total_loss'])
    for par,ix in ((dec.probability_embedding,(5,0)),
                   (dec.probability_adapter.W,(0,0)),
                   (head.W_back,(1,2))):
        numerical=_fd(par.data,ix,objective)
        np.testing.assert_allclose(par.grad[ix],numerical,atol=6e-8,rtol=6e-4,
            err_msg=f'combined final CE + 0.5 proposal CE gradient for {par.name}')
