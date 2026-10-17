"""0080BD: proposal latents are the only small decoder PLAN information path."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.block_proposal.checkpoint import ARCHITECTURE_V5, architecture_metadata
from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from test_block_proposal_0080l_baselines import _configuration

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='CPU reference')


def _net():
    model = DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='float64')
    dec = BlockPrefixDecoder(model,4,width=model.config.d_model,n_layers=2,n_heads=4,
        ff_multiplier=2,rng_seed=8082,
        soft_interface_config=SoftInterfaceConfig(mode='proposal_latents_direct'),
        experimental_cpu_full_vocab=True)
    head = HorizonProposalHead(model,4,hidden_width=12,rng_seed=8081)
    return TwoStageBlockModel(head,dec,lambda_proposal=0.5)


def test_no_probability_embedding_parameters_and_separate_checkpoint():
    net = _net()
    dec = net.decoder
    assert dec.direct_latents and not dec.direct_probability
    assert dec.width == net.proposal_head.model.config.d_model
    assert dec.probability_embedding is None
    assert dec.probability_expectation is None
    assert dec.probability_plan_norm is None
    assert dec.plan_projection is None and dec.token_projection is None and dec.up is None
    assert not any('probability_embedding' in p.name for p in net.parameters())
    metadata = architecture_metadata(net)
    assert metadata['type'] == ARCHITECTURE_V5
    assert 'probability_embedding' not in metadata
    assert metadata['direct_latent_decoder']['plan_scale'] == 0.02
    assert not metadata['direct_latent_decoder']['full_vocabulary_probability_embedding']


def test_missing_or_incorrect_latents_fail_loudly():
    net = _net()
    rng = np.random.default_rng(12)
    logits = rng.normal(size=(2,4,101))
    targets = rng.integers(0,101,(2,4))
    mask = np.ones_like(targets,dtype=bool)
    with pytest.raises(ValueError,match='proposal_latents'):
        net.decoder.forward(logits,targets,mask)
    with pytest.raises(ValueError,match='proposal_latents'):
        net.decoder.prepare_plan_cache(logits)
    with pytest.raises(ValueError,match='proposal_latents'):
        net.decoder.forward(logits,targets,mask,proposal_latents=np.ones((2,4,7)))


def test_teacher_forced_matches_cached_and_is_token_causal():
    net = _net()
    rng = np.random.default_rng(413)
    h = rng.normal(size=(2,16))
    y = rng.integers(0,101,(2,4),dtype=np.int64)
    valid = np.ones_like(y,dtype=bool)
    proposal, latents = net.proposal_head.forward(h,return_cache=False,return_latents=True)
    final,_ = net.decoder.forward(proposal,y,valid,proposal_latents=latents)
    state = net.decoder.prepare_plan_cache(proposal,proposal_latents=latents)
    for j in range(4):
        cached = net.decoder.cached_step(state,None if j==0 else y[:,j-1])
        np.testing.assert_allclose(cached,final[:,j,:],atol=3e-12,rtol=3e-11)
        altered=y.copy()
        altered[:,j:] = (altered[:,j:] + 41)%101
        new,_=net.decoder.forward(proposal,altered,valid,proposal_latents=latents)
        np.testing.assert_allclose(new[:,:j+1],final[:,:j+1],atol=0,rtol=0)


def test_latent_path_gradients_reach_horizon_parameters_and_boundary():
    net = _net()
    rng = np.random.default_rng(625)
    h = rng.normal(size=(2,16))
    y = rng.integers(0,101,(2,4),dtype=np.int64)
    mask = np.ones_like(y,dtype=bool)
    net.zero_grad()
    metrics, cache=net.forward(h,y,mask)
    dh=net.backward(cache)
    assert np.all(np.isfinite(dh))
    assert np.any(net.proposal_head.W_back.grad != 0)
    assert np.any(net.proposal_head.W_context.grad != 0)
    assert np.any(net.proposal_head.horizon_embedding.grad[1:] != 0)
    # Finite-difference of the complete final+proposal objective probes the
    # previously missing direct latent VJP, including decoder PLAN cross-token.
    def loss():
        return net.forward(h,y,mask)[0]['total_loss']
    eps=1e-5
    for i,j in [(0,0),(1,8)]:
        hplus=h.copy(); hminus=h.copy()
        hplus[i,j]+=eps; hminus[i,j]-=eps
        fd=(net.forward(hplus,y,mask)[0]['total_loss']-
            net.forward(hminus,y,mask)[0]['total_loss'])/(2*eps)
        np.testing.assert_allclose(dh[i,j],fd,atol=1e-7,rtol=1e-3)
    for param, idx in [(net.proposal_head.W_back,(0,0)),
                       (net.proposal_head.W_context,(0,0)),
                       (net.proposal_head.horizon_embedding,(1,0))]:
        old=param.data[idx].copy()
        try:
            param.data[idx]=old+eps;a=loss()
            param.data[idx]=old-eps;b=loss()
        finally:
            param.data[idx]=old
        np.testing.assert_allclose(param.grad[idx],(a-b)/(2*eps),atol=1e-7,rtol=3e-3,err_msg=param.name)


def test_original_proposal_logits_do_not_control_plan_anymore():
    net=_net()
    rng=np.random.default_rng(81)
    h=rng.normal(size=(2,16))
    logits,latents=net.proposal_head.forward(h,return_cache=False,return_latents=True)
    y=rng.integers(0,101,(2,4));v=np.ones_like(y,dtype=bool)
    out,_=net.decoder.forward(logits,y,v,proposal_latents=latents)
    # Change prior logits but preserve PLAN state; only additive prior changes.
    shifted=logits.copy(); shifted[:,:,9]+=0.123
    altered,_=net.decoder.forward(shifted,y,v,proposal_latents=latents)
    np.testing.assert_allclose(altered-out,shifted-logits,atol=1e-12,rtol=1e-11)


def test_v5_checkpoint_roundtrip_and_strict_v4_boundary(tmp_path):
    from mini_llm.block_proposal.checkpoint import save_two_stage_checkpoint, load_two_stage_checkpoint
    from mini_llm.optim.adamw import AdamW
    from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
    from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
    net=_net()
    rng=np.random.default_rng(32)
    h=rng.normal(size=(2,16))
    y=rng.integers(0,101,size=(2,4));v=np.ones_like(y,dtype=bool)
    opt=AdamW(net.parameters(),lr=1e-3,weight_decay=0.1)
    net.zero_grad()
    _,tape=net.forward(h,y,v)
    net.backward(tape)
    opt.step()
    save_two_stage_checkpoint(tmp_path/'v5',net,optimizer=opt,training_step=7,tokenizer_hash='same')
    next_net=_net()
    next_opt=AdamW(next_net.parameters(),lr=1e-3,weight_decay=0.1)
    loaded=load_two_stage_checkpoint(tmp_path/'v5',next_net,optimizer=next_opt,expected_tokenizer_hash='same')
    assert loaded['step']==7
    for a,b in zip(net.parameters(),next_net.parameters()):
        np.testing.assert_array_equal(a.data,b.data)
    v4model=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='float64')
    v4decoder=BlockPrefixDecoder(v4model,4,width=16,n_layers=2,n_heads=4,
        ff_multiplier=2,soft_interface_config=SoftInterfaceConfig(
            mode='probability_embedding_direct',probability_embedding_dim=16),
        experimental_cpu_full_vocab=True)
    v4net=TwoStageBlockModel(HorizonProposalHead(v4model,4,hidden_width=12),v4decoder)
    with pytest.raises(ValueError,match='architecture'):
        load_two_stage_checkpoint(tmp_path/'v5',v4net,expected_tokenizer_hash='same')


def test_cli_fresh_v5_and_same_architecture_resume_contract():
    from types import SimpleNamespace
    from train_model import build_training_objective
    model=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='float64')
    args=SimpleNamespace(objective='block-proposal',block_proposal_group_size=16,
        block_proposal_k=4,block_proposal_head_width=12,
        block_proposal_decoder_width=16,block_proposal_decoder_layers=2,
        block_proposal_decoder_heads=4,block_proposal_decoder_ff_multiplier=2,
        block_proposal_soft_interface='proposal_latents_direct',
        block_proposal_lambda=.5)
    obj=build_training_objective(model,args)
    meta=obj.checkpoint_metadata()
    assert meta['type']==ARCHITECTURE_V5
    saved={'architecture':meta,'prediction_objective':'block-proposal'}
    assert build_training_objective(model,args,checkpoint_state=saved).checkpoint_metadata()==meta
