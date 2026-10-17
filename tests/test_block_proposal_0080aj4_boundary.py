"""Independently controlled boundary condition; exact h-gradient and inference."""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
from test_block_proposal_0080l_baselines import _configuration

pytestmark=pytest.mark.skipif(BACKEND_NAME != 'numpy',reason='CPU finite differences')


def make_case(mode,active):
    model=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='float64')
    dec=BlockPrefixDecoder(model,4,width=12,n_layers=2,n_heads=3,rng_seed=8082,
        soft_interface_config=SoftInterfaceConfig(mode=mode,source_scope='all' if mode=='full_vocab' else 'own',
            read_heads=4,boundary_condition='dec_residual'),experimental_cpu_full_vocab=True)
    if active:
        dec.boundary_projection.W.data[:]=np.random.default_rng(91).normal(
            scale=.025,size=dec.boundary_projection.W.shape)
        if dec.vocab_read is not None:
            dec.vocab_read.output_projection.W.data[:]=np.random.default_rng(93).normal(
                scale=.03,size=dec.vocab_read.output_projection.W.shape)
    return model,dec


@pytest.mark.parametrize('mode',['expectation','full_vocab'])
@pytest.mark.parametrize('active',[False,True])
def test_cached_boundary_parity_and_no_future_labels(mode,active):
    model,dec=make_case(mode,active)
    gen=np.random.default_rng(17)
    logits=gen.normal(size=(2,4,101))
    h=gen.normal(size=(2,model.config.d_model))
    labels=gen.integers(0,101,size=(2,4),dtype=np.int64)
    forward,_=dec.forward(logits,labels,np.ones_like(labels,bool),boundary_h=h)
    state=dec.prepare_plan_cache(logits,boundary_h=h)
    for j in range(4):
        actual=dec.cached_step(state,None if j==0 else labels[:,j-1])
        np.testing.assert_allclose(actual,forward[:,j],rtol=5e-12,atol=5e-12)
    if active:
        with pytest.raises(ValueError,match='boundary'):
            dec.prepare_plan_cache(logits)
    changed=labels.copy();changed[:,-1]=(changed[:,-1]+1)%101
    modified,_=dec.forward(logits,changed,np.ones_like(changed,bool),boundary_h=h)
    np.testing.assert_array_equal(modified[:,:-1],forward[:,:-1])


@pytest.mark.parametrize('mode',['expectation','full_vocab'])
def test_exact_boundary_gradient_to_main_h(mode):
    model,dec=make_case(mode,True)
    head=HorizonProposalHead(model,4,hidden_width=12)
    net=TwoStageBlockModel(head,dec)
    gen=np.random.default_rng(217)
    h=gen.normal(size=(2,model.config.d_model))
    labels=gen.integers(0,101,size=(2,4),dtype=np.int64)
    valid=np.ones_like(labels,bool)
    def evaluate():
        return net.forward(h,labels,valid)[0]['total_loss']
    net.zero_grad()
    _,tape=net.forward(h,labels,valid)
    dh=net.backward(tape)
    eps=2e-5
    for idx in ((0,0),(1,7)):
        h[idx]+=eps; plus=evaluate(); h[idx]-=2*eps; minus=evaluate(); h[idx]+=eps
        np.testing.assert_allclose(dh[idx],(plus-minus)/(2*eps),rtol=5e-4,atol=5e-7)
    assert np.max(np.abs(dec.boundary_projection.W.grad))>0
