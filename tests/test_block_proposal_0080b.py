"""0080B: exact next-token recovery and handwritten gradient checks."""
import numpy as np
import pytest
from mini_llm.backend import asnumpy, xp
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.config import ModelConfig
from mini_llm.block_proposal.proposals import HorizonProposalHead


def _fixture(k=4):
    config = ModelConfig(tokenizer_vocab_size=17,context_length=16,
                         n_layers=1,d_model=8,n_q_heads=2,n_kv_heads=1,
                         d_head=4,d_ff=16,n_experts=2,top_k=1,dtype='float64')
    model = DecoderLanguageModel(config,rng_seed=11,dtype='float64')
    head = HorizonProposalHead(model,k,hidden_width=7,rng_seed=9)
    source = xp.asarray([[3,4,5,6,7,8]])
    hidden = model.forward_body(source,return_cache=False)
    return model,head,hidden


@pytest.mark.parametrize('k',[1,4,8,16])
def test_first_horizon_exact_conventional_lm_and_parameter_sharing(k):
    model, head, hidden = _fixture(k)
    h = hidden[:,2,:].copy()
    proposal = head.forward(h,return_cache=False)
    conventional = model.forward(xp.asarray([[3,4,5,6,7,8]]),return_cache=False)
    np.testing.assert_allclose(asnumpy(proposal[:,0,:]),asnumpy(conventional[:,2,:]),rtol=1e-14,atol=1e-14)
    assert head.model.output_proj.W is model.output_proj.W
    assert len(head.parameters()) == 3
    assert proposal.shape == (1,k,17)


def test_horizon_head_is_nonlinearly_context_dependent():
    model,head,hidden = _fixture(4)
    one = head.forward(hidden[:,1,:],return_cache=False)
    two = head.forward(hidden[:,2,:],return_cache=False)
    # Context-dependent differences; not a fixed added vocabulary bias.
    delta_one = asnumpy(one[:,1,:] - one[:,0,:])
    delta_two = asnumpy(two[:,1,:] - two[:,0,:])
    assert not np.allclose(delta_one,delta_two,atol=1e-12)


def test_proposal_all_horizons_loss_gradient_matches_finite_difference():
    model, head, hidden = _fixture(4)
    states = hidden[:,1,:].copy()
    targets = xp.asarray([[2,3,4,-100]])
    mask = xp.asarray([[1,1,1,0]],dtype=xp.bool_)
    logits,cache = head.forward(states)
    loss,ce_cache = head.supervised_loss(logits,targets,mask)
    d_logits = head.supervised_loss_backward(ce_cache)
    model.zero_grad(); head.zero_grad()
    d_h = head.backward(d_logits,cache)
    def loss_at():
        l = head.forward(states,return_cache=False)
        return head.supervised_loss(l,targets,mask)[0]
    eps=1e-5
    for param,idx in [
        (head.W_context,(2,3)),(head.W_back,(4,1)),
        (head.horizon_embedding,(2,3)),(model.output_proj.W,(2,5)),
    ]:
        prior = float(param.data[idx]); param.data[idx] = prior+eps
        plus = loss_at(); param.data[idx]=prior-eps
        minus=loss_at();param.data[idx]=prior
        numerical = (plus-minus)/(2*eps)
        np.testing.assert_allclose(asnumpy(param.grad)[idx],numerical,rtol=1e-5,atol=2e-8,err_msg=param.name)
    for i in (0,3,7):
        s = states.copy();s[0,i]+=eps
        plus = head.supervised_loss(head.forward(s,return_cache=False),targets,mask)[0]
        s[0,i]-=2*eps
        minus = head.supervised_loss(head.forward(s,return_cache=False),targets,mask)[0]
        np.testing.assert_allclose(asnumpy(d_h)[0,i],(plus-minus)/(2*eps),rtol=1e-5,atol=1e-8)
    assert np.isfinite(loss)
    assert np.linalg.norm(asnumpy(head.W_back.grad)) > 0
    assert np.linalg.norm(asnumpy(head.horizon_embedding.grad)[1:])>0
    np.testing.assert_array_equal(asnumpy(head.horizon_embedding.grad)[0], np.zeros(7))


def test_masked_tail_is_invariant_to_padding_sentinel():
    model,head,hidden = _fixture(4)
    logits = head.forward(hidden[:,0,:],return_cache=False)
    mask = xp.asarray([[1,1,0,0]],dtype=xp.bool_)
    a = head.supervised_loss(logits,xp.asarray([[2,3,-100,-100]]),mask)[0]
    b = head.supervised_loss(logits,xp.asarray([[2,3,16,7]]),mask)[0]
    np.testing.assert_allclose(a,b,atol=0,rtol=0)
