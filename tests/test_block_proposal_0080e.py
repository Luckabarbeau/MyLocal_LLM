"""0080E first half: K-position parallel vs real decoder KV-cache parity."""
import numpy as np
import pytest
from mini_llm.backend import xp,asnumpy
from tests.test_block_proposal_0080c import _joint

@pytest.mark.parametrize('k_valid',[1,2,4])
def test_full_prefix_decoder_matches_one_plan_prefill_and_k_cached_steps(k_valid):
    model,h,targets,valid=_joint()
    lp=model.proposal_head.forward(h,return_cache=False)
    full,_=model.decoder.forward(lp,targets,valid)
    state=model.decoder.prepare_plan_cache(lp)
    results=[]
    for j in range(4):
        prev=None if j==0 else targets[:,j-1]
        row=model.decoder.cached_step(state,prev)
        results.append(row)
    actual=xp.stack(results,axis=1)
    np.testing.assert_allclose(asnumpy(actual[:,:k_valid]),asnumpy(full[:,:k_valid]),rtol=1e-11,atol=1e-12)
    assert state.next_position==4
    assert all(k.shape[1]==8 for k in state.keys)
    with pytest.raises(ValueError,match='all block positions'):
        model.decoder.cached_step(state,targets[:,-1])


def test_cache_first_step_has_no_ground_truth_and_plan_prefix_remains_static():
    model,h,targets,valid=_joint()
    lp=model.proposal_head.forward(h,return_cache=False)
    state=model.decoder.prepare_plan_cache(lp)
    original=[asnumpy(k).copy() for k in state.keys]
    first=model.decoder.cached_step(state)
    for k_before,k_after in zip(original,state.keys):
        np.testing.assert_array_equal(k_before,asnumpy(k_after)[:,:4])
    # Last label does not affect first or second DEC decision.
    corrupted=targets.copy();corrupted[:,-1]=1
    alt=model.decoder.prepare_plan_cache(lp)
    np.testing.assert_allclose(asnumpy(first),asnumpy(model.decoder.cached_step(alt)),atol=1e-12,rtol=1e-12)


def test_main_dense_cached_block_append_matches_sequential_one_token_updates():
    from mini_llm.inference_model import InferenceModel
    model,h,targets,valid=_joint()
    large=model.proposal_head.model
    a=InferenceModel(large.config,dtype='float64')
    b=InferenceModel(large.config,dtype='float64')
    a.set_weights(large); b.set_weights(large)
    state_a=a.create_generation_state(1,16)
    state_b=b.create_generation_state(1,16)
    prompt=xp.asarray([[1,2,3,4]])
    a.prefill(prompt,state_a)
    b.prefill(prompt,state_b)
    committed=xp.asarray([[5,6,7,8]])
    a.append_causal_block(committed,state_a)
    for j in range(4):
        b.decode_one(committed[:,j],state_b)
    assert state_a.length==state_b.length==8
    np.testing.assert_allclose(asnumpy(state_a.k_cache[:,:,:,:8,:]),asnumpy(state_b.k_cache[:,:,:,:8,:]),rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(asnumpy(state_a.v_cache[:,:,:,:8,:]),asnumpy(state_b.v_cache[:,:,:,:8,:]),rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(asnumpy(a.last_hidden),asnumpy(b.last_hidden),rtol=1e-10,atol=1e-10)


def test_direct_generator_accepts_partial_block_and_continues_after_commit():
    from mini_llm.inference_model import InferenceModel
    from mini_llm.block_proposal.generation import DirectCachedBlockGenerator
    model,h,targets,valid=_joint()
    large=model.proposal_head.model
    inf=InferenceModel(large.config,dtype='float64');inf.set_weights(large)
    state=inf.create_generation_state(1,16)
    inf.prefill(xp.asarray([[1,2,3,4]]),state)
    gen=DirectCachedBlockGenerator(inf,model.proposal_head,model.decoder)
    out=gen.generate(state,max_new_tokens=6)
    assert len(out['tokens'])==6
    assert out['main_updates']==2
    assert state.length==10
    assert 0<=min(out['tokens']) and max(out['tokens'])<19
