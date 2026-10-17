"""0080AK: teacher-forced and cached DEC see exactly the same causal prefix."""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.block_proposal.decoder import BlockPrefixDecoder
from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel

pytestmark=pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='CPU reference')

from test_block_proposal_0080l_baselines import _configuration


def make_decoder(k, scope, layer, activated):
    model=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='float64')
    decoder=BlockPrefixDecoder(model,k,width=12,n_layers=2,n_heads=3,rng_seed=8082,
        soft_interface_config=SoftInterfaceConfig(mode='full_vocab',source_scope=scope,
                            read_heads=4,read_layer=layer),experimental_cpu_full_vocab=True)
    if activated:
        decoder.vocab_read.output_projection.W.data[:]=np.random.default_rng(43).normal(
            scale=.08,size=decoder.vocab_read.output_projection.W.shape)
        if decoder.vocab_read.source_bias is not None:
            decoder.vocab_read.source_bias.data[:]=np.random.default_rng(22).normal(
                scale=.25,size=decoder.vocab_read.source_bias.data.shape)
    return decoder


@pytest.mark.parametrize('k',[1,4])
@pytest.mark.parametrize('scope',['own','all'])
@pytest.mark.parametrize('layer',[0,1])
@pytest.mark.parametrize('activated',[False,True])
def test_cached_equals_teacher_forced(k,scope,layer,activated):
    decoder=make_decoder(k,scope,layer,activated)
    rng=np.random.default_rng(81)
    logits=rng.normal(size=(2,k,101)).astype('float64')
    targets=rng.integers(0,101,size=(2,k),dtype=np.int64)
    valid=np.ones((2,k),dtype=bool)
    out,_=decoder.forward(logits,targets,valid)
    state=decoder.prepare_plan_cache(logits)
    for j in range(k):
        previous=None if j==0 else targets[:,j-1]
        result=decoder.cached_step(state,previous)
        np.testing.assert_allclose(result,out[:,j,:],atol=3e-12,rtol=3e-12)


def test_future_committed_token_cannot_change_past_cached_logits():
    decoder=make_decoder(4,'all',0,True)
    logits=np.random.default_rng(23).normal(size=(1,4,101))
    for first in (3,81):
        state=decoder.prepare_plan_cache(logits)
        a=decoder.cached_step(state,None)
        b=decoder.cached_step(state,np.array([first]))
        if first==3:
            first_score=a.copy(); second_score=b.copy()
        else:
            np.testing.assert_array_equal(a,first_score)
            assert not np.array_equal(b,second_score)
