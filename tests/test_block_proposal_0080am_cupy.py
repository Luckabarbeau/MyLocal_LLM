"""RTX 5080 BF16 execution gate for v3 full-probability PLAN embeddings."""
import numpy as np
import pytest

from mini_llm.backend import xp, BACKEND_NAME, asnumpy
from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
from mini_llm.block_proposal.decoder import BlockPrefixDecoder
from mini_llm.model.decoder_lm import DecoderLanguageModel
from test_block_proposal_0080l_baselines import _configuration

pytestmark=pytest.mark.skipif(BACKEND_NAME != 'cupy',reason='requires CUDA/CuPy')


@pytest.mark.parametrize('dim',[64,256])
def test_bf16_prob_embedding_backward_and_teacher_forced_cached_parity(dim):
    import ml_dtypes
    rng=np.random.default_rng(839)
    model=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='bfloat16')
    decoder=BlockPrefixDecoder(model,4,width=12,n_layers=2,n_heads=3,rng_seed=8082,
        soft_interface_config=SoftInterfaceConfig(
            mode='probability_embedding',probability_embedding_dim=dim),
        experimental_gpu_full_vocab=True)
    logits=xp.asarray(rng.normal(size=(2,4,101)).astype('float32')).astype(ml_dtypes.bfloat16)
    targets=xp.asarray(rng.integers(0,101,size=(2,4),dtype=np.int64))
    valid=xp.ones_like(targets,dtype=xp.bool_)
    result,tape=decoder.forward(logits,targets,valid)
    decoder.zero_grad()
    model.embedding.W.zero_grad()
    model.output_proj.W.zero_grad()
    grad=xp.asarray(rng.normal(size=result.shape).astype('float32'))
    dl=decoder.backward(grad,tape)
    assert dl.shape==logits.shape
    for value in (dl,decoder.probability_embedding.grad,decoder.probability_adapter.W.grad):
        assert bool(asnumpy(xp.all(xp.isfinite(value))))
    assert bool(asnumpy(xp.any(decoder.probability_embedding.grad != 0)))
    state=decoder.prepare_plan_cache(logits)
    for j in range(4):
        prev=None if j==0 else targets[:,j-1]
        logit=decoder.cached_step(state,prev)
        np.testing.assert_allclose(asnumpy(logit).astype('float32'),
                                   asnumpy(result[:,j,:]).astype('float32'),
                                   atol=.10,rtol=.03)
