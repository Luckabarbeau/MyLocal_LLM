"""CuPy BF16 sanity gate for direct-width learned probability decoder."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME,asnumpy,xp
from mini_llm.block_proposal.decoder import BlockPrefixDecoder
from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from test_block_proposal_0080l_baselines import _configuration

pytestmark=pytest.mark.skipif(BACKEND_NAME!='cupy',reason='requires CUDA/CuPy')


def test_direct_width_bf16_backward_causality_and_cache_parity():
    import ml_dtypes
    model=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='bfloat16')
    d=int(model.config.d_model)
    dec=BlockPrefixDecoder(model,4,width=d,n_layers=2,n_heads=4,
        ff_multiplier=2,rng_seed=8082,
        soft_interface_config=SoftInterfaceConfig(
            mode='probability_embedding_direct',probability_embedding_dim=d),
        experimental_gpu_full_vocab=True)
    r=np.random.default_rng(123)
    logits=xp.asarray(r.normal(size=(2,4,101)).astype(np.float32)).astype(ml_dtypes.bfloat16)
    target=xp.asarray(r.integers(0,101,(2,4),dtype=np.int64))
    valid=xp.ones(target.shape,dtype=xp.bool_)
    result,tape=dec.forward(logits,target,valid)
    dec.zero_grad()
    model.embedding.W.zero_grad();model.output_proj.W.zero_grad()
    dlogits=dec.backward(xp.asarray(r.normal(size=result.shape).astype(np.float32)),tape)
    for arr in (dlogits,dec.probability_embedding.grad,
                dec.probability_plan_norm.gamma.grad,model.embedding.W.grad):
        assert bool(asnumpy(xp.all(xp.isfinite(arr))))
    assert bool(asnumpy(xp.any(dec.probability_embedding.grad!=0)))
    state=dec.prepare_plan_cache(logits)
    for j in range(4):
        got=dec.cached_step(state,None if j==0 else target[:,j-1])
        np.testing.assert_allclose(asnumpy(got).astype('float32'),
            asnumpy(result[:,j,:]).astype('float32'),atol=.15,rtol=.04)
