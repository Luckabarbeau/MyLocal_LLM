"""RTX 5080 gate: BF16, no probability table, latent VJP and cached parity."""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME, xp, asnumpy
from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from test_block_proposal_0080l_baselines import _configuration

pytestmark=pytest.mark.skipif(BACKEND_NAME!='cupy',reason='requires CuPy/CUDA')


def test_latent_bf16_gradients_and_cached_decoder():
    model=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='bfloat16')
    width=int(model.config.d_model)
    head=HorizonProposalHead(model,4,hidden_width=12,rng_seed=8081)
    dec=BlockPrefixDecoder(model,4,width=width,n_layers=2,n_heads=4,
        ff_multiplier=2,rng_seed=8082,
        soft_interface_config=SoftInterfaceConfig(mode='proposal_latents_direct'),
        experimental_gpu_full_vocab=True)
    net=TwoStageBlockModel(head,dec)
    rng=np.random.default_rng(39)
    h=xp.asarray(rng.normal(size=(2,width)).astype(np.float32)).astype(model.dtype)
    ids=xp.asarray(rng.integers(0,101,(2,4),dtype=np.int64))
    valid=xp.ones((2,4),dtype=xp.bool_)
    logits,latent=head.forward(h,return_cache=False,return_latents=True)
    expected,_=dec.forward(logits,ids,valid,proposal_latents=latent)
    cache=dec.prepare_plan_cache(logits,proposal_latents=latent)
    for j in range(4):
        out=dec.cached_step(cache,None if j==0 else ids[:,j-1])
        np.testing.assert_allclose(asnumpy(out).astype(np.float32),
            asnumpy(expected[:,j,:]).astype(np.float32),atol=0.2,rtol=0.05)
    net.zero_grad()
    metrics,tape=net.forward(h,ids,valid)
    dh=net.backward(tape)
    assert bool(asnumpy(xp.all(xp.isfinite(dh))))
    assert bool(asnumpy(xp.any(head.W_back.grad != 0)))
    assert not any('probability_embedding' in p.name for p in net.parameters())
    assert np.isfinite(float(metrics['total_loss']))
