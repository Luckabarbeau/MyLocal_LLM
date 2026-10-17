"""RTX 5080 gate: BF16 teacher forcing vs sequential cached full-soft decoding.

These tests intentionally exercise the actual RawKernel read implementation,
not NumPy or a simplified inference network.
"""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME, xp, asnumpy
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.block_proposal.decoder import BlockPrefixDecoder
from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
from test_block_proposal_0080l_baselines import _configuration

pytestmark=pytest.mark.skipif(BACKEND_NAME!='cupy',reason='CuPy CUDA required')


@pytest.mark.parametrize('k',[1,4])
@pytest.mark.parametrize('boundary',['off','dec_residual'])
@pytest.mark.parametrize('scope',['own','all'])
def test_full_soft_bf16_teacher_forced_cached_parity(k,boundary,scope):
    import ml_dtypes
    rg=np.random.default_rng(870)
    main=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='bfloat16')
    dec=BlockPrefixDecoder(main,k,width=12,n_layers=2,n_heads=3,rng_seed=8082,
        soft_interface_config=SoftInterfaceConfig(mode='full_vocab',source_scope=scope,
                    read_heads=4,read_layer=0,boundary_condition=boundary),
        experimental_gpu_full_vocab=True,
        vocab_read_algorithm='fused_mixture' if scope=='all' else 'source_mixture',
        vocab_read_tile=32,vocab_read_block_tile=2)
    dec.vocab_read.output_projection.W.data[:]=xp.asarray(.03,dtype=ml_dtypes.bfloat16)
    if dec.boundary_projection is not None:
        dec.boundary_projection.W.data[:]=xp.asarray(.02,dtype=ml_dtypes.bfloat16)
    logits=xp.asarray(rg.normal(size=(2,k,101)).astype('float32')).astype(ml_dtypes.bfloat16)
    targets=xp.asarray(rg.integers(0,101,size=(2,k),dtype=np.int64))
    boundary_h=xp.asarray(rg.normal(size=(2,16)).astype('float32')).astype(ml_dtypes.bfloat16)
    forced,_=dec.forward(logits,targets,xp.ones_like(targets,dtype=xp.bool_),
                         boundary_h=boundary_h if boundary!='off' else None)
    cache=dec.prepare_plan_cache(logits,boundary_h=boundary_h if boundary!='off' else None)
    assert cache.next_position==0
    for j in range(k):
        previous=None if j==0 else targets[:,j-1]
        step=dec.cached_step(cache,previous)
        # Outputs involve BF16 projection/reductions; rely on an absolute
        # mixed-precision tolerance, not a claim of bitwise parity.
        np.testing.assert_allclose(asnumpy(step).astype('float32'),
                                  asnumpy(forced[:,j,:]).astype('float32'),
                                  atol=.10,rtol=.03)
    assert cache.next_position==k
