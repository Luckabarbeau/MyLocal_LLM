"""0080AJ3 RTX 5080 correctness checks for the fused RawKernel path.

Run with MINI_LLM_BACKEND=cupy. These are not executed in CPU CI.
"""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.parameter import Parameter
from mini_llm.block_proposal.soft_vocab_read import SoftVocabularyRead

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CuPy required')


@pytest.mark.parametrize('dtype_name', ['float32', 'bfloat16'])
@pytest.mark.parametrize('k', [1, 4])
@pytest.mark.parametrize('vocab_tile', [7, 32, 257])
def test_fused_mixture_matches_batched_all_vjps(dtype_name,k,vocab_tile):
    import ml_dtypes
    rg=np.random.default_rng(803)
    dt = xp.float32 if dtype_name == 'float32' else ml_dtypes.bfloat16
    g,v,h,dh,decoder_width=3,277,4,3,7
    e=Parameter(xp.asarray(rg.normal(size=(v,h*dh)).astype('float32')).astype(dt),name='E')
    obj=SoftVocabularyRead(e, decoder_width, n_heads=h, block_size=k,
                           source_scope='all', execution='tiled', vocab_tile=vocab_tile,
                           block_tile=2, read_scale=1.2, temperature=1.1,
                           all_source_algorithm='batched_mixture')
    obj.output_projection.W.data[:]=xp.asarray(.05,dtype=dt)
    obj.source_bias.data[:]=xp.asarray(rg.normal(size=(h,k,k)).astype('float32')*.2).astype(dt)
    x=xp.asarray(rg.normal(size=(g,k,decoder_width)).astype('float32'))
    # Preserve a substantial dynamic range to test stable source log-softmax.
    logits=xp.asarray((rg.normal(size=(g,k,v))*4).astype('float32')).astype(dt)
    dy=xp.asarray(rg.normal(size=x.shape).astype('float32'))
    comparisons=[]
    for algorithm in ('batched_mixture','fused_mixture'):
        obj.all_source_algorithm=algorithm
        obj.zero_grad();e.zero_grad()
        y,cache=obj.forward(x,logits)
        dx,dl=obj.backward(dy,cache)
        comparisons.append([xp.asnumpy(z).copy() for z in (y,dx,dl,e.grad,
                            *[p.grad for p in obj.parameters()])])
    rtol,atol=(8e-4,8e-5) if dtype_name=='float32' else (3e-3,3e-4)
    for before,after in zip(*comparisons):
        assert np.isfinite(after).all()
        np.testing.assert_allclose(before,after,rtol=rtol,atol=atol)


def test_full_vocab_fused_backward_is_bounded():
    import ml_dtypes
    rg=np.random.default_rng(803)
    g,k,v,h,dh,dec=8,4,65280,4,16,32
    e=Parameter(xp.asarray(rg.normal(size=(v,h*dh)).astype('float32')).astype(
                ml_dtypes.bfloat16),name='E')
    read=SoftVocabularyRead(e,dec,n_heads=h,block_size=k,source_scope='all',
        execution='tiled',vocab_tile=2048,block_tile=8,
        all_source_algorithm='fused_mixture')
    read.output_projection.W.data[:]=xp.asarray(.025,dtype=ml_dtypes.bfloat16)
    states=xp.asarray(rg.normal(size=(g,k,dec)).astype('float32'))
    logits=xp.asarray(rg.normal(size=(g,k,v)).astype('float32')).astype(ml_dtypes.bfloat16)
    before=xp.get_default_memory_pool().used_bytes()
    y,cache=read.forward(states,logits)
    dx,dl=read.backward(xp.ones_like(y),cache)
    assert y.shape==(g,k,dec) and dl.shape==(g,k,v)
    assert bool(xp.asnumpy(xp.all(xp.isfinite(dx))))
    assert bool(xp.asnumpy(xp.all(xp.isfinite(dl))))
    assert bool(xp.asnumpy(xp.all(xp.isfinite(e.grad))))
    assert xp.get_default_memory_pool().used_bytes()-before < 512*1024*1024
