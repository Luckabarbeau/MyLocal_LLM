"""GPU-only 0080AJ2 numerical and full-vocab memory-bound tests (RTX 5080)."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.parameter import Parameter
from mini_llm.block_proposal.soft_vocab_read import SoftVocabularyRead

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CuPy CUDA required')


@pytest.mark.parametrize('dtype_name', ['float32', 'bfloat16'])
@pytest.mark.parametrize('blocks', [1, 4, 8])
def test_cupy_batched_and_original_mixture_match_all_gradient_paths(dtype_name, blocks):
    import ml_dtypes
    dtype = xp.float32 if dtype_name == 'float32' else ml_dtypes.bfloat16
    rg = np.random.default_rng(2525)
    v, k, h, d, dec = 37, 4, 4, 3, 7
    e = Parameter(xp.asarray(rg.normal(size=(v,h*d)).astype('float32')).astype(dtype),name='E')
    read = SoftVocabularyRead(e,dec,n_heads=h,source_scope='all',block_size=k,
                              execution='tiled',vocab_tile=13,block_tile=3,
                              temperature=1.2,read_scale=.8)
    read.output_projection.W.data[:]=xp.asarray(.07,dtype=dtype)
    read.source_bias.data[:]=xp.asarray(rg.normal(scale=.13,size=(h,k,k)).astype('float32')).astype(dtype)
    x=xp.asarray(rg.normal(size=(blocks,k,dec)).astype('float32'))
    l=xp.asarray(rg.normal(size=(blocks,k,v)).astype('float32')).astype(dtype)
    dy=xp.asarray(rg.normal(size=x.shape).astype('float32'))
    res=[]
    for alg in ('source_mixture','batched_mixture'):
        read.all_source_algorithm=alg
        read.zero_grad();e.zero_grad()
        out,cache=read.forward(x,l)
        dx,dl=read.backward(dy,cache)
        res.append((out.copy(),dx.copy(),dl.copy(),e.grad.copy(),
                    *[p.grad.copy() for p in read.parameters()]))
    rtol,atol=(3e-4,5e-5) if dtype_name=='float32' else (3e-3,3e-4)
    for before,after in zip(*res):
        np.testing.assert_allclose(xp.asnumpy(after),xp.asnumpy(before),
                                   rtol=rtol,atol=atol)


def test_full_vocab_bf16_memory_bounded_and_gradient_finite():
    import ml_dtypes
    rg=np.random.default_rng(81)
    v,k,g,heads,dh=65280,4,8,4,16
    e=Parameter(xp.asarray(rg.normal(size=(v,heads*dh)).astype('float32')).astype(
        ml_dtypes.bfloat16),name='E')
    read=SoftVocabularyRead(e,32,n_heads=heads,source_scope='all',block_size=k,
                            execution='tiled',vocab_tile=2048,block_tile=8,
                            all_source_algorithm='batched_mixture')
    read.output_projection.W.data[:]=xp.asarray(.025,dtype=ml_dtypes.bfloat16)
    x=xp.asarray(rg.normal(size=(g,k,32)).astype('float32'))
    logits=xp.asarray(rg.normal(size=(g,k,v)).astype('float32')).astype(ml_dtypes.bfloat16)
    pool=xp.get_default_memory_pool()
    before=pool.used_bytes()
    output,cache=read.forward(x,logits)
    dx,dl=read.backward(xp.ones_like(output),cache)
    assert output.shape==(g,k,32) and dl.shape==(g,k,v)
    assert bool(xp.asnumpy(xp.all(xp.isfinite(dl))))
    assert bool(xp.asnumpy(xp.all(xp.isfinite(dx))))
    assert bool(xp.asnumpy(xp.all(xp.isfinite(e.grad))))
    # This measures retained CuPy pool allocations, not true peak device VRAM.
    assert pool.used_bytes()-before < 512*1024*1024
