"""0080AJ source-mixture parity on a CUDA workstation (not run on CPU CI)."""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.parameter import Parameter
from mini_llm.block_proposal.soft_vocab_read import SoftVocabularyRead

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CUDA CuPy required')


@pytest.mark.parametrize('dtype_name', ['float32', 'bfloat16'])
@pytest.mark.parametrize('group_blocks', [1, 4])
def test_cupy_factor_matches_source_loop_all_gradient_branches(dtype_name, group_blocks):
    import ml_dtypes
    np_rng = np.random.default_rng(5080)
    v, k, width, d = 41, 4, 12, 8
    dtype = xp.float32 if dtype_name == 'float32' else ml_dtypes.bfloat16
    e = Parameter(xp.asarray(np_rng.normal(size=(v, d)).astype('float32')).astype(dtype), name='E')
    read = SoftVocabularyRead(e, width, n_heads=2, source_scope='all',
                              block_size=k, execution='tiled',
                              vocab_tile=11, block_tile=2,
                              temperature=1.3, read_scale=1.1)
    read.output_projection.W.data[:] = xp.asarray(.06, dtype=dtype)
    read.source_bias.data[:] = xp.asarray(
        np_rng.normal(scale=.2, size=(2,k,k)).astype('float32'), dtype=dtype)
    x = xp.asarray(np_rng.normal(size=(group_blocks,k,width)).astype('float32'))
    l = xp.asarray(np_rng.normal(size=(group_blocks,k,v)).astype('float32')).astype(dtype)
    dy = xp.asarray(np_rng.normal(size=(group_blocks,k,width)).astype('float32'))
    results = []
    for algorithm in ['source_loop', 'source_mixture']:
        read.all_source_algorithm=algorithm
        read.zero_grad();e.zero_grad()
        out, cache = read.forward(x,l)
        dx, dl = read.backward(dy,cache)
        results.append((out.copy(),dx.copy(),dl.copy(),e.grad.copy(),
                        *[p.grad.copy() for p in read.parameters()]))
    rtol,atol=(1e-4,2e-5) if dtype_name=='float32' else (2e-3,2e-4)
    for before,after in zip(*results):
        np.testing.assert_allclose(xp.asnumpy(after),xp.asnumpy(before),
                                   rtol=rtol,atol=atol)
    # Explicit cross-horizon gradient from DEC 1 into proposal 4.
    first = xp.zeros_like(dy);first[:,0]=dy[:,0]
    read.zero_grad();e.zero_grad()
    _, d_logit = read.backward(first, read.forward(x,l)[1])
    assert bool(xp.asnumpy(xp.any(xp.abs(d_logit[:,3,:])>1e-8)))


def test_cupy_mixture_full_vocab_bf16_streaming():
    import ml_dtypes
    np_rng=np.random.default_rng(901)
    v,k,g,d=65280,4,4,8
    e=Parameter(xp.asarray(np_rng.normal(size=(v,d)).astype('float32')).astype(
        ml_dtypes.bfloat16),name='E')
    read=SoftVocabularyRead(e,6,n_heads=2,source_scope='all',block_size=k,
                            execution='tiled',vocab_tile=2048,block_tile=2)
    read.output_projection.W.data[:]=xp.asarray(.02,dtype=ml_dtypes.bfloat16)
    x=xp.asarray(np_rng.normal(size=(g,k,6)).astype('float32'))
    l=xp.asarray(np_rng.normal(size=(g,k,v)).astype('float32')).astype(ml_dtypes.bfloat16)
    pool=xp.get_default_memory_pool();baseline=pool.used_bytes()
    y,cache=read.forward(x,l)
    dx,dl=read.backward(xp.ones_like(y),cache)
    assert y.shape==(g,k,6) and dl.shape==(g,k,v)
    assert bool(xp.asnumpy(xp.all(xp.isfinite(dl))))
    assert bool(xp.asnumpy(xp.all(xp.isfinite(e.grad))))
    assert pool.used_bytes()-baseline < 512*1024*1024
