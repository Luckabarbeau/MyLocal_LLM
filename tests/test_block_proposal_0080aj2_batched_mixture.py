"""0080AJ2 batched-head/query GEMM, exact VJP versus pre-existing source_mixture.

The CPU path exercises identical chunking math, with tiny vocabularies and
finite differences. GPU performance tests are in the separate CuPy suite.
"""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.parameter import Parameter
from mini_llm.block_proposal.soft_vocab_read import SoftVocabularyRead

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='NumPy reference')


def _case(k=4, g=3, heads=4, dh=3, tile=5, block_tile=2,
          dtype='float64', seed=502, beta=1.3, temperature=1.1):
    rng = np.random.default_rng(seed)
    width = 7
    vocab = 19
    data = rng.normal(size=(vocab, heads*dh)).astype(dtype)
    x = rng.normal(size=(g, k, width)).astype(dtype)
    logits = rng.normal(size=(g, k, vocab)).astype(dtype)
    dy = rng.normal(size=x.shape).astype(dtype)
    reads = []
    for algorithm in ('source_mixture', 'batched_mixture'):
        read = SoftVocabularyRead(Parameter(data.copy(), name='shared.E'), width,
                                  n_heads=heads, source_scope='all', block_size=k,
                                  execution='tiled', vocab_tile=tile, block_tile=block_tile,
                                  temperature=temperature, read_scale=beta,
                                  all_source_algorithm=algorithm)
        read.output_projection.W.data[:] = .03
        reads.append(read)
    for p, q in zip(reads[0].parameters(), reads[1].parameters()):
        q.data[:] = p.data
    # Nonzero target/source bias is essential to expose axis mistakes.
    bias = rng.normal(scale=.3, size=(heads,k,k)).astype(dtype)
    for obj in reads:
        obj.source_bias.data[:] = bias
    return reads, x, logits, dy


def _run(obj, x, logits, dy):
    obj.zero_grad(); obj.embedding.zero_grad()
    out, cache = obj.forward(x, logits)
    dx, dl = obj.backward(dy, cache)
    return (out, cache.read, cache.attention_norm, cache.source_norm,
            dx, dl, obj.embedding.grad.copy(),
            *[p.grad.copy() for p in obj.parameters()])


@pytest.mark.parametrize('k', [1, 4])
@pytest.mark.parametrize('tile', [1, 5, 19])
@pytest.mark.parametrize('heads', [1, 4])
@pytest.mark.parametrize('beta', [0., 1.3])
def test_batched_matches_original_all_gradients(k, tile, heads, beta):
    (baseline, optimized), x, logits, dy = _case(k=k, heads=heads, tile=tile,
                                                  beta=beta)
    for a,b in zip(_run(baseline,x,logits,dy), _run(optimized,x,logits,dy)):
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=2e-12)


@pytest.mark.parametrize('dtype', ['float32','float64'])
def test_multiple_block_tiles_and_extreme_logits(dtype):
    (baseline, optimized), x, logits, dy = _case(g=5, block_tile=3, dtype=dtype)
    logits *= 40
    logits[:,0] += 1000
    logits[:,3] -= 500
    rtol, atol = (1e-4, 1e-4) if dtype == 'float32' else (1e-11, 1e-11)
    for a,b in zip(_run(baseline,x,logits,dy), _run(optimized,x,logits,dy)):
        assert np.all(np.isfinite(b))
        np.testing.assert_allclose(a,b,rtol=rtol,atol=atol)


def _finite_diff(array, coord, f, eps=1e-5):
    original = array[coord]
    try:
        array[coord] = original+eps
        plus = f()
        array[coord] = original-eps
        minus = f()
    finally:
        array[coord] = original
    return (plus-minus)/(2*eps)


def test_batched_manual_vjp_finite_difference_and_cross_horizon():
    (_, obj), x, logits, dy = _case(g=2, seed=404)
    out, cache = obj.forward(x,logits)
    dx, dl = obj.backward(dy,cache)
    def loss():
        return float(np.sum(obj.forward(x,logits)[0]*dy))
    for index in ((0,0,0),(1,3,9)):
        np.testing.assert_allclose(dl[index],_finite_diff(logits,index,loss),rtol=1e-5,atol=1e-10)
    for index in ((0,2,0),(1,3,5)):
        np.testing.assert_allclose(dx[index],_finite_diff(x,index,loss),rtol=1e-5,atol=1e-10)
    for index in ((0,0),(10,11)):
        np.testing.assert_allclose(obj.embedding.grad[index],_finite_diff(obj.embedding.data,index,loss),rtol=1e-5,atol=1e-10)
    for par,indices in (
        (obj.query_projection.W,((0,0),(6,11))),
        (obj.output_projection.W,((0,0),(11,6))),
        (obj.source_bias,((0,0,3),(3,3,1)))):
        for index in indices:
            np.testing.assert_allclose(par.grad[index],_finite_diff(par.data,index,loss),rtol=1e-5,atol=1e-10)
    zero_other=np.zeros_like(dy);zero_other[:,0]=dy[:,0]
    obj.zero_grad();obj.embedding.zero_grad()
    _, dl_first = obj.backward(zero_other, obj.forward(x,logits)[1])
    assert np.linalg.norm(dl_first[:,3]) > 1e-10


def test_reject_switching_algorithm_after_forward():
    (_,obj),x,logits,dy=_case()
    _,cache=obj.forward(x,logits)
    obj.all_source_algorithm='source_mixture'
    with pytest.raises(ValueError,match='between forward and backward'):
        obj.backward(dy,cache)


def test_batch_cache_is_compact_and_does_not_hold_scores():
    (_,obj),x,logits,_=_case(g=5)
    _,cache=obj.forward(x,logits)
    assert cache.queries.shape == (5,4,4,3)
    assert cache.read.shape == (5,4,12)
    assert cache.attention_norm.shape == (5,4,4)
    assert not hasattr(cache, 'attention_weights')


def test_batched_algorithm_validates_scope_and_execution():
    emb = Parameter(np.ones((5,8),dtype='float64'),name='E')
    with pytest.raises(ValueError,match='all-source'):
        SoftVocabularyRead(emb,6,n_heads=2,source_scope='own',
                           execution='tiled',all_source_algorithm='batched_mixture')
    with pytest.raises(ValueError,match='tiled execution'):
        SoftVocabularyRead(emb,6,n_heads=2,source_scope='all',block_size=4,
                           execution='reference',all_source_algorithm='batched_mixture')
