"""0080AJ: exact source-mixture reduction against source-loop and dense oracle.

All CPU tests execute the same tiled NumPy/CuPy kernel algebra.  They test
outputs, every manual gradient, and cross-horizon derivatives.  They do not
claim GPU speed or generation/checkpoint parity.
"""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.block_proposal.soft_vocab_read import SoftVocabularyRead
from mini_llm.parameter import Parameter

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='NumPy reference')


def _pair(k=4, v=11, beta=1.2, temperature=1.1, seed=108, dtype='float64'):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(v, 8)).astype(dtype)
    x = rng.normal(size=(2, k, 6)).astype(dtype)
    logits = rng.normal(size=(2, k, v)).astype(dtype)
    gradient = rng.normal(size=x.shape).astype(dtype)
    modules = []
    for algorithm in ('source_loop', 'source_mixture'):
        read = SoftVocabularyRead(Parameter(E.copy(), name='shared.E'), 6,
                                  n_heads=2, source_scope='all', block_size=k,
                                  execution='tiled', vocab_tile=3, block_tile=1,
                                  read_scale=beta, temperature=temperature,
                                  all_source_algorithm=algorithm)
        read.output_projection.W.data[:] = .071
        modules.append(read)
    for p, q in zip(modules[0].parameters(), modules[1].parameters()):
        q.data[:] = p.data
    bias = rng.normal(scale=.2, size=(2,k,k)).astype(dtype)
    for obj in modules:
        obj.source_bias.data[:] = bias
    return modules, x, logits, gradient


def _run(obj, x, logits, gradient):
    obj.zero_grad()
    obj.embedding.zero_grad()
    output, cache = obj.forward(x, logits)
    d_states, d_logits = obj.backward(gradient, cache)
    return (output, cache.read, cache.attention_norm, cache.source_norm,
            d_states, d_logits, obj.embedding.grad.copy(),
            *[p.grad.copy() for p in obj.parameters()])


@pytest.mark.parametrize('k', [1, 4])
@pytest.mark.parametrize('beta', [0., 1.3])
@pytest.mark.parametrize('temperature', [.7, 1.7])
def test_exact_forward_and_every_gradient_matches_source_loop(k, beta, temperature):
    (old, new), x, logits, grad = _pair(k, beta=beta, temperature=temperature)
    reference = _run(old, x, logits, grad)
    collapsed = _run(new, x, logits, grad)
    for a, b in zip(reference, collapsed):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=2e-12)
    # Every source, including r != target j, receives gradient.
    if k > 1 and beta:
        masked = np.zeros_like(grad)
        masked[:, 0] = grad[:, 0]
        _, dlogit = new.backward(masked, new.forward(x, logits)[1])
        assert np.linalg.norm(dlogit[:, -1, :]) > 1.e-12


def _fd(tensor, pos, callback, eps=1e-5):
    value = tensor[pos]
    try:
        tensor[pos] = value + eps
        plus = callback()
        tensor[pos] = value - eps
        minus = callback()
    finally:
        tensor[pos] = value
    return (plus - minus)/(2*eps)


def test_mixture_gradients_finite_difference_all_paths():
    (old, obj), x, logits, grad = _pair(k=4, v=9, seed=55)
    out, cache = obj.forward(x, logits)
    dx, dl = obj.backward(grad, cache)
    def loss():
        return float(np.sum(obj.forward(x, logits)[0] * grad))
    for at in ((0, 0, 0), (1, 3, 8), (0, 2, 4)):
        np.testing.assert_allclose(dl[at], _fd(logits, at, loss), rtol=2e-5, atol=2e-10)
    for at in ((0,0,0), (1,3,5)):
        np.testing.assert_allclose(dx[at], _fd(x, at, loss), rtol=2e-5, atol=2e-10)
    for at in ((0,0), (7,6)):
        np.testing.assert_allclose(obj.embedding.grad[at],
                                   _fd(obj.embedding.data,at,loss),rtol=2e-5,atol=2e-10)
    for par, positions in (
        (obj.query_projection.W, ((0,0), (4,7))),
        (obj.output_projection.W, ((0,0), (6,5))),
        (obj.source_bias, ((0,0,3), (1,2,1)))):
        for at in positions:
            np.testing.assert_allclose(par.grad[at], _fd(par.data, at, loss),
                                       rtol=2e-5, atol=2e-10)
    # A DEC_1-only loss still backpropagates into all proposal sources.
    one = np.zeros_like(grad)
    one[:,0,:] = grad[:,0,:]
    _, isolated = obj.backward(one,cache)
    assert np.linalg.norm(isolated[:, 3, :]) > 1e-12


def test_source_logit_shift_invariance_and_extreme_logits():
    (old, new), x, logits, grad = _pair(seed=63, temperature=.9)
    logits[0, 0] = -1000
    logits[0, 0, 2] = 850
    logits[1, 2] = 400
    logits[1, 2, 7] = -950
    reference = _run(old, x, logits, grad)
    collapsed = _run(new, x, logits, grad)
    for a, b in zip(reference, collapsed):
        assert np.all(np.isfinite(b))
        np.testing.assert_allclose(a, b, rtol=2e-11, atol=3e-11)
    logits[:, 1, :] += 35
    after = _run(new, x, logits, grad)
    # The cached *normalizer* must shift with L; the probabilities, read,
    # downstream derivatives and all other parameter gradients must not.
    np.testing.assert_allclose(after[3][:,1]-collapsed[3][:,1],35/.9,atol=3e-11)
    for i, (a,b) in enumerate(zip(collapsed,after)):
        if i == 3:
            continue
        np.testing.assert_allclose(a,b,rtol=1e-11,atol=3e-11)


def test_mixture_parameter_contract_and_single_source():
    with pytest.raises(ValueError, match='all_source_algorithm'):
        SoftVocabularyRead(Parameter(np.zeros((7,8),dtype='float64'),name='E'),6,
                           source_scope='all',block_size=4,all_source_algorithm='incorrect')
    (old, new), x, logits, grad = _pair(k=1)
    np.testing.assert_allclose(_run(old, x, logits, grad)[0],
                               _run(new, x, logits, grad)[0],atol=1e-13)


def test_switching_algorithm_between_forward_and_backward_is_rejected():
    (old, new), x, logits, grad = _pair()
    _, cache = new.forward(x, logits)
    new.all_source_algorithm = 'source_loop'
    with pytest.raises(ValueError, match='between forward and backward'):
        new.backward(grad, cache)
