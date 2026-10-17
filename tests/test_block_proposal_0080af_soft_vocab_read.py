"""0080AF: full-vocabulary, own-source CPU reference and manual-VJP audits.

This exercises an isolated module, NOT the v1 decoder training path. CUDA and
all-source cross-horizon access are gated for their later dedicated patches.
"""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.block_proposal.soft_vocab_read import SoftVocabularyRead
from mini_llm.parameter import Parameter

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='CPU reference')


def _make(*, vocab=7, width=4, dec_width=3, n_heads=2, beta=1.3,
          temperature=1.7, dtype='float64', seed=80):
    rng = np.random.default_rng(seed)
    e = Parameter(rng.normal(size=(vocab, width)).astype(dtype), name='main.embedding.W')
    read = SoftVocabularyRead(e, dec_width, n_heads=n_heads, read_scale=beta,
                              temperature=temperature, rng_seed=88)
    decoder = rng.normal(size=(2, 2, dec_width)).astype(dtype)
    logits = rng.normal(size=(2, 2, vocab)).astype(dtype)
    grad = rng.normal(size=decoder.shape).astype(dtype)
    return e, read, decoder, logits, grad


def _loss(read, decoder, logits, upstream):
    out, _ = read.forward(decoder, logits)
    return float(np.sum(out * upstream))


def _finite_difference(arr, fun, eps=1e-6):
    numeric = np.zeros_like(arr)
    for index in np.ndindex(arr.shape):
        original = arr[index]
        try:
            arr[index] = original + eps
            plus = fun()
            arr[index] = original - eps
            minus = fun()
        finally:
            arr[index] = original
        numeric[index] = (plus - minus) / (2 * eps)
    return numeric


def _activate_output(read):
    # The actual integration will start at zero output. After an update the
    # weights are nonzero, allowing exact VJP checks for every upstream path.
    rng = np.random.default_rng(509)
    read.output_projection.W.data[:] = rng.normal(
        scale=0.27, size=read.output_projection.W.shape)


def test_module_isolated_and_zero_residual_initialization():
    embed, read, dec, logits, g = _make()
    params = read.parameters()
    assert len(params) == 2
    assert len({id(p) for p in params}) == 2
    assert len({p.name for p in params}) == 2
    assert all(p is not embed for p in params)
    assert all(p.name.startswith('block_decoder.soft_vocab_read.') for p in params)
    np.testing.assert_array_equal(read.output_projection.W.data, 0)
    correction, cache = read.forward(dec, logits)
    np.testing.assert_array_equal(correction, 0)
    d_dec, d_logits = read.backward(g, cache)
    # A zero output projection blocks upstream gradients on the first update;
    # it must not block the output projection's own nonzero gradient.
    assert np.any(read.output_projection.W.grad)
    np.testing.assert_array_equal(read.query_projection.W.grad, 0)
    np.testing.assert_array_equal(embed.grad, 0)
    np.testing.assert_array_equal(d_dec, 0)
    np.testing.assert_array_equal(d_logits, 0)


def test_beta_zero_gives_full_softmax_expectation_head_by_head():
    for dtype in ('float32', 'float64'):
        e, read, dec, logits, _ = _make(beta=0.0, dtype=dtype)
        _, cache = read.forward(dec, logits)
        scaled = logits / read.temperature
        weights = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        reference = weights @ e.data
        np.testing.assert_allclose(cache.read, reference, rtol=5e-6 if dtype=='float32' else 1e-14,
                                   atol=1e-6 if dtype=='float32' else 1e-14)
        _activate_output(read)
        read.zero_grad()
        e.zero_grad()
        d_dec, d_logits = read.backward(np.ones_like(dec), read.forward(dec, logits)[1])
        # With beta=0, proposals still receive softmax/value gradients, but
        # query/decoder gradients vanish exactly.
        assert np.any(d_logits)
        np.testing.assert_array_equal(d_dec, 0)
        np.testing.assert_array_equal(read.query_projection.W.grad, 0)


def test_queries_resolve_different_modes_of_same_proposal_marginal():
    # All queries have *exactly* the same prior P, hence the same P@E.
    e = Parameter(np.asarray([[2., 0.], [-2., 0.], [0., 1.], [0., -1.]]), name='embed')
    read = SoftVocabularyRead(e, 2, n_heads=1, temperature=1., read_scale=3.)
    read.query_projection.W.data[:] = np.eye(2)
    logits = np.zeros((1, 2, 4))
    dec = np.asarray([[[1., 0.], [-1., 0.]]])
    _, cache = read.forward(dec, logits)
    np.testing.assert_allclose(np.mean(e.data, axis=0), [0., 0.])
    assert cache.read[0, 0, 0] > 1.5
    assert cache.read[0, 1, 0] < -1.5
    # The negative and positive modes are selected from all 4 options, not top-k.


def test_manual_backward_all_parameters_inputs_and_shared_key_values_fd():
    e, read, dec, logits, g = _make(vocab=5, width=4, dec_width=3, beta=1.5)
    _activate_output(read)
    correction, cache = read.forward(dec, logits)
    assert correction.shape == dec.shape
    d_dec, d_logits = read.backward(g, cache)
    numeric_dec = _finite_difference(dec, lambda: _loss(read, dec, logits, g))
    numeric_logits = _finite_difference(logits, lambda: _loss(read, dec, logits, g))
    numeric_emb = _finite_difference(e.data, lambda: _loss(read, dec, logits, g))
    numeric_query = _finite_difference(read.query_projection.W.data,
                                      lambda: _loss(read, dec, logits, g))
    numeric_out = _finite_difference(read.output_projection.W.data,
                                    lambda: _loss(read, dec, logits, g))
    for name, analytic, numeric in (
        ('DEC query input', d_dec, numeric_dec),
        ('proposal logits', d_logits, numeric_logits),
        ('shared vocabulary E key+value', e.grad, numeric_emb),
        ('W_query', read.query_projection.W.grad, numeric_query),
        ('W_read_out', read.output_projection.W.grad, numeric_out),
    ):
        np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=2e-9,
                                   err_msg=name)
    assert np.any(e.grad)
    assert np.any(read.query_projection.W.grad)
    assert np.any(read.output_projection.W.grad)


def test_embedding_key_and_value_terms_have_distinct_gradients():
    e, read, dec, logits, g = _make()
    _activate_output(read)
    _, cache = read.forward(dec, logits)
    read.backward(g, cache)
    total = e.grad.copy()
    # At beta=0 there can be no key gradient. With beta!=0 the key term
    # ordinarily changes the result; both configurations are FD audited.
    e0, r0, d0, l0, _ = _make(beta=0.0)
    r0.output_projection.W.data[:] = read.output_projection.W.data
    r0.query_projection.W.data[:] = read.query_projection.W.data
    r0.backward(g, r0.forward(d0, l0)[1])
    assert not np.allclose(total, e0.grad)
    fd = _finite_difference(e0.data, lambda: _loss(r0, d0, l0, g))
    np.testing.assert_allclose(e0.grad, fd, atol=2e-9, rtol=1e-7)


def test_own_source_horizon_isolation_and_query_dependency():
    e, read, dec, logits, _ = _make()
    _activate_output(read)
    _, baseline = read.forward(dec, logits)
    altered_l = logits.copy()
    altered_l[:, 1] += np.linspace(-4, 4, logits.shape[-1])[None, :]
    _, changed_l = read.forward(dec, altered_l)
    np.testing.assert_array_equal(changed_l.read[:, 0], baseline.read[:, 0])
    altered_dec = dec.copy()
    altered_dec[:, 1] *= -9.
    _, changed_dec = read.forward(altered_dec, logits)
    np.testing.assert_array_equal(changed_dec.read[:, 0], baseline.read[:, 0])
    assert not np.array_equal(changed_dec.read[:, 1], baseline.read[:, 1])
    # The module consumes no labels and cannot inspect future true tokens.


def test_logit_shift_invariance_and_no_cross_horizon_logit_gradients():
    e, read, dec, logits, g = _make()
    _activate_output(read)
    out, cache = read.forward(dec, logits)
    # Arbitrary independent row offsets must cancel during softmax.
    shift = np.asarray([[[120.], [-230.]], [[8.], [-95.]]])
    out2, _ = read.forward(dec, logits + shift)
    np.testing.assert_allclose(out, out2, rtol=0., atol=1e-14)
    only_first = np.zeros_like(g)
    only_first[:, 0] = g[:, 0]
    _, d_logits = read.backward(only_first, cache)
    np.testing.assert_array_equal(d_logits[:, 1], 0)
    # Softmax's Jacobian has zero row sum in logit space.
    np.testing.assert_allclose(d_logits.sum(axis=-1), 0., atol=5e-16)


def test_extreme_logits_duplicate_embeddings_and_tiny_probabilities():
    e, read, dec, logits, g = _make(vocab=7, dtype='float64')
    _activate_output(read)
    e.data[1] = e.data[0]  # degenerate keys and values must be well defined
    for extreme in (np.full_like(logits, 1e4),
                    np.tile(np.array([1500., -1500., -1200., 0., -80., 3., -2.]), (2, 2, 1)),
                    np.tile(np.array([-1e3, 1e3, 50., -30., 600., 20., 0.]), (2, 2, 1))):
        out, cache = read.forward(dec, extreme)
        read.zero_grad(); e.zero_grad()
        d_dec, d_logits = read.backward(g, cache)
        for arr in (out, cache.read, d_dec, d_logits, e.grad):
            assert np.all(np.isfinite(arr))
        assert np.all(np.isfinite(cache.queries))
        for m in range(read.n_heads):
            s = slice(m * read.head_width, (m+1) * read.head_width)
            weights, _ = read._one_head(cache.queries[:, :, m], extreme, e.data[:, s])
            np.testing.assert_allclose(weights.sum(axis=-1), 1., atol=2e-15)
            assert np.all(weights >= 0)


def test_two_independent_forward_tapes_accumulate_gradients():
    e, read, dec, logits, g = _make()
    _activate_output(read)
    _, cache1 = read.forward(dec, logits)
    _, cache2 = read.forward(dec * 0.3, logits * -0.7)
    d2, l2 = read.backward(g * 1.3, cache2)
    d1, l1 = read.backward(g, cache1)
    final_grads = [p.grad.copy() for p in read.parameters()] + [e.grad.copy()]
    assert np.any(d1) and np.any(d2) and np.any(l1) and np.any(l2)
    read.zero_grad(); e.zero_grad()
    read.backward(g, read.forward(dec, logits)[1])
    read.backward(g * 1.3, read.forward(dec * .3, logits * -.7)[1])
    for p, expected in zip(read.parameters() + [e], final_grads):
        np.testing.assert_allclose(p.grad, expected, rtol=1e-13, atol=1e-14)
    # The small V input bank is copied on forward, preventing a later mutation
    # of caller logits from corrupting the current forward tape.
    _, cache = read.forward(dec, logits)
    frozen = cache.proposal_logits.copy()
    logits[:] = 31.0
    np.testing.assert_array_equal(cache.proposal_logits, frozen)


@pytest.mark.parametrize('wrong_args,exception,match', [
    ({'n_heads': 3}, ValueError, 'divisible'),
    ({'temperature': 0.}, ValueError, 'temperature'),
    ({'temperature': float('nan')}, ValueError, 'temperature'),
    ({'read_scale': -1.}, ValueError, 'read_scale'),
    ({'read_scale': float('inf')}, ValueError, 'read_scale'),
    ({'source_scope': 'all'}, ValueError, 'block_size'),
    ({'max_reference_vocab': 4}, ValueError, 'tiny vocabulary'),
])
def test_invalid_reference_configuration_fails_loudly(wrong_args, exception, match):
    e = Parameter(np.zeros((7, 4), dtype=np.float64), name='E')
    with pytest.raises(exception, match=match):
        SoftVocabularyRead(e, 3, **{**{"n_heads": 2}, **wrong_args})


def test_dense_reference_cannot_be_accidentally_used_at_production_scale():
    huge_e = Parameter(np.zeros((65280, 4), dtype=np.float32), name='E')
    with pytest.raises(ValueError, match='tiny vocabulary'):
        SoftVocabularyRead(huge_e, 3, n_heads=2)
    e, read, dec, logits, _ = _make()
    read.max_reference_scores = 2
    with pytest.raises(ValueError, match='score budget'):
        read.forward(dec, logits)
    with pytest.raises(ValueError, match='expected DEC'):
        read.forward(dec[:, :1], logits)
    read.max_reference_scores = 500_000
    with pytest.raises(TypeError, match='share float32/float64'):
        read.forward(dec.astype(np.float32), logits)
    with pytest.raises(ValueError, match='nonfinite'):
        bad = logits.copy(); bad[0, 1, 0] = float('nan')
        read.max_reference_scores = 500_000
        read.forward(dec, bad)
