"""0080A: precision, alignment, leakage and decoder cache contracts."""
import numpy as np
import pytest

from mini_llm.backend import asnumpy, xp, RandomStream
from mini_llm.parameter import Parameter
from mini_llm.predictive_refinement.soft_embeddings import SoftTokenExpectation
from mini_llm.block_proposal.contracts import (
    block_schedule, prefix_lm_allow_mask, shifted_decoder_tokens,
)
from mini_llm.block_proposal.reference import (
    reference_attention, reference_incremental_decoding,
)


def test_full_probability_embedding_gradient_matches_finite_difference():
    rng = np.random.default_rng(8)
    weight = Parameter(xp.asarray(rng.normal(size=(9, 5))), name='test.embedding')
    expectation = SoftTokenExpectation(weight)
    logits = xp.asarray(rng.normal(size=(2, 3, 9)))
    upstream = xp.asarray(rng.normal(size=(2, 3, 5)))
    z, cache = expectation.forward(logits)
    dz = expectation.backward(upstream, cache)
    # Full-vocabulary, exact, no top-k and no stop-gradient.
    def loss(l):
        return float(xp.sum(expectation.forward(xp.asarray(l))[0] * upstream))
    at = asnumpy(logits).copy()
    eps = 1e-5
    for idx in [(0, 0, 0), (0, 1, 7), (1, 2, 4)]:
        plus = at.copy(); plus[idx] += eps
        minus = at.copy(); minus[idx] -= eps
        numerical = (loss(plus) - loss(minus)) / (2 * eps)
        np.testing.assert_allclose(asnumpy(dz)[idx], numerical, rtol=1e-6, atol=1e-8)
    original = asnumpy(weight.data).copy()
    for idx in [(0, 0), (4, 2), (8, 3)]:
        plus = original.copy(); plus[idx] += eps
        minus = original.copy(); minus[idx] -= eps
        weight.data[...] = plus; a = loss(at)
        weight.data[...] = minus; b = loss(at)
        weight.data[...] = original
        np.testing.assert_allclose(asnumpy(weight.grad)[idx], (a-b)/(2*eps), atol=1e-8, rtol=1e-6)
    assert z.shape == (2, 3, 5)


@pytest.mark.parametrize('k', [1, 2, 4, 8, 16])
def test_block_schedule_real_next_token_alignment_short_documents_and_tails(k):
    labels = xp.asarray([[21,22,23,24,25,26,27,28,29], [31,32,-100,-100,-100,-100,-100,-100,-100]])
    valid = xp.asarray([[1]*9, [1,1]+[0]*7], dtype=xp.bool_)
    blocks = block_schedule(labels, valid, k, vocab_size=40)
    assert blocks.unique_supervised_tokens == 11
    assert int(xp.sum(blocks.valid)) == 11
    for b, s, seq, mask in zip(asnumpy(blocks.source_row), asnumpy(blocks.boundary_working_index),
                                asnumpy(blocks.targets), asnumpy(blocks.valid)):
        # hidden[b, prefix + s] represents x[s] and predicts labels[s].
        np.testing.assert_array_equal(seq[mask], asnumpy(labels)[b, s:s+k][:sum(mask)])
    assert not np.any(asnumpy(blocks.targets)[~asnumpy(blocks.valid)])
    shifted = shifted_decoder_tokens(blocks.targets, blocks.valid)
    assert np.all(asnumpy(shifted)[:, 0] == -1)
    for row in range(shifted.shape[0]):
        for j in range(1, k):
            if bool(asnumpy(blocks.valid)[row, j]):
                assert int(asnumpy(shifted)[row,j]) == int(asnumpy(blocks.targets)[row,j-1])


def test_reject_nonprefix_validity_and_invalid_targets():
    with pytest.raises(ValueError, match='prefix-contiguous'):
        block_schedule(xp.asarray([[1,2,3]]), xp.asarray([[1,0,1]],dtype=xp.bool_),2)
    with pytest.raises(ValueError, match='vocabulary'):
        block_schedule(xp.asarray([[1,99]]), None, 2, vocab_size=10)


@pytest.mark.parametrize('k', [2, 4])
def test_plan_bidirectional_decoder_strictly_causal_and_cached_parity(k):
    mask = asnumpy(prefix_lm_allow_mask(k))
    assert mask[:k,:k].all() and not mask[:k,k:].any()
    assert mask[k:,:k].all()
    np.testing.assert_array_equal(mask[k:,k:], np.tri(k,dtype=bool))
    rng = np.random.default_rng(101)
    inputs = xp.asarray(rng.normal(size=(2*k,7)))
    q = xp.asarray(rng.normal(size=(7,5)))
    key = xp.asarray(rng.normal(size=(7,5)))
    value = xp.asarray(rng.normal(size=(7,6)))
    full = reference_attention(inputs, q, key, value, block_size=k)
    plan, dec = reference_incremental_decoding(inputs[:k],inputs[k:],q,key,value)
    np.testing.assert_allclose(asnumpy(full[:k]), asnumpy(plan), rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(asnumpy(full[k:]), asnumpy(dec), rtol=1e-12,atol=1e-12)
    # Future teacher-forced commitments must not affect earlier decoder outputs.
    corrupted = inputs.copy()
    corrupted[k+1:] += 500.0
    changed = reference_attention(corrupted,q,key,value,block_size=k)
    np.testing.assert_allclose(asnumpy(changed[k]),asnumpy(full[k]),rtol=0,atol=1e-12)
    # Plans cannot read decoder input, even if its values change arbitrarily.
    np.testing.assert_allclose(asnumpy(changed[:k]),asnumpy(full[:k]),rtol=0,atol=1e-12)


def test_production_causal_hidden_boundary_and_future_corruption():
    # Production Transformer forward_body, not the discarded old refinement model.
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    cfg = ModelConfig(tokenizer_vocab_size=43,context_length=16,n_layers=2,
                      d_model=16,n_q_heads=2,n_kv_heads=1,d_head=8,
                      d_ff=32,n_experts=2,top_k=1,dtype='float64')
    model = DecoderLanguageModel(cfg,rng_seed=71,dtype='float64')
    inputs = xp.asarray([[3,4,5,6,7,8,9,10]])
    h = model.forward_body(inputs,return_cache=False)
    changed = inputs.copy(); changed[:,4:] = xp.asarray([30,31,32,33])
    hc = model.forward_body(changed,return_cache=False)
    for idx in (0,2):
        # Any later truth is causally forbidden from the boundary representation.
        np.testing.assert_allclose(asnumpy(h[:,idx]),asnumpy(hc[:,idx]),rtol=0,atol=1e-12)
        logits = h[:,idx] @ model.output_proj.W.data
        standard = model.forward(inputs, return_cache=False)[:,idx]
        np.testing.assert_allclose(asnumpy(logits),asnumpy(standard),rtol=0,atol=1e-12)
