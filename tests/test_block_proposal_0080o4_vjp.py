"""0080O4: memory-bounded full-vocabulary probability feedback derivatives."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp, asnumpy
from mini_llm.parameter import Parameter
from mini_llm.predictive_refinement.soft_embeddings import (
    SoftTokenExpectation, softmax_vjp,
)


@pytest.mark.parametrize('dtype', ['float64', 'float32'])
@pytest.mark.parametrize('temperature', [0.65, 1.0, 1.8])
@pytest.mark.parametrize('chunk_rows', [1, 3, 64])
def test_tiled_vjp_full_vocabulary_matches_reference(monkeypatch, dtype, temperature, chunk_rows):
    rng = np.random.default_rng(8094)
    logits = rng.normal(size=(2, 5, 257)).astype(dtype)
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=-1, keepdims=True)
    g = rng.normal(size=p.shape).astype(dtype)
    p, g = xp.asarray(p), xp.asarray(g)
    monkeypatch.delenv('MINI_LLM_FUSED_SOFTMAX_VJP', raising=False)
    monkeypatch.delenv('MINI_LLM_SOFTMAX_VJP_CHUNK_ROWS', raising=False)
    baseline = asnumpy(softmax_vjp(p, g, temperature))
    monkeypatch.setenv('MINI_LLM_SOFTMAX_VJP_CHUNK_ROWS', str(chunk_rows))
    got = asnumpy(softmax_vjp(p, g, temperature))
    eps = 1e-12 if dtype == 'float64' else 6e-7
    np.testing.assert_allclose(got, baseline, rtol=eps, atol=eps)


def test_tiled_vjp_non_contiguous_layout_and_invalid_settings(monkeypatch):
    rng = np.random.default_rng(82)
    p = xp.asarray(np.abs(rng.normal(size=(4, 7, 16))).astype(np.float64))
    p /= xp.sum(p, axis=-1, keepdims=True)
    p = p[:, ::2, ::2]
    g = xp.asarray(rng.normal(size=(4, 7, 16)))[:, ::2, ::2]
    monkeypatch.delenv('MINI_LLM_SOFTMAX_VJP_CHUNK_ROWS', raising=False)
    baseline = asnumpy(softmax_vjp(p, g, 0.75))
    monkeypatch.setenv('MINI_LLM_SOFTMAX_VJP_CHUNK_ROWS', '2')
    result = asnumpy(softmax_vjp(p, g, 0.75))
    np.testing.assert_allclose(result, baseline, rtol=1e-12, atol=1e-12)
    monkeypatch.setenv('MINI_LLM_SOFTMAX_VJP_CHUNK_ROWS', '-1')
    with pytest.raises(ValueError, match='negative'):
        softmax_vjp(p, g)
    monkeypatch.setenv('MINI_LLM_SOFTMAX_VJP_CHUNK_ROWS', 'oops')
    with pytest.raises(ValueError, match='integer'):
        softmax_vjp(p, g)


def test_tiled_expectation_end_to_end_logit_and_embedding_gradients(monkeypatch):
    rng = np.random.default_rng(84)
    token_weights = rng.normal(size=(31, 6))
    logits = rng.normal(size=(2, 4, 31))
    upstream = rng.normal(size=(2, 4, 6))
    monkeypatch.delenv('MINI_LLM_FUSED_SOFTMAX_VJP', raising=False)
    outputs = []
    for chunk in ('0', '1', '3'):
        monkeypatch.setenv('MINI_LLM_SOFTMAX_VJP_CHUNK_ROWS', chunk)
        W = Parameter(xp.asarray(token_weights.copy()))
        op = SoftTokenExpectation(W)
        y, cache = op.forward(xp.asarray(logits), temperature=0.8)
        grad = op.backward(xp.asarray(upstream), cache)
        outputs.append((asnumpy(y), asnumpy(grad), asnumpy(W.grad)))
    for got in outputs[1:]:
        for a, b in zip(got, outputs[0]):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires real CUDA/CuPy GPU')
@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_cuda_fused_vjp_full_vocabulary_gradient_parity(monkeypatch, dtype):
    import ml_dtypes
    rng = np.random.default_rng(85)
    # Preserve the real vocabulary width; small row count keeps GPU CI light.
    vocab = 65280
    p_np = rng.random((2, 4, vocab), dtype=np.float32)
    p_np /= p_np.sum(axis=-1, keepdims=True)
    p = xp.asarray(p_np)
    g_np = rng.normal(size=p_np.shape).astype(np.float32)
    g = xp.asarray(g_np, dtype=ml_dtypes.bfloat16 if dtype == 'bfloat16' else xp.float32)
    monkeypatch.delenv('MINI_LLM_FUSED_SOFTMAX_VJP', raising=False)
    monkeypatch.setenv('MINI_LLM_SOFTMAX_VJP_CHUNK_ROWS', '2')
    reference = asnumpy(softmax_vjp(p, g, temperature=0.77))
    monkeypatch.setenv('MINI_LLM_FUSED_SOFTMAX_VJP', '1')
    actual = asnumpy(softmax_vjp(p, g, temperature=0.77))
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=3e-8)
    # Invariance: softmax logit gradients sum to zero in each row.
    np.testing.assert_allclose(actual.sum(axis=-1), 0., atol=2e-6)
    assert np.isfinite(actual).all()
