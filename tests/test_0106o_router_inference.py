"""Inference-only selected-mass normalization must not change native defaults."""

import numpy as np
import pytest

from mini_llm.backend import asnumpy, xp
from mini_llm.ops.router import expert_topk_forward
from mini_llm.ops.router_inference import RouterInference


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('k', [1, 2, 4])
def test_opt_in_normalization_and_exact_legacy_default(dtype, k):
    rng = np.random.default_rng(10615)
    x = xp.asarray(rng.normal(0, .2, (2, 3, 5)).astype(dtype))
    weight = xp.asarray(rng.normal(0, .3, (5, 4)).astype(dtype))
    bias = xp.asarray(np.array([.1, -.1, .2, -.2], dtype=dtype))
    native = RouterInference(5, 4, k, dtype)
    normalized = RouterInference(5, 4, k, dtype, renormalize_selected=True)
    for router in (native, normalized):
        router.W_router, router.b_router = weight, bias
    logits = (x.reshape(-1, 5) @ weight + bias).reshape(2, 3, 4)
    baseline, expected_ids, _ = expert_topk_forward(logits, k, output_dtype=x.dtype,
                                                   return_cache=False)
    actual, ids = native.forward(x)
    np.testing.assert_array_equal(asnumpy(actual), asnumpy(baseline))
    np.testing.assert_array_equal(asnumpy(ids), asnumpy(expected_ids))
    actual, ids = normalized.forward(x)
    selected = np.take_along_axis(asnumpy(logits).astype(np.float64), asnumpy(ids), -1)
    expected = np.exp(selected - selected.max(axis=-1, keepdims=True))
    expected /= expected.sum(axis=-1, keepdims=True)
    tol = 1e-3 if dtype == 'float16' else 2e-7 if dtype == 'float32' else 2e-15
    np.testing.assert_allclose(asnumpy(actual), expected, rtol=tol, atol=tol)
    np.testing.assert_array_equal(asnumpy(ids), asnumpy(expected_ids))
    assert actual.dtype == x.dtype
    if k < 4:
        assert np.all(asnumpy(baseline).sum(axis=-1) < 1)


@pytest.mark.parametrize('kwargs', [dict(renormalize_selected=1),
    dict(normalization_floor=0), dict(normalization_floor=-1),
    dict(normalization_floor=np.nan), dict(normalization_floor=np.inf)])
def test_normalization_configuration_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        RouterInference(4, 4, 2, **kwargs)
