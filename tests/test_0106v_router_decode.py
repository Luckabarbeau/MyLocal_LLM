"""Single-row normalized routing, exact selection and existing arithmetic floor."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.router import expert_topk_forward
from mini_llm.ops.router_decode import decode_enabled, try_normalized_topk
from mini_llm.ops.router_inference import RouterInference


@pytest.mark.parametrize('shape,k,floor', [((2, 512), 10, 1e-6), ((1, 8), 8, 1e-6),
    ((1, 512), 33, 1e-6), ((1, 4097), 10, 1e-6), ((1, 16), 3, 1e-50)])
def test_unsupported_returns_none(shape, k, floor):
    a = xp.zeros(shape, dtype=xp.float32)
    if shape == (1, 8):
        a = a.astype(xp.float64)
    assert try_normalized_topk(a, k, floor) is None


def test_escape_hatch(monkeypatch):
    monkeypatch.setenv('MINI_LLM_ROUTER_DECODE', '0')
    assert not decode_enabled()
    r = RouterInference(3, 4, 2, renormalize_selected=True)
    assert not r._decode_fastpath_enabled


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Requires CUDA')
@pytest.mark.parametrize('n,k', [(1, 1), (3, 2), (32, 4), (512, 10), (4096, 32)])
@pytest.mark.parametrize('floor', [2**-14, .9, 2.0])
def test_exact_ids_and_probability_parity(n, k, floor):
    logits = xp.asarray(np.random.default_rng(n).normal(0, 5, (1, 1, n)).astype(np.float32))
    before = logits.copy()
    expected, ids, _ = expert_topk_forward(logits, k, return_cache=False)
    expected /= xp.maximum(xp.sum(expected, -1, keepdims=True), floor)
    actual, actual_ids = try_normalized_topk(logits, k, floor)
    np.testing.assert_array_equal(asnumpy(actual_ids), asnumpy(ids))
    np.testing.assert_allclose(asnumpy(actual), asnumpy(expected), rtol=2e-7, atol=2e-7)
    np.testing.assert_array_equal(asnumpy(logits), asnumpy(before))
    assert actual.dtype == xp.float32 and actual_ids.dtype == xp.int64


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Requires CUDA')
@pytest.mark.parametrize('values', [
    [0.] * 512,
    [2., 1., 2., -1., -np.inf, 2.],
    [1., np.nextafter(np.float32(1), np.float32(2)), 1., -10.],
    [10000., -10000., 0., 1.],
    [np.nan, 1., -np.inf, 1., np.nan],
    [np.inf, 2., np.inf, -np.inf],
    [-np.inf] * 4,
])
def test_ties_extremes_and_nonfinite_match_reference(values):
    logits = xp.asarray([values], dtype=xp.float32)
    k = min(4, len(values))
    with np.errstate(invalid='ignore', divide='ignore'):
        expected, ids, _ = expert_topk_forward(logits, k, return_cache=False)
        expected /= xp.maximum(xp.sum(expected, -1, keepdims=True), 2**-14)
    actual, actual_ids = try_normalized_topk(logits, k, 2**-14)
    np.testing.assert_array_equal(asnumpy(actual_ids), asnumpy(ids))
    np.testing.assert_allclose(asnumpy(actual), asnumpy(expected), rtol=2e-7, atol=2e-7, equal_nan=True)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Requires CUDA')
def test_integration_current_stream_and_exact_native_default():
    rng = np.random.default_rng(10622)
    with xp.cuda.Stream(non_blocking=True) as stream:
        x = xp.asarray(rng.normal(0, .1, (1, 1, 32)).astype(np.float32))
        weight = xp.asarray(rng.normal(0, .1, (32, 512)).astype(np.float32))
        bias = xp.zeros(512, xp.float32)
        native = RouterInference(32, 512, 10)
        fused = RouterInference(32, 512, 10, renormalize_selected=True, normalization_floor=2**-14)
        for r in (native, fused):
            r.W_router, r.b_router = weight, bias
        base, ids, _ = expert_topk_forward(x @ weight, 10, return_cache=False)
        a, i = native.forward(x)
        np.testing.assert_array_equal(asnumpy(a), asnumpy(base))
        np.testing.assert_array_equal(asnumpy(i), asnumpy(ids))
        expected = base / xp.maximum(xp.sum(base, -1, keepdims=True), 2**-14)
        a, i = fused.forward(x)
        stream.synchronize()
        np.testing.assert_array_equal(asnumpy(i), asnumpy(ids))
        np.testing.assert_allclose(asnumpy(a), asnumpy(expected), rtol=2e-7, atol=2e-7)
