"""0102Q: view/scatter fast path must equal indexed blocks, including EOS masks."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata
from tests.test_matrix_pool_dual_resolution_0102o import _attention


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="NumPy deterministic layout reference")
@pytest.mark.parametrize("tokens,expected", [
    ([1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 10], True),
    ([1, 2, 0, 4, 5, 6, 7, 0, 9, 10, 11, 12], False),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], True),
])
def test_view_pool_matches_indexed_forward_backward(tokens, expected, monkeypatch):
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    op = _attention()
    tokens = np.asarray(tokens, dtype=np.int64)[None, :]
    metadata = op._prepare_metadata(
        build_dual_resolution_metadata(tokens, eos_token_id=0, block_size=4),
        1, tokens.shape[1], None,
    )
    assert metadata["contiguous_pool_blocks"] is expected
    forced_reference = dict(metadata, contiguous_pool_blocks=False)
    rng = np.random.default_rng(1029)
    hidden = rng.normal(size=(1, tokens.shape[1], 16))
    out, cache = op._pool_forward(hidden, metadata, True)
    old, old_cache = op._pool_forward(hidden, forced_reference, True)
    np.testing.assert_allclose(out, old, rtol=0, atol=0)
    if expected and cache["blocks"].size:
        assert np.shares_memory(cache["blocks"], hidden)
    if cache["blocks"].size == 0:
        return
    for current in (cache, old_cache):
        current["block_indices"] = metadata["block_indices"][current["rows"], current["slots"]]
    ds = rng.normal(size=out.shape)
    op.zero_grad()
    dfast = op._pool_backward(ds, cache, hidden.shape, True)
    fast_grads = [op.W_pool_score.grad.copy(), op.pool_position.grad.copy()]
    op.zero_grad()
    dref = op._pool_backward(ds, old_cache, hidden.shape, True)
    np.testing.assert_allclose(dfast, dref, rtol=1e-12, atol=1e-12)
    for got, expected_grad in zip(fast_grads, (op.W_pool_score.grad, op.pool_position.grad)):
        np.testing.assert_allclose(got, expected_grad, rtol=1e-12, atol=1e-12)
    seed_dx = rng.normal(size=hidden.shape)
    dadd = op._pool_backward(ds, cache, hidden.shape, False, initial_dx=seed_dx.copy())
    np.testing.assert_allclose(dadd, seed_dx + dref, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="NumPy reference")
def test_regular_metadata_has_no_gather(monkeypatch):
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    op = _attention()
    data = np.zeros((2, 8, 16), dtype=np.float64)
    metadata = op._prepare_metadata(None, 2, 8, None)
    assert metadata["contiguous_pool_blocks"]
    _, cache = op._pool_forward(data, metadata, True)
    assert cache["blocks"].shape == (4, 4, 16)
    assert np.shares_memory(cache["blocks"], data)
    irregular = op._prepare_metadata(None, 2, 10, None)
    assert not irregular["contiguous_pool_blocks"]


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="NumPy reference")
def test_contiguous_view_opt_out_preserves_pooling(monkeypatch):
    operator = _attention()
    metadata = operator._prepare_metadata(None, 1, 8, None)
    hidden = np.random.default_rng(1044).normal(size=(1, 8, 16))
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_CONTIGUOUS_VIEW", "1")
    fast, fast_cache = operator._pool_forward(hidden, metadata, True)
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_CONTIGUOUS_VIEW", "0")
    slow, slow_cache = operator._pool_forward(hidden, metadata, True)
    assert fast_cache["contiguous"] and not slow_cache["contiguous"]
    np.testing.assert_array_equal(fast, slow)
