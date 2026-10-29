"""0102R: cache expanded summary metadata and avoid absent global buffers."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata
from tests.test_matrix_pool_dual_resolution_0102o import _attention


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="NumPy deterministic reference")
def test_expanded_metadata_reuses_all_four_arrays_across_replays(monkeypatch):
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    a = _attention()
    tokens = np.arange(12, dtype=np.int64)[None, :]
    metadata = a._prepare_metadata(
        build_dual_resolution_metadata(tokens, eos_token_id=99, block_size=4),
        1, 12, None,
    )
    first = a._summary_metadata(metadata)
    second = a._summary_metadata(metadata)
    assert second is first
    assert all(first[name] is second[name] for name in first)
    assert first["valid"].shape == (1, 3 * 4)
    for key in ("valid", "documents", "ends", "positions"):
        assert np.all(np.isfinite(first[key]))
    a.summaries_per_block = 2
    different = a._summary_metadata(metadata)
    assert different is not first
    assert different["valid"].shape == (1, 6)
    a.summaries_per_block = 4
    assert a._summary_metadata(metadata) is first


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="NumPy deterministic reference")
def test_local_only_output_identical_for_short_and_explicit_off(monkeypatch):
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    rng = np.random.default_rng(1032)
    for length in (3, 8):
        a = _attention(global_enabled=False)
        x = rng.normal(size=(1, length, 16))
        residual = rng.normal(size=x.shape)
        out, cache = a.forward(x, residual_input=residual)
        assert cache["global"] is None
        gradient, residual_gradient = a.backward(np.ones_like(out), cache)
        assert np.all(np.isfinite(gradient))
        np.testing.assert_array_equal(residual_gradient, np.zeros_like(residual))


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="NumPy reference")
def test_summary_metadata_cache_opt_out(monkeypatch):
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_REUSE_SUMMARY_METADATA", "0")
    operator = _attention()
    metadata = operator._prepare_metadata(None, 1, 12, None)
    left, right = operator._summary_metadata(metadata), operator._summary_metadata(metadata)
    assert left is not right
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])
