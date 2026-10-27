"""Contract and optional GPU parity tests for PATCH 0088A."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from benchmarks.benchmark_0088a_custom_attention import _compare, _summary


def test_summary_reports_independent_distribution_statistics():
    summary = _summary([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["median_ms"] == 2.5
    assert summary["min_ms"] == 1.0
    assert summary["max_ms"] == 4.0
    assert summary["p10_ms"] < summary["median_ms"] < summary["p90_ms"]


def test_parity_gate_requires_exact_zero_inactive_heads(tmp_path):
    shape = (1, 3, 16, 2)
    tensors = {
        "context": np.zeros(shape, dtype=np.float32),
        "d_q": np.zeros(shape, dtype=np.float32),
        "d_k": np.zeros((1, 3, 4, 2), dtype=np.float32),
        "d_v": np.zeros((1, 3, 4, 2), dtype=np.float32),
    }
    reference = tmp_path / "reference.npz"
    np.savez(
        reference,
        metadata_json=np.asarray(json.dumps({"candidate": "grouped"})),
        **tensors,
    )
    accepted = _compare(tensors, reference, (1, 3, 5))
    assert accepted["accepted"] is True
    tensors["context"][0, 0, 2, 0] = 1e-8
    rejected = _compare(tensors, reference, (1, 3, 5))
    assert rejected["accepted"] is False
    assert rejected["inactive_output_max_abs"] == pytest.approx(1e-8)


@pytest.mark.gpu
@pytest.mark.skipif(
    importlib.util.find_spec("triton") is None,
    reason="requires isolated Triton benchmark environment",
)
def test_custom_forward_backward_matches_fp64_oracle_and_inactive_heads(monkeypatch):
    monkeypatch.setenv("MINI_LLM_BACKEND", "cupy")
    import cupy as cp
    import mini_llm.backend  # noqa: F401

    from mini_llm.ops.triton_mixed_attention import (
        mixed_attention_forward_backward,
    )
    from benchmarks.reference_fused_attention_oracle import (
        backward_reference,
        forward_reference,
    )

    rng = np.random.default_rng(88001)
    q_host = rng.normal(0.0, 0.2, (1, 33, 16, 64)).astype(np.float32)
    k_host = rng.normal(0.0, 0.2, (1, 33, 4, 64)).astype(np.float32)
    v_host = rng.normal(0.0, 0.2, (1, 33, 4, 64)).astype(np.float32)
    d_out_host = rng.normal(0.0, 0.2, q_host.shape).astype(np.float32)
    active_local = (1, 3, 4, 7, 9, 12, 15)
    inactive = sorted(set(range(1, 16)) - set(active_local))
    d_out_host[:, :, inactive] = 0.0

    q = cp.asarray(q_host, dtype="bfloat16")
    k = cp.asarray(k_host, dtype="bfloat16")
    v = cp.asarray(v_host, dtype="bfloat16")
    d_out = cp.asarray(d_out_host)
    output, gradients = mixed_attention_forward_backward(
        q, k, v, d_out, active_local
    )
    cp.cuda.get_current_stream().synchronize()

    q_rounded, k_rounded, v_rounded = (
        cp.asnumpy(value).astype(np.float32) for value in (q, k, v)
    )
    mapping = tuple(head // 4 for head in range(16))
    windows = (33,) + (512,) * 15
    active = (0,) + active_local
    expected_output, _ = forward_reference(
        q_rounded, k_rounded, v_rounded, mapping, windows, active_heads=active
    )
    expected_gradients = backward_reference(
        q_rounded,
        k_rounded,
        v_rounded,
        d_out_host,
        mapping,
        windows,
        active_heads=active,
    )

    np.testing.assert_allclose(cp.asnumpy(output), expected_output, rtol=1e-2, atol=3e-3)
    for actual, expected in zip(gradients, expected_gradients):
        np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=3e-2, atol=4e-3)
    assert np.max(np.abs(cp.asnumpy(output)[:, :, inactive])) == 0.0
    assert np.max(np.abs(cp.asnumpy(gradients[0])[:, :, inactive])) == 0.0


@pytest.mark.gpu
@pytest.mark.skipif(
    importlib.util.find_spec("triton") is None,
    reason="requires isolated Triton benchmark environment",
)
def test_custom_full_head_only_skips_empty_local_launches(monkeypatch):
    monkeypatch.setenv("MINI_LLM_BACKEND", "cupy")
    import cupy as cp
    import mini_llm.backend  # noqa: F401

    from mini_llm.ops.triton_mixed_attention import (
        mixed_attention_backward,
        mixed_attention_forward,
    )

    rng = np.random.default_rng(88002)
    q = cp.asarray(rng.normal(0.0, 0.2, (1, 33, 16, 64)), dtype="bfloat16")
    k = cp.asarray(rng.normal(0.0, 0.2, (1, 33, 4, 64)), dtype="bfloat16")
    v = cp.asarray(rng.normal(0.0, 0.2, (1, 33, 4, 64)), dtype="bfloat16")
    upstream = cp.zeros(q.shape, dtype=cp.float32)
    upstream[:, :, 0] = 1.0

    output, cache = mixed_attention_forward(q, k, v, ())
    d_q, d_k, d_v = mixed_attention_backward(upstream, cache)
    cp.cuda.get_current_stream().synchronize()

    assert bool(cp.all(cp.isfinite(output[:, :, 0])))
    assert bool(cp.all(cp.isfinite(d_q[:, :, 0])))
    assert bool(cp.all(cp.isfinite(d_k[:, :, 0])))
    assert bool(cp.all(cp.isfinite(d_v[:, :, 0])))
    assert float(cp.max(cp.abs(output[:, :, 1:]))) == 0.0
    assert float(cp.max(cp.abs(d_q[:, :, 1:]))) == 0.0
    assert float(cp.max(cp.abs(d_k[:, :, 1:]))) == 0.0
    assert float(cp.max(cp.abs(d_v[:, :, 1:]))) == 0.0


@pytest.mark.gpu
@pytest.mark.skipif(
    importlib.util.find_spec("triton") is None,
    reason="requires isolated Triton benchmark environment",
)
def test_custom_local_window_has_exact_query_519_boundary(monkeypatch):
    monkeypatch.setenv("MINI_LLM_BACKEND", "cupy")
    import cupy as cp
    import mini_llm.backend  # noqa: F401

    from mini_llm.ops.triton_mixed_attention import mixed_attention_forward

    q = cp.zeros((1, 520, 16, 64), dtype="bfloat16")
    k = cp.zeros((1, 520, 4, 64), dtype="bfloat16")
    v_host = np.zeros((1, 520, 4, 64), dtype=np.float32)
    v_host[0, 7, 0, 0] = 100.0
    v_host[0, 8, 0, 1] = 1.0
    v = cp.asarray(v_host, dtype="bfloat16")
    output, _ = mixed_attention_forward(q, k, v, tuple(range(1, 16)))
    cp.cuda.get_current_stream().synchronize()
    last_full = cp.asnumpy(output[0, 519, 0])
    last_local = cp.asnumpy(output[0, 519, 1])
    assert last_local[0] == 0.0
    np.testing.assert_allclose(last_local[1], 1.0 / 512.0, rtol=2e-3, atol=2e-6)
    np.testing.assert_allclose(last_full[0], 100.0 / 520.0, rtol=2e-3, atol=2e-6)
