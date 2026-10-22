"""0086B GPU correctness gates for the opt-in fused local forward."""

import numpy as np
import pytest

from benchmarks.reference_fused_attention_oracle import forward_reference
from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.fused_local_attention import fused_local_forward
from mini_llm.ops.indexed_attention import (
    local_window_attention_backward,
    local_window_attention_forward,
)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CUDA/CuPy"),
]


def _enable_grouped_recompute(monkeypatch):
    monkeypatch.setenv("MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM", "1")
    monkeypatch.setenv("MINI_LLM_LOCAL_CUBLAS_MODE", "grouped")
    monkeypatch.setenv("MINI_LLM_BF16_LOCAL_GEMM", "1")
    monkeypatch.setenv("MINI_LLM_FUSED_BF16_LOCAL_PIPELINE", "1")
    monkeypatch.setenv("MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS", "1")
    monkeypatch.setenv("MINI_LLM_BF16_MIXED_CONTEXT", "0")


def _bf16(array):
    return xp.asarray(np.asarray(array, dtype=np.float32), dtype="bfloat16")


def test_fused_forward_and_lse_match_fp64_oracle_for_strided_q(monkeypatch):
    _enable_grouped_recompute(monkeypatch)
    monkeypatch.setenv("MINI_LLM_TRAIN_ATTN_STRICT", "1")
    rng = np.random.default_rng(86201)
    q_parent = _bf16(rng.normal(0.0, 0.2, (2, 33, 8, 64)))
    q = q_parent[:, :, 1:4, :]
    k = _bf16(rng.normal(0.0, 0.2, (2, 33, 4, 64)))
    v = _bf16(rng.normal(0.0, 0.2, (2, 33, 4, 64)))
    mapping = (0, 2, 3)
    kv_map = xp.asarray(mapping, dtype=xp.int64)

    result = fused_local_forward(
        q, k, v, 33, 0.125, kv_map, output_dtype=xp.float32
    )
    assert result is not None
    output, lse = result
    xp.cuda.get_current_stream().synchronize()
    expected, expected_lse = forward_reference(
        xp.asnumpy(q).astype(np.float32),
        xp.asnumpy(k).astype(np.float32),
        xp.asnumpy(v).astype(np.float32),
        mapping,
        (33, 33, 33),
        scale=0.125,
    )
    np.testing.assert_allclose(
        xp.asnumpy(output), expected, rtol=5e-3, atol=3e-3
    )
    np.testing.assert_allclose(
        xp.asnumpy(lse), expected_lse, rtol=8e-4, atol=8e-4
    )


def test_public_fused_forward_reuses_grouped_recompute_backward(monkeypatch):
    _enable_grouped_recompute(monkeypatch)
    rng = np.random.default_rng(86202)
    q_parent = _bf16(rng.normal(0.0, 0.2, (1, 65, 16, 64)))
    d_parent = _bf16(rng.normal(0.0, 0.2, (1, 65, 16, 64)))
    q = q_parent[:, :, 1:16, :]
    d_out = d_parent[:, :, 1:16, :]
    k = _bf16(rng.normal(0.0, 0.2, (1, 65, 4, 64)))
    v = _bf16(rng.normal(0.0, 0.2, (1, 65, 4, 64)))
    mapping = tuple(head // 4 for head in range(1, 16))

    monkeypatch.setenv("MINI_LLM_TRAIN_ATTN_BACKEND", "grouped")
    grouped_output, grouped_cache = local_window_attention_forward(
        q, k, v, 512, kv_head_indices=mapping, return_cache=True
    )
    grouped_gradients = local_window_attention_backward(d_out, grouped_cache)

    monkeypatch.setenv("MINI_LLM_TRAIN_ATTN_BACKEND", "fused")
    monkeypatch.setenv("MINI_LLM_TRAIN_ATTN_STRICT", "1")
    fused_output, fused_cache = local_window_attention_forward(
        q, k, v, 512, kv_head_indices=mapping, return_cache=True
    )
    assert fused_cache["attention_backend"] == "fused_local_forward_0086b"
    assert fused_cache["lse"].shape == q.shape[:3]
    fused_gradients = local_window_attention_backward(d_out, fused_cache)
    xp.cuda.get_current_stream().synchronize()

    np.testing.assert_allclose(
        xp.asnumpy(fused_output), xp.asnumpy(grouped_output), rtol=8e-3, atol=4e-3
    )
    for fused, grouped in zip(fused_gradients, grouped_gradients):
        np.testing.assert_allclose(
            xp.asnumpy(fused), xp.asnumpy(grouped), rtol=0.0, atol=0.0
        )


def test_fused_window_512_has_exact_inclusive_left_boundary(monkeypatch):
    _enable_grouped_recompute(monkeypatch)
    monkeypatch.setenv("MINI_LLM_TRAIN_ATTN_BACKEND", "fused")
    monkeypatch.setenv("MINI_LLM_TRAIN_ATTN_STRICT", "1")
    q = _bf16(np.zeros((1, 520, 1, 64), dtype=np.float32))
    k = _bf16(np.zeros((1, 520, 1, 64), dtype=np.float32))
    v = np.zeros((1, 520, 1, 64), dtype=np.float32)
    v[0, 0, 0, 0] = 100.0
    v[0, 8, 0, 1] = 1.0
    output = local_window_attention_forward(
        q, k, _bf16(v), 512, kv_head_indices=(0,), return_cache=False
    )
    xp.cuda.get_current_stream().synchronize()
    last = xp.asnumpy(output[0, -1, 0])
    assert last[0] == 0.0
    np.testing.assert_allclose(last[1], 1.0 / 512.0, rtol=2e-3, atol=2e-6)


def test_strict_fused_mode_rejects_unsupported_head_dimension(monkeypatch):
    _enable_grouped_recompute(monkeypatch)
    monkeypatch.setenv("MINI_LLM_TRAIN_ATTN_BACKEND", "fused")
    monkeypatch.setenv("MINI_LLM_TRAIN_ATTN_STRICT", "1")
    q = _bf16(np.zeros((1, 8, 2, 32), dtype=np.float32))
    k = _bf16(np.zeros((1, 8, 1, 32), dtype=np.float32))
    v = _bf16(np.zeros((1, 8, 1, 32), dtype=np.float32))
    with pytest.raises(RuntimeError, match="D=64"):
        local_window_attention_forward(
            q, k, v, 8, kv_head_indices=(0, 0), return_cache=False
        )
