"""Production-dispatch tests for the opt-in 0088A Triton attention backend."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME


def test_triton_backend_is_not_a_default_runtime_dependency(monkeypatch):
    monkeypatch.delenv("MINI_LLM_TRAIN_ATTN_BACKEND", raising=False)
    from mini_llm.ops.attention import _triton_mixed_attention_requested

    assert _triton_mixed_attention_requested() is False


@pytest.mark.gpu
@pytest.mark.skipif(
    BACKEND_NAME != "cupy" or importlib.util.find_spec("triton") is None,
    reason="requires CuPy plus the isolated optional Triton environment",
)
@pytest.mark.parametrize("historical_skip_heads", ((), (2, 5, 7, 11, 14, 15)))
def test_production_mixed_dispatch_matches_grouped_forward_backward(
    monkeypatch, historical_skip_heads
):
    import cupy as cp

    from benchmarks.reference_0087b_torch_attention import error_metrics
    from mini_llm.backend import RandomStream
    from mini_llm.config import AttentionLayerConfig, LocalAttentionConfig
    from mini_llm.ops.attention import GQAAttention

    for name, value in {
        "MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM": "1",
        "MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT": "0",
        "MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD": "1",
        "MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS": "1",
        "MINI_LLM_LOCAL_CUBLAS_MODE": "grouped",
        "MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS": "1",
        "MINI_LLM_LOCAL_ATTN_QUERY_CHUNK": "auto",
        "MINI_LLM_BF16_MIXED_CONTEXT": "0",
    }.items():
        monkeypatch.setenv(name, value)

    attention = GQAAttention(
        d_model=1024,
        n_q_heads=16,
        n_kv_heads=4,
        d_head=64,
        input_std=0.02,
        output_std=0.02,
        rng=RandomStream(88020),
        dtype="bfloat16",
        attention_config=AttentionLayerConfig(
            heads=(
                LocalAttentionConfig(window=4096),
                *(LocalAttentionConfig(window=512) for _ in range(15)),
            )
        ),
    )
    rng = np.random.default_rng(88021)
    q = cp.asarray(rng.normal(0.0, 0.2, (1, 129, 16, 64)), dtype="bfloat16")
    k = cp.asarray(rng.normal(0.0, 0.2, (1, 129, 4, 64)), dtype="bfloat16")
    v = cp.asarray(rng.normal(0.0, 0.2, (1, 129, 4, 64)), dtype="bfloat16")
    x = cp.zeros((1, 129, 1024), dtype="bfloat16")
    dcontext = cp.asarray(
        rng.normal(0.0, 0.2, q.shape).astype(np.float32)
    )

    monkeypatch.setenv("MINI_LLM_TRAIN_ATTN_BACKEND", "grouped")
    grouped_output, grouped_groups, _ = attention._mixed_context_forward(
        x,
        q,
        k,
        v,
        True,
        historical_skip_heads=historical_skip_heads,
    )
    grouped_gradients = attention._mixed_context_backward(
        dcontext,
        {"x": x, "q": q, "k": k, "v": v, "pattern_caches": grouped_groups},
    )[:3]

    monkeypatch.setenv("MINI_LLM_TRAIN_ATTN_BACKEND", "triton")
    monkeypatch.setenv("MINI_LLM_TRAIN_ATTN_STRICT", "1")
    triton_output, triton_groups, _ = attention._mixed_context_forward(
        x,
        q,
        k,
        v,
        True,
        historical_skip_heads=historical_skip_heads,
    )
    assert [group["kind"] for group in triton_groups] == ["triton_mixed_0088a"]
    triton_gradients = attention._mixed_context_backward(
        dcontext,
        {"x": x, "q": q, "k": k, "v": v, "pattern_caches": triton_groups},
    )[:3]
    cp.cuda.get_current_stream().synchronize()

    pairs = ((triton_output, grouped_output), *zip(triton_gradients, grouped_gradients))
    for actual, expected in pairs:
        metrics = error_metrics(cp.asnumpy(actual), cp.asnumpy(expected))
        assert metrics["relative_l2"] <= 0.02
        assert metrics["cosine"] >= 0.999
    if historical_skip_heads:
        inactive = list(historical_skip_heads)
        assert float(cp.max(cp.abs(triton_output[:, :, inactive, :]))) == 0.0
        assert float(cp.max(cp.abs(triton_gradients[0][:, :, inactive, :]))) == 0.0
