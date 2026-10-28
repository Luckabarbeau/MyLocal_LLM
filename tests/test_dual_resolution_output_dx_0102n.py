"""0102N BF16 attention output-projection input-gradient parity."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, RandomStream
from mini_llm.config import DualResolutionConfig
from mini_llm.ops.dual_resolution_attention import (
    DualResolutionAttention,
    _bf16_output_dx_enabled,
    build_dual_resolution_metadata,
)


def test_bf16_output_dx_is_independently_opt_in(monkeypatch):
    monkeypatch.delenv("MINI_LLM_DUAL_RESOLUTION_BF16_OUTPUT_DX", raising=False)
    assert not _bf16_output_dx_enabled()
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_BF16_OUTPUT_DX", "1")
    assert _bf16_output_dx_enabled() == (BACKEND_NAME == "cupy")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_BF16_OUTPUT_DX", "0")
    assert not _bf16_output_dx_enabled()


def _model():
    config = DualResolutionConfig(
        enabled=True,
        block_size=128,
        local_window=512,
        compression_dim=512,
        compression_heads=8,
        compression_depth=1,
        max_blocks=8,
        global_query_chunk=128,
        recompute_attention_probs=True,
    )
    return DualResolutionAttention(
        d_model=1024,
        n_q_heads=16,
        n_kv_heads=4,
        d_head=64,
        input_std=0.01,
        output_std=0.01,
        rng=RandomStream(10214),
        dtype="bfloat16",
        config=config,
    )


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CuPy/CUDA")
@pytest.mark.parametrize("batch,sequence", [(1, 257), (2, 260)])
def test_bf16_output_dx_full_attention_backward_parity(monkeypatch, batch, sequence):
    import cupy as cp

    pytest.importorskip("triton")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL", "triton")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL", "triton")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL", "triton")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "1")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_BF16_PROJECTION_GRADS", "1")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FUSED_POOLING", "1")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FUSED_RMSNORM", "1")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FUSED_GATE", "1")

    tokens = np.full((batch, sequence), 3, dtype=np.int32)
    if batch == 2:
        tokens[0, 255] = 0
        tokens[1, 127] = 0
    metadata = build_dual_resolution_metadata(tokens, eos_token_id=0, block_size=128)
    rng = np.random.default_rng(10215 + batch)
    x = cp.asarray(
        rng.normal(scale=0.04, size=(batch, sequence, 1024)).astype("float32")
    ).astype("bfloat16")
    dy = cp.asarray(
        rng.normal(scale=0.04, size=x.shape).astype("float32")
    ).astype("bfloat16")

    def run(enabled):
        monkeypatch.setenv(
            "MINI_LLM_DUAL_RESOLUTION_BF16_OUTPUT_DX", str(int(enabled))
        )
        model = _model()
        output, cache = model.forward(x, dual_resolution_metadata=metadata)
        model.zero_grad()
        dx = model.backward(dy, cache)
        return model, output.copy(), dx.copy(), [p.grad.copy() for p in model.parameters()]

    reference = run(False)
    optimized = run(True)
    cp.testing.assert_array_equal(optimized[1], reference[1])
    cp.testing.assert_allclose(
        optimized[2].astype("float32"), reference[2].astype("float32"),
        rtol=0.12, atol=0.10,
    )
    cp.testing.assert_array_equal(optimized[0].Wo.grad, reference[0].Wo.grad)
    for parameter, got, expected in zip(
        optimized[0].parameters(), optimized[3], reference[3]
    ):
        assert bool(cp.all(cp.isfinite(got))), parameter.name
        cp.testing.assert_allclose(got, expected, rtol=0.18, atol=0.18)

    # Reusing the same backward tape must add the same contribution rather than
    # overwrite the canonical FP32 accumulation buffers.
    once = [p.grad.copy() for p in optimized[0].parameters()]
    optimized[0].backward(dy, optimized[0].forward(
        x, dual_resolution_metadata=metadata
    )[1])
    for parameter, first in zip(optimized[0].parameters(), once):
        cp.testing.assert_allclose(parameter.grad, first * 2, rtol=2e-4, atol=2e-4)
