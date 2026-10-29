"""0102S: experimental fused Triton matrix-pool forward and reference backward."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata
from tests.test_matrix_pool_dual_resolution_0102o import _attention


def test_cpu_fused_flag_does_not_change_reference(monkeypatch):
    if BACKEND_NAME != "numpy":
        pytest.skip("CPU-only reference fallback")
    rng = np.random.default_rng(1040)
    op = _attention()
    hidden = rng.normal(size=(1, 8, 16))
    tokens = np.arange(8)[None, :]
    metadata = op._prepare_metadata(
        build_dual_resolution_metadata(tokens, eos_token_id=99, block_size=4),
        1, 8, None,
    )
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_FUSED_FORWARD", "0")
    ref, _ = op._pool_forward(hidden, metadata, True)
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_FUSED_FORWARD", "1")
    result, _ = op._pool_forward(hidden, metadata, True)
    np.testing.assert_array_equal(result, ref)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires RTX/CuPy + Triton")
def test_fused_pool_cuda_forward_and_backward_reference(monkeypatch):
    import cupy as cp
    pytest.importorskip("triton")
    rng = np.random.default_rng(1041)
    # Production shapes exercise the 128x1024 x4 learned compression.
    op = _attention(dtype="float32")
    op.d_model = 16
    tokens = np.arange(12, dtype=np.int64)[None, :]
    metadata = op._prepare_metadata(
        build_dual_resolution_metadata(tokens, eos_token_id=99, block_size=4),
        1, 12, None,
    )
    hidden = cp.asarray(rng.normal(scale=0.1, size=(1, 12, 16)).astype('float32'))
    up = cp.asarray(rng.normal(scale=0.1, size=(1, 3, 4, 16)).astype('float32'))

    def evaluate(flag):
        monkeypatch.setenv("MINI_LLM_MATRIX_POOL_FUSED_FORWARD", flag)
        op.zero_grad()
        summaries, cache = op._pool_forward(hidden, metadata, True)
        cache['block_indices'] = metadata['block_indices'][cache['rows'], cache['slots']]
        dx = op._pool_backward(up, cache, hidden.shape, True)
        grads = [p.grad.copy() for p in (op.W_pool_score, op.pool_position)]
        return summaries, cache['probabilities'], dx, grads

    ref = evaluate('0')
    fused = evaluate('1')
    for a, b in zip(ref[:3], fused[:3]):
        cp.testing.assert_allclose(b, a, rtol=3e-4, atol=3e-5)
    for a, b in zip(ref[3], fused[3]):
        cp.testing.assert_allclose(b, a, rtol=5e-4, atol=3e-5)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires RTX/CuPy + Triton")
def test_fused_pool_production_width_bf16(monkeypatch):
    import cupy as cp
    pytest.importorskip('triton')
    from mini_llm.backend import RandomStream
    from mini_llm.config import ModelConfig
    from mini_llm.ops.matrix_pool_dual_resolution_attention import MatrixPoolDualResolutionAttention
    cfg = ModelConfig.dual_resolution_matrix_pool_1024_20l().dual_resolution
    operator = MatrixPoolDualResolutionAttention(
        d_model=1024, n_q_heads=16, n_kv_heads=4, d_head=64,
        input_std=0.01, output_std=0.01, rng=RandomStream(1042),
        dtype='bfloat16', config=cfg,
    )
    rng = np.random.default_rng(1043)
    hidden = cp.asarray(rng.normal(scale=0.1, size=(1, 256, 1024)).astype('float32'))
    metadata = operator._prepare_metadata(None, 1, 256, None)
    monkeypatch.setenv('MINI_LLM_MATRIX_POOL_FUSED_FORWARD', '0')
    ref, _ = operator._pool_forward(hidden, metadata, True)
    monkeypatch.setenv('MINI_LLM_MATRIX_POOL_FUSED_FORWARD', '1')
    fast, _ = operator._pool_forward(hidden, metadata, True)
    cp.testing.assert_allclose(fast, ref, rtol=0.02, atol=0.008)
