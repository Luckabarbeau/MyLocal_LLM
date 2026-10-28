"""0102J: packed BF16 is default; experimental fused pooling parity."""
import os
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, RandomStream
from mini_llm.config import DualResolutionConfig
from mini_llm.ops.dual_resolution_attention import (
    DualResolutionAttention, _fast_projection_enabled,
    _bf16_projection_matmul_enabled, _bf16_compressor_qkv_enabled,
    _fused_pooling_enabled, build_dual_resolution_metadata,
)


def test_packed_projection_is_default_and_can_be_explicitly_disabled(monkeypatch):
    monkeypatch.delenv('MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS', raising=False)
    monkeypatch.delenv('MINI_LLM_DUAL_RESOLUTION_BF16_PROJECTION_GRADS', raising=False)
    monkeypatch.delenv('MINI_LLM_DUAL_RESOLUTION_BF16_COMPRESSOR_QKV', raising=False)
    monkeypatch.delenv('MINI_LLM_DUAL_RESOLUTION_FUSED_POOLING', raising=False)
    assert _fast_projection_enabled() == (BACKEND_NAME == 'cupy')
    assert _bf16_projection_matmul_enabled()
    assert _bf16_compressor_qkv_enabled()
    assert not _fused_pooling_enabled()
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS', '0')
    assert not _fast_projection_enabled()
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS', '1')
    assert _fast_projection_enabled()


def test_cpu_reference_default_matches_explicit_reference(monkeypatch):
    if BACKEND_NAME != 'numpy':
        pytest.skip('CPU default-parity test')
    cfg = DualResolutionConfig(
        enabled=True, block_size=4, local_window=8,
        compression_dim=16, compression_heads=2,
        compression_depth=1, max_blocks=8, global_query_chunk=8,
    )
    def model():
        return DualResolutionAttention(
            d_model=16, n_q_heads=2, n_kv_heads=1, d_head=8,
            input_std=0.05, output_std=0.05, rng=RandomStream(81),
            dtype='float32', config=cfg,
        )
    tokens = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 0, 3, 4, 5, 6, 7, 8, 9]])
    metadata = build_dual_resolution_metadata(tokens, eos_token_id=0, block_size=4)
    x = np.random.default_rng(82).standard_normal((1, 16, 16)).astype('float32')
    dy = np.random.default_rng(83).standard_normal((1, 16, 16)).astype('float32')
    def run():
        m = model()
        y, cache = m.forward(x, dual_resolution_metadata=metadata)
        dx = m.backward(dy, cache)
        return y, dx, [p.grad.copy() for p in m.parameters()]
    monkeypatch.delenv('MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS', raising=False)
    default = run()
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS', '0')
    explicit = run()
    for a, b in zip(default[:2], explicit[:2]):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(default[2], explicit[2]):
        np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CuPy/CUDA')
@pytest.mark.parametrize('blocks', [1, 11, 64])
def test_fused_pooling_direct_forward_backward(blocks):
    import cupy as cp
    pytest.importorskip('triton')
    from mini_llm.ops.triton_dual_resolution_pooling import forward, backward
    from mini_llm.ops.attention import softmax_forward, softmax_backward
    rng = np.random.default_rng(808 + blocks)
    k = cp.asarray(rng.normal(size=(blocks, 8, 128, 64)).astype('float32'))
    v = cp.asarray(rng.normal(size=(blocks, 8, 128, 64)).astype('float32'))
    query = cp.asarray(rng.normal(size=(8, 64)).astype('float32'))
    dc = cp.asarray(rng.normal(size=(blocks, 8, 64)).astype('float32'))
    scale = 0.125
    scores = cp.sum(k * query[None, :, None, :], axis=-1) * scale
    p_ref = softmax_forward(scores, axis=-1)
    c_ref = cp.sum(p_ref[..., None] * v, axis=2)
    c, p = forward(k, v, query, scale)
    cp.testing.assert_allclose(c, c_ref, rtol=2e-5, atol=2e-5)
    cp.testing.assert_allclose(p, p_ref, rtol=2e-5, atol=2e-5)
    dv_ref = p_ref[..., None] * dc[:, :, None, :]
    dp = cp.sum(dc[:, :, None, :] * v, axis=-1)
    ds = softmax_backward(dp, p_ref, axis=-1)
    dk_ref = ds[..., None] * query[None, :, None, :] * scale
    dq_ref = cp.sum(ds[..., None] * k, axis=(0, 2)) * scale
    dk, dv, dq = backward(dc, k, v, query, p, scale)
    cp.testing.assert_allclose(dk, dk_ref, rtol=3e-5, atol=3e-5)
    cp.testing.assert_allclose(dv, dv_ref, rtol=3e-5, atol=3e-5)
    # The final reduction aggregates up to 64 independent block-head partials;
    # CuPy and Triton legitimately accumulate FP32 sums in different orders.
    cp.testing.assert_allclose(dq, dq_ref, rtol=3e-4, atol=3e-4)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CuPy/CUDA')
def test_fused_pooling_complete_model_production_width(monkeypatch):
    import cupy as cp
    pytest.importorskip('triton')
    cfg = DualResolutionConfig(
        enabled=True, block_size=128, local_window=512,
        compression_dim=512, compression_heads=8, compression_depth=1,
        max_blocks=8, global_query_chunk=128,
    )
    def model():
        return DualResolutionAttention(
            d_model=1024, n_q_heads=16, n_kv_heads=4, d_head=64,
            input_std=0.01, output_std=0.01, rng=RandomStream(994),
            dtype='bfloat16', config=cfg,
        )
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL', 'triton')
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL', 'triton')
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL', 'triton')
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS', '1')
    token = np.full((2, 258), 3, dtype=np.int64)
    token[0, 255] = 0
    token[1, 127] = 0
    token[1, 193] = 0
    metadata = build_dual_resolution_metadata(token, eos_token_id=0, block_size=128)
    rng = np.random.default_rng(995)
    x = cp.asarray(rng.normal(scale=0.04, size=(2, 258, 1024)).astype('float32')).astype(cp.dtype('bfloat16'))
    dy = cp.asarray(rng.normal(scale=0.04, size=(2, 258, 1024)).astype('float32'))
    def run(fused):
        monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FUSED_POOLING', str(int(fused)))
        m = model()
        y, cache = m.forward(x, dual_resolution_metadata=metadata)
        dx = m.backward(dy, cache)
        return y.astype(cp.float32), dx.astype(cp.float32), [p.grad.copy() for p in m.parameters()]
    ref = run(False)
    opt = run(True)
    cp.testing.assert_allclose(opt[0], ref[0], atol=0.04, rtol=0.04)
    cp.testing.assert_allclose(opt[1], ref[1], atol=0.08, rtol=0.10)
    for got, expected in zip(opt[2], ref[2]):
        assert bool(cp.all(cp.isfinite(got)))
        cp.testing.assert_allclose(got, expected, atol=0.12, rtol=0.12)
