"""Patch 0102U: matrix-pool backward reference parity and CUDA coverage."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata
from tests.test_matrix_pool_dual_resolution_0102o import _attention


def _run(op, hidden, metadata, upstream, *, fused, monkeypatch, initial=None):
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_FUSED_BACKWARD", "1" if fused else "0")
    op.zero_grad()
    _, cache = op._pool_forward(hidden, metadata, True)
    cache["block_indices"] = metadata["block_indices"][cache["rows"], cache["slots"]]
    dx = op._pool_backward(upstream, cache, hidden.shape, True,
                           initial_dx=None if initial is None else initial.copy())
    grads = [op.W_pool_score.grad.copy(), op.pool_position.grad.copy()]
    return dx, grads


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="NumPy backend regression")
def test_disabled_cuda_backend_uses_reference_even_with_flag(monkeypatch):
    op = _attention()
    rng = np.random.default_rng(1048)
    hidden = rng.normal(size=(1, 8, 16))
    metadata = op._prepare_metadata(None, 1, 8, None)
    ds = rng.normal(size=(1, 2, 4, 16))
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_FUSED_FORWARD", "0")
    for seed_dx in (None, rng.normal(size=hidden.shape)):
        a = _run(op, hidden, metadata, ds, fused=False, monkeypatch=monkeypatch, initial=seed_dx)
        b = _run(op, hidden, metadata, ds, fused=True, monkeypatch=monkeypatch, initial=seed_dx)
        np.testing.assert_array_equal(a[0], b[0])
        for ga, gb in zip(a[1], b[1]):
            np.testing.assert_array_equal(ga, gb)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CuPy/Triton GPU")
@pytest.mark.parametrize("length, seed_add", [(8, False), (8, True), (12, True)])
def test_fused_backward_cuda_small_parity(monkeypatch, length, seed_add):
    import cupy as cp
    pytest.importorskip("triton")
    op = _attention(dtype="float32")
    rng = np.random.default_rng(1050 + length)
    hidden = cp.asarray(rng.normal(scale=0.2, size=(1, length, 16)).astype('float32'))
    tokens = np.arange(length, dtype=np.int64)[None, :]
    metadata = op._prepare_metadata(
        build_dual_resolution_metadata(tokens, eos_token_id=99, block_size=4),
        1, length, None,
    )
    ds = cp.asarray(rng.normal(scale=0.1, size=(1, length//4, 4, 16)).astype('float32'))
    initial = cp.asarray(rng.normal(size=hidden.shape).astype('float32')) if seed_add else None
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_FUSED_FORWARD", "0")
    ref = _run(op, hidden, metadata, ds, fused=False, monkeypatch=monkeypatch, initial=initial)
    trial = _run(op, hidden, metadata, ds, fused=True, monkeypatch=monkeypatch, initial=initial)
    cp.testing.assert_allclose(trial[0], ref[0], rtol=4e-4, atol=4e-5)
    for x, y in zip(trial[1], ref[1]):
        cp.testing.assert_allclose(x, y, rtol=4e-4, atol=4e-5)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CuPy/Triton GPU")
def test_fused_backward_cuda_production_width(monkeypatch):
    import cupy as cp
    pytest.importorskip("triton")
    from mini_llm.backend import RandomStream
    from mini_llm.config import ModelConfig
    from mini_llm.ops.matrix_pool_dual_resolution_attention import MatrixPoolDualResolutionAttention

    cfg = ModelConfig.dual_resolution_matrix_pool_1024_20l().dual_resolution
    op = MatrixPoolDualResolutionAttention(
        d_model=1024, n_q_heads=16, n_kv_heads=4, d_head=64,
        input_std=0.01, output_std=0.01, rng=RandomStream(1055),
        dtype='bfloat16', config=cfg,
    )
    rng = np.random.default_rng(1056)
    hidden = cp.asarray(rng.normal(scale=0.1, size=(1, 256, 1024)).astype('float32'))
    ds = cp.asarray(rng.normal(scale=0.1, size=(1, 2, 4, 1024)).astype('float32'))
    metadata = op._prepare_metadata(None, 1, 256, None)
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_FUSED_FORWARD", "0")
    ref = _run(op, hidden, metadata, ds, fused=False, monkeypatch=monkeypatch)
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_R4_LOGITS_BACKWARD", "0")
    trial = _run(op, hidden, metadata, ds, fused=True, monkeypatch=monkeypatch)
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_R4_LOGITS_BACKWARD", "1")
    r4_trial = _run(op, hidden, metadata, ds, fused=True, monkeypatch=monkeypatch)
    for candidate in (trial, r4_trial):
        cp.testing.assert_allclose(candidate[0], ref[0], rtol=0.02, atol=0.006)
        for x, y in zip(candidate[1], ref[1]):
            cp.testing.assert_allclose(x, y, rtol=0.025, atol=0.006)
