"""HIP split-pass KV key-tile parity through the real local attention API."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME

pytestmark = pytest.mark.skipif(
    BACKEND_NAME != 'cupy', reason='HIP Triton parity requires CuPy backend',
)


@pytest.fixture
def hip_local():
    pytest.importorskip('triton')
    cp = pytest.importorskip('cupy')
    if not cp.cuda.runtime.is_hip or not cp.cuda.is_available():
        pytest.skip('HIP device required')
    from mini_llm.ops.triton_dual_resolution_local import (
        local_attention_forward, local_attention_backward,
    )
    return cp, local_attention_forward, local_attention_backward


def _assert_tile_parity(monkeypatch, cp, backward, dy, cache):
    monkeypatch.setenv('MINI_LLM_HIP_SPLIT_KV_BACKWARD', '2')
    monkeypatch.setenv('MINI_LLM_HIP_LOCAL_KV_KEY_TILE', '32')
    baseline = backward(dy, cache)
    monkeypatch.setenv('MINI_LLM_HIP_LOCAL_KV_KEY_TILE', '64')
    tiled = backward(dy, cache)
    repeated = backward(dy, cache)
    owned = np.arange(16) != cache.skip_head
    for index, (want, got, again) in enumerate(zip(baseline, tiled, repeated)):
        assert want.shape == got.shape == again.shape
        assert all(str(x.dtype) == 'float32' for x in (want, got, again))
        want, got, again = map(cp.asnumpy, (want, got, again))
        if index == 0:
            # A skipped dQ head belongs to the caller and is unwritten.
            want, got, again = (x[:, :, owned] for x in (want, got, again))
            np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(got, again)


@pytest.mark.parametrize('batch,length', [(2, 65), (1, 641)])
@pytest.mark.parametrize('skip_head', [-1, 0, 5])
def test_key_tile_parity_packed(monkeypatch, hip_local, batch, length, skip_head):
    cp, forward, backward = hip_local
    rng = np.random.default_rng(407 + length)
    # Packed offsets plus noncontiguous token and head strides for Q/K/V.
    packed = cp.asarray(
        rng.normal(0, 0.2, (batch, length * 2, 48, 64)), dtype=cp.float32,
    ).astype('bfloat16')[:, ::2, ::2]
    q, k, v = packed[:, :, :16], packed[:, :, 16:20], packed[:, :, 20:24]
    dy = cp.asarray(
        rng.normal(0, 0.1, (batch, length * 2, 32, 64)), dtype=cp.float32,
    )[:, ::2, ::2]
    docs = cp.zeros((batch, length * 2), dtype=cp.int32)[:, ::2]
    # Boundaries inside key tiles, including one near the 641-token tail.
    docs[:, 17:] = 1
    docs[:, 599:] = 2
    _, cache = forward(
        q, k, v, docs, long_window=4096, long_heads=3, skip_head=skip_head,
    )
    _assert_tile_parity(monkeypatch, cp, backward, dy, cache)


def test_key_tile_parity_complete_long_sequence(monkeypatch, hip_local):
    cp, forward, backward = hip_local
    rng = np.random.default_rng(8192)
    q, k, v = (
        cp.asarray(rng.normal(0, 0.2, (1, 8192, heads, 64)), dtype=cp.float32)
        .astype('bfloat16')
        for heads in (16, 4, 4)
    )
    dy = cp.asarray(rng.normal(0, 0.1, q.shape), dtype=cp.float32)
    docs = cp.zeros((1, 8192), dtype=cp.int32)
    _, cache = forward(
        q, k, v, docs, long_window=4096, long_heads=1, skip_head=-1,
    )
    # Compare every token/head, including both sides of the window cutoff.
    _assert_tile_parity(monkeypatch, cp, backward, dy, cache)


@pytest.mark.parametrize('tile', ['0', '16', '128', 'invalid'])
def test_invalid_key_tile_raises(monkeypatch, hip_local, tile):
    cp, forward, backward = hip_local
    q = cp.zeros((1, 1, 16, 64), dtype=cp.float32).astype('bfloat16')
    k = cp.zeros((1, 1, 4, 64), dtype=cp.float32).astype('bfloat16')
    v = cp.zeros_like(k)
    dy = cp.zeros(q.shape, dtype=cp.float32)
    docs = cp.zeros((1, 1), dtype=cp.int32)
    _, cache = forward(q, k, v, docs)
    monkeypatch.setenv('MINI_LLM_HIP_SPLIT_KV_BACKWARD', '2')
    monkeypatch.setenv('MINI_LLM_HIP_LOCAL_KV_KEY_TILE', tile)
    with pytest.raises(ValueError):
        backward(dy, cache)
