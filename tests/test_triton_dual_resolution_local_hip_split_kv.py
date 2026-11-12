"""Focused HIP split-query-head KV backward schedule parity."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME

pytestmark = pytest.mark.skipif(
    BACKEND_NAME != 'cupy', reason='HIP Triton parity requires CuPy backend',
)


@pytest.mark.parametrize('batch,length', [(2, 65), (1, 641)])
@pytest.mark.parametrize('skip_head', [-1, 0, 5])
def test_split_kv_matches_baseline_fp32_packed(monkeypatch, batch, length, skip_head):
    pytest.importorskip('triton')
    cp = pytest.importorskip('cupy')
    if not cp.cuda.runtime.is_hip or not cp.cuda.is_available():
        pytest.skip('HIP device required')
    from mini_llm.ops.triton_dual_resolution_local import (
        local_attention_forward, local_attention_backward,
    )

    rng = np.random.default_rng(407 + length)
    # Exercise noncontiguous token/head strides as well as packed Q/K/V offsets.
    packed = cp.asarray(
        rng.normal(0, 0.2, (batch, length * 2, 48, 64)),
        dtype=cp.float32,
    ).astype('bfloat16')[:, ::2, ::2]
    q, k, v = packed[:, :, :16], packed[:, :, 16:20], packed[:, :, 20:24]
    dy = cp.asarray(
        rng.normal(0, 0.1, (batch, length * 2, 16, 64)), dtype=cp.float32,
    )[:, ::2]
    docs = cp.zeros((batch, length * 2), dtype=cp.int32)[:, ::2]
    docs[:, 17:] = 1
    docs[:, 599:] = 2
    _, cache = local_attention_forward(
        q, k, v, docs, long_window=4096, long_heads=3, skip_head=skip_head,
    )

    monkeypatch.setenv('MINI_LLM_HIP_SPLIT_KV_BACKWARD', '0')
    baseline = local_attention_backward(dy, cache)
    monkeypatch.setenv('MINI_LLM_HIP_SPLIT_KV_BACKWARD', '1')
    split = local_attention_backward(dy, cache)
    repeated = local_attention_backward(dy, cache)
    owned = np.arange(16) != skip_head
    for index, (want, got, again) in enumerate(zip(baseline, split, repeated)):
        assert str(got.dtype) == 'float32'
        want, got, again = map(cp.asnumpy, (want, got, again))
        if index == 0:
            # Skipped dQ belongs to the caller and is intentionally unwritten.
            want, got, again = (x[:, :, owned] for x in (want, got, again))
            np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_allclose(got, want, atol=2e-6, rtol=2e-5)
        np.testing.assert_array_equal(got, again)
