"""General projected attention geometry/cache contract and shared native math."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.attention_inference import GQAAttentionInference
from mini_llm.ops.projected_attention_inference import (
    ProjectedGQAAttentionInference, grouped_attention_forward,
)


def scalar_attention(query, key, value, scale):
    """Independent FP64 causal oracle, with scalar batch/head/time loops."""
    b, time, hq, _ = query.shape
    hkv, dv = value.shape[2:]
    out = np.empty((b, time, hq, dv), dtype=np.float64)
    for row in range(b):
        for head in range(hq):
            kv = head // (hq // hkv)
            for t in range(time):
                scores = np.array([np.dot(query[row, t, head].astype(np.float64),
                    key[row, j, kv].astype(np.float64)) * scale for j in range(t + 1)])
                probs = np.exp(scores - scores.max())
                probs /= probs.sum()
                out[row, t, head] = sum(probs[j] * value[row, j, kv].astype(np.float64)
                                       for j in range(t + 1))
    return out


def inputs(dtype=np.float64):
    rng = np.random.default_rng(10614)
    return [rng.normal(0, .3, shape).astype(dtype) for shape in
            ((2, 7, 6, 5), (2, 7, 2, 5), (2, 7, 2, 3))]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_independent_dimensions_causal_scalar_reference_and_streaming(dtype):
    raw = inputs(dtype)
    op = ProjectedGQAAttentionInference(6, 2, 5, 3, dtype=dtype)
    expected = scalar_attention(*raw, op.scale)
    arrays = [xp.asarray(a) for a in raw]
    tolerance = 2e-6 if dtype == np.float32 else 2e-14
    for counts in ((7,), (3, 2, 2), (1,) * 7):
        state = op.create_state(2, 9)
        values, start = [], 0
        for count in counts:
            values.append(op.forward(*(a[:, start:start+count] for a in arrays), state))
            start += count
        output = asnumpy(xp.concatenate(values, axis=1))
        np.testing.assert_allclose(output, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_array_equal(asnumpy(state.k_cache[:, :, :7]), raw[1].transpose(0, 2, 1, 3))
        np.testing.assert_array_equal(asnumpy(state.v_cache[:, :, :7]), raw[2].transpose(0, 2, 1, 3))
        np.testing.assert_array_equal(asnumpy(state.k_cache[:, :, 7:]), 0)
        assert state.length == 7
        assert state.k_cache.shape == (2, 2, 9, 5) and state.v_cache.shape == (2, 2, 9, 3)


def test_no_future_leakage_and_grouped_not_tiled_heads():
    raw = inputs(np.float64)
    op = ProjectedGQAAttentionInference(6, 2, 5, 3, dtype="float64")
    base = asnumpy(op.forward(*(xp.asarray(a) for a in raw), op.create_state(2, 9)))
    changed = [a.copy() for a in raw]
    changed[1][:, 4:] += 100
    changed[2][:, 4:] -= 100
    actual = asnumpy(op.forward(*(xp.asarray(a) for a in changed), op.create_state(2, 9)))
    np.testing.assert_array_equal(actual[:, :4], base[:, :4])
    # At the first token softmax has one key, so its value identifies head mapping.
    for head in range(6):
        np.testing.assert_array_equal(base[:, 0, head], raw[2][:, 0, head // 3])


def test_native_attention_delegates_same_grouped_kernel_and_dtype_policy(monkeypatch):
    import mini_llm.ops.attention_inference as module
    native = GQAAttentionInference(24, 4, 2, 6, dtype="float32")
    rng = np.random.default_rng(4)
    q = xp.asarray(rng.normal(size=(2, 3, 4, 6)).astype(np.float32))
    k = xp.asarray(rng.normal(size=(2, 2, 3, 6)).astype(np.float32))
    v = xp.asarray(rng.normal(size=(2, 2, 3, 6)).astype(np.float32))
    mask = xp.asarray(np.tri(3, dtype=bool))
    calls = []
    def capture(*args, **kwargs):
        calls.append(kwargs)
        return grouped_attention_forward(*args, **kwargs)
    monkeypatch.setattr(module, "grouped_attention_forward", capture)
    actual = native._attention(q, k, v, mask)
    expected = scalar_attention(asnumpy(q), asnumpy(k).transpose(0, 2, 1, 3),
                                 asnumpy(v).transpose(0, 2, 1, 3), native.scale)
    np.testing.assert_allclose(asnumpy(actual), expected, rtol=2e-6, atol=2e-6)
    assert len(calls) == 1 and calls[0]["probability_dtype"] == "float32"
    assert "softmax_dtype" not in calls[0]  # Native default stays FP32.


def test_copy_reset_empty_and_invalid_calls_preserve_cache():
    op = ProjectedGQAAttentionInference(6, 2, 5, 3)
    arrays = [xp.asarray(a) for a in inputs(np.float32)]
    state = op.create_state(2, 8)
    op.forward(*(a[:, :3] for a in arrays), state)
    copied = op.copy_state(state)
    copied.k_cache.fill(9)
    assert not np.array_equal(asnumpy(copied.k_cache), asnumpy(state.k_cache))
    k_before, v_before = asnumpy(state.k_cache).copy(), asnumpy(state.v_cache).copy()
    empty = op.forward(*(a[:, :0] for a in arrays), state)
    assert empty.shape == (2, 0, 6, 3) and state.length == 3
    for malformed in (arrays[0][:, :, :1], arrays[0].astype(xp.int32)):
        with pytest.raises((ValueError, TypeError)):
            op.forward(malformed, arrays[1], arrays[2], state)
    with pytest.raises(ValueError, match="capacity"):
        op.forward(*arrays, state)
    np.testing.assert_array_equal(asnumpy(state.k_cache), k_before)
    np.testing.assert_array_equal(asnumpy(state.v_cache), v_before)
    assert state.length == 3
    k_id, v_id = id(state.k_cache), id(state.v_cache)
    state.reset()
    assert state.length == 0 and id(state.k_cache) == k_id and id(state.v_cache) == v_id
    result = op.forward(*(a[:, :1] for a in arrays), state)
    np.testing.assert_array_equal(asnumpy(result), np.repeat(inputs(np.float32)[2][:, :1], 3, axis=2))


@pytest.mark.parametrize("kwargs", [dict(n_q_heads=5), dict(key_dim=0), dict(dtype="int8"),
                                     dict(value_dim=False), dict(scale=np.inf)])
def test_invalid_configuration(kwargs):
    params = dict(n_q_heads=6, n_kv_heads=2, key_dim=5, value_dim=3)
    params.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        ProjectedGQAAttentionInference(**params)


@pytest.mark.gpu
@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="Requires CuPy")
def test_nondefault_stream_matches_fp64_without_native_kv_replication():
    raw = inputs(np.float32)
    op = ProjectedGQAAttentionInference(6, 2, 5, 3)
    stream = xp.cuda.Stream(non_blocking=True)
    with stream:
        state = op.create_state(2, 9)
        arrays = [xp.asarray(a) for a in raw]
        result = op.forward(*arrays, state)
        host = result.get(stream=stream)
    np.testing.assert_allclose(host, scalar_attention(*raw, op.scale), rtol=2e-6, atol=2e-6)
    assert state.k_cache.shape[1] == state.v_cache.shape[1] == 2
