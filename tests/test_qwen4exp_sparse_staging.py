"""Sparse staging reuses KV groups without changing per-query arithmetic."""
import numpy as np

from mini_llm.backend import asnumpy, xp
from mini_llm.external.qwen4exp_attention import Qwen4ExpAttention
from mini_llm.external import qwen4exp_hoststate
from test_0230_qsa_gpu_decode import fixture


def test_dead_index_is_uploaded_once_and_reset_invalidates(monkeypatch):
    config, weights, rng = fixture()
    model = Qwen4ExpAttention(config, **weights, qsa_state_storage='host')
    state = model.create_state(24)
    calls = []
    original = qwen4exp_hoststate.upload

    def upload(value):
        if value.shape == (config.index_dim,):
            calls.append(value.copy())
        return original(value)

    monkeypatch.setattr(qwen4exp_hoststate, 'upload', upload)
    hidden = xp.asarray(rng.normal(size=(24, config.d_model)), dtype=xp.float32)
    for i in range(12):
        model.step(hidden[i], state, position=i)
    assert len(calls) == 1
    cached = state.dead_index
    for i in range(12, 24):
        model.step(hidden[i], state, position=i)
        assert state.dead_index is cached
    assert len(calls) == 1
    model.reset(state)
    assert state.dead_index is None
    fresh = model.create_state(24)
    for i in range(8):
        actual = model.step(hidden[-1-i], state, position=i)
        expected = model.step(hidden[-1-i], fresh, position=i)
        np.testing.assert_array_equal(asnumpy(actual), asnumpy(expected))
    assert len(calls) == 3


def test_selected_group_matrices_have_reference_head_outputs():
    config, weights, rng = fixture()
    model = Qwen4ExpAttention(config, **weights, qsa_state_storage='host')
    state = model.create_state(27)
    for t in range(27):
        hidden = xp.asarray(rng.normal(size=config.d_model), dtype=xp.float32)
        actual, diagnostic = model.step(hidden, state, position=t, diagnostics=True)
        qg = model._project('query_gate', hidden).reshape(config.n_q_heads, 2*config.head_dim)
        q = model._rotate(model.norms[0].forward(qg[:, :config.head_dim]), t)
        ids = asnumpy(diagnostic['selected_ids'])
        contexts = []
        scale = xp.float32(config.attention_scale or 1/np.sqrt(config.head_dim))
        for h in range(config.n_q_heads):
            kh = h // (config.n_q_heads // config.n_kv_heads)
            # Original contiguous per-head copy, independent of grouped staging.
            keys = xp.array(state.keys[ids, kh], dtype=xp.float32, copy=True)
            values = xp.array(state.values[ids, kh], dtype=xp.float32, copy=True)
            logits = (keys @ q[h]) * scale
            probability = xp.exp(logits-xp.max(logits))
            probability /= xp.sum(probability, dtype=xp.float32)
            contexts.append(probability @ values)
        from mini_llm.external.qwen4exp_hyperconnection import _sigmoid
        gated = xp.stack(contexts) * _sigmoid(qg[:, config.head_dim:])
        expected = model._project('output', gated.reshape(-1))
        np.testing.assert_array_equal(asnumpy(actual), asnumpy(expected))
