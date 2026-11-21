"""Placement-independent paired projection callback contract, CPU fixture."""
import os
import numpy as np
import pytest

if os.environ.get('MINI_LLM_BACKEND', 'numpy').lower() != 'numpy':
    pytest.skip('NumPy callback contract tests', allow_module_level=True)

from test_0207_qwen4exp_mixer import fixture
from mini_llm.external.qwen4exp_mixer import Qwen4ExpRecurrentMixerInference
from mini_llm.external.qwen4exp_weights import Qwen4ExpWeights


def test_pair_preserves_complete_mixer_and_state():
    spec, weights, hidden, original, state = fixture()
    _, _, _, _, paired_state = fixture()
    calls = []
    def pair(x):
        calls.append(x.shape)
        return x @ weights['qkv'].T, x @ weights['z'].T
    paired = Qwen4ExpRecurrentMixerInference(spec, **weights, qkv_z_projection=pair)
    def forbidden(_):
        raise AssertionError('Individual QKV/Z called despite explicit pair')
    paired.projections['qkv'] = paired.projections['z'] = forbidden
    expected = original.forward(hidden, state)
    got = paired.forward(hidden, paired_state)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(paired_state.conv.history, state.conv.history)
    np.testing.assert_array_equal(paired_state.delta.memory, state.delta.memory)
    assert paired_state.length == state.length
    assert calls == [hidden.shape]


@pytest.mark.parametrize('bad', [True, 1, np.zeros(1)])
def test_pair_requires_callable(bad):
    spec, weights, _, _, _ = fixture()
    with pytest.raises(TypeError, match='callable'):
        Qwen4ExpRecurrentMixerInference(spec, **weights, qkv_z_projection=bad)


def test_pair_output_validation_precedes_state_update():
    spec, weights, hidden, _, state = fixture()
    initial = state.conv.history.copy()
    paired = Qwen4ExpRecurrentMixerInference(spec, **weights,
        qkv_z_projection=lambda x: (x, x))
    with pytest.raises(ValueError, match='paired qkv'):
        paired.forward(hidden, state)
    np.testing.assert_array_equal(state.conv.history, initial)


def test_numpy_never_materializes_gpu_pair(monkeypatch):
    monkeypatch.setenv('MINI_LLM_LINEAR_GPU_Q8', '1')
    assert Qwen4ExpWeights.paired_projection(lambda x: x, lambda x: x) is None
