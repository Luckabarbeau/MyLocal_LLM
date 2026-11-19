"""Whole recurrent segment capture: arithmetic, explicit state and ownership."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.external.qwen4exp_mixer import Qwen4ExpMixerSpec, Qwen4ExpRecurrentMixerInference

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CUDA graph execution')


def mixers(fast=True):
    rng = np.random.default_rng(265)
    spec = Qwen4ExpMixerSpec(64, 2, 4, 16, 4, 1e-5)
    def rand(shape):
        return xp.asarray(rng.normal(0, .1, shape).astype(np.float32))
    weights = dict(qkv=rand((spec.qkv_channels, 64)), z=rand((spec.inner_size, 64)),
        beta=rand((4, 64)), alpha=rand((4, 64)), out=rand((64, spec.inner_size)),
        conv_weight=rand((spec.qkv_channels, 4)), norm_weight=xp.ones(16, xp.float32),
        ssm_a=-xp.exp(rand((4,))), dt_bias=rand((4,)), fast_coefficients=fast)
    # Compare replay against the same capture-capable projection arithmetic,
    # not cuBLAS's distinct FP32 summation tree.
    from mini_llm.ops.inference_matmul import matmul_inference
    for name in ('qkv', 'z', 'beta', 'alpha', 'out'):
        weight = weights[name]
        weights[name] = lambda x, weight=weight: matmul_inference(x, weight.T)
    old = Qwen4ExpRecurrentMixerInference(spec, **weights)
    new = Qwen4ExpRecurrentMixerInference(spec, **weights, graph_decode=True)
    old.mixer._decode_fastpath_enabled = new.mixer._decode_fastpath_enabled = True
    return old, new, rng


@pytest.mark.parametrize('fast', [False, True])
def test_sequence_and_interleaved_sessions_exact(fast):
    old, new, rng = mixers(fast)
    old_states = [old.create_state(1), old.create_state(1)]
    new_states = [new.create_state(1), new.create_state(1)]
    for i in range(32):
        session = i % 2
        x = xp.asarray(rng.normal(0, .2, (1, 64)).astype(np.float32))
        reference = old.step(x, old_states[session])
        got = new.step(x, new_states[session])
        np.testing.assert_array_equal(got.get(), reference.get())
        for attr, data in (('conv', 'history'), ('delta', 'memory')):
            a, b = getattr(new_states[session], attr), getattr(old_states[session], attr)
            assert a.length == b.length == i//2+1
            np.testing.assert_array_equal(getattr(a, data).get(), getattr(b, data).get())
    assert new.graph_statistics()['calls'] == 32
    assert new.graph_statistics()['pool_reserved_bytes'] <= 16*1024**2
    fresh_old, fresh_new = old.create_state(1), new.create_state(1)
    x = xp.zeros((1, 64), xp.float32)
    np.testing.assert_array_equal(new.step(x, fresh_new).get(), old.step(x, fresh_old).get())


def test_capture_failure_does_not_advance_caller_state():
    _, new, _ = mixers()
    state = new.create_state(1)
    state.conv.history.fill(.1)
    state.delta.memory.fill(.2)
    history, memory = state.conv.history.copy(), state.delta.memory.copy()
    def fail(x):
        raise RuntimeError('output projection failed after private recurrence')
    new.projections['out'] = fail
    with pytest.raises(RuntimeError, match='output projection failed'):
        new.step(xp.ones((1, 64), xp.float32), state)
    np.testing.assert_array_equal(state.conv.history.get(), history.get())
    np.testing.assert_array_equal(state.delta.memory.get(), memory.get())
    assert state.conv.length == state.delta.length == 0
    assert new._decode_graph is None


def test_no_global_pending_refs_from_graph_and_stream_binding():
    from mini_llm.ops.gated_delta_decode import _PENDING
    _, new, _ = mixers()
    existing_keys = set(_PENDING)
    state = new.create_state(1)
    stream = xp.cuda.Stream(non_blocking=True)
    with stream:
        x = xp.ones((1, 64), xp.float32)
        new.step(x, state)
        stream.synchronize()
    capture_key = (xp.cuda.runtime.getDevice(), new._decode_graph.capture_stream.ptr)
    assert capture_key not in _PENDING or capture_key in existing_keys
    with pytest.raises(ValueError, match='replay stream'):
        new.step(xp.ones((1, 64), xp.float32), state)
    assert state.length == 1


def test_graph_state_metadata_rejected_before_replay():
    _, new, _ = mixers()
    state = new.create_state(1)
    state.conv.length = 1
    with pytest.raises(ValueError, match='length mismatch'):
        new.step(xp.ones((1, 64), xp.float32), state)
    assert new.graph_statistics()['calls'] == 0
