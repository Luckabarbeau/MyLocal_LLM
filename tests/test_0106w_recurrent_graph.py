"""Default graph replay must preserve explicit state and fresh output ownership."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.inference_graph import CapturedArrayInference


@pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='CPU rejection contract')
def test_cpu_rejects_graph_without_mutation():
    x, s = np.ones((1, 4), np.float32), np.zeros((1, 4), np.float32)
    with pytest.raises(RuntimeError, match='CuPy'):
        CapturedArrayInference(lambda a, state: a, x, (s,))
    np.testing.assert_array_equal(s, 0)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Requires CUDA graphs')
def test_private_capture_state_replay_outputs_and_stream_contract():
    x = xp.ones((1, 32), dtype=xp.float32)
    state = xp.full((1, 32), 2, dtype=xp.float32)
    def enqueue(a, states):
        states[0][...] += a * 2
        return states[0] * 3
    graph = CapturedArrayInference(enqueue, x, (state,))
    np.testing.assert_array_equal(asnumpy(state), 2)
    first = graph.forward(x, (state,))
    second = graph.forward(x * 2, (state,))
    np.testing.assert_array_equal(asnumpy(first), 12)
    np.testing.assert_array_equal(asnumpy(second), 24)
    np.testing.assert_array_equal(asnumpy(state), 8)
    assert graph.statistics()['calls'] == 2
    with xp.cuda.Stream(non_blocking=True):
        with pytest.raises(ValueError, match='replay stream'):
            graph.forward(x, (state,))
    assert not graph.matches(x[:, ::2], (state,))


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Requires CUDA graphs')
def test_failed_graph_allocation_never_changes_caller_state():
    x = xp.ones((1, 32), dtype=xp.float32)
    state = xp.full((1, 32), 2, dtype=xp.float32)
    with pytest.raises(xp.cuda.memory.OutOfMemoryError):
        CapturedArrayInference(lambda a, states: states[0].fill(0), x, (state,), pool_limit_bytes=1)
    np.testing.assert_array_equal(asnumpy(state), 2)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Requires CUDA graphs')
def test_full_tiny_hybrid_model_graph_parity_reset_fork_and_new_requests(tmp_path, monkeypatch):
    from benchmarks.fixture_0106q_qwen_reference import write_fixture
    from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
    from mini_llm.external.qwen3next_model import Qwen3NextGGUFModel
    fixture = write_fixture(tmp_path / 'fixture')
    index = Qwen3NextGGUFIndex.open(fixture['model_path'])
    monkeypatch.setenv('MINI_LLM_RECURRENT_GRAPH', '0')
    reference = Qwen3NextGGUFModel.from_index(index, expert_cache_budget_bytes=1024**2)
    monkeypatch.delenv('MINI_LLM_RECURRENT_GRAPH', raising=False)
    graph_model = Qwen3NextGGUFModel.from_index(index, expert_cache_budget_bytes=1024**2)
    try:
        assert all(not b.mixer._graph_decode_enabled for b in reference.blocks if not b.is_attention)
        assert all(b.mixer._graph_decode_enabled for b in graph_model.blocks if not b.is_attention)
        for request in range(2):
            a = reference.create_generation_state(1, 20)
            b = graph_model.create_generation_state(1, 20)
            ids = xp.asarray([[256, 32, 97]], dtype=xp.int32)
            np.testing.assert_allclose(asnumpy(graph_model.prefill(ids, b)),
                asnumpy(reference.prefill(ids, a)), rtol=4e-4, atol=8e-6)
            for token in (10, 47, 219, 210, 204):
                actual = graph_model.decode_one(xp.asarray([token], xp.int32), b)
                expected = reference.decode_one(xp.asarray([token], xp.int32), a)
                np.testing.assert_allclose(asnumpy(actual), asnumpy(expected), rtol=4e-4, atol=8e-6)
                assert a.length == b.length
            fork_a, fork_b = reference.copy_generation_state(a), graph_model.copy_generation_state(b)
            expected = reference.decode_one(xp.asarray([32], xp.int32), fork_a)
            actual = graph_model.decode_one(xp.asarray([32], xp.int32), fork_b)
            np.testing.assert_allclose(asnumpy(actual), asnumpy(expected), rtol=4e-4, atol=8e-6)
            for block, ls in zip(graph_model.blocks, b.layer_states):
                if not block.is_attention:
                    assert block.mixer._decode_graph.statistics()['calls'] >= 6
                    assert ls.mixer_state.conv.length == ls.mixer_state.delta.length == b.length
            a.reset(); b.reset()
            expected = reference.prefill(ids, a)
            actual = graph_model.prefill(ids, b)
            np.testing.assert_allclose(asnumpy(actual), asnumpy(expected), rtol=4e-4, atol=8e-6)
            counts = [block.mixer._decode_graph.calls for block in graph_model.blocks if not block.is_attention]
            ready = xp.cuda.Event(disable_timing=True)
            ready.record(xp.cuda.get_current_stream())
            with xp.cuda.Stream(non_blocking=True) as other:
                other.wait_event(ready)
                expected = reference.decode_one(xp.asarray([32], xp.int32), a)
                actual = graph_model.decode_one(xp.asarray([32], xp.int32), b)
                other.synchronize()
                np.testing.assert_allclose(asnumpy(actual), asnumpy(expected), rtol=4e-4, atol=8e-6)
            assert counts == [block.mixer._decode_graph.calls for block in graph_model.blocks if not block.is_attention]
    finally:
        graph_model.close()
        reference.close()
