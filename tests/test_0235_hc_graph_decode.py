"""Static HC graph replay, fresh ownership, capped pools and budget admission."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp, asnumpy
from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection
from test_0229_resident_budget import capacity, built_model

GPU = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='actual CuPy graph required')


def pair(d=31, rank=7, inject=True):
    rng = np.random.default_rng(235)
    arrays = [xp.asarray(rng.normal(1, .1, (4, d)), dtype=xp.float32),
              xp.asarray(rng.normal(0, .1/np.sqrt(d), (rank, 4*d)), dtype=xp.float32),
              xp.asarray(rng.normal(0, .1, (4*d, rank)), dtype=xp.float32)]
    if inject:
        arrays.append(xp.asarray(rng.normal(0, .1/np.sqrt(d), (4, 4*d)), dtype=xp.float32))
    return (Qwen4ExpHyperConnection(*arrays, epsilon=1e-6, fast_decode=True),
            Qwen4ExpHyperConnection(*arrays, epsilon=1e-6, fast_decode=True, graph_decode=True), rng)


def assert_close(a, b):
    np.testing.assert_allclose(asnumpy(a), asnumpy(b), atol=1e-6, rtol=2e-6)


@GPU
@pytest.mark.parametrize('inject', [True, False])
@pytest.mark.parametrize('d,rank', [(31, 7), (2560, 320)])
def test_replay_changes_input_fresh_outputs_and_bounded_pool(inject, d, rank):
    baseline, graph, rng = pair(d, rank, inject)
    saved = []
    for token in range(4):
        x = xp.asarray(rng.normal(size=(1, 4, d)), dtype=xp.float32)
        original = asnumpy(x).copy()
        actual, gains = graph.read(x)
        expected, expected_gains = baseline.read(x)
        assert_close(actual, expected)
        if inject:
            assert_close(gains, expected_gains)
            actual_write = graph.write(x, actual, gains)
            assert_close(actual_write, baseline.write(x, expected, expected_gains))
        else:
            assert gains is expected_gains is None
            assert_close(graph.collapse(x), baseline.collapse(x))
        np.testing.assert_array_equal(asnumpy(x), original)
        saved.append((actual, asnumpy(actual).copy()))
    for array, snapshot in saved:
        np.testing.assert_array_equal(asnumpy(array), snapshot)
    stats = graph.graph_statistics()
    assert stats['calls'] == (4 if inject else 8)
    assert 0 < stats['pool_reserved_bytes'] <= stats['pool_limit_bytes'] == 2*1024**2


@GPU
def test_other_shape_or_stream_falls_back_without_replacing_live_graph():
    baseline, graph, rng = pair()
    stream = xp.cuda.Stream(non_blocking=True)
    xp.cuda.get_current_stream().synchronize()
    with stream:
        x = xp.asarray(rng.normal(size=(1, 4, 31)), dtype=xp.float32)
        result = graph.read(x)[0]
        dependency = result * 2
    stream.synchronize()
    assert_close(dependency, baseline.read(x)[0] * 2)
    old_graph = graph._read_graph
    count = graph.graph_statistics()['calls']
    # Changed stream and a shape mismatch must not discard graph-owned buffers.
    for value in (x, x.reshape(4, 31), xp.repeat(x, 2, axis=0)):
        actual, gain = graph.read(value)
        expected, expected_gain = baseline.read(value)
        assert_close(actual, expected)
        assert_close(gain, expected_gain)
    assert graph._read_graph is old_graph
    assert graph.graph_statistics()['calls'] == count


@GPU
def test_undersized_private_pool_fails_without_mutating_input():
    _, graph, rng = pair()
    graph.graph_pool_bytes = 1
    x = xp.asarray(rng.normal(size=(4, 31)), dtype=xp.float32)
    original = asnumpy(x).copy()
    with pytest.raises(xp.cuda.memory.OutOfMemoryError):
        graph.read(x)
    assert graph._read_graph is None
    np.testing.assert_array_equal(asnumpy(x), original)


def test_graph_options_validate_and_numpy_is_reference_only():
    weights = [xp.ones((4, 3), dtype=xp.float32), xp.ones((2, 12), dtype=xp.float32),
               xp.ones((12, 2), dtype=xp.float32)]
    with pytest.raises(TypeError, match='graph_decode'):
        Qwen4ExpHyperConnection(*weights, epsilon=1e-6, graph_decode='yes')
    for budget in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='graph_pool_bytes'):
            Qwen4ExpHyperConnection(*weights, epsilon=1e-6, graph_pool_bytes=budget)
    if BACKEND_NAME == 'numpy':
        baseline, graph, _ = pair()
        x = xp.ones((4, 31), dtype=xp.float32)
        np.testing.assert_array_equal(graph.read(x)[0], baseline.read(x)[0])
        assert graph._read_graph is None


def test_graph_budget_is_reserved_and_replanning_credits_only_owned_pools(capacity):
    weights, memory, budgets = capacity
    initial = weights.memory_plan(16, **budgets)
    budgets['graph_workspace_bytes'] = 4096
    planned = weights.memory_plan(16, **budgets)
    assert planned['device_reserved_bytes'] == initial['device_reserved_bytes'] + 4096
    memory.free = planned['device_reserved_bytes'] - 1
    with pytest.raises(MemoryError, match='free device memory'):
        weights.memory_plan(16, **budgets)
    memory.free = 1 << 40
    model = built_model(weights, budgets)
    from types import SimpleNamespace
    model._hc_graph_owners = (SimpleNamespace(graph_statistics=lambda: {'pool_reserved_bytes': 1024}),)
    remaining = planned['device_reserved_bytes'] - planned['device_static_bytes'] - 1024
    memory.free = remaining
    assert model.memory_plan(1, 16)['graph_workspace_bytes'] == 4096
    memory.free -= 1
    with pytest.raises(MemoryError, match='free device memory'):
        model.memory_plan(1, 16)
