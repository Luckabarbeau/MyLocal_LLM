"""Retire only dead-request graph bindings before admitting replacement state."""
from types import SimpleNamespace
import threading

import pytest

from mini_llm.external import qwen4exp_model as module


def fixture_model(monkeypatch, *, live=False, bound=True):
    events = []
    graph = SimpleNamespace(bound_arrays=lambda: (object(),) if bound else (),
                            close=lambda: events.append('retire_graph'))
    segment = SimpleNamespace(graph=graph)
    model = module.Qwen4ExpGGUFModel.__new__(module.Qwen4ExpGGUFModel)
    model._call_lock = threading.Lock()
    model._states = {object()} if live else set()
    model._pre_moe_segments = {0: segment}
    model._qsa_block_segments = {}
    monkeypatch.setattr(module, 'BACKEND_NAME', 'cupy')
    monkeypatch.setattr(module, 'xp', SimpleNamespace(get_default_memory_pool=lambda:
        SimpleNamespace(free_all_blocks=lambda: events.append('release_unused'))))
    return model, segment, events


def test_orphaned_bindings_retire_before_unused_storage_release(monkeypatch):
    model, segment, events = fixture_model(monkeypatch)
    model._retire_orphaned_state_graphs()
    assert segment.graph is None
    assert events == ['retire_graph', 'release_unused']
    assert model._call_lock.acquire(blocking=False)
    model._call_lock.release()
    model._retire_orphaned_state_graphs()
    assert events == ['retire_graph', 'release_unused']


@pytest.mark.parametrize('live,bound', [(True, True), (True, False), (False, False)])
def test_live_requests_and_unbound_graphs_are_not_retired(monkeypatch, live, bound):
    model, segment, events = fixture_model(monkeypatch, live=live, bound=bound)
    model._retire_orphaned_state_graphs()
    assert segment.graph is not None and not events


def test_qsa_bindings_use_same_retirement_lifecycle(monkeypatch):
    model, segment, events = fixture_model(monkeypatch)
    model._pre_moe_segments = {}
    model._qsa_block_segments = {3: segment}
    model._retire_orphaned_state_graphs()
    assert segment.graph is None
    assert events == ['retire_graph', 'release_unused']


def test_concurrent_forward_rejects_retirement(monkeypatch):
    model, segment, events = fixture_model(monkeypatch)
    model._call_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='during forward'):
            model._retire_orphaned_state_graphs()
        assert segment.graph is not None and not events
    finally:
        model._call_lock.release()


def test_retirement_failure_does_not_release_unretired_storage(monkeypatch):
    model, segment, events = fixture_model(monkeypatch)
    def fail():
        raise RuntimeError('retirement failure')
    segment.graph.close = fail
    with pytest.raises(RuntimeError, match='retirement failure'):
        model._retire_orphaned_state_graphs()
    assert segment.graph is not None and not events
    assert model._call_lock.acquire(blocking=False)
    model._call_lock.release()


def test_cpu_model_is_unchanged(monkeypatch):
    model, segment, events = fixture_model(monkeypatch)
    monkeypatch.setattr(module, 'BACKEND_NAME', 'numpy')
    model._retire_orphaned_state_graphs()
    assert segment.graph is not None and not events
