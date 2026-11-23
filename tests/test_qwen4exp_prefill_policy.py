"""Unsupported prefill policies must fail rather than silently execute fallback."""
from types import SimpleNamespace

import pytest

from mini_llm.external.qwen4exp_prefill import _routed


@pytest.mark.parametrize('mode', ['strata', 'unknown', ''])
def test_removed_or_invalid_backend_cannot_silently_fallback(monkeypatch, mode):
    monkeypatch.setenv('MINI_LLM_PREFILL_EXPERTS', mode)
    def forbidden(*args):
        pytest.fail('Unsupported policy invoked model execution')
    experts = SimpleNamespace(expert_executor=None, forward=forbidden)
    with pytest.raises(ValueError, match='Unsupported prefill expert mode'):
        _routed(experts, None, None, None)


def test_reference_backend_preserves_dispatch(monkeypatch):
    monkeypatch.setenv('MINI_LLM_PREFILL_EXPERTS', 'decode')
    seen = []
    experts = SimpleNamespace(expert_executor=None,
        forward=lambda *args: seen.append(args) or 'result')
    assert _routed(experts, 'x', 'weights', 'ids') == 'result'
    assert seen == [('x', 'weights', 'ids')]
