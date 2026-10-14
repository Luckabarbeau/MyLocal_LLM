"""0080O12: cap shape-dependent metadata while preserving route geometry."""
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import asnumpy
from mini_llm.config import ContextRouterConfig
from mini_llm.ops.context_router import ContextRouter
from mini_llm.diagnostics.gpu_cache_audit import persistent_gpu_cache_inventory


def _router():
    from mini_llm.backend import RandomStream
    return ContextRouter(
        d_model=16,
        config=ContextRouterConfig(
            history_block_size=4, routing_stride=4, query_window=8,
            top_k_blocks=2, exclude_recent_tokens=4,
            router_dim=4, num_queries=1,
        ),
        rng=RandomStream(12), dtype="float32",
    )


def _snapshot(meta):
    return {key: asnumpy(value).copy() if hasattr(value, 'shape') else value
            for key, value in meta.items()}


def test_per_layer_cache_is_lru_bounded_and_geometry_exact(monkeypatch):
    monkeypatch.setenv('MINI_LLM_ROUTER_METADATA_CACHE_ENTRIES', '2')
    router = _router()
    original = _snapshot(router._routing_metadata(28))
    router._routing_metadata(32)
    router._routing_metadata(28)  # refresh 28 recency
    router._routing_metadata(36)
    assert list(router._routing_metadata_cache) == [28, 36]
    assert 32 not in router._routing_metadata_cache
    for name, ref in original.items():
        np.testing.assert_array_equal(asnumpy(router._routing_metadata(28)[name])
                                      if hasattr(ref, 'shape') else router._routing_metadata(28)[name],
                                      ref)
    router._routing_metadata(32)  # reconstruct the evicted metadata
    assert len(router._routing_metadata_cache) == 2


def test_zero_entry_setting_disables_metadata_retention(monkeypatch):
    monkeypatch.setenv('MINI_LLM_ROUTER_METADATA_CACHE_ENTRIES', '0')
    router = _router()
    router._routing_metadata(28)
    assert len(router._routing_metadata_cache) == 0
    router._routing_metadata(36)
    assert len(router._routing_metadata_cache) == 0


@pytest.mark.parametrize('setting', ['-1', 'unlimited'])
def test_reject_invalid_cache_limit(monkeypatch, setting):
    monkeypatch.setenv('MINI_LLM_ROUTER_METADATA_CACHE_ENTRIES', setting)
    with pytest.raises(ValueError, match='router metadata cache entries'):
        _router()


def test_gpu_inventory_includes_per_layer_retrieval_caches(monkeypatch):
    monkeypatch.setenv('MINI_LLM_ROUTER_METADATA_CACHE_ENTRIES', '2')
    a, b = _router(), _router()
    a._routing_metadata(28)
    a._routing_metadata(32)
    b._routing_metadata(28)
    modules = [SimpleNamespace(router=a), SimpleNamespace(router=b)]
    attn = SimpleNamespace(_retrieval_groups=[{'module': mod} for mod in modules])
    # Same first router appears twice: do not double count shared groups.
    attn._retrieval_groups.append({'module': modules[0]})
    block = SimpleNamespace(attention=attn)
    fake = SimpleNamespace(blocks=[block])
    inventory = persistent_gpu_cache_inventory(fake)
    assert inventory['layer_retrieval_routing_metadata']['entries'] == 3
    # NumPy backend cannot have CUDA arrays, but cache-entry count is correct.
