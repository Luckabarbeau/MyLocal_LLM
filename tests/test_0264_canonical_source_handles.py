"""Reuse metadata only, never duplicate expert payloads or bypass closed stores."""
import gc
import weakref

import numpy as np
import pytest

from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
from mini_llm.ops.expert_residency import ExpertKey
from test_0204_sparse_iq_execution import sparse_store


def test_source_identity_zero_copy_and_explicit_retirement(tmp_path, monkeypatch):
    store = sparse_store(tmp_path)
    index = Qwen4ExpGGUFIndex(store)
    original = store.rows
    calls = []
    def tracked(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(store, 'rows', tracked)
    key = ExpertKey(index.model_id, 0, 0)
    bundle = index.source(key)
    for _ in range(1000):
        assert index.source(key) is bundle
        assert index.provider(key) is bundle.weights
    assert len(calls) == 3
    for w, name in zip(bundle.weights, ('gate', 'up', 'down')):
        raw = original(f'blk.0.ffn_{name}_exps.weight', 0, w.shape[0], outer=(0,))
        assert np.shares_memory(w.data, raw)
        assert not w.data.flags.writeable
    weak = weakref.ref(bundle)
    index.clear_source_cache()
    del bundle
    gc.collect()
    assert weak() is None
    bundle = index.source(key)
    assert len(calls) == 6
    store.close()
    with pytest.raises(RuntimeError, match='closed'):
        index.source(key)
    assert bundle.weights[0].data.size > 0  # Exported views remain valid.
    index.clear_source_cache()


def test_opt_out_and_namespace_validation(tmp_path):
    with sparse_store(tmp_path) as store:
        index = Qwen4ExpGGUFIndex(store, cache_source_handles=False)
        key = ExpertKey(index.model_id, 47, 511)
        assert index.source(key) is not index.source(key)
        assert not index._sources
        for bad in (ExpertKey(index.model_id, 48, 0), ExpertKey(index.model_id, 0, 512)):
            with pytest.raises(IndexError):
                index.source(bad)
        with pytest.raises(ValueError, match='identity'):
            index.source(ExpertKey('other', 0, 0))
