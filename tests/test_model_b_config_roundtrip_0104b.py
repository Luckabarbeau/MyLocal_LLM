"""Canonical architecture serialization checks for the production Model B."""
from dataclasses import asdict
import json
from dataclasses import replace
import pytest

from mini_llm.config import DualResolutionConfig, ModelConfig
from mini_llm.model_contracts import ArchitectureSpec


def test_model_b_exact_json_config_roundtrip():
    original = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    loaded = ModelConfig(**json.loads(json.dumps(asdict(original))))
    assert loaded == original
    assert isinstance(loaded.dual_resolution.sparse_retrieval_layers, tuple)
    assert ArchitectureSpec.from_config(loaded) == ArchitectureSpec.from_config(original)
    assert loaded.dual_resolution.sparse_retrieval_layers == tuple(range(2, 18))


def test_reject_invalid_sparse_layer_sequences():
    with pytest.raises(TypeError, match="list or tuple"):
        DualResolutionConfig(sparse_retrieval_layers="2,3")
    with pytest.raises(ValueError, match="duplicate"):
        DualResolutionConfig(sparse_retrieval_layers=[2, 2])
