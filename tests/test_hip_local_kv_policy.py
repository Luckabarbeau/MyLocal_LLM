"""CPU-only coverage of the HIP Model B local-KV runtime policy."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mini_llm import backend
from mini_llm.model_registry import registry
from mini_llm.runtime_defaults import apply_runtime_defaults


KEY_TILE = "MINI_LLM_HIP_LOCAL_KV_KEY_TILE"
SPLIT_KV = "MINI_LLM_HIP_SPLIT_KV_BACKWARD"


@pytest.fixture(autouse=True)
def _isolated_environment():
    with patch.dict(os.environ):
        for name in tuple(os.environ):
            if name.startswith("MINI_LLM_"):
                del os.environ[name]
        yield


@pytest.fixture
def model_b():
    return registry.config("model_b_348m_sparse16")


@pytest.fixture
def mock_runtime(monkeypatch):
    def install(*, is_hip):
        monkeypatch.setattr(
            backend,
            "xp",
            SimpleNamespace(cuda=SimpleNamespace(
                runtime=SimpleNamespace(is_hip=is_hip)
            )),
        )
    return install


@pytest.mark.parametrize("precision", ["bf16-mixed", "bfloat16", "bf16"])
def test_hip_consumer_model_b_sparse_materializes_key_tile(model_b, mock_runtime, precision):
    mock_runtime(is_hip=True)
    assert model_b.dual_resolution.sparse_retrieval_stride == 16

    result = apply_runtime_defaults(
        "consumer-gpu", backend_name="cupy", config=model_b,
        training=True, precision=precision,
    )

    assert result.resolved == "consumer-gpu"
    assert os.environ[KEY_TILE] == "64"
    assert os.environ[SPLIT_KV] == "2"
    assert dict(result.applied)[KEY_TILE] == "64"
    assert dict(result.applied)[SPLIT_KV] == "2"


def test_explicit_key_tile_32_is_preserved_not_applied(model_b, mock_runtime):
    mock_runtime(is_hip=True)
    os.environ[KEY_TILE] = "32"

    result = apply_runtime_defaults(
        "consumer-gpu", backend_name="cupy", config=model_b,
        training=True, precision="bf16-mixed",
    )

    assert os.environ[KEY_TILE] == "32"
    assert KEY_TILE not in dict(result.applied)
    assert os.environ[SPLIT_KV] == "2"
    assert dict(result.applied)[SPLIT_KV] == "2"


def test_cuda_consumer_model_b_does_not_materialize_hip_policy(model_b, mock_runtime):
    mock_runtime(is_hip=False)

    result = apply_runtime_defaults(
        "consumer-gpu", backend_name="cupy", config=model_b,
        training=True, precision="bf16-mixed",
    )

    assert KEY_TILE not in os.environ
    assert SPLIT_KV not in os.environ
    assert KEY_TILE not in dict(result.applied)
    assert SPLIT_KV not in dict(result.applied)


@pytest.mark.parametrize(
    "profile,backend_name,training,precision",
    [
        ("consumer-gpu", "numpy", True, "bf16-mixed"),
        ("auto", "numpy", True, "bf16-mixed"),
        ("reference", "cupy", True, "bf16-mixed"),
        ("consumer-gpu", "cupy", False, "bf16-mixed"),
        ("gpu-fast", "cupy", True, "bf16-mixed"),
        ("consumer-gpu", "cupy", True, "float32"),
    ],
    ids=["numpy", "numpy-auto", "reference", "inference", "gpu-fast", "non-bf16"],
)
def test_other_runtime_paths_do_not_materialize_hip_policy(
    model_b, mock_runtime, profile, backend_name, training, precision,
):
    mock_runtime(is_hip=True)

    result = apply_runtime_defaults(
        profile, backend_name=backend_name, config=model_b,
        training=training, precision=precision,
    )

    assert KEY_TILE not in os.environ
    assert SPLIT_KV not in os.environ
    assert KEY_TILE not in dict(result.applied)
    assert SPLIT_KV not in dict(result.applied)
    if backend_name == "numpy" or profile == "reference":
        assert result.applied == ()
