"""Strict native-checkpoint interoperability tests for the PyTorch backend."""

from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np
import pytest


try:
    import torch
except ImportError:
    torch = None

pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch is not installed")

from mini_llm.backends.pytorch.checkpoint import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
)
from mini_llm.checkpoint import save_checkpoint as save_native_checkpoint  # noqa: E402
from mini_llm.config import ModelConfig  # noqa: E402
from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer  # noqa: E402


if torch is not None:
    class TinyTorchModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.weight = torch.nn.Parameter(torch.zeros(3, 2))
            self.bias = torch.nn.Parameter(torch.zeros(3))
            self.tied_weight = self.weight
            self.active_layers = 0

        def canonical_named_parameters(self):
            yield "tiny.weight", self.weight
            yield "tiny.bias", self.bias

        def set_active_layers(self, count):
            self.active_layers = int(count)
else:
    TinyTorchModel = None


def _config():
    return ModelConfig.tiny_inspection()


def _write_ancillary(path, config):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(asdict(config)), encoding="utf-8"
    )
    tokenizer = SimpleBPETokenizer(vocab_size=8)
    tokenizer.train(["abc cab"], vocab_size=8)
    tokenizer.save(str(path / "tokenizer.json"))


def test_strict_load_save_roundtrip_preserves_schema_and_tied_weight(tmp_path):
    source = tmp_path / "source"
    config = _config()
    _write_ancillary(source, config)
    expected = {
        "tiny.weight": np.arange(6, dtype=np.float32).reshape(3, 2),
        "tiny.bias": np.array([0.5, -1.0, 2.0], dtype=np.float32),
    }
    optimizer = {
        "step": 7,
        "m": {name: np.full(value.shape, 0.25, np.float32)
              for name, value in expected.items()},
    }
    training = {"step": 7, "progressive_active_layers": 1}
    save_native_checkpoint(source, expected, optimizer, training)

    loaded = load_checkpoint(
        source,
        model_factory=TinyTorchModel,
        device="cpu",
        dtype=torch.float64,
    )
    assert loaded.model.weight.dtype == torch.float64
    assert loaded.model.tied_weight is loaded.model.weight
    assert loaded.model.active_layers == 1
    assert loaded.training_state == training
    assert loaded.optimizer_state["step"] == 7
    assert loaded.optimizer_state["m"]["tiny.weight"].dtype == torch.float32
    np.testing.assert_array_equal(
        loaded.model.weight.detach().cpu().numpy(), expected["tiny.weight"]
    )

    exported = tmp_path / "exported"
    save_checkpoint(
        exported,
        loaded.model,
        optimizer_state=loaded.optimizer_state,
        training_state=loaded.training_state,
        tokenizer=loaded.tokenizer,
    )
    reloaded = load_checkpoint(
        exported,
        TinyTorchModel(config),
        device=torch.device("cpu"),
        dtype="float64",
    )
    assert reloaded.model.tied_weight is reloaded.model.weight
    assert reloaded.training_state == training
    assert reloaded.optimizer_state["step"] == 7
    for name, parameter in reloaded.model.canonical_named_parameters():
        np.testing.assert_array_equal(
            parameter.detach().cpu().numpy(), expected[name]
        )


def test_load_rejects_nonexact_parameter_names_before_mutation(tmp_path):
    checkpoint = tmp_path / "bad-names"
    config = _config()
    _write_ancillary(checkpoint, config)
    save_native_checkpoint(
        checkpoint,
        {
            "tiny.weight": np.ones((3, 2), np.float32),
            "unexpected": np.ones((3,), np.float32),
        },
    )
    model = TinyTorchModel(config)
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="do not exactly match"):
        load_checkpoint(checkpoint, model, device="cpu", dtype="float32")
    torch.testing.assert_close(model.weight, before)


def test_bfloat16_void_representation_loads_and_saves(tmp_path):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    source = tmp_path / "bf16-source"
    config = _config()
    _write_ancillary(source, config)
    expected_weight = np.array(
        [[1.0, -2.0], [3.5, 0.25], [-0.5, 8.0]], dtype=ml_dtypes.bfloat16
    )
    expected_bias = np.array([1.0, 2.0, 3.0], dtype=ml_dtypes.bfloat16)
    save_native_checkpoint(
        source,
        {"tiny.weight": expected_weight, "tiny.bias": expected_bias},
    )
    assert np.load(source / "model_params" / "tiny.weight.npy").dtype.kind == "V"

    loaded = load_checkpoint(
        source,
        model_factory=TinyTorchModel,
        device="cpu",
        dtype="bfloat16",
        skip_optimizer=True,
    )
    assert loaded.model.weight.dtype == torch.bfloat16
    torch.testing.assert_close(
        loaded.model.weight.float(),
        torch.from_numpy(expected_weight.astype(np.float32)),
        rtol=0,
        atol=0,
    )

    exported = tmp_path / "bf16-exported"
    save_checkpoint(exported, loaded.model, tokenizer=loaded.tokenizer)
    serialized = np.load(exported / "model_params" / "tiny.weight.npy")
    assert serialized.dtype.kind == "V"
    assert serialized.dtype.itemsize == 2
    np.testing.assert_array_equal(
        serialized.view(ml_dtypes.bfloat16), expected_weight
    )
