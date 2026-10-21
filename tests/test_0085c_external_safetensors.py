"""Single-file and indexed Safetensors imported without training modules."""

import json
import numpy as np
import pytest
from safetensors.numpy import save_file

from mini_llm.external import ExternalLlamaModel, ExternalLlamaConfig
from mini_llm.external.safetensors_loader import (
    load_safetensors_model, SafetensorsWeightMap,
)
from test_0085b_external_llama import make_config, make_weights


def setup_hf_checkpoint(path, *, shards=False, tamper=None):
    path.mkdir()
    payload = {
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "hidden_size": 24, "intermediate_size": 48,
        "num_hidden_layers": 2, "num_attention_heads": 3,
        "num_key_value_heads": 1, "head_dim": 8,
        "vocab_size": 49, "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5, "rope_theta": 100000.0,
        "tie_word_embeddings": True, "rope_interleaved": False,
    }
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    expected_model = ExternalLlamaModel(ExternalLlamaConfig.from_dict(payload))
    tensors = make_weights(expected_model)
    if tamper == "shape":
        tensors["model.layers.1.self_attn.o_proj.weight"] = np.zeros((7, 7), np.float32)
    if shards:
        names = sorted(tensors)
        first = {n: tensors[n] for n in names[::2]}
        second = {n: tensors[n] for n in names[1::2]}
        save_file(first, str(path / "model-00001-of-00002.safetensors"))
        save_file(second, str(path / "model-00002-of-00002.safetensors"))
        index = {"weight_map": {name: ("model-00001-of-00002.safetensors"
                                      if name in first else "model-00002-of-00002.safetensors")
                                for name in tensors}}
        (path / "model.safetensors.index.json").write_text(json.dumps(index))
    else:
        save_file(tensors, str(path / "model.safetensors"))
    return expected_model, tensors


@pytest.mark.parametrize("shards", [False, True])
def test_load_single_or_sharded_without_gradients(tmp_path, shards):
    expected_model, tensors = setup_hf_checkpoint(tmp_path / "hf", shards=shards)
    reader = SafetensorsWeightMap(tmp_path / "hf")
    assert len(reader) == len(tensors)
    for name, tensor in tensors.items():
        assert reader.shape_of(name) == tensor.shape
    expected_model.set_weights(tensors)
    model, provenance = load_safetensors_model(tmp_path / "hf", hash_weights=True)
    tokens = np.array([[1, 4, 7, 3]], dtype=np.int32)
    np.testing.assert_allclose(model.forward(tokens), expected_model.forward(tokens), rtol=1e-6, atol=1e-6)
    assert len(provenance["weight_sha256"]) == (2 if shards else 1)
    assert provenance["original_dtypes"] == ["F32"]
    assert provenance["runtime_dtype"] == "float32"
    assert not hasattr(model.engine.embedding.W, "grad")
    assert not hasattr(model.engine.blocks[0].attention, "Wq_grad")


def test_reject_malformed_index_and_wrong_shapes(tmp_path):
    setup_hf_checkpoint(tmp_path / "hf", shards=True)
    index_file = tmp_path / "hf" / "model.safetensors.index.json"
    index = json.loads(index_file.read_text())
    index["weight_map"]["model.norm.weight"] = "../wrong.safetensors"
    index_file.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Unsafe"):
        SafetensorsWeightMap(tmp_path / "hf")

    setup_hf_checkpoint(tmp_path / "bad_shape", tamper="shape")
    with pytest.raises(ValueError, match="expected shape"):
        load_safetensors_model(tmp_path / "bad_shape")


def test_reject_missing_shard_tensor(tmp_path):
    setup_hf_checkpoint(tmp_path / "hf", shards=True)
    index_file = tmp_path / "hf" / "model.safetensors.index.json"
    index = json.loads(index_file.read_text())
    index["weight_map"].pop("model.norm.weight")
    index_file.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="disagrees"):
        SafetensorsWeightMap(tmp_path / "hf")


def test_import_bf16_serialized_weights_to_fp32(tmp_path):
    """Official SmolLM2 files store BF16; NumPy loader must understand it."""
    import ml_dtypes
    path = tmp_path / "bf16"
    expected_model, tensors = setup_hf_checkpoint(path)
    serialized = {name: values.astype(ml_dtypes.bfloat16)
                  for name, values in tensors.items()}
    save_file(serialized, str(path / "model.safetensors"))
    model, provenance = load_safetensors_model(path, dtype="float32")
    assert provenance["original_dtypes"] == ["BF16"]
    expected_model.set_weights({name: values.astype("float32")
                                for name, values in serialized.items()})
    tokens = np.array([[2, 9, 6]], dtype=np.int32)
    np.testing.assert_allclose(model.forward(tokens), expected_model.forward(tokens),
                               rtol=1e-6, atol=1e-6)
