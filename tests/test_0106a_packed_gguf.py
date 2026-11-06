"""Packed GGUF representation and Q3/Q4 reference dequantization."""

from types import SimpleNamespace
import numpy as np
import pytest

from mini_llm.ops.quantized_weight import PackedGGUFWeight, BLOCK_FORMATS
from mini_llm.external.gguf_loader import GGUFWeightMap
from test_0085f_external_gguf import fake_reader
from test_0085b_external_llama import make_config, make_weights
from mini_llm.external.llama_model import ExternalLlamaModel


def _f16(value):
    return np.array([value], dtype="<f2").view(np.uint8).tobytes()


def test_reference_q4_0_signed_nibbles():
    # GGML nibble order is low 16 followed by high 16.
    raw = np.frombuffer(_f16(0.5) + bytes([0x83] * 16), dtype=np.uint8).copy()
    packed = PackedGGUFWeight(raw, (1, 32), "Q4_0")
    np.testing.assert_array_equal(packed.dequantize_reference()[0], [-2.5] * 16 + [0.0] * 16)
    assert np.shares_memory(packed.data, raw)
    assert packed.nbytes == 18


def test_reference_q4_k_scales_and_minima():
    # Eight 32-element groups; low/high nibbles in successive groups.
    # d=1, dmin=0. First 4 groups have scale 1, last 4 scale 0.
    raw = _f16(1) + _f16(0) + bytes([1] * 4 + [0] * 8) + bytes([0x12] * 128)
    packed = PackedGGUFWeight(np.frombuffer(raw, np.uint8), (1, 256), "Q4_K")
    expected = np.array(([2.0] * 32 + [1.0] * 32) * 2 + [0] * 128)
    np.testing.assert_array_equal(packed.dequantize_reference().ravel(), expected)


def test_reference_q3_k_bit_masks_and_scales():
    # Q3 six-bit scale 33 represents signed scale +1. Every 2-bit q=2.
    # Mask bit 1 means no offset; zero means subtract four.
    scales = bytes([0x11] * 8 + [0xAA] * 4)
    for hmask, expected in ((0xFF, 2.0), (0x00, -2.0)):
        raw = bytes([hmask] * 32 + [0xAA] * 64) + scales + _f16(1)
        packed = PackedGGUFWeight(np.frombuffer(raw, np.uint8), (1, 256), "Q3_K")
        np.testing.assert_array_equal(packed.dequantize_reference(), np.full((1, 256), expected))


def test_select_expert_from_3d_packed_array_without_copy():
    rows = [(_f16(float(i)) + bytes([0x99] * 16)) for i in (1, 2, 3)]
    raw = np.frombuffer(b"".join(rows), dtype=np.uint8)
    all_experts = PackedGGUFWeight(raw, (3, 1, 32), "Q4_0")
    expert = all_experts.select_outer(1)
    assert expert.shape == (1, 32)
    assert expert.nbytes == 18
    assert np.shares_memory(expert.data, all_experts.data)
    np.testing.assert_array_equal(expert.dequantize_reference(), np.full((1, 32), 2.0))
    with pytest.raises(IndexError):
        all_experts.select_outer(3)


def test_invalid_packed_shapes_and_lengths():
    for fmt, (block, nbytes) in BLOCK_FORMATS.items():
        with pytest.raises(ValueError, match="innermost"):
            PackedGGUFWeight(np.zeros(nbytes, dtype=np.uint8), (2, block // 2), fmt)
        with pytest.raises(ValueError, match="expected"):
            PackedGGUFWeight(np.zeros(nbytes - 1, dtype=np.uint8), (1, block), fmt)
    with pytest.raises(ValueError, match="Unsupported"):
        PackedGGUFWeight(np.zeros(1, np.uint8), (1, 32), "IQ3_S")


def test_legacy_gguf_map_adds_packed_view_without_changing_dense_semantics():
    cfg = make_config(hidden_size=32, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = ExternalLlamaModel(cfg)
    weights = make_weights(model)
    reader = fake_reader(cfg, weights, quantize=True)
    mapping = GGUFWeightMap(reader)
    key = next(name for name in mapping if mapping._tensors[name].tensor_type.name == "Q8_0")
    packed = mapping.get_packed(key)
    assert packed.quant_type == "Q8_0"
    if not (key.endswith("self_attn.q_proj.weight") or key.endswith("self_attn.k_proj.weight")):
        np.testing.assert_array_equal(mapping[key], packed.dequantize_reference())
    assert np.shares_memory(packed.data, np.asarray(mapping._tensors[key].data))
    with pytest.raises(TypeError, match="not a packed"):
        mapping.get_packed("model.norm.weight")


def test_gguf_map_accepts_q4_0_tensor_and_unpacks_for_legacy_llama():
    cfg = make_config(hidden_size=32, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = ExternalLlamaModel(cfg)
    reader = fake_reader(cfg, make_weights(model))
    target = next(t for t in reader.tensors if t.name.endswith("ffn_gate.weight") and
                  int(np.prod(t.shape)) % 32 == 0)
    length = int(np.prod(target.shape))
    target.tensor_type = SimpleNamespace(name="Q4_0")
    target.data = np.frombuffer((_f16(0.25) + bytes([0x99] * 16)) * (length // 32), np.uint8)
    mapped = GGUFWeightMap(reader)
    assert mapped.get_packed(f"model.layers.0.mlp.gate_proj.weight").nbytes == length // 32 * 18
    np.testing.assert_array_equal(mapped["model.layers.0.mlp.gate_proj.weight"],
                                  np.full(target.shape[::-1], .25))


@pytest.mark.parametrize("fmt", ["Q3_K", "Q4_K"])
def test_against_optional_gguf_reference(fmt):
    gguf = pytest.importorskip("gguf")
    from gguf.quants import dequantize
    from gguf.constants import GGMLQuantizationType
    rng = np.random.default_rng(1606)
    block, nbytes = BLOCK_FORMATS[fmt]
    raw = rng.integers(0, 256, size=nbytes * 4, dtype=np.uint8)
    # Stable scales, avoiding NaNs from random half-float exponent patterns.
    if fmt == "Q4_K":
        raw.reshape(4, nbytes)[:, :4] = np.array([0, 60, 0, 52], dtype=np.uint8)
    else:
        raw.reshape(4, nbytes)[:, 108:110] = np.array([0, 60], dtype=np.uint8)
    packed = PackedGGUFWeight(raw, (4, block), fmt)
    expected = dequantize(raw.reshape(4, nbytes), GGMLQuantizationType[fmt]).reshape(4, block)
    np.testing.assert_array_equal(packed.dequantize_reference(), expected)
