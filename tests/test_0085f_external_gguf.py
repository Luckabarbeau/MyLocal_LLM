"""GGUF Q8_0 + tensor mapping tested without downloading pretrained weights.

Uses a GGUFReader-shaped fake to validate metadata, tensor orientation,
quantization, and the crucial conversion Q/K permutation independently.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.external import GGUFWeightMap, ExternalLlamaModel
from mini_llm.external.gguf_loader import _dequantize_q8_0, _restore_hf_rope_rows
from test_0085b_external_llama import make_config, make_weights


class Field:
    def __init__(self, value):
        self.value = value
    def contents(self, index):
        return self.value


def fake_reader(cfg, canonical, *, quantize=False, wrong_arch=False, extras=()):
    fields = dict(
        **{"general.architecture": "unknown" if wrong_arch else "llama"},
        **{"llama.embedding_length": cfg.hidden_size,
           "llama.feed_forward_length": cfg.intermediate_size,
           "llama.block_count": cfg.num_hidden_layers,
           "llama.attention.head_count": cfg.num_attention_heads,
           "llama.attention.head_count_kv": cfg.num_key_value_heads,
           "llama.rope.dimension_count": cfg.head_dim,
           "llama.context_length": cfg.max_position_embeddings,
           "llama.rope.freq_base": cfg.rope_theta,
           "llama.attention.layer_norm_rms_epsilon": cfg.rms_norm_eps,
           "llama.vocab_size": cfg.vocab_size}
    )
    tensors = []
    inv_base = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    inv = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }
    for hf_name, weight in canonical.items():
        if hf_name in inv_base:
            gguf_name = inv_base[hf_name]
        else:
            _, _, layer, rest = hf_name.split(".", 3)
            gguf_name = f"blk.{layer}.{inv[rest]}"
        matrix = weight.copy()
        if gguf_name.endswith("attn_q.weight") or gguf_name.endswith("attn_k.weight"):
            heads = (cfg.num_attention_heads if gguf_name.endswith("attn_q.weight")
                     else cfg.num_key_value_heads)
            # Forward HF -> GGUF conversion, independently of the tested reverse.
            matrix = matrix.reshape(heads, 2, -1, matrix.shape[1]).swapaxes(1, 2).reshape(matrix.shape)
        if quantize and matrix.size % 32 == 0 and matrix.ndim == 2:
            blocks = matrix.reshape(-1, 32)
            scales = np.max(abs(blocks), axis=1) / 127.0
            scales[scales == 0] = 1
            scales = scales.astype("<f2")
            q = np.clip(np.round(blocks / scales.astype(np.float32)[:, None]), -127, 127).astype(np.int8)
            chunks = np.concatenate([scales.view(np.uint8).reshape(-1, 2),
                                     q.view(np.uint8)], axis=-1)
            matrix = chunks.reshape(-1)
            tensor_type = "Q8_0"
        else:
            tensor_type = "F32"
        tensors.append(SimpleNamespace(
            name=gguf_name, shape=np.asarray(weight.shape[::-1]), data=matrix,
            tensor_type=SimpleNamespace(name=tensor_type),
        ))
    for key in extras:
        tensors.append(SimpleNamespace(name=key, shape=np.array([4]),
                                       data=np.zeros(4, dtype=np.float32),
                                       tensor_type=SimpleNamespace(name="F32")))
    return SimpleNamespace(fields={key: Field(value) for key, value in fields.items()},
                           tensors=tensors)


def test_q8_0_block_dequantization():
    block = np.arange(-16, 16, dtype=np.int8)
    buf = np.array([0.25], dtype="<f2").view(np.uint8).tobytes() + block.tobytes()
    np.testing.assert_array_equal(_dequantize_q8_0(np.frombuffer(buf, np.uint8), 32), block * 0.25)
    with pytest.raises(ValueError):
        _dequantize_q8_0(np.zeros(33, dtype=np.uint8), 32)


def test_qk_permutation_roundtrip_and_float32_logits():
    cfg = make_config()
    expected = ExternalLlamaModel(cfg)
    weights = make_weights(expected)
    reader = GGUFWeightMap(fake_reader(cfg, weights))
    assert reader.config == cfg
    for name, weight in weights.items():
        np.testing.assert_array_equal(reader[name], weight)
    actual = ExternalLlamaModel(reader.config)
    actual.set_weights(reader)
    expected.set_weights(weights)
    ids = np.array([[1, 2, 3, 4]], dtype=np.int32)
    np.testing.assert_allclose(actual.forward(ids), expected.forward(ids), rtol=1e-6, atol=1e-6)


def test_q8_0_logits_close_to_unquantized():
    cfg = make_config()
    original = ExternalLlamaModel(cfg)
    weights = make_weights(original)
    # Small config includes some non-multiple of 32 weights. Quantize where allowed.
    reader = GGUFWeightMap(fake_reader(cfg, weights, quantize=True))
    assert "Q8_0" in reader.tensor_types
    imported = ExternalLlamaModel(cfg)
    imported.set_weights(reader)
    original.set_weights(weights)
    ids = np.array([[1, 3, 7, 11]], dtype=np.int32)
    max_abs = np.max(abs(imported.forward(ids) - original.forward(ids)))
    assert max_abs < 0.06, max_abs


def test_reject_unsupported_metadata_and_weights():
    cfg = make_config()
    model = ExternalLlamaModel(cfg)
    weights = make_weights(model)
    with pytest.raises(ValueError, match="Unsupported GGUF architecture"):
        GGUFWeightMap(fake_reader(cfg, weights, wrong_arch=True))
    with pytest.raises(ValueError, match="Unsupported GGUF tensor name"):
        GGUFWeightMap(fake_reader(cfg, weights, extras=["blk.0.attn_q_norm.weight"]))
    reader = fake_reader(cfg, weights)
    reader.fields["llama.rope.scaling.type"] = Field("linear")
    with pytest.raises(ValueError, match="RoPE scaling"):
        GGUFWeightMap(reader)
    np.testing.assert_array_equal(
        _restore_hf_rope_rows(np.zeros((24, 16)), 3), np.zeros((24, 16))
    )


@pytest.mark.parametrize("q_heads,kv_heads", [(9, 3), (15, 5)])
def test_smollm2_non_power_of_two_gqa_heads_execute(q_heads, kv_heads):
    cfg = make_config(hidden_size=q_heads * 8, num_attention_heads=q_heads,
                      num_key_value_heads=kv_heads, head_dim=8,
                      intermediate_size=64, num_hidden_layers=1)
    model = ExternalLlamaModel(cfg)
    weights = make_weights(model)
    model.set_weights(GGUFWeightMap(fake_reader(cfg, weights, quantize=True)))
    tokens = np.asarray([[1, 8, 4]], dtype=np.int32)
    reference = model.forward(tokens)
    assert reference.shape == (1, 3, cfg.vocab_size)
    state = model.create_generation_state(1, 4)
    logits = model.prefill(tokens[:, :2], state)
    np.testing.assert_allclose(logits, reference[:, 1], atol=2e-5, rtol=1e-4)
    logits = model.decode_one(tokens[:, 2], state)
    np.testing.assert_allclose(logits, reference[:, 2], atol=2e-5, rtol=1e-4)
