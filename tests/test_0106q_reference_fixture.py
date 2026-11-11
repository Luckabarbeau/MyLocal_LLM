"""CPU-only fixture checks; independent llama.cpp execution belongs to its harness.

MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
    tests/test_0106q_reference_fixture.py
"""

import os
from pathlib import Path
import runpy

import numpy as np
import pytest


if os.environ.get("MINI_LLM_BACKEND", "numpy").lower() != "numpy":
    pytest.skip("0106q fixture validation is CPU-only", allow_module_level=True)

gguf = pytest.importorskip("gguf")
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = runpy.run_path(str(ROOT / "benchmarks/fixture_0106q_qwen_reference.py"))
write_fixture = FIXTURE["write_fixture"]


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    result = write_fixture(tmp_path_factory.mktemp("0106q"))
    original = gguf.GGUFReader(result["model_path"])
    reference = gguf.GGUFReader(result["reference_path"])
    return result, original, reference


def _shape(tensor):
    return tuple(int(d) for d in reversed(tensor.shape))


def _values(tensor):
    return gguf.quants.dequantize(tensor.data, tensor.tensor_type).reshape(_shape(tensor))


def test_api_deterministic_files_and_seed(pair, tmp_path):
    result, original, reference = pair
    assert set(result) == {"model_path", "reference_path", "token_ids", "metadata"}
    assert all(isinstance(result[key], Path) for key in ("model_path", "reference_path"))
    assert result["token_ids"] == [256, 32, 97, 10, 256]
    assert all(type(token) is int for token in result["token_ids"])
    assert all(type(value) in (str, int, float, bool) for value in result["metadata"].values())
    assert result["metadata"]["seed"] == 10617
    assert result["metadata"]["tensor_count"] == len(original.tensors) == len(reference.tensors) == 73
    same = write_fixture(tmp_path / "same", seed=10617)
    changed = write_fixture(tmp_path / "changed", seed=10618)
    assert result["metadata"] == same["metadata"]
    for key in ("model_path", "reference_path"):
        assert 0 < result[key].stat().st_size < 2 * 1024**2
        assert result[key].read_bytes() == same[key].read_bytes()
        assert result[key].read_bytes() != changed[key].read_bytes()
        # Compare actual tensors too, not only the seed provenance bytes.
        reread = gguf.GGUFReader(same[key])
        for first, second in zip((original if key == "model_path" else reference).tensors,
                                 reread.tensors):
            assert first.name == second.name
            np.testing.assert_array_equal(first.data, second.data)
    changed_reader = gguf.GGUFReader(changed["model_path"])
    assert not np.array_equal(original.tensors[0].data, changed_reader.tensors[0].data)
    before = result["model_path"].read_bytes()
    with pytest.raises(FileExistsError):
        write_fixture(result["model_path"].parent)
    assert result["model_path"].read_bytes() == before


def test_every_tensor_exact_dequantized_equivalence(pair):
    _, original, reference = pair
    refs = {tensor.name: tensor for tensor in reference.tensors}
    assert len(refs) == len(reference.tensors)
    assert set(refs) == {tensor.name for tensor in original.tensors}
    packed_count = 0
    for tensor in original.tensors:
        other = refs[tensor.name]
        assert other.tensor_type == gguf.GGMLQuantizationType.F32
        assert _shape(tensor) == _shape(other)
        decoded = _values(tensor)
        assert decoded.dtype == np.float32
        assert np.isfinite(decoded).all()
        np.testing.assert_array_equal(decoded, _values(other), err_msg=tensor.name)
        packed = "_exps.weight" in tensor.name or tensor.name in (
            "token_embd.weight", "output.weight")
        assert tensor.tensor_type == (gguf.GGMLQuantizationType.Q8_0 if packed
                                      else gguf.GGMLQuantizationType.F32)
        if packed:
            packed_count += 1
            # A second, literal decoder is independent of both gguf and production.
            blocks = tensor.data.reshape(-1, 34)
            scales = blocks[:, :2].copy().view("<f2").astype(np.float32)
            quants = blocks[:, 2:].view(np.int8)
            assert np.all(scales == np.float32(1 / 128))
            assert quants.min() >= -7 and quants.max() <= 7
            literal = (scales * quants.astype(np.float32)).reshape(_shape(tensor))
            np.testing.assert_array_equal(literal, decoded, err_msg=tensor.name)
    assert packed_count == 14  # 3 banks x 4 layers + embedding + untied output.


def test_complete_metadata_and_llama_compatible_shapes(pair):
    from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
    from mini_llm.external.qwen3next_model import inspect_qwen3next_model

    result, original, reference = pair
    expected = {
        "general.architecture": "qwen3next", "qwen3next.context_length": 32,
        "qwen3next.embedding_length": 32, "qwen3next.block_count": 4,
        "qwen3next.vocab_size": 260, "qwen3next.feed_forward_length": 128,
        "qwen3next.expert_count": 4, "qwen3next.expert_used_count": 2,
        "qwen3next.expert_feed_forward_length": 32,
        "qwen3next.expert_shared_feed_forward_length": 64,
        "qwen3next.full_attention_interval": 4,
        "qwen3next.attention.head_count": 4, "qwen3next.attention.head_count_kv": 2,
        "qwen3next.attention.key_length": 32, "qwen3next.attention.value_length": 32,
        "qwen3next.attention.layer_norm_rms_epsilon": np.float32(1e-6),
        "qwen3next.rope.dimension_count": 8, "qwen3next.rope.freq_base": 5e6,
        "qwen3next.rope.scaling.type": "none", "qwen3next.ssm.conv_kernel": 4,
        "qwen3next.ssm.inner_size": 128, "qwen3next.ssm.state_size": 32,
        "qwen3next.ssm.time_step_rank": 4, "qwen3next.ssm.group_count": 2,
        "fixture.seed": 10617, "fixture.id": "0106q",
        "fixture.llama_cpp_revision": "171e8846b4af9766c354064cb776cb34a50f053f",
    }
    shapes = {"token_embd.weight": (260, 32), "output.weight": (260, 32),
              "output_norm.weight": (32,)}
    for layer in range(4):
        common = {
            "attn_norm.weight": (32,), "post_attention_norm.weight": (32,),
            "ffn_gate_inp.weight": (4, 32), "ffn_gate_inp_shexp.weight": (32,),
            "ffn_gate_shexp.weight": (64, 32), "ffn_up_shexp.weight": (64, 32),
            "ffn_down_shexp.weight": (32, 64),
            "ffn_gate_exps.weight": (4, 32, 32), "ffn_up_exps.weight": (4, 32, 32),
            "ffn_down_exps.weight": (4, 32, 32),
        }
        mixer = ({
            "attn_q.weight": (256, 32), "attn_k.weight": (64, 32),
            "attn_v.weight": (64, 32), "attn_output.weight": (32, 128),
            "attn_q_norm.weight": (32,), "attn_k_norm.weight": (32,),
        } if layer == 3 else {
            "attn_qkv.weight": (256, 32), "attn_gate.weight": (128, 32),
            "ssm_ba.weight": (8, 32), "ssm_conv1d.weight": (256, 4),
            "ssm_dt.bias": (4,), "ssm_a": (4,), "ssm_norm.weight": (32,),
            "ssm_out.weight": (32, 128),
        })
        shapes.update({f"blk.{layer}." + key: value for key, value in (common | mixer).items()})
    for reader, is_reference in ((original, False), (reference, True)):
        for key, value in expected.items():
            assert reader.fields[key].contents(0) == value, key
        assert {tensor.name: _shape(tensor) for tensor in reader.tensors} == shapes
        assert reader.fields["fixture.all_f32_reference"].contents(0) == is_reference
        assert reader.fields["general.file_type"].contents(0) == (0 if is_reference else 7)
        index = Qwen3NextGGUFIndex(reader)
        assert (index.layers, index.d_model, index.d_ff, index.expert_count) == (4, 32, 32, 4)
    index = Qwen3NextGGUFIndex.open(result["model_path"])
    config, attention, recurrent, _ = inspect_qwen3next_model(index)
    assert (config.vocab_size, config.max_position_embeddings) == (260, 32)
    assert not config.tie_word_embeddings
    assert recurrent.key_dim == recurrent.value_dim == attention.key_dim == attention.value_dim == 32
    assert recurrent.n_value_heads % recurrent.n_key_heads == 0
    for layer in range(4):
        for expert in range(4):
            for weight in index.expert_weights(layer, expert):
                assert weight.quant_type == "Q8_0" and weight.shape == (32, 32)
        if layer != 3:
            assert np.all(index.non_expert_weight(f"blk.{layer}.ssm_a") < 0)


def test_tokenizer_explicit_matched_raw_ids_and_byte_alphabet(pair):
    pytest.importorskip("tokenizers")
    from tokenizers.pre_tokenizers import ByteLevel
    from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer

    result, original, reference = pair
    for reader in (original, reference):
        tokenizer = load_gguf_tokenizer(reader, vocab_size=260)
        assert set(tokenizer.spec.tokens[:256]) == set(ByteLevel.alphabet())
        assert tokenizer.spec.tokens[256:] == (
            "ab", "<|endoftext|>", "<|im_start|>", "<|im_end|>")
        assert tokenizer.spec.merges == ("a b",)
        assert (tokenizer.spec.model, tokenizer.spec.pre) == ("gpt2", "qwen2")
        assert tokenizer.spec.bos_id == 257 and tokenizer.spec.eos_id == 259
        assert tokenizer.spec.add_bos is False
        assert tokenizer.encode("ab").ids == [256]
        assert tokenizer.encode("ab a\nab").ids == result["token_ids"] == [256, 32, 97, 10, 256]
        assert tokenizer.decode(result["token_ids"]) == result["metadata"]["prompt_text"]
        for text, token in (("<|endoftext|>", 257), ("<|im_start|>", 258), ("<|im_end|>", 259)):
            assert tokenizer.encode(text).ids == [token]
            assert tokenizer.decode([token]) == text
        text = "café 中文 🙂\t\r\n"
        assert tokenizer.decode(tokenizer.encode(text).ids) == text


@pytest.mark.parametrize("packed_backend", ["reference", "native"])
def test_original_cpu_full_prefill_step_and_existing_fp64_oracle(pair, packed_backend):
    from mini_llm.backend import BACKEND_NAME
    from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
    from mini_llm.external.qwen3next_model import Qwen3NextGGUFModel

    assert BACKEND_NAME == "numpy"
    result, original, reference = pair
    index = Qwen3NextGGUFIndex(original, source_path=result["model_path"])
    # Reuse the established independent source oracles, not a new reference engine.
    oracle = runpy.run_path(str(ROOT / "tests/test_0106p_qwen3next_model.py"))
    values = {tensor.name: _values(tensor) for tensor in reference.tensors}
    with Qwen3NextGGUFModel.from_index(index, cpu_packed_backend=packed_backend,
                                     cpu_workers=0, expert_cache_budget_bytes=0) as model:
        weights = {}
        for block in model.blocks:
            stem = f"blk.{block.layer_id}."
            weights[block.layer_id] = {
                "mixer": {suffix: values[stem + suffix]
                          for suffix in block.mixer.spec.tensor_shapes()},
                "ffn": {name[len(stem):]: value for name, value in values.items()
                        if name.startswith(stem) and (name[len(stem):].startswith("ffn_")
                        or name[len(stem):] in ("attn_norm.weight", "post_attention_norm.weight"))},
            }
        ids = np.array([result["token_ids"]], dtype=np.int32)
        expected = oracle["reference"](ids, model, weights, values)
        full = model.forward(ids)
        assert full.shape == (1, 5, 260) and full.dtype == np.float32
        assert np.isfinite(full).all() and np.max(np.abs(full)) < 10
        np.testing.assert_allclose(full, expected, rtol=4e-4, atol=8e-6)
        state = model.create_generation_state(1, 8)
        np.testing.assert_allclose(model.prefill(ids[:, :2], state), expected[:, 1],
                                   rtol=4e-4, atol=8e-6)
        for t in range(2, ids.shape[1]):
            step = model.decode_one(ids[:, t], state)
            assert np.isfinite(step).all()
            np.testing.assert_allclose(step, full[:, t], rtol=4e-4, atol=8e-6)
        assert state.length == 5 and all(layer.length == 5 for layer in state.layer_states)
        state.reset()
        np.testing.assert_allclose(model.prefill(ids, state), full[:, -1], rtol=4e-4, atol=8e-6)
        changed = ids.copy()
        changed[:, 3:] = 98
        # Suffix routes change expert GEMM row counts; FP32 reduction roundoff
        # need not be bitwise identical even though the graph remains causal.
        np.testing.assert_allclose(model.forward(changed)[:, :3], full[:, :3],
                                   rtol=2e-6, atol=3e-7)
