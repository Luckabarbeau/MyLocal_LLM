"""Per-tensor CPU BF16 decoding and lazy adapters; no pretrained models."""

from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.external.gguf_loader import GGUFWeightMap
from mini_llm.external.gguf_tensor import decode_gguf_dense
from mini_llm.external.llama_model import ExternalLlamaModel
from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from test_0085b_external_llama import make_config, make_weights
from test_0085f_external_gguf import fake_reader
from test_0106f_qwen3next_gguf import Field, Reader


def tensor(data, shape, kind="BF16", name="test.weight"):
    return SimpleNamespace(name=name, shape=np.asarray(shape[::-1]), data=data,
                           tensor_type=SimpleNamespace(name=kind))


def bf16_bytes(words):
    return np.asarray(words, dtype="<u2").view(np.uint8)


class Unreadable:
    def __array__(self, *args, **kwargs):
        raise AssertionError("unrequested tensor payload accessed")


@pytest.mark.parametrize("storage", ["bytes", "<u2", ">u2"])
def test_bf16_exact_expansion_all_bit_patterns(storage):
    # Includes +/-0, subnormals, infinities, and every quiet/signaling NaN
    # payload and sign. Compare integer bits, never NaN numeric equality.
    words = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    data = bf16_bytes(words) if storage == "bytes" else words.astype(storage)
    before = data.tobytes()
    actual = decode_gguf_dense(tensor(data, words.shape))
    assert actual.dtype == np.float32
    assert actual.shape == words.shape
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual.view(np.uint32), words.astype(np.uint32) << 16)
    assert data.tobytes() == before


@pytest.mark.parametrize("shape", [(6,), (2, 3), (1, 2, 3)])
@pytest.mark.parametrize("flat", [False, True])
def test_bf16_little_endian_packed_shape_and_order(shape, flat):
    # Explicit bytes avoid making expected byte order depend on the host.
    raw = np.array([0x80, 0x3F, 0x20, 0xC0, 0, 0x80,
                    0x80, 0x7F, 0x80, 0xFF, 0x81, 0x7F], dtype=np.uint8)
    data = raw if flat else raw.reshape((*shape[:-1], shape[-1] * 2))
    actual = decode_gguf_dense(tensor(data, shape))
    expected_bits = np.array([0x3F800000, 0xC0200000, 0x80000000,
                              0x7F800000, 0xFF800000, 0x7F810000], dtype=np.uint32)
    np.testing.assert_array_equal(actual.view(np.uint32), expected_bits.reshape(shape))


def test_bf16_noncontiguous_byte_fixture():
    raw = np.zeros((2, 12), dtype=np.uint8)
    raw[:, ::2] = bf16_bytes(np.array([[0x3F80, 0x4000, 0x4040],
                                      [0x4080, 0x40A0, 0x40C0]], dtype=np.uint16))
    actual = decode_gguf_dense(tensor(raw[:, ::2], (2, 3)))
    np.testing.assert_array_equal(actual, np.arange(1, 7, dtype=np.float32).reshape(2, 3))


@pytest.mark.parametrize("nbytes", [0, 1, 10, 11, 13, 14])
def test_bf16_rejects_inexact_payload_sizes(nbytes):
    with pytest.raises(ValueError, match=r"BF16 payload.*expected 12"):
        decode_gguf_dense(tensor(np.zeros(nbytes, dtype=np.uint8), (2, 3)))


@pytest.mark.parametrize("dtype", [np.float16, np.int16, np.float32, np.uint32])
def test_bf16_rejects_non_payload_dtypes(dtype):
    # Even an exact byte count must not allow numeric F16/integer casts.
    data = np.zeros(12 // np.dtype(dtype).itemsize, dtype=dtype)
    with pytest.raises(ValueError, match="uint8 bytes or uint16 words"):
        decode_gguf_dense(tensor(data, (2, 3)))


@pytest.mark.parametrize("shape", [(0, 3), (-1, 3)])
def test_bf16_rejects_invalid_shape(shape):
    with pytest.raises(ValueError, match="invalid BF16 shape"):
        decode_gguf_dense(tensor(np.zeros(0, dtype=np.uint8), shape))


@pytest.mark.parametrize("kind,dtype", [("F32", np.float32), ("F16", np.float16)])
def test_dense_float_preserves_views_dtype_shape_and_validation(kind, dtype):
    data = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    actual = decode_gguf_dense(tensor(data, data.shape, kind))
    assert actual is data
    assert np.shares_memory(actual, data)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, data)
    # Same element count is insufficient; F32/F16 must not be reshaped.
    with pytest.raises(ValueError, match="tensor data.*metadata"):
        decode_gguf_dense(tensor(data.reshape(-1), data.shape, kind))


@pytest.mark.parametrize("kind", ["Q4_0", "IQ3_S", "F64", "I16", "UNKNOWN"])
def test_dense_helper_failclosed_before_payload_access(kind):
    with pytest.raises(NotImplementedError, match=kind):
        decode_gguf_dense(tensor(Unreadable(), (2, 3), kind))


def make_bf16_llama():
    cfg = make_config()
    weights = make_weights(ExternalLlamaModel(cfg))
    reader = fake_reader(cfg, weights)
    for item in reader.tensors:
        words = (item.data.view(np.uint32) >> 16).astype("<u2")
        item.tensor_type = SimpleNamespace(name="BF16")
        item.data = bf16_bytes(words)
    expected = {name: ((value.view(np.uint32) >> 16) << 16).view(np.float32)
                for name, value in weights.items()}
    return cfg, reader, expected


def test_llama_bf16_loading_restores_qk_rows_and_logits():
    cfg, reader, expected_weights = make_bf16_llama()
    mapping = GGUFWeightMap(reader)
    assert mapping.config == cfg
    assert mapping.tensor_types == ["BF16"]
    for name, expected in expected_weights.items():
        assert mapping.shape_of(name) == expected.shape
        np.testing.assert_array_equal(mapping[name].view(np.uint32), expected.view(np.uint32))
    with pytest.raises(TypeError, match="not a packed"):
        mapping.get_packed("model.embed_tokens.weight")
    actual = ExternalLlamaModel(cfg)
    actual.set_weights(mapping)
    reference = ExternalLlamaModel(cfg)
    reference.set_weights(expected_weights)
    ids = np.array([[1, 2, 3, 4]], dtype=np.int32)
    np.testing.assert_allclose(actual.forward(ids), reference.forward(ids), rtol=1e-6, atol=1e-6)


def test_llama_bf16_construction_is_lazy():
    _, reader, expected = make_bf16_llama()
    for item in reader.tensors:
        item.data = Unreadable()
    mapping = GGUFWeightMap(reader)
    assert len(mapping) == len(expected)
    for name, value in expected.items():
        assert mapping.shape_of(name) == value.shape


@pytest.mark.parametrize("kind", ["F64", "I16", "IQ3_S"])
def test_llama_unsupported_arch_and_formats_stay_failclosed(kind):
    _, reader, _ = make_bf16_llama()
    reader.fields["general.architecture"] = Field("qwen3next")
    with pytest.raises(ValueError, match="Unsupported GGUF architecture"):
        GGUFWeightMap(reader)
    reader.fields["general.architecture"] = Field("llama")
    reader.tensors[0].tensor_type = SimpleNamespace(name=kind)
    with pytest.raises(ValueError, match="Unsupported GGUF tensor type"):
        GGUFWeightMap(reader)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("kind", ["F32", "F16", "BF16"])
def test_qwen_dense_accessor_is_lazy_uncached_and_preserves_expert_shapes(monkeypatch, fused, kind):
    reader = Reader(fused=fused)
    # Keep only one explicitly requested payload readable.
    for item in reader.tensors:
        item.data = Unreadable()
    shape = (3, 256)  # Reuse the existing router's metadata shape.
    words = np.full(shape, 0x3F80, dtype=np.uint16)
    data = bf16_bytes(words) if kind == "BF16" else np.ones(shape, dtype=kind.lower().replace("f", "float"))
    target = tensor(data, shape, kind, "blk.0.ffn_gate_inp.weight")
    reader.tensors = [target if t.name == target.name else t for t in reader.tensors]
    import mini_llm.external.qwen3next_gguf as module
    calls = []

    def decode(item):
        calls.append(item.name)
        return decode_gguf_dense(item)

    monkeypatch.setattr(module, "decode_gguf_dense", decode)
    index = Qwen3NextGGUFIndex(reader)
    assert calls == []
    assert (index.expert_count, index.d_model, index.d_ff) == (3, 256, 256)
    assert tuple(reversed(index.tensors["blk.0.ffn_down_exps.weight"].shape)) == (3, 256, 256)
    np.testing.assert_array_equal(index.non_expert_weight(target.name), np.ones(shape))
    if kind == "BF16":
        target.data = bf16_bytes(np.full(shape, 0x4000, dtype=np.uint16))
    else:
        data[:] = 2
    np.testing.assert_array_equal(index.non_expert_weight(target.name), np.full(shape, 2))
    assert calls == [target.name, target.name]


@pytest.mark.parametrize("kind", ["Q4_0", "Q8_0"])
def test_qwen_quantized_accessor_decodes_only_requested_tensor_without_cache(monkeypatch, kind):
    reader = Reader(kind=kind)
    target = next(t for t in reader.tensors if t.name == "blk.0.ffn_gate_inp.weight")
    for item in reader.tensors:
        if item is not target:
            item.data = Unreadable()
    calls = []
    original = PackedGGUFWeight.dequantize_reference

    def decode(weight):
        calls.append(weight.shape)
        return original(weight)

    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", decode)
    index = Qwen3NextGGUFIndex(reader)
    assert not calls
    first = index.non_expert_weight(target.name)
    second = index.non_expert_weight(target.name)
    assert calls == [(3, 256), (3, 256)]
    assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(first, np.zeros((3, 256), dtype=np.float32))
    np.testing.assert_array_equal(second, first)


@pytest.mark.parametrize("fused", [False, True])
def test_qwen_accessor_rejects_all_expert_banks_before_decode(fused):
    reader = Reader(fused=fused)
    for item in reader.tensors:
        item.data = Unreadable()
    index = Qwen3NextGGUFIndex(reader)
    names = [name for name in index.tensors if name.endswith("_exps.weight")]
    # Also reject future routed-bank names, even absent from the directory.
    names.append("blk.0.future_exps.weight")
    for name in names:
        with pytest.raises(ValueError, match="routed expert banks"):
            index.non_expert_weight(name)


def test_qwen_accessor_unsupported_formats_ngram_arch_and_missing_name():
    reader = Reader()
    reader.tensors.extend([
        tensor(Unreadable(), (3, 256), "IQ3_S", "blk.0.attn_q.weight"),
        tensor(Unreadable(), (256,), "F64", "output_norm.weight"),
        tensor(Unreadable(), (3, 256), "BF16", "token_embd_ngram.weight"),
    ])
    index = Qwen3NextGGUFIndex(reader)
    for name, message in [("blk.0.attn_q.weight", "IQ3_S"),
                          ("output_norm.weight", "F64"),
                          ("token_embd_ngram.weight", "ngram tables")]:
        with pytest.raises(NotImplementedError, match=message):
            index.non_expert_weight(name)
    with pytest.raises(KeyError):
        index.non_expert_weight("absent.weight")
    reader.fields["general.architecture"] = Field("llama")
    with pytest.raises(ValueError, match="Expected Qwen3-Next"):
        Qwen3NextGGUFIndex(reader)


def test_optional_real_gguf_writer_reader_dense_roundtrip(tmp_path):
    gguf = pytest.importorskip("gguf")
    if not hasattr(gguf.GGMLQuantizationType, "BF16"):
        pytest.skip("installed gguf has no BF16 support")
    words = np.array([0, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80,
                      0x7FC1, 0x7F81, 0xFFC2, 1, 0x7F7F, 0x0080], dtype="<u2").reshape(2, 2, 3)
    path = tmp_path / "tiny-bf16.gguf"
    writer = gguf.GGUFWriter(path, "llama")
    try:
        writer.add_tensor("bf16.weight", bf16_bytes(words),
                          raw_dtype=gguf.GGMLQuantizationType.BF16)
        writer.add_tensor("f32.weight", np.arange(6, dtype=np.float32).reshape(2, 3))
        writer.add_tensor("f16.weight", np.arange(6, dtype=np.float16).reshape(2, 3))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    reader = gguf.GGUFReader(str(path))
    bf16, f32, f16 = reader.tensors
    assert tuple(bf16.shape) == (3, 2, 2)
    assert bf16.data.dtype == np.uint8
    assert bf16.data.shape == (2, 2, 6)
    np.testing.assert_array_equal(decode_gguf_dense(bf16).view(np.uint32),
                                  words.astype(np.uint32) << 16)
    for item in (f32, f16):
        actual = decode_gguf_dense(item)
        assert np.shares_memory(actual, item.data)
        np.testing.assert_array_equal(actual, np.arange(6).reshape(2, 3))
