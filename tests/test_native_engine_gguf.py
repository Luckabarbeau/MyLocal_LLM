"""Native header/mmap tests; no dense model loading or tensor decoding."""
import ctypes as C
import os
import struct

import pytest

from mini_llm.native import Runtime, NativeError
from mini_llm.external.gguf_store import MultiShardTensorStore
from test_0201_gguf_store import write_gguf, split, STRING, ARRAY, U32, pair


@pytest.fixture
def runtime():
    return Runtime()


def test_descriptor_metadata_parity(runtime, tmp_path):
    path = tmp_path / "tiny.gguf"
    metadata = [("general.architecture", STRING, "test"),
                ("tokens", ARRAY, (STRING, ["hello", "λ"])),
                ("general.alignment", U32, 64)]
    write_gguf(path, [("dense", (4, 3), 0, 0, bytes(48)),
                      ("packed", (32, 2, 3), 8, 64, bytes(204)),
                      ("q2", (64, 2), 42, 320, bytes(36))], metadata, alignment=64)
    with runtime.gguf(path) as native, MultiShardTensorStore.open(path) as reference:
        assert native.paths == reference.paths
        for key, value in native.metadata.items():
            assert value == reference.metadata[key]
        for name, actual in native.tensors.items():
            expected = reference.tensors[name]
            for field in ("name", "shard_index", "shape", "ggml_shape", "relative_offset",
                          "data_offset", "nbytes", "row_bytes", "byte_shape", "byte_strides"):
                assert getattr(actual, field) == getattr(expected, field)
            assert actual.type_id == expected.format.type_id
            assert actual.block_elements == expected.format.block_elements
            assert actual.block_bytes == expected.format.block_bytes
        # One read-only OS mapping; no tensor payload copies or exported raw views.
        maps = open("/proc/self/maps").read().splitlines()
        assert any(str(path) in line and "r--p" in line for line in maps)
    assert not any(str(path) in line for line in open("/proc/self/maps"))


@pytest.mark.parametrize("name,type_id,elements,block_bytes", [
    ("IQ3_XXS", 18, 256, 98), ("IQ4_NL", 20, 32, 18),
    ("IQ3_S", 21, 256, 110), ("IQ2_S", 22, 256, 82),
    ("IQ4_XS", 23, 256, 136),
])
def test_iq_descriptor_parity(runtime, tmp_path, name, type_id, elements, block_bytes):
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
    from mini_llm.ops.quantized_weight import BLOCK_FORMATS

    assert int(GGMLQuantizationType[name]) == type_id
    assert GGML_QUANT_SIZES[GGMLQuantizationType[name]] == (elements, block_bytes)
    assert BLOCK_FORMATS[name] == (elements, block_bytes)
    path = tmp_path / "iq.gguf"
    size = 2 * 3 * 2 * block_bytes
    write_gguf(path, [(name, (2 * elements, 3, 2), type_id, 32, bytes(size))])
    with runtime.gguf(path) as native, MultiShardTensorStore.open(path) as reference:
        actual, expected = native.tensors[name], reference.tensors[name]
        for field in ("name", "shard_index", "shape", "ggml_shape", "relative_offset",
                      "data_offset", "nbytes", "row_bytes", "byte_shape", "byte_strides"):
            assert getattr(actual, field) == getattr(expected, field)
        assert actual.type_id == expected.format.type_id == type_id
        assert actual.block_elements == expected.format.block_elements == elements
        assert actual.block_bytes == expected.format.block_bytes == block_bytes
    write_gguf(path, [(name, (elements - 1,), type_id, 0, None)])
    with pytest.raises(NativeError, match="row alignment"):
        runtime.gguf(path)
    write_gguf(path, [(name, (elements,), type_id, 0, bytes(block_bytes - 1))])
    with pytest.raises(NativeError, match="truncated"):
        runtime.gguf(path)


def test_shards_parity_and_identity(runtime, tmp_path):
    a, b = pair(tmp_path)
    with runtime.gguf(b) as native, MultiShardTensorStore.open(a) as reference:
        assert native.paths == reference.paths
        assert {n: (d.shard_index, d.shape, d.nbytes) for n, d in native.tensors.items()} == {
            n: (d.shard_index, d.shape, d.nbytes) for n, d in reference.tensors.items()}
    paths = (C.c_char_p * 2)(os.fsencode(a), os.fsencode(a))
    out = C.c_uint64(123)
    assert runtime._lib.ml_gguf_open(paths, 2, C.byref(out)) == 1
    assert out.value == 0
    assert b"identity" in runtime._lib.ml_last_error()


@pytest.mark.parametrize("case", ["duplicate", "split", "metadata"])
def test_invalid_shard_namespace(runtime, tmp_path, case):
    a, b = pair(tmp_path)
    name = "experts" if case == "duplicate" else "other"
    md = split(0 if case == "split" else 1)
    if case == "metadata":
        write_gguf(a, [("experts", (32,), 8, 0, bytes(34))], split(0) + [("x", U32, 1)])
        md += [("x", U32, 2)]
    write_gguf(b, [(name, (32,), 8, 0, bytes(34))], md)
    with pytest.raises(NativeError):
        runtime.gguf(a)


@pytest.mark.parametrize("dims,kind,offset,payload,match", [
    ((4,), 0, 1, bytes(16), "unaligned"),
    ((4,), 0, 0, None, "truncated"),
    ((0,), 0, 0, None, "dimension"),
    ((1,) * 5, 0, 0, None, "rank"),
    ((31,), 8, 0, None, "row alignment"),
    ((4,), 999, 0, None, "unsupported"),
    ((2**63, 4), 0, 0, None, "overflow"),
    ((4,), 0, 2**64 - 32, None, "overflow"),
])
def test_invalid_tensor_layout(runtime, tmp_path, dims, kind, offset, payload, match):
    path = tmp_path / "bad.gguf"
    write_gguf(path, [("x", dims, kind, offset, payload)])
    with pytest.raises(NativeError, match=match):
        runtime.gguf(path)


def test_overlap_alignment_header_and_version(runtime, tmp_path):
    path = tmp_path / "bad.gguf"
    write_gguf(path, [("a", (16,), 0, 0, bytes(64)), ("b", (4,), 0, 32, bytes(16))])
    with pytest.raises(NativeError, match="overlapping"):
        runtime.gguf(path)
    for alignment in (0, 3):
        write_gguf(path, metadata=[("general.alignment", U32, alignment)])
        with pytest.raises(NativeError, match="alignment"):
            runtime.gguf(path)
    for raw in (b"GGUF" + struct.pack("<IQQ", 1, 0, 0),
                b"GGUF" + struct.pack(">IQQ", 3, 0, 0),
                b"GGUF" + struct.pack("<IQQ", 3, 0, 2**64 - 1)):
        path.write_bytes(raw)
        with pytest.raises(NativeError):
            runtime.gguf(path)


def test_cleanup_and_abi(runtime, tmp_path):
    path = tmp_path / "tiny.gguf"
    write_gguf(path)
    baseline = len(os.listdir("/proc/self/fd"))
    for _ in range(10):
        store = runtime.gguf(path)
        stale = store._handle
        store.close()
        store.close()
        assert runtime._lib.ml_gguf_destroy(stale) == 1
    assert len(os.listdir("/proc/self/fd")) == baseline
    assert runtime._lib.ml_gguf_destroy(0) == 0
    assert runtime._lib.ml_gguf_open(None, 1, None) == 1
    out = C.c_uint64(12)
    assert runtime._lib.ml_gguf_open(None, 1, C.byref(out)) == 1
    assert out.value == 0
    write_gguf(path, [("x", (4,), 0, 0, None)])
    for _ in range(10):
        with pytest.raises(NativeError):
            runtime.gguf(path)
    assert len(os.listdir("/proc/self/fd")) == baseline
    assert not any(str(path) in line for line in open("/proc/self/maps"))
