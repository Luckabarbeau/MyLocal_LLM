"""Real minimal GGUF binaries, written without GGUFWriter or checkpoints."""

import gc
import math
import struct
import weakref

import numpy as np
import pytest


gguf = pytest.importorskip("gguf")

from mini_llm.external.gguf_store import MultiShardTensorStore, _reader_api
from mini_llm.ops.quantized_weight import PackedGGUFWeight


U16, U32, I32, STRING, ARRAY = 2, 4, 5, 8, 9


def string(value, endian="<"):
    value = value.encode("utf-8")
    return struct.pack(endian + "Q", len(value)) + value


def value_bytes(kind, value, endian):
    if kind == STRING:
        return string(value, endian)
    if kind == ARRAY:
        subtype, items = value
        return (struct.pack(endian + "IQ", subtype, len(items)) +
                b"".join(value_bytes(subtype, item, endian) for item in items))
    return struct.pack(endian + {U16: "H", U32: "I", I32: "i"}[kind], value)


def write_gguf(path, tensors=(), metadata=(), *, endian="<", alignment=32):
    """Tensor tuple: name, GGML dimensions, raw type, relative offset, payload.

    payload=None writes only the descriptor (for corrupt-range fixtures).
    Sparse seeks allow testing >4GiB offsets without allocating those bytes.
    """
    header = b"GGUF" + struct.pack(endian + "IQQ", 3, len(tensors), len(metadata))
    for key, kind, value in metadata:
        header += string(key, endian) + struct.pack(endian + "I", kind)
        header += value_bytes(kind, value, endian)
    for name, dims, kind, offset, _ in tensors:
        header += string(name, endian) + struct.pack(endian + "I", len(dims))
        header += struct.pack(endian + "Q" * len(dims), *dims)
        header += struct.pack(endian + "IQ", kind, offset)
    data_offset = (len(header) + alignment - 1) // alignment * alignment
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(b"\x00" * (data_offset - len(header)))
        for _, _, _, offset, payload in tensors:
            if payload is not None:
                stream.seek(data_offset + offset)
                stream.write(payload)
    return data_offset


def split(no, count=2, tensors=2):
    return [("split.no", U16, no), ("split.count", U16, count),
            ("split.tensors.count", I32, tensors)]


def pair(tmp_path, *, metadata0=(), metadata1=(), tensor1=None):
    a = tmp_path / "model-00001-of-00002.gguf"
    b = tmp_path / "model-00002-of-00002.gguf"
    write_gguf(a, [("experts", (32, 2, 3), 8, 0, bytes(range(204)))],
               split(0) + list(metadata0))
    write_gguf(b, [tensor1 or ("ple", (64, 3), 42, 0, bytes(range(54)))],
               split(1) + list(metadata1))
    return a, b


def test_multishard_namespace_zero_copy_and_lifetime(tmp_path):
    a, b = pair(tmp_path, metadata0=[("general.architecture", STRING, "qwen4exp"),
                                     ("tokens", ARRAY, (STRING, ["a", "b", "c"]))])
    store = MultiShardTensorStore.open(b)  # either numbered shard discovers both
    assert store.paths == (a, b)
    assert len(store) == 2
    assert store.metadata["tokens"] == ["a", "b", "c"]
    assert store.metadata["general.architecture"] == "qwen4exp"
    assert store.shard_metadata[1]["GGUF.kv_count"] == 3
    assert store.shard_metadata[1]["split.no"] == 1
    assert len(store._owners) == 2
    assert all(isinstance(owner, np.memmap) and owner.mode == "r" for owner in store._owners)
    assert store._owners[0]._mmap is not store._owners[1]._mmap
    raw = store.tensor_bytes("experts")
    expert = store.packed_expert("experts", 2)
    assert expert.shape == (2, 32)
    assert np.shares_memory(raw, expert.data)
    np.testing.assert_array_equal(expert.data, np.arange(136, 204, dtype=np.uint8))
    assert not raw.flags.writeable and not expert.data.flags.writeable
    with pytest.raises(ValueError):
        raw[0] = 1
    owner = weakref.ref(store._owners[0])
    store.close()
    store.close()
    gc.collect()
    assert owner() is not None and not owner()._mmap.closed
    np.testing.assert_array_equal(expert.data, np.arange(136, 204, dtype=np.uint8))
    with pytest.raises(RuntimeError, match="closed"):
        store.tensor_bytes("experts")
    del raw, expert
    gc.collect()
    assert owner() is None
    with MultiShardTensorStore.open(tmp_path) as directory_store:
        assert len(directory_store) == 2
    assert directory_store.closed


def test_header_only_reader_and_no_decode(tmp_path, monkeypatch):
    a, _ = pair(tmp_path)
    requests = []
    get = gguf.GGUFReader._get

    def track(self, offset, dtype, count=1, override_order=None):
        requests.append((self, int(offset), int(offset) + np.dtype(dtype).itemsize * int(count)))
        return get(self, offset, dtype, count, override_order)

    def forbidden(*args, **kwargs):
        pytest.fail("Normal payload builder/dequantizer must not run")

    monkeypatch.setattr(gguf.GGUFReader, "_get", track)
    monkeypatch.setattr(gguf.GGUFReader, "_build_tensors", forbidden)
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", forbidden)
    with MultiShardTensorStore.open(a) as store:
        assert len(store) == 2
        assert requests
        assert all(0 <= begin <= end <= reader.data_offset for reader, begin, end in requests)
        assert all(reader.tensors == [] for reader, _, _ in requests)
        reader = requests[0][0]
        with pytest.raises(ValueError, match="payload header read"):
            reader._get(reader.data_offset, np.uint8)


def test_formats_shapes_strides_and_bounded_ple_rows(tmp_path, monkeypatch):
    formats = [(0, (4, 3), 48), (42, (64, 3), 54), (21, (256, 3), 330),
               (22, (256, 3), 246)]
    tensors = []
    offset = 0
    for kind, dims, size in formats:
        tensors.append((str(kind), dims, kind, offset, bytes(i % 256 for i in range(size))))
        offset = (offset + size + 31) // 32 * 32
    path = tmp_path / "formats.gguf"
    write_gguf(path, tensors)
    enum_members = dict(gguf.GGMLQuantizationType.__members__)
    # Indexing and raw rows do not consult the packed backend's format table.
    monkeypatch.setattr("mini_llm.ops.quantized_weight.BLOCK_FORMATS", {})
    with MultiShardTensorStore.open(path) as store:
        for kind, dims, size in formats:
            desc = store.tensors[str(kind)]
            assert desc.shape == dims[::-1]
            assert desc.ggml_shape == dims
            assert desc.nbytes == size
            assert desc.byte_shape == (3, size // 3)
            assert desc.byte_strides == (size // 3, 1)
            rows = store.rows(str(kind), 1, 3)
            assert rows.shape == (2, size // 3)
            assert np.shares_memory(rows, store.tensor_bytes(str(kind)))
            np.testing.assert_array_equal(rows.ravel(), store.tensor_bytes(str(kind))[size // 3:])
            with pytest.raises(NotImplementedError, match="not supported yet"):
                store.packed_weight(str(kind))
        assert store.tensors["42"].quant_type == "Q2_0"
        assert store.tensors["42"].format.block_elements == 64
        assert store.tensors["42"].format.block_bytes == 18
    assert dict(gguf.GGMLQuantizationType.__members__) == enum_members


def test_outer_rows_and_packed_expert_reuse(tmp_path, monkeypatch):
    a, _ = pair(tmp_path)
    select = PackedGGUFWeight.select_outer
    calls = []

    def selected(self, index):
        calls.append(index)
        return select(self, index)

    monkeypatch.setattr(PackedGGUFWeight, "select_outer", selected)
    with MultiShardTensorStore.open(a) as store:
        desc = store.tensors["experts"]
        assert desc.shape == (3, 2, 32)
        assert desc.byte_strides == (68, 34, 1)
        rows = store.rows("experts", 1, 2, outer=(2,))
        np.testing.assert_array_equal(rows.ravel(), np.arange(170, 204, dtype=np.uint8))
        assert np.shares_memory(rows, store.packed_expert("experts", 2).data)
        assert calls == [2]
        for first, last, outer, error in [(0, 1, (), ValueError), (-1, 1, (0,), IndexError),
                (0, 3, (0,), IndexError), (0, 0, (0,), IndexError),
                (0, 1, (3,), IndexError), (True, 1, (0,), ValueError)]:
            with pytest.raises(error):
                store.rows("experts", first, last, outer=outer)
        with pytest.raises(IndexError):
            store.packed_expert("experts", 3)
        with pytest.raises(ValueError, match="expert dimension"):
            store.packed_expert("ple", 0)


def test_metadata_overlap_union_and_arrays(tmp_path):
    common = [("tags", ARRAY, (STRING, ["one", "two"])),
              ("numbers", ARRAY, (U32, [1, 2, 3]))]
    a, _ = pair(tmp_path, metadata0=common, metadata1=common + [("later", U32, 7)])
    with MultiShardTensorStore.open(a) as store:
        assert store.metadata["tags"] == ["one", "two"]
        assert store.metadata["numbers"] == [1, 2, 3]
        assert store.metadata["later"] == 7


@pytest.mark.parametrize("later", [[("tags", ARRAY, (STRING, ["one", "DIFFERENT"]))],
                                   [("tags", STRING, "one")]])
def test_inconsistent_metadata(tmp_path, later):
    a, _ = pair(tmp_path, metadata0=[("tags", ARRAY, (STRING, ["one", "two"]))], metadata1=later)
    with pytest.raises(ValueError, match="overlapping metadata: tags"):
        MultiShardTensorStore.open(a)


@pytest.mark.parametrize("changes", [split(0), split(1, count=3), split(1, tensors=3),
                                      split(1, tensors=1), split(1)[:2], []])
def test_invalid_split_metadata(tmp_path, changes):
    a, b = pair(tmp_path)
    write_gguf(b, [("ple", (64, 3), 42, 0, bytes(54))], changes)
    with pytest.raises(ValueError, match="split|metadata"):
        MultiShardTensorStore.open(a)


@pytest.mark.parametrize("across_shards", [False, True])
def test_duplicate_tensor_names(tmp_path, across_shards):
    if across_shards:
        path, _ = pair(tmp_path, tensor1=("experts", (64, 3), 42, 0, bytes(54)))
    else:
        path = tmp_path / "duplicate.gguf"
        write_gguf(path, [("same", (1,), 0, 0, bytes(4)), ("same", (1,), 0, 32, bytes(4))])
    with pytest.raises(ValueError, match="Duplicate tensor name"):
        MultiShardTensorStore.open(path)


@pytest.mark.parametrize("tensors,message", [
    ([("x", (), 0, 0, bytes(4))], "dimensions/layout"),
    ([("x", (0, 3), 0, 0, bytes(4))], "dimensions/layout"),
    ([("x", (1, 1, 1, 1, 1), 0, 0, bytes(4))], "dimensions/layout"),
    ([("x", (16, 2), 8, 0, bytes(34))], "row alignment"),
    ([("x", (1,), 999, 0, bytes(4))], "unknown GGML"),
    ([("x", (1,), 0, 1, bytes(4))], "unaligned tensor"),
    ([("x", (2,), 0, 0, bytes(4))], "truncated tensor"),
    ([("x", (1,), 0, 2**64 - 32, None)], "truncated tensor"),
    ([("x", (16,), 0, 0, bytes(64)), ("y", (1,), 0, 32, bytes(4))], "Overlapping"),
])
def test_invalid_descriptors(tmp_path, tensors, message):
    path = tmp_path / "invalid.gguf"
    write_gguf(path, tensors)
    with pytest.raises(ValueError, match=message):
        MultiShardTensorStore.open(path)


def test_large_sparse_uint64_offset(tmp_path):
    path = tmp_path / "sparse.gguf"
    offset = 2**32 + 32
    data_offset = write_gguf(path, [("far", (2,), 0, offset, b"abcdefgh")])
    assert path.stat().st_blocks * 512 < 1024 * 1024
    with MultiShardTensorStore.open(path) as store:
        desc = store.tensors["far"]
        assert type(desc.data_offset) is int and type(desc.relative_offset) is int
        assert desc.data_offset == data_offset + offset
        assert store.tensor_bytes("far").tobytes() == b"abcdefgh"
        assert np.shares_memory(store.tensor_bytes("far"), store._owners[0])


@pytest.mark.parametrize("endian", ["<", ">"])
def test_endian_and_alignment(tmp_path, endian):
    path = tmp_path / "endian.gguf"
    write_gguf(path, [("x", (4,), 0, 0, bytes(16))],
               [("general.alignment", U32, 64)], endian=endian, alignment=64)
    if endian == ">":
        with pytest.raises(NotImplementedError, match="little-endian"):
            MultiShardTensorStore.open(path)
    else:
        with MultiShardTensorStore.open(path) as store:
            assert store.tensors["x"].data_offset % 64 == 0


@pytest.mark.parametrize("corruption", ["short", "magic", "dims", "alignment", "duplicate_metadata"])
def test_malformed_header(tmp_path, corruption):
    path = tmp_path / "malformed.gguf"
    if corruption == "short":
        path.write_bytes(b"GGUF")
    elif corruption == "magic":
        path.write_bytes(b"NOPE" + bytes(28))
    elif corruption == "dims":
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + string("x") + struct.pack("<I", 2**32 - 1))
    elif corruption == "alignment":
        write_gguf(path, metadata=[("general.alignment", U32, 3)])
    else:
        write_gguf(path, metadata=[("a", U32, 1), ("a", U32, 1)])
    with pytest.raises(ValueError):
        MultiShardTensorStore.open(path)


def test_discovery_missing_invalid_and_ambiguous(tmp_path):
    a, b = pair(tmp_path)
    b.unlink()
    with pytest.raises(ValueError, match="Missing"):
        MultiShardTensorStore.open(a)
    a.rename(tmp_path / "model-00000-of-00001.gguf")
    with pytest.raises(ValueError, match="invalid numbered"):
        MultiShardTensorStore.open(tmp_path)


def test_discovery_mixed_counts(tmp_path):
    a, b = pair(tmp_path)
    b.rename(tmp_path / "model-00002-of-00003.gguf")
    with pytest.raises(ValueError, match="filename count"):
        MultiShardTensorStore.open(a)


def test_discovery_ambiguous_and_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        MultiShardTensorStore.open(tmp_path)
    write_gguf(tmp_path / "a.gguf")
    write_gguf(tmp_path / "b.gguf")
    with pytest.raises(ValueError, match="Ambiguous"):
        MultiShardTensorStore.open(tmp_path)
    with pytest.raises(FileNotFoundError):
        MultiShardTensorStore.open(tmp_path / "missing.gguf")


def test_unnumbered_split_rejected(tmp_path):
    path = tmp_path / "unnumbered.gguf"
    write_gguf(path, metadata=split(0))
    with pytest.raises(ValueError, match="split.no/count"):
        MultiShardTensorStore.open(path)


def test_installed_size_mapping():
    _, formats = _reader_api()
    for kind, (elements, size) in gguf.GGML_QUANT_SIZES.items():
        if int(kind) != 42:
            assert (formats[int(kind)].name, formats[int(kind)].block_elements,
                    formats[int(kind)].block_bytes) == (kind.name, elements, size)
    assert math.prod((formats[42].block_elements, formats[42].block_bytes)) == 64 * 18
