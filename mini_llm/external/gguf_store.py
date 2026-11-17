"""Header-only GGUF indexing with independent, read-only shard byte owners.

No package enums are modified and no tensor payload views are built by the
reader. Logical shapes are canonical NumPy order (reversed GGML dimensions).
"""

from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path
import re
import sys

import numpy as np


_NUMBERED = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf\Z")
_SPLIT = ("split.no", "split.count", "split.tensors.count")


@dataclass(frozen=True)
class GGUFFormat:
    type_id: int
    name: str
    block_elements: int
    block_bytes: int


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    shard_index: int
    shape: tuple[int, ...]
    ggml_shape: tuple[int, ...]
    format: GGUFFormat
    relative_offset: int
    data_offset: int
    nbytes: int
    row_bytes: int
    byte_shape: tuple[int, ...]
    byte_strides: tuple[int, ...]

    @property
    def quant_type(self):
        return self.format.name


def _reader_api():
    try:
        from gguf import GGML_QUANT_SIZES, GGUFReader
    except ImportError as exc:
        raise RuntimeError("Install the optional 'gguf' package to open GGUF files") from exc

    formats = {int(k): GGUFFormat(int(k), k.name, int(v[0]), int(v[1]))
               for k, v in GGML_QUANT_SIZES.items()}
    # Verified GGML Q2_0 layout: one fp16 scale and 16 packed bytes per 64 values.
    formats[42] = GGUFFormat(42, "Q2_0", 64, 18)

    class HeaderReader(GGUFReader):
        def _get(self, offset, dtype, count=1, override_order=None):
            offset, count = int(offset), int(count)
            end = offset + np.dtype(dtype).itemsize * count
            limit = getattr(self, "_header_limit", int(self.data.size))
            if offset < 0 or count < 0 or end > limit:
                raise ValueError("Truncated GGUF header or attempted payload header read")
            return super()._get(offset, dtype, count, override_order)

        def _build_tensors(self, start_offs, fields):
            self._header_limit = int(start_offs)
            if self._header_limit > self.data.size:
                raise ValueError("Truncated GGUF header alignment padding")
            self.header_descriptors = fields
            self.tensors = []

    return HeaderReader, formats


def _discover(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob("*.gguf"))
        if not candidates:
            raise FileNotFoundError(f"No GGUF files in {path}")
        first = _NUMBERED.fullmatch(candidates[0].name)
        if len(candidates) > 1 and (first is None or any(
                (m := _NUMBERED.fullmatch(p.name)) is None or m[1] != first[1]
                for p in candidates)):
            raise ValueError("Ambiguous GGUF directory: select a single model file")
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    match = _NUMBERED.fullmatch(path.name)
    if match is None:
        return (path,), False
    total = int(match[3])
    siblings = []
    for sibling in path.parent.glob("*.gguf"):
        other = _NUMBERED.fullmatch(sibling.name)
        if other is not None and other[1] == match[1]:
            if int(other[3]) != total:
                raise ValueError("Inconsistent shard filename count")
            siblings.append((int(other[2]), sibling))
    siblings.sort()
    if total < 1 or [n for n, _ in siblings] != list(range(1, total + 1)):
        raise ValueError("Missing or invalid numbered GGUF shard set")
    return tuple(p for _, p in siblings), True


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    return int(value)


class MultiShardTensorStore:
    """An additive tensor namespace, not a model runtime or full-file decoder.

    ``metadata`` is the first-occurrence union; ``shard_metadata`` retains
    per-file GGUF.* and split.no values. Missing metadata on later shards is
    allowed, but overlapping model metadata must agree in type and value.
    """

    def __init__(self):
        self.paths = ()
        self.metadata = {}
        self.shard_metadata = []
        self.tensors = {}
        self.formats = {}
        self._owners = []
        self.closed = False

    @classmethod
    def open(cls, path):
        paths, numbered = _discover(path)
        reader_type, formats = _reader_api()
        store = cls()
        store.paths, store.formats = paths, formats
        signatures = {}
        split_totals = []
        try:
            for shard, source in enumerate(paths):
                # Explicitly reject unsupported endian layouts before reader parsing.
                with source.open("rb") as stream:
                    prefix = stream.read(8)
                if sys.byteorder != "little" or (prefix[:4] == b"GGUF" and
                        prefix[4:8] in (b"\x00\x00\x00\x02", b"\x00\x00\x00\x03")):
                    raise NotImplementedError("Only little-endian GGUF on little-endian hosts is supported")
                reader = reader_type(str(source), mode="r")
                store._owners.append(reader.data)
                metadata = {key: field.contents() for key, field in reader.fields.items()}
                store.shard_metadata.append(metadata)
                for key, value in metadata.items():
                    signature = (tuple(reader.fields[key].types), value)
                    if key != "split.no" and not key.startswith("GGUF."):
                        if key in signatures and signatures[key] != signature:
                            raise ValueError(f"Inconsistent overlapping metadata: {key}")
                        signatures[key] = signature
                    store.metadata.setdefault(key, value)
                has_split = any(key in metadata for key in _SPLIT)
                if numbered or has_split:
                    if not all(key in metadata for key in _SPLIT):
                        raise ValueError("Missing split metadata")
                    no, count, tensors = (_integer(metadata[k], k) for k in _SPLIT)
                    if no != shard or count != len(paths) or tensors < 0:
                        raise ValueError("Invalid split.no/count/tensors.count metadata")
                    split_totals.append(tensors)
                ranges = []
                for field in reader.header_descriptors:
                    _, name_data, _, dims, raw_type, raw_offset = field.parts
                    name = bytes(name_data).decode("utf-8")
                    if name in store.tensors:
                        raise ValueError(f"Duplicate tensor name: {name}")
                    ggml_shape = tuple(int(d) for d in dims)
                    if not 1 <= len(ggml_shape) <= 4 or any(d <= 0 for d in ggml_shape):
                        raise ValueError(f"{name}: invalid dimensions/layout (GGML rank must be 1..4)")
                    type_id = int(raw_type[0])
                    if type_id not in formats:
                        raise ValueError(f"{name}: unknown GGML type {type_id}")
                    fmt = formats[type_id]
                    if ggml_shape[0] % fmt.block_elements:
                        raise ValueError(f"{name}: invalid quantized row alignment")
                    shape = ggml_shape[::-1]
                    row_bytes = shape[-1] // fmt.block_elements * fmt.block_bytes
                    nbytes = math.prod(shape[:-1]) * row_bytes
                    # Convert BEFORE arithmetic: np.uint64 + int can lose offset precision.
                    relative = int(raw_offset[0])
                    absolute = int(reader.data_offset) + relative
                    if relative % int(reader.alignment):
                        raise ValueError(f"{name}: unaligned tensor offset/layout")
                    if absolute + nbytes > int(reader.data.size):
                        raise ValueError(f"{name}: truncated tensor range")
                    byte_shape = shape[:-1] + (row_bytes,)
                    strides = tuple(math.prod(byte_shape[i + 1:]) for i in range(len(shape)))
                    store.tensors[name] = TensorDescriptor(
                        name, shard, shape, ggml_shape, fmt, relative, absolute,
                        nbytes, row_bytes, byte_shape, strides)
                    ranges.append((absolute, absolute + nbytes, name))
                ranges.sort()
                for previous, current in zip(ranges, ranges[1:]):
                    if current[0] < previous[1]:
                        raise ValueError(f"Overlapping tensor ranges: {previous[2]}, {current[2]}")
                # Reader fields can retain millions of tiny header views. Keep only
                # detached metadata/descriptors and this shard's single byte owner.
                del reader
            if split_totals and (len(split_totals) != len(paths) or
                                 any(n != len(store.tensors) for n in split_totals)):
                raise ValueError("split.tensors.count does not match total tensor count")
        except (IndexError, KeyError, OverflowError) as exc:
            store.close()
            raise ValueError(f"Malformed GGUF header: {exc}") from exc
        except Exception:
            store.close()
            raise
        return store

    def __len__(self):
        return len(self.tensors)

    def _descriptor(self, name):
        if self.closed:
            raise RuntimeError("GGUF store is closed")
        return self.tensors[name]

    def tensor_bytes(self, name):
        """Lazy, flat, read-only uint8 view; never concatenate shard payloads."""
        desc = self._descriptor(name)
        return self._owners[desc.shard_index][desc.data_offset:desc.data_offset + desc.nbytes]

    def packed_weight(self, name):
        from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight

        desc = self._descriptor(name)
        if BLOCK_FORMATS.get(desc.quant_type) != (desc.format.block_elements, desc.format.block_bytes):
            raise NotImplementedError(f"{name}: packed {desc.quant_type} is not supported yet (Patch0202)")
        return PackedGGUFWeight(self.tensor_bytes(name), desc.shape, desc.quant_type)

    def packed_expert(self, name, expert):
        """Select one outer expert using the existing packed zero-copy slicer."""
        desc = self._descriptor(name)
        expert = _integer(expert, "expert")
        if len(desc.shape) < 3:
            raise ValueError("Expert access requires an outer expert dimension")
        return self.packed_weight(name).select_outer(expert)

    def rows(self, name, first, last, *, outer=()):
        """Bounded packed/dense row bytes for PLE, without decoding or gathering.

        ``outer`` fixes all axes preceding [rows, width]. The returned uint8
        matrix has shape [last-first, row_bytes], even for unsupported decoders.
        """
        desc = self._descriptor(name)
        first, last = _integer(first, "first"), _integer(last, "last")
        outer = tuple(_integer(i, "outer index") for i in outer)
        if len(desc.shape) < 2 or len(outer) != len(desc.shape) - 2:
            raise ValueError("Row access requires all outer axes fixed")
        if not 0 <= first < last <= desc.shape[-2]:
            raise IndexError("Invalid bounded row range")
        if any(not 0 <= i < size for i, size in zip(outer, desc.shape[:-2])):
            raise IndexError("Invalid outer row index")
        begin = desc.data_offset + sum(i * stride for i, stride in zip(outer, desc.byte_strides))
        begin += first * desc.row_bytes
        size = (last - first) * desc.row_bytes
        return self._owners[desc.shard_index][begin:begin + size].reshape(last - first, desc.row_bytes)

    def close(self):
        """Release ownership, never forcibly close an mmap with exported views.

        Outstanding NumPy/PackedGGUFWeight views retain their base chain and
        remain valid. Actual unmapping waits for the final view to be released.
        """
        self._owners.clear()
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError("GGUF store is closed")
        return self

    def __exit__(self, *exc):
        self.close()
