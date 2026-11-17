"""Packed GGUF weight representation for shared inference backend operators.

Unlike the older GGUF mapping, constructing this object NEVER materializes a
floating-point weight matrix. The payload can remain a NumPy memmap view and
can later be transferred/decoded one expert (or tile) at a time.

`dequantize_reference` is a *CPU correctness oracle*, not a fast inference
kernel. It must never be used as the default way to load an 80B checkpoint.
K-format unpacking follows the GGML GGUF block layouts (see llama.cpp's
MIT-licensed gguf-py/gguf/quants.py). IQ tables and decoding follow pinned
GGML references; see docs/qwen4exp/0202_quantization.md for provenance.
"""

from dataclasses import dataclass
import math
import numpy as np

from .gguf_codebooks import (
    iq2s_grid, iq3s_grid, iq3xxs_grid, ksigns_iq2xs, kvalues_iq4nl,
)

# Elements per block, bytes per block. GGUF always packs along the innermost
# (GGML dim0) dimension; after canonical orientation, this is shape[-1].
BLOCK_FORMATS = {
    "Q8_0": (32, 34),
    "Q4_0": (32, 18),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
    "IQ2_S": (256, 82),
    "IQ3_S": (256, 110),
    "IQ3_XXS": (256, 98),
    "IQ4_NL": (32, 18),
    "IQ4_XS": (256, 136),
    "Q2_0": (64, 18),
}


def _f16_from_bytes(data):
    return np.frombuffer(data.copy().tobytes(), dtype="<f2").astype(np.float32)


@dataclass(frozen=True)
class PackedGGUFWeight:
    """A validated packed tensor; `data` references the original byte owner.

    Shape is in canonical row-major order, **not** the reversed dimensions
    returned by a GGUFReader tensor header. Slicing arbitrary packed rows is
    valid only at whole-block boundaries; kernels must honour the format.
    """

    data: np.ndarray
    shape: tuple[int, ...]
    quant_type: str

    def __post_init__(self):
        if self.quant_type not in BLOCK_FORMATS:
            raise ValueError(f"Unsupported packed GGUF type: {self.quant_type}")
        shape = tuple(int(d) for d in self.shape)
        if not shape or any(d <= 0 for d in shape):
            raise ValueError("Packed GGUF shape must have positive dimensions")
        block_elems, block_bytes = BLOCK_FORMATS[self.quant_type]
        if shape[-1] % block_elems:
            raise ValueError(f"GGUF {self.quant_type}: innermost dimension must be a multiple of {block_elems}")
        raw = np.asarray(self.data)
        if not raw.flags.c_contiguous:
            raise ValueError("GGUF packed data must be contiguous")
        # `.view` retains the underlying mmap owner if provided by GGUFReader.
        raw = raw.view(np.uint8).reshape(-1)
        expected = math.prod(shape) // block_elems * block_bytes
        if raw.size != expected:
            raise ValueError(f"GGUF {self.quant_type}: expected {expected} bytes, got {raw.size}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "data", raw)

    @property
    def nbytes(self):
        return int(self.data.nbytes)

    @property
    def block_elements(self):
        return BLOCK_FORMATS[self.quant_type][0]

    def select_outer(self, index: int):
        """Zero-copy index along leading dimension (e.g. one MoE expert).

        Intended for expert tensors stored as [experts, out, in]. The final
        dimension must remain block-aligned; no dequantization occurs.
        """
        if len(self.shape) < 2:
            raise ValueError("select_outer requires at least two dimensions")
        if not 0 <= int(index) < self.shape[0]:
            raise IndexError(index)
        _, block_bytes = BLOCK_FORMATS[self.quant_type]
        chunk_bytes = (math.prod(self.shape[1:]) // self.block_elements) * block_bytes
        begin = int(index) * chunk_bytes
        return PackedGGUFWeight(self.data[begin:begin + chunk_bytes],
                                self.shape[1:], self.quant_type)

    def dequantize_rows_reference(self, indices, *, tile_rows=32):
        """Decode selected rows of a matrix, preserving order and duplicates.

        Scratch decoding is bounded by tile_rows * shape[1] FP32 values;
        only the explicitly requested output rows are materialized.
        """
        if len(self.shape) != 2:
            raise ValueError("selected rows require a matrix; select_outer first")
        tile_rows = self._reference_tile_rows(tile_rows)
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("indices must be a one-dimensional integer array")
        if np.any(indices >= self.shape[0]) or np.any(indices < 0):
            raise IndexError("packed row index out of range")
        result = np.empty((len(indices), self.shape[1]), dtype=np.float32)
        row_bytes = self.nbytes // self.shape[0]
        raw = self.data.reshape(self.shape[0], row_bytes)
        for start in range(0, len(indices), tile_rows):
            selected = indices[start:start + tile_rows]
            tile = PackedGGUFWeight(raw[selected], (len(selected), self.shape[1]), self.quant_type)
            result[start:start + len(selected)] = tile.dequantize_reference()
        return result

    @staticmethod
    def _reference_tile_rows(tile_rows):
        if isinstance(tile_rows, (bool, np.bool_)) or not isinstance(tile_rows, (int, np.integer)) or tile_rows <= 0:
            raise ValueError("tile_rows must be a positive integer")
        return int(tile_rows)

    def matvec_reference(self, vector, *, tile_rows=32):
        """CPU FP32 W @ vector, decoding at most tile_rows complete rows.

        This deliberate correctness path never expands an entire expert unless
        the caller explicitly chooses a tile at least as large as the matrix.
        """
        if len(self.shape) != 2:
            raise ValueError("matvec requires a matrix; select_outer first")
        tile_rows = self._reference_tile_rows(tile_rows)
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.shape[1],):
            raise ValueError("vector shape must equal (matrix columns,)")
        result = np.empty(self.shape[0], dtype=np.float32)
        row_bytes = self.nbytes // self.shape[0]
        for start in range(0, self.shape[0], tile_rows):
            end = min(start + tile_rows, self.shape[0])
            tile = PackedGGUFWeight(self.data[start * row_bytes:end * row_bytes],
                                    (end - start, self.shape[1]), self.quant_type)
            result[start:end] = tile.dequantize_reference() @ vector
        return result

    def dequantize_reference(self):
        """Materialize an FP32 ndarray for small tests / deliberate fallback."""
        rows = self.data.reshape(-1, BLOCK_FORMATS[self.quant_type][1])
        n = len(rows)
        if self.quant_type in ("IQ2_S", "IQ3_S", "IQ3_XXS", "IQ4_NL", "IQ4_XS", "Q2_0"):
            d = _f16_from_bytes(rows[:, :2]).reshape(n, 1)
            if self.quant_type == "Q2_0":
                # Strata raw type 42: four adjacent two-bit lanes per byte.
                qs = rows[:, 2:, None] >> np.array([0, 2, 4, 6], np.uint8)
                return (d * ((qs & 3).reshape(n, 64).astype(np.float32) - 1)).reshape(self.shape)
            if self.quant_type == "IQ4_NL":
                qs = rows[:, 2:]
                values = kvalues_iq4nl[np.concatenate((qs & 15, qs >> 4), axis=1)]
                return (d * values).reshape(self.shape)
            out = np.empty((n, 256), dtype=np.float32)
            for group in range(8):
                if self.quant_type == "IQ4_XS":
                    high = rows[:, 2].astype(np.uint16) | (rows[:, 3].astype(np.uint16) << 8)
                    scale = ((rows[:, 4 + group // 2] >> (4 * (group % 2))) & 15).astype(np.int16)
                    scale |= (((high >> (2 * group)) & 3) << 4).astype(np.int16)
                    dl = d * (scale[:, None] - 32)
                    qs = rows[:, 8 + 16 * group:24 + 16 * group]
                    values = kvalues_iq4nl[np.concatenate((qs & 15, qs >> 4), axis=1)]
                    out[:, 32 * group:32 * group + 32] = dl * values
                    continue
                if self.quant_type == "IQ3_XXS":
                    aux = rows[:, 66 + 4 * group:70 + 4 * group].copy().view("<u4").reshape(n)
                    dl = (d * (np.float32(0.5) + (aux >> 28).astype(np.float32))[:, None]) * np.float32(0.5)
                elif self.quant_type == "IQ3_S":
                    scale = (rows[:, 106 + group // 2] >> (4 * (group % 2))) & 15
                    dl = d * (1 + 2 * scale.astype(np.float32))[:, None]
                for lane in range(4):
                    if self.quant_type == "IQ2_S":
                        index = rows[:, 2 + 4 * group + lane].astype(np.uint16)
                        index |= ((rows[:, 66 + group].astype(np.uint16) >> (2 * lane)) & 3) << 8
                        values = iq2s_grid[index]
                        signs = rows[:, 34 + 4 * group + lane]
                        scale = (rows[:, 74 + group] >> (4 * (lane // 2))) & 15
                        dl = (d * (np.float32(0.5) + scale.astype(np.float32))[:, None]) * np.float32(0.25)
                    else:
                        pos = 2 + 8 * group + 2 * lane
                        indices = rows[:, pos:pos + 2].astype(np.uint16)
                        if self.quant_type == "IQ3_S":
                            high = rows[:, 66 + group].astype(np.uint16)
                            indices[:, 0] |= ((high >> (2 * lane)) & 1) << 8
                            indices[:, 1] |= ((high >> (2 * lane + 1)) & 1) << 8
                            grid = iq3s_grid
                            signs = rows[:, 74 + 4 * group + lane]
                        else:
                            grid = iq3xxs_grid
                            signs = ksigns_iq2xs[(aux >> (7 * lane)) & 127]
                        values = grid[indices].reshape(n, 8)
                    sign = np.where((signs[:, None] >> np.arange(8, dtype=np.uint8)) & 1,
                                    np.float32(-1), np.float32(1))
                    begin = 32 * group + 8 * lane
                    out[:, begin:begin + 8] = (dl * values) * sign
            return out.reshape(self.shape)
        if self.quant_type == "Q8_0":
            d = _f16_from_bytes(rows[:, :2]).reshape(n, 1)
            values = rows[:, 2:].copy().view(np.int8).astype(np.float32)
            return (d * values).reshape(self.shape)
        if self.quant_type == "Q4_0":
            d = _f16_from_bytes(rows[:, :2]).reshape(n, 1)
            qs = rows[:, 2:]
            values = np.concatenate((qs & 15, qs >> 4), axis=1).astype(np.float32) - 8
            return (d * values).reshape(self.shape)
        if self.quant_type in ("Q4_K", "Q5_K"):
            # Q5_K adds a 32-byte high-bit field between the same 12-byte
            # Q4_K scale/min metadata and the low-nibble payload. All 256
            # quants still form eight sequential groups of 32 elements.
            d = _f16_from_bytes(rows[:, 0:2]).reshape(n, 1, 1)
            dmin = _f16_from_bytes(rows[:, 2:4]).reshape(n, 1, 1)
            packed_scales = rows[:, 4:16].reshape(n, 3, 4)
            lo, mid, high = np.split(packed_scales, 3, axis=1)
            scales = np.concatenate((lo & 63, (high & 15) | ((lo >> 2) & 48)), axis=-1)
            minima = np.concatenate((mid & 63, (high >> 4) | ((mid >> 2) & 48)), axis=-1)
            offset = 48 if self.quant_type == "Q5_K" else 16
            qs = rows[:, offset:].reshape(n, -1, 1, 32)
            qs = (qs >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & 15
            qs = qs.reshape(n, 8, 32)
            if self.quant_type == "Q5_K":
                high = rows[:, 16:48].reshape(n, 1, 1, 32)
                high = (high >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)) & 1
                qs = qs | (high.reshape(n, 8, 32) << np.uint8(4))
            qs = qs.astype(np.float32)
            return ((d * scales.reshape(n, 8, 1)) * qs
                    - (dmin * minima.reshape(n, 8, 1))).reshape(self.shape)
        if self.quant_type == "Q6_K":
            # Two 128-value halves: low nibbles cover 64 lanes each,
            # high two-bit planes cover 32. Signed scales apply per 16 lanes.
            lane = np.arange(256)
            low = rows[:, (lane // 128) * 64 + lane % 64]
            low = (low >> ((lane % 128) // 64 * 4)) & 15
            high = rows[:, 128 + (lane // 128) * 32 + lane % 32]
            high = (high >> ((lane % 128) // 32 * 2)) & 3
            quants = (low | (high << 4)).astype(np.float32) - 32
            scales = rows[:, 192:208].copy().view(np.int8).astype(np.float32)
            d = _f16_from_bytes(rows[:, 208:210]).reshape(n, 1, 1)
            return ((d * scales.reshape(n, 16, 1))
                    * quants.reshape(n, 16, 16)).reshape(self.shape)
        # Q3_K: low two bits in qs and a sign/third-bit mask, 16 six-bit scales.
        hmask = rows[:, :32]
        qs = rows[:, 32:96]
        scale_bytes = rows[:, 96:108]
        d = _f16_from_bytes(rows[:, 108:110]).reshape(n, 1, 1)
        lo = scale_bytes[:, :8].reshape(n, 1, 8) >> np.array([0, 4], dtype=np.uint8).reshape(1, 2, 1)
        hi = scale_bytes[:, 8:].reshape(n, 1, 4) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 4, 1)
        scales = ((lo.reshape(n, 16) & 15) | ((hi.reshape(n, 16) & 3) << 4))
        scales = scales.astype(np.float32) - 32
        low_bits = qs.reshape(n, -1, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
        low_bits = (low_bits.reshape(n, 16, 16) & 3).astype(np.int8)
        high_bits = hmask.reshape(n, -1, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)
        high_bits = (high_bits.reshape(n, 16, 16) & 1) ^ 1
        signed = low_bits - (high_bits << 2).astype(np.int8)
        return ((d * scales.reshape(n, 16, 1)) * signed).reshape(self.shape)
