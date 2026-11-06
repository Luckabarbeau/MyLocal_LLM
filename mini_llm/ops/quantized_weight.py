"""Packed GGUF weight representation for shared inference backend operators.

Unlike the older GGUF mapping, constructing this object NEVER materializes a
floating-point weight matrix. The payload can remain a NumPy memmap view and
can later be transferred/decoded one expert (or tile) at a time.

`dequantize_reference` is a *CPU correctness oracle*, not a fast inference
kernel. It must never be used as the default way to load an 80B checkpoint.
K-format unpacking follows the GGML GGUF block layouts (see llama.cpp's
MIT-licensed gguf-py/gguf/quants.py). No llama.cpp code is vendored here.
"""

from dataclasses import dataclass
import math
import numpy as np

# Elements per block, bytes per block. GGUF always packs along the innermost
# (GGML dim0) dimension; after canonical orientation, this is shape[-1].
BLOCK_FORMATS = {
    "Q8_0": (32, 34),
    "Q4_0": (32, 18),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
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

    def dequantize_reference(self):
        """Materialize an FP32 ndarray for small tests / deliberate fallback."""
        rows = self.data.reshape(-1, BLOCK_FORMATS[self.quant_type][1])
        n = len(rows)
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
