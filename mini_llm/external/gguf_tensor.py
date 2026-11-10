"""CPU decoding for individual, unquantized GGUF tensors only."""

import math
from numbers import Integral
from types import SimpleNamespace

import numpy as np


def decode_gguf_dense(tensor):
    """Return values in canonical row-major (reversed GGUF metadata) shape.

    F32/F16 retain the reader's array/view and strict shape validation. BF16
    readers normally expose packed uint8 bytes, not numeric float16 values.
    Unsigned 16-bit fixtures are also accepted, respecting their dtype endian.
    No quantized formats or model-wide conversions are performed here.
    """
    kind = tensor.tensor_type.name
    if kind not in {"F32", "F16", "BF16"}:
        raise NotImplementedError(f"{tensor.name}: unsupported dense GGUF type {kind}")
    shape = tuple(int(d) for d in reversed(tensor.shape))
    values = np.asarray(tensor.data)
    if kind != "BF16":
        if values.shape != shape:
            raise ValueError(
                f"GGUF {tensor.name}: tensor data {values.shape} != metadata {shape}"
            )
        return values

    if any(d <= 0 for d in shape):
        raise ValueError(f"GGUF {tensor.name}: invalid BF16 shape {shape}")
    expected_bytes = math.prod(shape) * 2
    if values.nbytes != expected_bytes:
        raise ValueError(
            f"GGUF {tensor.name}: BF16 payload has {values.nbytes} bytes; "
            f"expected {expected_bytes} for shape {shape}"
        )
    if values.dtype == np.dtype("uint8"):
        # Byte-packed reader dimensions need not match the logical shape.
        words = np.ascontiguousarray(values).reshape(-1).view("<u2")
    elif values.dtype.kind == "u" and values.dtype.itemsize == 2:
        words = values.reshape(-1)
    else:
        raise ValueError(
            f"GGUF {tensor.name}: BF16 payload must be uint8 bytes or uint16 words, "
            f"not {values.dtype}"
        )
    # Integer expansion followed by a bit view preserves signed zero and all
    # NaN payloads (including signaling NaNs); no floating-point cast occurs.
    bits = words.astype(np.uint32) << 16
    return bits.view(np.float32).reshape(shape)


def decode_gguf_row(tensor, row):
    """Read one canonical [rows,width] row without expanding the whole tensor.

    Packed blocks stay mmap-backed until the selected row is decoded. BF16 is
    sliced as bytes before bit expansion. Returned dense rows may be views;
    callers that cache or mutate them must take ownership explicitly.
    """
    from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight

    shape = tuple(int(d) for d in reversed(tensor.shape))
    if len(shape) != 2 or any(d <= 0 for d in shape):
        raise ValueError(f'{tensor.name}: row access requires a positive matrix shape')
    if isinstance(row, bool) or not isinstance(row, Integral):
        raise TypeError('row must be an integer')
    if not 0 <= row < shape[0]:
        raise IndexError(row)
    kind = tensor.tensor_type.name
    if kind in BLOCK_FORMATS:
        return PackedGGUFWeight(tensor.data, shape, kind).select_outer(row).dequantize_reference()
    if kind in {'F32', 'F16'}:
        return decode_gguf_dense(tensor)[row]
    if kind != 'BF16':
        raise NotImplementedError(f'{tensor.name}: unsupported row type {kind}')
    data = np.asarray(tensor.data)
    if not data.flags.c_contiguous or data.nbytes != math.prod(shape) * 2:
        raise ValueError(f'{tensor.name}: invalid contiguous BF16 matrix payload')
    # Preserve endian handling of uint16 fixtures through the shared decoder.
    width = shape[1]
    if data.dtype.kind == 'u' and data.dtype.itemsize == 2:
        payload = data.reshape(shape)[row]
    elif data.dtype == np.dtype('uint8'):
        payload = data.reshape(-1)[row * width * 2:(row + 1) * width * 2]
    else:
        raise ValueError(f'{tensor.name}: BF16 row payload must be uint8 or uint16')
    return decode_gguf_dense(SimpleNamespace(name=tensor.name, shape=(width,),
                             tensor_type=tensor.tensor_type, data=payload))
