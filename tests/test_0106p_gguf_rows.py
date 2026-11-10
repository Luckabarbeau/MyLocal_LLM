"""Sparse GGUF row access must never expand the complete embedding matrix."""

from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.external.gguf_tensor import decode_gguf_row
from mini_llm.ops.quantized_weight import PackedGGUFWeight


def tensor(kind):
    values = (np.arange(5 * 32).reshape(5, 32) % 17 - 8).astype(np.float32) / 16
    if kind == 'Q8_0':
        q = (values * 16).astype(np.int8)
        scale = np.full((5, 1), 1 / 16, dtype='<f2')
        data = np.concatenate((scale.view(np.uint8), q.view(np.uint8)), axis=1)
    elif kind == 'BF16':
        data = (values.view(np.uint32) >> 16).astype('<u2').view(np.uint8)
    else:
        data = values.astype(np.float16 if kind == 'F16' else np.float32)
    return SimpleNamespace(name='embedding', shape=(32, 5), data=data,
                           tensor_type=SimpleNamespace(name=kind)), values


@pytest.mark.parametrize('kind', ['F32', 'F16', 'BF16', 'Q8_0'])
def test_selected_rows_only_and_owner_unchanged(kind, monkeypatch):
    weight, expected = tensor(kind)
    before = weight.data.copy()
    writable = weight.data.flags.writeable
    decode = PackedGGUFWeight.dequantize_reference
    shapes = []
    def selected_only(self):
        shapes.append(self.shape)
        assert self.shape == (32,), 'Whole embedding table expansion is forbidden'
        return decode(self)
    monkeypatch.setattr(PackedGGUFWeight, 'dequantize_reference', selected_only)
    for row in (4, 0, 2):
        np.testing.assert_array_equal(decode_gguf_row(weight, row), expected[row])
    assert len(shapes) == (3 if kind == 'Q8_0' else 0)
    np.testing.assert_array_equal(weight.data, before)
    assert weight.data.flags.writeable == writable


@pytest.mark.parametrize('row,error', [(-1, IndexError), (5, IndexError), (True, TypeError), (1., TypeError)])
def test_row_validation(row, error):
    weight, _ = tensor('Q8_0')
    with pytest.raises(error):
        decode_gguf_row(weight, row)


def test_bf16_big_endian_words_and_invalid_payload():
    weight, expected = tensor('BF16')
    weight.data = weight.data.view('<u2').astype('>u2')
    np.testing.assert_array_equal(decode_gguf_row(weight, 3), expected[3])
    weight.data = weight.data[:, :-1]
    with pytest.raises(ValueError, match='payload'):
        decode_gguf_row(weight, 0)
