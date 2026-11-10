"""Packed projections with vocabulary rows above CUDA's grid-Y limit."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.quantized_linear import packed_to_device, quantized_linear
from mini_llm.ops.quantized_weight import PackedGGUFWeight


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Requires real CuPy/Triton execution')
@pytest.mark.parametrize('rows', [65536, 151936])
@pytest.mark.parametrize('tokens', [1, 2])
def test_large_output_projection_nondefault_stream(rows, tokens):
    q = np.broadcast_to((np.arange(rows, dtype=np.int64) % 15 - 7).astype(np.int8)[:, None],
                         (rows, 32)).copy()
    scale = np.full((rows, 1), .125, '<f2')
    raw = np.concatenate((scale.view(np.uint8), q.view(np.uint8)), axis=1).reshape(-1)
    weight = PackedGGUFWeight(raw, (rows, 32), 'Q8_0')
    hidden = np.stack([(np.arange(32, dtype=np.float32) + t) / 32 for t in range(tokens)])
    expected = hidden.sum(axis=1, keepdims=True) * (q[:, 0].astype(np.float32) * .125)[None]
    stream = xp.cuda.Stream(non_blocking=True)
    with stream:
        actual = quantized_linear(xp.asarray(hidden), packed_to_device(weight))
        host = actual.get(stream=stream)
    np.testing.assert_array_equal(host, expected)
