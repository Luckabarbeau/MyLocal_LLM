"""Public inference matmul: broadcasting/strides, FP64 oracle and graph replay."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.inference_matmul import matmul_inference, supports_inference_gemv


@pytest.mark.parametrize('shapes', [((1, 7), (7, 5)), ((1, 1, 32), (32, 19)),
    ((2, 3, 1, 17), (2, 3, 17, 11)), ((2, 1, 1, 17), (1, 3, 17, 11)),
    ((1, 1, 128), (3, 128, 67)), ((1, 513), (513, 41)), ((1, 2048), (2048, 512))])
@pytest.mark.parametrize('layout', ['normal', 'transposed_weight', 'strided_reversed'])
def test_fp32_geometry_strides_and_oracle(shapes, layout):
    rng = np.random.default_rng(10624)
    a = rng.normal(0, .1, shapes[0]).astype(np.float32)
    b = rng.normal(0, .1, shapes[1]).astype(np.float32)
    if layout == 'transposed_weight':
        b = xp.asarray(np.swapaxes(b, -1, -2).copy()).swapaxes(-1, -2)
        a = xp.asarray(a)
    elif layout == 'strided_reversed':
        large = np.zeros((*shapes[1][:-1], shapes[1][-1] * 2), np.float32)
        large[..., ::2] = b
        b = xp.asarray(large)[..., ::2][..., ::-1]
        a = xp.asarray(a)[..., ::-1]
        b = b[..., ::-1, :]
    else:
        a, b = xp.asarray(a), xp.asarray(b)
    aa, bb = asnumpy(a), asnumpy(b)
    expected = aa.astype(np.float64) @ bb.astype(np.float64)
    before_a, before_b = aa.copy(), bb.copy()
    actual = matmul_inference(a, b)
    np.testing.assert_allclose(asnumpy(actual), expected, rtol=3e-5, atol=2e-6)
    assert actual.shape == expected.shape and actual.dtype == xp.float32
    np.testing.assert_array_equal(asnumpy(a), before_a)
    np.testing.assert_array_equal(asnumpy(b), before_b)
    assert supports_inference_gemv(a, b) == (BACKEND_NAME == 'cupy')
    out = xp.empty(expected.shape, dtype=xp.float32)
    assert matmul_inference(a, b, out=out) is out
    np.testing.assert_array_equal(asnumpy(out), asnumpy(actual))


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_general_matrix_fallback(dtype):
    a = xp.asarray(np.arange(15).reshape(3, 5), dtype=dtype)
    b = xp.asarray(np.arange(20).reshape(5, 4), dtype=dtype)
    assert not supports_inference_gemv(a, b)
    np.testing.assert_array_equal(asnumpy(matmul_inference(a, b)), asnumpy(a @ b))


def test_alias_and_strided_output_fallback():
    a = xp.asarray([[1., 2., 3.]], dtype=xp.float32)
    b = xp.eye(3, dtype=xp.float32) * 2
    matmul_inference(a, b, out=a)
    np.testing.assert_array_equal(asnumpy(a), [[2, 4, 6]])
    full = xp.zeros((1, 6), dtype=xp.float32)
    result = matmul_inference(a, b, out=full[:, ::2])
    np.testing.assert_array_equal(asnumpy(result), [[4, 8, 12]])
    np.testing.assert_array_equal(asnumpy(full[:, 1::2]), 0)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Requires CUDA graphs')
def test_public_api_graph_replay_and_nondefault_stream():
    from mini_llm.ops.inference_graph import CapturedArrayInference
    with xp.cuda.Stream(non_blocking=True) as stream:
        a = xp.ones((1, 32), dtype=xp.float32)
        b = xp.asarray(np.random.default_rng(24).normal(size=(32, 11)).astype(np.float32))
        graph = CapturedArrayInference(lambda x, states: matmul_inference(x, b), a, ())
        for factor in (1, -2, .25):
            value = a * factor
            actual = graph.forward(value, ())
            stream.synchronize()
            np.testing.assert_allclose(asnumpy(actual), asnumpy(value).astype(np.float64) @
                asnumpy(b).astype(np.float64), rtol=3e-5, atol=2e-6)
        assert graph.statistics()['calls'] == 3
