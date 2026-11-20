"""Exact tiled warp-dot dispatch through a single runtime-position CUDA graph."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Actual CUDA runtime positions')


def test_runtime_warp_dots_capture_advances_without_recapture(monkeypatch):
    from mini_llm.ops.qsa_graph_gpu import qsa_decode_graph_gpu, set_position_gpu
    from mini_llm.ops.inference_graph import CapturedArrayInference
    rng = np.random.default_rng(320)
    def rand(shape):
        return xp.asarray(rng.normal(0, .2, shape).astype(np.float32))
    q, qg = rand((24, 256)), rand((24, 512))
    k, v = rand((2, 256)), rand((2, 256))
    keys, values = rand((1024, 2, 256)), rand((1024, 2, 256))
    position = xp.zeros((), dtype=xp.int64)
    scale = np.float32(1/16)
    monkeypatch.setenv('MINI_LLM_QSA_WARP_DOTS', '1')
    graph = CapturedArrayInference(
        lambda query, arrays: qsa_decode_graph_gpu(query, qg, k, v, keys, values, arrays[0], scale),
        q, (position,), copy_back_mutable=False, pool_limit_bytes=2*1024**2)
    try:
        for t in (0, 1, 76, 140, 434, 1023, 1):
            set_position_gpu(position, t, 1023)
            with monkeypatch.context() as context:
                context.setenv('MINI_LLM_QSA_WARP_DOTS', '0')
                expected = qsa_decode_graph_gpu(q, qg, k, v, keys, values, position, scale)
            result = graph.forward(q, (position,))
            np.testing.assert_array_equal(result.get().view(np.uint32), expected.get().view(np.uint32))
        assert graph.calls == 7
    finally:
        graph.close()


@pytest.mark.parametrize('n', [1, 141, 435, 1024])
def test_normal_dispatch_retains_exact_serial_output(monkeypatch, n):
    from mini_llm.ops.qsa_decode_gpu import qsa_decode_gpu
    rng = np.random.default_rng(320+n)
    arrays = [xp.asarray(rng.normal(0, .2, shape).astype(np.float32)) for shape in
        ((24, 256), (24, 512), (2, 256), (2, 256), (n, 2, 256), (n, 2, 256))]
    arrays[4][-1] = xp.nan
    arrays[5][-1] = xp.nan
    monkeypatch.setenv('MINI_LLM_QSA_WARP_DOTS', '0')
    expected = qsa_decode_gpu(*arrays, n, np.float32(1/16))
    monkeypatch.setenv('MINI_LLM_QSA_WARP_DOTS', '1')
    actual = qsa_decode_gpu(*arrays, n, np.float32(1/16))
    np.testing.assert_array_equal(actual.get().view(np.uint32), expected.get().view(np.uint32))
