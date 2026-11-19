"""Original SIMD integer dot versus independently loaded pinned GGML oracle."""
import ctypes as C
import os
import time

import numpy as np
import pytest

from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPUExecutor, GGMLPackedCPURowPool
from test_0256_ggml_packed_cpu import selected_library, packed, DOT


@pytest.fixture(scope='module')
def pair(selected_library):
    old = GGMLPackedCPUExecutor(selected_library, q2_simd=False)
    if not old._bridge.mylocal_q2_avx2_available():
        pytest.skip('AVX2/F16C unavailable')
    return old, GGMLPackedCPUExecutor(selected_library, q2_simd=True)


@pytest.mark.parametrize('k', [64, 128, 192, 320, 640, 768, 2560])
@pytest.mark.parametrize('seed', [261, 262, 263])
def test_bit_exact_linear(pair, k, seed):
    old, new = pair
    weight = packed('Q2_0', n=31, k=k, seed=seed)
    x = np.random.default_rng(seed).normal(0, .1, (5, k)).astype(np.float32)
    x[0] = 0
    np.testing.assert_array_equal(new.linear(x, weight), old.linear(x, weight))


def test_exact_arbitrary_q8_codes_and_scales(pair):
    old, new = pair
    ref = DOT(old._functions['Q2_0'][2])
    simd = DOT(new._functions['Q2_0'][2])
    rng = np.random.default_rng(264)
    for blocks in (1, 2, 3, 5, 10, 40):
        for _ in range(100):
            w = rng.integers(0, 256, (blocks, 18), np.uint8)
            a = rng.integers(0, 256, (blocks*2, 34), np.uint8)
            # Includes negative scales/subnormals, but avoids nonfinite scales.
            for raw in (w, a):
                scales = rng.integers(0, 0x10000, len(raw), np.uint16)
                scales = np.where((scales & 0x7c00) == 0x7c00, scales & 0x83ff, scales).astype(np.uint16)
                raw[:, :2] = scales.view(np.uint8).reshape(-1, 2)
            got, expected = np.zeros(1, np.float32), np.zeros(1, np.float32)
            ref(blocks*64, expected.ctypes.data, 0, w.ctypes.data, 0, a.ctypes.data, 0, 1)
            simd(blocks*64, got.ctypes.data, 0, w.ctypes.data, 0, a.ctypes.data, 0, 1)
            np.testing.assert_array_equal(got.view(np.uint32), expected.view(np.uint32))


def test_mixed_experts_and_pool_bit_exact(pair):
    old, new = pair
    ws = (packed('IQ3_S', 640, 2560, 265), packed('IQ3_S', 640, 2560, 266),
          packed('Q2_0', 2560, 640, 267))
    x = np.random.default_rng(268).normal(0, .1, (1, 2560)).astype(np.float32)
    reference = old.expert(x, ws)
    np.testing.assert_array_equal(new.expert(x, ws), reference)
    with GGMLPackedCPURowPool(new, 2560, 640, 15) as pool:
        got = pool.submit(x, [ws]*10).result()
        np.testing.assert_array_equal(got, np.repeat(reference, 10, axis=0))


def test_optional_timing(pair):
    if os.environ.get('MINI_LLM_Q2_SIMD_TIMING') != '1':
        pytest.skip('Opt-in warmed operator timing')
    old, new = pair
    w = packed('Q2_0', 2560, 640, 270)
    x = np.random.default_rng(271).normal(0, .1, (1, 640)).astype(np.float32)
    times = {}
    for name, executor in (('ggml_scalar', old), ('avx2', new)):
        for _ in range(10):
            executor.linear(x, w)
        samples = []
        for _ in range(100):
            t = time.perf_counter()
            executor.linear(x, w)
            samples.append(time.perf_counter()-t)
        times[name] = float(np.median(samples))
    print('0261 Q2_0 D2560/F640 median seconds', times, 'speedup', times['ggml_scalar']/times['avx2'])
