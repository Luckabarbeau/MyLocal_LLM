# SPDX-License-Identifier: LGPL-3.0-or-later
"""0256 independent direct-ctypes GGML oracle, not a dequantized-dot oracle.

Set MINI_LLM_GGML_CPU_LIBRARY explicitly. Optional real expert: set
MINI_LLM_GGML_CPU_EXPERT_NPZ to an allow_pickle=False NPZ with gate/up/down
uint8 packed bytes, {name}_shape int arrays, and {name}_type scalar strings.
This can be an externally extracted single expert; tests never modify weights.
Set MINI_LLM_GGML_CPU_TIMING=1 to run the warmed tiny CPU timing comparison.
"""
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import pytest

from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPUExecutor
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight

# Independent ABI declarations and dispatch; do not reuse executor internals.
DOT = C.CFUNCTYPE(None, C.c_int, C.c_void_p, C.c_size_t, C.c_void_p,
                 C.c_size_t, C.c_void_p, C.c_size_t, C.c_int)
QUANT = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.c_int64)
FORMATS = ('IQ2_S', 'IQ3_S', 'IQ3_XXS', 'IQ4_XS', 'IQ4_NL', 'Q2_0')
TRIPLES = (
    ('IQ2_S', 'IQ2_S', 'IQ4_NL'),
    ('IQ3_S', 'IQ3_S', 'Q2_0'),
    ('IQ3_XXS', 'IQ3_XXS', 'IQ4_NL'),
    ('IQ4_XS', 'IQ4_XS', 'Q2_0'),
    ('IQ2_S', 'IQ3_S', 'Q2_0'),
    ('IQ3_XXS', 'IQ4_XS', 'IQ4_NL'),
    ('IQ4_NL', 'IQ3_S', 'Q2_0'),  # different gate/up activation types
)


@pytest.fixture(scope='module')
def selected_library():
    selected = os.environ.get('MINI_LLM_GGML_CPU_LIBRARY')
    if not selected:
        pytest.skip('Set MINI_LLM_GGML_CPU_LIBRARY explicitly to the pinned libggml-cpu.so')
    if not Path(selected).expanduser().is_file():
        pytest.skip(f'Explicit GGML CPU library is missing: {selected}')
    return str(Path(selected).expanduser().resolve())


@pytest.fixture(scope='module')
def executor(selected_library):
    return GGMLPackedCPUExecutor(selected_library)


@pytest.fixture(scope='module')
def oracle_library(selected_library):
    lib = C.CDLL(selected_library)
    initialize = lib.ggml_cpu_init
    initialize.argtypes, initialize.restype = [], None
    initialize()
    return lib


def packed(fmt, n=7, k=512, seed=256):
    qk, qb = BLOCK_FORMATS[fmt]
    raw = np.random.default_rng(seed).integers(0, 256, (n * k // qk, qb), np.uint8)
    raw[:, :2] = np.frombuffer(np.float16(1 / 1024).tobytes(), np.uint8)
    raw.flags.writeable = False
    return PackedGGUFWeight(raw, (n, k), fmt)


def test_heterogeneous_cpu_dispatch_selected_ggml(executor):
    from mini_llm.backend import BACKEND_NAME
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
    if BACKEND_NAME != 'numpy':
        pytest.skip('CPU adapter integration test')
    weights = (packed('IQ3_S', 320, 256), packed('IQ3_S', 320, 256, 257),
               packed('IQ4_NL', 256, 320, 258))
    x = np.random.default_rng(256).normal(0, .1, (1, 256)).astype(np.float32)
    expected = executor.expert(x, weights)
    managed = HeterogeneousExpertExecutor(lambda key: weights, model_id='0256',
        d_model=256, d_ff=320, compute_dtype='float32', cpu_workers=2,
        cpu_packed_backend='ggml', execution_mode='cpu')
    try:
        np.testing.assert_array_equal(managed.resolve(managed.submit(0, 0, x)), expected)
        assert managed.counters['ggml_cpu_experts'] == 1
        assert not managed.host_cache.budget_bytes
    finally:
        managed.close()


def row_oracle(lib, x, w):
    """Call GGML directly for each row, with no bridge or production dispatch."""
    q8k = w.quant_type in ('IQ2_S', 'IQ3_S', 'IQ3_XXS', 'IQ4_XS')
    suffix = 'q8_K' if q8k else 'q8_0'
    quantize = QUANT(('quantize_row_' + suffix, lib))
    dot = DOT(('ggml_vec_dot_' + w.quant_type.lower() + '_' + suffix, lib))
    n, k = w.shape
    qk, qb = BLOCK_FORMATS[w.quant_type]
    stride = (k // qk) * qb
    # ctypes allocator provides natural alignment for block float/int16 fields.
    q = C.create_string_buffer((k // 256) * 292 if q8k else (k // 32) * 34)
    x = np.ascontiguousarray(x.reshape(-1, k), dtype=np.float32)
    result = np.empty((len(x), n), np.float32)
    for b in range(len(x)):
        quantize(x[b].ctypes.data, C.addressof(q), k)
        for r in range(n):
            dot(k, result.ctypes.data + 4 * (b * n + r), 0,
                w.data.ctypes.data + stride * r, 0, C.addressof(q), 0, 1)
    return result


def expert_oracle(lib, x, weights):
    g, u = (row_oracle(lib, x, w) for w in weights[:2])
    hidden = (g / (np.float32(1) + np.exp(-g))) * u
    return row_oracle(lib, hidden, weights[2])


def compare(got, expected, label):
    error = np.abs(got.astype(np.float64) - expected.astype(np.float64))
    relative = error / np.maximum(np.abs(expected), 1e-12)
    print(f'{label}: FP32 max_abs={error.max():.9g}, max_rel={relative.max():.9g}; '
          'atol=2e-7 rtol=3e-5; direct GGML Q8 oracle, no finite differences')
    np.testing.assert_allclose(got, expected, atol=2e-7, rtol=3e-5)


@pytest.mark.parametrize('fmt', FORMATS)
@pytest.mark.parametrize('k', [256, 768])
def test_linear_direct_ggml(executor, oracle_library, fmt, k, monkeypatch, tmp_path):
    source = packed(fmt, k=k)
    path = tmp_path / 'immutable.bin'
    path.write_bytes(source.data.tobytes())
    w = PackedGGUFWeight(np.memmap(path, mode='r', dtype=np.uint8), source.shape, fmt)
    before = hashlib.sha256(w.data).hexdigest()
    x = np.random.default_rng(257).normal(0, .02, (2, 3, k * 2)).astype(np.float32)[..., ::2]
    original = x.copy()
    expected = row_oracle(oracle_library, x, w).reshape(2, 3, 7)
    def forbidden(*args, **kwargs):
        raise AssertionError('Production must not dequantize weights')
    monkeypatch.setattr(PackedGGUFWeight, 'dequantize_reference', forbidden)
    got = executor.linear(x, w)
    assert got.dtype == np.float32 and got.shape == (2, 3, 7)
    compare(got, expected, fmt)
    np.testing.assert_array_equal(x, original)
    assert hashlib.sha256(w.data).hexdigest() == before and not w.data.flags.writeable
    assert executor.linear(np.empty((0, k), np.float32), w).shape == (0, 7)


@pytest.mark.parametrize('formats', TRIPLES)
def test_seven_expert_triples(executor, oracle_library, formats, monkeypatch):
    # Real-model-like FF width, deliberately not divisible by 256.
    d, f = 256, 320
    ws = (packed(formats[0], f, d, 260), packed(formats[1], f, d, 261),
          packed(formats[2], d, f, 262))
    x = np.random.default_rng(258).normal(0, .02, (2, d)).astype(np.float32)
    before = [hashlib.sha256(w.data).hexdigest() for w in ws]
    expected = expert_oracle(oracle_library, x, ws)
    def forbidden(*args, **kwargs):
        raise AssertionError('Production must not dequantize weights')
    monkeypatch.setattr(PackedGGUFWeight, 'dequantize_reference', forbidden)
    got = executor.expert(x, ws)
    assert got.dtype == np.float32 and got.shape == x.shape
    compare(got, expected, str(formats) + ' unweighted')
    assert [hashlib.sha256(w.data).hexdigest() for w in ws] == before
    assert executor.expert(x[:0], ws).shape == (0, d)


@pytest.mark.parametrize('fmt', FORMATS)
def test_dequant_fp32_baseline_is_separate(executor, fmt):
    from mini_llm.ops.native_packed_cpu import packed_cpu_linear
    w = packed(fmt)
    x = np.random.default_rng(259).normal(0, .02, (2, 512)).astype(np.float32)
    fp32 = packed_cpu_linear(x, w, backend='native')
    decoded = x @ w.dequantize_reference().T
    np.testing.assert_allclose(fp32, decoded, atol=1e-6, rtol=5e-5)
    quantized = executor.linear(x, w)
    rel_l2 = np.linalg.norm(quantized - fp32) / max(np.linalg.norm(fp32), 1e-12)
    print(f'{fmt}: Q8 versus dequant FP32 relative L2={rel_l2:.9g} (not exact)')
    assert 0 < rel_l2 < .05


def test_threadpool_shared_executor(executor, oracle_library):
    ws = (packed('IQ2_S', 320, 256, 270), packed('IQ3_S', 320, 256, 271),
          packed('Q2_0', 256, 320, 272))
    inputs = [np.random.default_rng(i).normal(0, .02, (1, 256)).astype(np.float32)
              for i in range(8)]
    expected = [expert_oracle(oracle_library, x, ws) for x in inputs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(lambda x: executor.expert(x, ws), inputs))
    for y, reference in zip(got, expected):
        compare(y, reference, 'concurrent expert')


def test_quantization_reuse(executor):
    # Instrument just quantizer calls at the native ABI, retaining callback lifetimes.
    quantizations = []
    functions = executor._functions.copy()
    callbacks = []
    try:
        for fmt in ('IQ2_S', 'IQ3_S', 'IQ4_NL'):
            activation, quant, dot = functions[fmt]
            original = QUANT(quant)
            def counted(x, y, k, original=original, activation=activation):
                quantizations.append((activation, k))
                original(x, y, k)
            callback = QUANT(counted)
            callbacks.append(callback)
            executor._functions[fmt] = (activation, C.cast(callback, C.c_void_p).value, dot)
        ws = (packed('IQ2_S', 320, 256), packed('IQ3_S', 320, 256),
              packed('IQ4_NL', 256, 320))
        executor.expert(np.ones((2, 256), np.float32), ws)
        assert quantizations == [(15, 256), (8, 320)] * 2
    finally:
        executor._functions = functions


def test_explicit_selection(monkeypatch):
    monkeypatch.delenv('MINI_LLM_GGML_CPU_LIBRARY', raising=False)
    with pytest.raises(RuntimeError, match='Explicitly set'):
        GGMLPackedCPUExecutor()


def test_validation_and_provenance(executor, selected_library):
    p = executor.provenance
    assert p['library_path'] == selected_library
    assert len(p['library_sha256']) == 64 and not p['library_revision_verified']
    print('GGML provenance:', p)
    w = packed('IQ2_S', k=256)
    with pytest.raises(TypeError, match='FP32'):
        executor.linear(np.ones((1, 256), np.float16), w)
    with pytest.raises(ValueError, match='matching'):
        executor.linear(np.ones((1, 512), np.float32), w)
    with pytest.raises(ValueError, match='projection shapes'):
        executor.expert(np.ones((1, 256), np.float32), (w, w, w))
    # A valid packed 3D tensor is not a matrix.
    with pytest.raises(TypeError, match='matrix'):
        executor.linear(np.ones((1, 256), np.float32),
                        PackedGGUFWeight(w.data, (1, 7, 256), w.quant_type))
    # Native ABI guard: invalid Q8_K K must return before touching pointers.
    for k in (32, 640):
        args = executor._projection(w)
        assert executor._bridge.mylocal_ggml_linear(*args, 7, k, 1, 1, 1) == 1


def test_optional_real_expert(executor, oracle_library):
    selected = os.environ.get('MINI_LLM_GGML_CPU_EXPERT_NPZ')
    if not selected:
        pytest.skip('Set MINI_LLM_GGML_CPU_EXPERT_NPZ for an extracted real expert')
    with np.load(selected, allow_pickle=False) as data:
        ws = tuple(PackedGGUFWeight(data[name], tuple(data[name + '_shape']),
                                   str(data[name + '_type'].item()))
                   for name in ('gate', 'up', 'down'))
    x = np.random.default_rng(273).normal(0, .02, (1, ws[0].shape[1])).astype(np.float32)
    compare(executor.expert(x, ws), expert_oracle(oracle_library, x, ws), 'real expert')


def test_optional_warmed_timing(executor):
    if os.environ.get('MINI_LLM_GGML_CPU_TIMING') != '1':
        pytest.skip('Set MINI_LLM_GGML_CPU_TIMING=1 for warmed tiny-expert timing')
    from mini_llm.ops.native_packed_cpu import packed_cpu_expert
    ws = (packed('IQ2_S', 320, 256, 280), packed('IQ3_S', 320, 256, 281),
          packed('Q2_0', 256, 320, 282))
    x = np.random.default_rng(274).normal(0, .02, (1, 256)).astype(np.float32)
    timings = {}
    for label, run in (('GGML Q8 integer dot', lambda: executor.expert(x, ws)),
                       ('native dequant FP32', lambda: packed_cpu_expert(x, ws, backend='native'))):
        for _ in range(5):
            run()
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(50):
                run()
            samples.append((time.perf_counter() - start) / 50)
        timings[label] = float(np.median(samples))
    print('Warmed tiny D256/F320 single CPU expert median seconds:', timings,
          'speedup=', timings['native dequant FP32'] / timings['GGML Q8 integer dot'])
    assert all(value > 0 for value in timings.values())
