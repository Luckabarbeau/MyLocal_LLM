"""Standalone exact-arithmetic pair tests; run CUDA only through the 0290 guard."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops import mixed_q8_gpu as mixed
from mini_llm.ops import q8_projection_bundle as bundle

cp = pytest.importorskip('cupy')
pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Actual CUDA Q8 projections')


@pytest.fixture(autouse=True)
def isolated_cuda():
    from scripts.run_0290_isolated_benchmark import require_isolation
    require_isolation()
    if cp.cuda.runtime.is_hip:
        pytest.skip('Native Q8_1 requires CUDA')


def weight(fmt, n=13, k=768, seed=338):
    qk, qb = mixed.iq.BLOCK_FORMATS[fmt]
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, (n * k // qk, qb), dtype=np.uint8)
    offset = 208 if fmt == 'Q6_K' else 0
    data[:, offset:offset + 2] = rng.uniform(-.02, .02, len(data)).astype('<f2').view(np.uint8).reshape(-1, 2)
    if fmt in ('Q4_K', 'Q5_K'):
        data[:, 2:4] = rng.uniform(-.02, .02, len(data)).astype('<f2').view(np.uint8).reshape(-1, 2)
    return SimpleNamespace(data=cp.asarray(data), shape=(n, k), quant_type=fmt)


def exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == cp.float32
    np.testing.assert_array_equal(actual.get().view(np.uint32), expected.get().view(np.uint32))


def reference(x, w):
    return mixed.linear(x.reshape(-1, x.shape[-1]), w).reshape(x.shape[:-1] + (w.shape[0],))


FORMATS = ('Q4_K', 'Q5_K', 'Q6_K', 'Q8_0', 'IQ3_S')


@pytest.mark.parametrize('shared', ['0', '1'])
@pytest.mark.parametrize('f0', FORMATS)
@pytest.mark.parametrize('f1', FORMATS)
@pytest.mark.parametrize('shape,n0,n1', [((1, 256), 1, 3), ((2, 3, 768), 5, 9)])
def test_bit_exact_formats_and_cta_tails(monkeypatch, shared, f0, f1, shape, n0, n1):
    monkeypatch.setenv('MINI_LLM_Q8_SHARED_GRID', shared)
    x = cp.asarray(np.random.default_rng(3381).normal(0, .1, shape).astype(np.float32))
    weights = (weight(f0, n0, shape[-1]), weight(f1, n1, shape[-1], 339))
    original = tuple(w.data.get() for w in weights)
    outputs = bundle.linear_pair(x, *weights)
    assert isinstance(outputs, tuple) and len(outputs) == 2
    for result, w, saved in zip(outputs, weights, original):
        exact(result, reference(x, w))
        np.testing.assert_array_equal(w.data.get(), saved)


@pytest.mark.parametrize('fmt', mixed.FORMATS)
@pytest.mark.parametrize('shared', ['0', '1'])
def test_all_supported_formats_long_rows(monkeypatch, fmt, shared):
    monkeypatch.setenv('MINI_LLM_Q8_SHARED_GRID', shared)
    x = cp.asarray(np.random.default_rng(3382).normal(0, .2, (2, 2560)).astype(np.float32))
    # Different IQ table sizes/formats exercise CTA-local selected initialization.
    other = 'IQ3_S' if fmt != 'IQ3_S' else 'IQ2_S'
    weights = (weight(fmt, 7, 2560), weight(other, 12, 2560, 340))
    for y, w in zip(bundle.linear_pair(x, *weights), weights):
        exact(y, reference(x, w))


@pytest.mark.parametrize('shape', [(0, 768), (2, 0, 768)])
def test_empty_no_launch(monkeypatch, shape):
    weights = (weight('Q6_K'), weight('IQ3_S', 5))
    x = cp.empty(shape, cp.float32)
    def forbidden(*args, **kwargs):
        raise AssertionError('Empty batch must not compile or launch')
    monkeypatch.setattr(bundle, '_module', forbidden)
    monkeypatch.setattr(bundle.q8, 'quantize_q8_1', forbidden)
    outputs = bundle.linear_pair(x, *weights)
    assert tuple(y.shape for y in outputs) == (shape[:-1] + (13,), shape[:-1] + (5,))


def test_one_quantizer_one_projection_launch(monkeypatch):
    x = cp.ones((2, 3, 768), cp.float32)
    w0, w1 = weight('Q6_K', 5), weight('Q5_K', 9)
    out = (cp.empty((2, 3, 5), cp.float32), cp.empty((2, 3, 9), cp.float32))
    expected = (reference(x, w0), reference(x, w1))
    quantize, module = bundle.q8.quantize_q8_1, bundle._module
    calls = []
    def pack(value, **kwargs):
        calls.append(('quantize', value.shape))
        return quantize(value, **kwargs)
    def compile_pair(*args):
        mod = module(*args)
        def get_function(name):
            kernel = mod.get_function(name)
            def launch(grid, block, arguments):
                calls.append(('linear_pair', grid, block))
                return kernel(grid, block, arguments)
            return launch
        return SimpleNamespace(get_function=get_function)
    monkeypatch.setattr(bundle.q8, 'quantize_q8_1', pack)
    monkeypatch.setattr(bundle, '_module', compile_pair)
    actual = bundle.linear_pair(x, w0, w1, out=out)
    assert actual[0] is out[0] and actual[1] is out[1]
    assert calls == [('quantize', (6, 768)), ('linear_pair', (5, 6), (128,))]
    for a, e in zip(actual, expected):
        exact(a, e)


@pytest.mark.parametrize('shared', ['0', '1'])
@pytest.mark.parametrize('formats', [('Q6_K', 'Q5_K'), ('IQ2_S', 'IQ3_S'), ('IQ3_S', 'IQ3_XXS')])
def test_capture_replay_changed_input_same_address(monkeypatch, shared, formats):
    monkeypatch.setenv('MINI_LLM_Q8_SHARED_GRID', shared)
    x = cp.ones((2, 1, 768), cp.float32)
    weights = (weight(formats[0], 5), weight(formats[1], 7))
    outputs = (cp.empty((2, 1, 5), cp.float32), cp.empty((2, 1, 7), cp.float32))
    ptrs = (x.data.ptr, *(y.data.ptr for y in outputs))
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()
    with stream:
        bundle.linear_pair(x, *weights, out=outputs)
    stream.synchronize()
    with stream:
        stream.begin_capture()
        captured = bundle.linear_pair(x, *weights, out=outputs)
        graph = stream.end_capture()
    snapshots = []
    for scale in (0., -.7, 1.25, .1):
        with stream:
            x[:] = cp.arange(x.size, dtype=cp.float32).reshape(x.shape) * cp.float32(scale)
            graph.launch(stream)
            expected = tuple(reference(x, w) for w in weights)
        stream.synchronize()
        assert ptrs == (x.data.ptr, *(y.data.ptr for y in captured))
        for a, e in zip(captured, expected):
            exact(a, e)
        snapshots.append(captured[0].get())
    assert not np.array_equal(snapshots[0], snapshots[2])


@pytest.mark.parametrize('case', [
    'x_dtype', 'x_rank', 'x_contiguous', 'x_width', 'x_zero_width',
    'weight_missing', 'weight_format', 'weight_rank', 'weight_zero',
    'weight_width', 'weight_alignment', 'weight_length', 'weight_dtype',
    'weight_contiguous', 'iq_length', 'iq_dtype', 'out_tuple', 'out_shape',
    'out_dtype', 'out_contiguous', 'out_input0', 'out_input1', 'out_each_other',
    'out_weight00', 'out_weight01', 'out_weight10', 'out_weight11',
])
def test_validation_before_quantization_or_compile(monkeypatch, case):
    x = cp.ones((1, 768), cp.float32)
    weights = [weight('Q6_K'), weight('Q5_K')]
    outputs = [cp.empty((1, 13), cp.float32), cp.empty((1, 13), cp.float32)]
    out = None
    error = ValueError
    if case == 'x_dtype':
        x = x.astype(cp.float16)
        error = TypeError
    elif case == 'x_rank':
        x = x.reshape(-1)
    elif case == 'x_contiguous':
        x = cp.ones((1, 1536), cp.float32)[:, ::2]
    elif case == 'x_width':
        x = cp.ones((1, 256), cp.float32)
    elif case == 'x_zero_width':
        x = cp.empty((1, 0), cp.float32)
    elif case == 'weight_missing':
        weights[1] = object()
        error = TypeError
    elif case == 'weight_format':
        weights[1].quant_type = 'F16'
        error = NotImplementedError
    elif case == 'weight_rank':
        weights[1].shape = (13, 1, 768)
    elif case == 'weight_zero':
        weights[1].shape = (0, 768)
    elif case == 'weight_width':
        weights[1].shape = (13, 256)
    elif case == 'weight_alignment':
        x = cp.ones((1, 800), cp.float32)
        for w in weights:
            w.shape = (13, 800)
    elif case == 'weight_length':
        weights[1].data = weights[1].data.reshape(-1)[:-1]
    elif case == 'weight_dtype':
        weights[1].data = weights[1].data.astype(cp.int8)
    elif case == 'weight_contiguous':
        weights[1].data = weights[1].data[:, ::2]
    elif case in ('iq_length', 'iq_dtype'):
        weights[1] = weight('IQ3_S')
        if case == 'iq_length':
            weights[1].data = weights[1].data.reshape(-1)[:-1]
        else:
            weights[1].data = weights[1].data.astype(cp.int8)
            error = TypeError
    elif case == 'out_tuple':
        out = outputs
    else:
        if case == 'out_shape':
            outputs[1] = cp.empty((13,), cp.float32)
        elif case == 'out_dtype':
            outputs[1] = outputs[1].astype(cp.float16)
            error = TypeError
        elif case == 'out_contiguous':
            outputs[1] = cp.empty((1, 26), cp.float32)[:, ::2]
        elif case.startswith('out_input'):
            outputs[int(case[-1])] = x[:, :13]
        elif case == 'out_each_other':
            storage = cp.empty(20, cp.float32)
            outputs = [storage[:13].reshape(1, 13), storage[7:].reshape(1, 13)]
        elif case.startswith('out_weight'):
            i, j = map(int, case[-2:])
            outputs[i] = weights[j].data.reshape(-1)[:52].view(cp.float32).reshape(1, 13)
        out = tuple(outputs)
    def forbidden(*args, **kwargs):
        raise AssertionError('Validation must precede compilation and quantization')
    monkeypatch.setattr(bundle.q8, 'quantize_q8_1', forbidden)
    monkeypatch.setattr(bundle, '_module', forbidden)
    with pytest.raises(error):
        bundle.linear_pair(x, *weights, out=out)


def test_other_device_rejected():
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip('Requires two CUDA devices')
    device = cp.cuda.runtime.getDevice()
    x = cp.ones((1, 768), cp.float32)
    w0 = weight('Q6_K')
    with cp.cuda.Device((device + 1) % cp.cuda.runtime.getDeviceCount()):
        w1 = weight('IQ3_S')
        wrong_out = cp.empty((1, 13), cp.float32)
    with pytest.raises(ValueError, match='same device'):
        bundle.linear_pair(x, w0, w1)
    with pytest.raises(ValueError, match='current device'):
        bundle.linear_pair(x, w0, w0, out=(wrong_out, None))


@pytest.mark.skipif(os.environ.get('MINI_LLM_Q8_PAIR_TIMING') != '1', reason='Opt-in synthetic timing only')
def test_optional_standalone_timing():
    # Synthetic sizes only; no model, captures from disk, or native CPU work.
    x = cp.ones((1, 2560), cp.float32)
    w0, w1 = weight('Q6_K', 10240, 2560), weight('Q5_K', 6144, 2560)
    outputs = (cp.empty((1, 10240), cp.float32), cp.empty((1, 6144), cp.float32))
    def separate():
        return (mixed.linear(x, w0, out=outputs[0]), mixed.linear(x, w1, out=outputs[1]))
    def reused():
        packed = mixed.q8.quantize_q8_1(x)
        return (mixed.linear_packed(packed, w0, out=outputs[0]),
                mixed.linear_packed(packed, w1, out=outputs[1]))
    def paired():
        return bundle.linear_pair(x, w0, w1, out=outputs)
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()
    graphs = []
    with stream:
        for operation in (separate, reused, paired):
            operation()
    stream.synchronize()
    with stream:
        for operation in (separate, reused, paired):
            stream.begin_capture()
            operation()
            graphs.append(stream.end_capture())
        start, end = cp.cuda.Event(), cp.cuda.Event()
        for label, graph in zip(('separate', 'reused', 'paired'), graphs):
            samples = []
            for _ in range(5):
                start.record()
                for _ in range(50):
                    graph.launch(stream)
                end.record()
                end.synchronize()
                samples.append(cp.cuda.get_elapsed_time(start, end) / 50)
            print(f'0338 synthetic captured {label}: median_ms={np.median(samples):.6f}, samples={samples}')
