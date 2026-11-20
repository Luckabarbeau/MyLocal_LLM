"""CPU source/API checks; GPU bitwise tests are for the parent's isolated run."""

import builtins
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.ops import qsa_decode_gpu as serial
from mini_llm.ops import qsa_graph_gpu as graph
from mini_llm.ops import qsa_staged_gpu as staged


@pytest.mark.parametrize('source', [serial._SOURCE, graph._SOURCE], ids=['host-n', 'device-n'])
def test_source_adapter_changes_only_shared_declaration_and_dot(source):
    generated = staged.staged_source(source)
    # Invert exactly the two permitted edits: all remaining bytes (including
    # softmax tree, 128-cell order, V loop, gate, and dynamic signature) match.
    assert generated.count(staged._STAGED_SHARED) == 1
    assert generated.count(staged._STAGED_DOT) == 1
    restored = generated.replace(staged._STAGED_SHARED, staged._SHARED, 1)
    restored = restored.replace(staged._STAGED_DOT, staged._SERIAL_DOT, 1)
    assert restored == source
    assert 'for (int d = 0; d < 256; ++d)' in generated
    assert 'dot += q[h * qs0 + d * qs1] * key;' in generated
    assert staged._SERIAL_DOT not in generated
    assert generated.count('__syncthreads();') == source.count('__syncthreads();') + 2
    assert 'position' in generated if source == graph._SOURCE else 'position' not in generated
    assert staged._OPTIONS == ('--std=c++14', '--fmad=false')


@pytest.mark.parametrize('fragment', [
    staged._SHARED, staged._SERIAL_DOT,
    '    const int h = blockIdx.x, kh = h / groups, lane = threadIdx.x;',
    '    for (int base = 0; base < n; base += 128) {',
    '        const int cell = base + lane, count = min(128, n - base);',
    '        float logit = -INFINITY;', '        reduction[lane] = logit;',
])
@pytest.mark.parametrize('drift', ['missing', 'duplicate'])
def test_source_adapter_fails_closed(fragment, drift):
    source = serial._SOURCE.replace(fragment, '', 1) if drift == 'missing' else serial._SOURCE + fragment
    with pytest.raises(ValueError, match='source contract changed'):
        staged.staged_source(source)


def test_source_adapter_rejects_reapplication_and_non_source():
    with pytest.raises(ValueError, match='source contract changed'):
        staged.staged_source(staged._SOURCE)
    with pytest.raises(TypeError, match='string'):
        staged.staged_source(None)


def test_cooperative_index_mapping_and_bank_padding():
    writes = []
    for lane in range(128):
        for index in range(lane, 8192, 128):
            row, d = divmod(index, 256)
            writes.append((d * 33 + row, row, d))
    assert len({address for address, _, _ in writes}) == 8192
    assert max(address for address, _, _ in writes) < 256 * 33
    # A warp writes consecutive dimensions and reads consecutive cell columns.
    assert len({(d * 33) % 32 for d in range(32)}) == 32
    assert len({(123 * 33 + col) % 32 for col in range(32)}) == 32
    for g in range(4):
        owners = [lane for lane in range(128) if lane // 32 == g]
        assert [g * 32 + lane % 32 for lane in owners] == owners


def test_cpu_module_import_never_imports_cuda_or_triton(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'cupy', 'triton', 'torch'}:
            raise AssertionError(f'unexpected optional import: {name}')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    spec = importlib.util.spec_from_file_location('mini_llm.ops._staged_cpu_check', staged.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._SOURCE == staged._SOURCE
    assert not module._KERNELS


class Array:
    """No data, driver, or CUDA dependency: validation reads metadata only."""

    def __init__(self, shape, dtype=np.float32, device=0, strides=None):
        self.shape, self.ndim = shape, len(shape)
        self.dtype, self.itemsize = np.dtype(dtype), np.dtype(dtype).itemsize
        self.device = SimpleNamespace(id=device)
        self.strides = np.empty(shape, dtype=dtype).strides if strides is None else strides


def metadata(n=141, dim=256):
    return [Array(s) for s in ((4, dim), (4, 2*dim), (2, dim), (2, dim),
                              (n-1, 2, dim), (n-1, 2, dim))]


@pytest.mark.parametrize('n', [1, 2, 77, 128, 129, 141, 435, 1024, 2051, 8192, 8193])
def test_metadata_accepts_n_minus_one_cache_and_host_scalars(n):
    actual_n, scale = staged._validate(SimpleNamespace(ndarray=Array), *metadata(n), np.int64(n), .0625)
    assert actual_n == n
    assert isinstance(scale, np.float32) and scale == np.float32(.0625)


@pytest.mark.parametrize('kind,exc,match', [
    ('host', TypeError, 'CuPy'), ('dtype', TypeError, 'FP32'),
    ('device', ValueError, 'same device'), ('dim', ValueError, 'only dim=256'),
    ('rank', ValueError, 'shape'), ('gate', ValueError, 'shapes'),
    ('groups', ValueError, 'grouped heads'), ('cache', ValueError, 'cache shapes'),
    ('capacity', ValueError, 'capacity'), ('stride', ValueError, 'whole FP32'),
    ('nzero', ValueError, 'n must'), ('nbig', ValueError, 'n must'),
    ('nbool', TypeError, 'integer'), ('nfloat', TypeError, 'integer'),
    ('nan', ValueError, 'scale'), ('inf', ValueError, 'scale'),
    ('negative', ValueError, 'scale'), ('zero', ValueError, 'scale'),
    ('overflow', ValueError, 'scale'), ('underflow', ValueError, 'scale'),
    ('scale_bool', TypeError, 'host scalar'), ('scale_array', TypeError, 'host scalar'),
])
def test_bad_metadata_is_rejected_before_allocation(kind, exc, match):
    arrays, n, scale = metadata(), 141, .0625
    if kind == 'host': arrays[0] = np.empty((4, 256), np.float32)
    if kind == 'dtype': arrays[0] = Array((4, 256), np.float16)
    if kind == 'device': arrays[-1] = Array((140, 2, 256), device=1)
    if kind == 'dim': arrays = metadata(dim=128)
    if kind == 'rank': arrays[0] = Array((1024,))
    if kind == 'gate': arrays[1] = Array((4, 256))
    if kind == 'groups': arrays[2:4] = [Array((3, 256)), Array((3, 256))]
    if kind == 'cache': arrays[-1] = Array((140, 256))
    if kind == 'capacity': arrays[-1] = Array((139, 2, 256))
    if kind == 'stride': arrays[0] = Array((4, 256), strides=(1024, 1))
    if kind == 'nzero': n = 0
    if kind == 'nbig': n = np.iinfo(np.int32).max
    if kind == 'nbool': n = True
    if kind == 'nfloat': n = 1.5
    if kind == 'nan': scale = float('nan')
    if kind == 'inf': scale = float('inf')
    if kind == 'negative': scale = -1
    if kind == 'zero': scale = 0
    if kind == 'overflow': scale = 1e100
    if kind == 'underflow': scale = 1e-100
    if kind == 'scale_bool': scale = True
    if kind == 'scale_array': scale = np.array(.0625)
    with pytest.raises(exc, match=match):
        staged._validate(SimpleNamespace(ndarray=Array), *arrays, n, scale)


def test_launch_cache_device_strides_current_stream_and_output_only(monkeypatch):
    state = SimpleNamespace(device=7, stream=111)
    compilations, launches, allocations = [], [], []

    class Device:
        def __init__(self, device): self.device = device
        def __enter__(self):
            self.previous = state.device
            state.device = self.device
        def __exit__(self, *args): state.device = self.previous

    def raw(source, name, options):
        compilations.append((state.device, source, name, options))
        return lambda grid, block, args: launches.append((state.device, state.stream, grid, block, args))

    def empty(size, dtype):
        allocations.append((state.device, size, dtype))
        return Array((size,), dtype=dtype, device=state.device)

    cp = SimpleNamespace(ndarray=Array, float32=np.float32, empty=empty, RawKernel=raw,
                         cuda=SimpleNamespace(Device=Device))
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    monkeypatch.setattr(staged, '_KERNELS', {})
    arrays = metadata()
    arrays[0].strides = (-2048, 8)
    arrays[1].strides = (0, -4)
    for device, stream in [(0, 111), (0, 222), (1, 333)]:
        state.stream = stream
        for a in arrays: a.device.id = device
        output = staged.qsa_staged_gpu(*arrays, 141, .0625)
        assert output.shape == (1024,) and output.device.id == device
        assert state.device == 7 and state.stream == stream
    assert len(compilations) == 2
    assert all(c[1:] == (staged._SOURCE, 'qsa_decode', staged._OPTIONS) for c in compilations)
    assert len(allocations) == 3  # output only, no global staging scratch
    assert [c[:4] for c in launches] == [(0, 111, (4,), (128,)), (0, 222, (4,), (128,)),
                                        (1, 333, (4,), (128,))]
    assert tuple(launches[0][-1][11:15]) == (-512, 2, 0, -1)
    with pytest.raises(ValueError, match='serial fallback'):
        staged.qsa_staged_gpu(*metadata(dim=64), 141, .0625)
    assert len(allocations) == 3


@pytest.fixture
def gpu():
    # Never run a GPU probe from the editor/desktop cgroup.
    from scripts.run_0290_isolated_benchmark import require_isolation
    require_isolation()
    cp = pytest.importorskip('cupy')
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip('CuPy GPU unavailable')
    old = cp.RawKernel(serial._SOURCE, 'qsa_decode', options=staged._OPTIONS)
    return cp, old


def make_case(cp, n, *, heads=24, kv=2, layout='plain', capacity=None):
    rng, bases = np.random.default_rng(319 + n), []
    capacity = n + 3 if capacity is None else capacity

    def array(shape):
        strided = layout in {'strided', 'reverse'}
        base_shape = tuple(2*s for s in shape) if strided else shape
        base = cp.asarray(rng.normal(size=base_shape).astype(np.float32))
        bases.append(base)
        view = base[tuple(slice(None, None, 2) for _ in shape)] if strided else base
        return view[tuple(slice(None, None, -1) for _ in shape)] if layout == 'reverse' else view

    inputs = [array(s) for s in ((heads, 256), (heads, 512), (kv, 256), (kv, 256),
                                  (capacity, kv, 256), (capacity, kv, 256))]
    inputs[1][:, :256] = cp.nan
    inputs[-2][n-1:] = cp.nan
    inputs[-1][n-1:] = cp.nan
    if layout == 'broadcast':
        for i in (0, 2, 3):
            inputs[i] = cp.broadcast_to(inputs[i][:1], inputs[i].shape)
    return inputs, bases


def launch_source(cp, kernel, inputs, n, scale, position=None):
    q, qg, k, v, keys, values = inputs
    output = cp.empty(q.size, cp.float32)
    strides = tuple(np.int64(s // a.itemsize) for a in inputs for s in a.strides)
    operand = np.int32(n) if position is None else position
    kernel((q.shape[0],), (128,), (*inputs, output, operand, np.int32(256),
                                  np.int32(q.shape[0] // k.shape[0]), np.float32(scale), *strides))
    return output


def check_exact(cp, old, inputs, bases, n, scale=.0625):
    before = [cp.asnumpy(a).view(np.uint32).copy() for a in bases]
    device, stream = cp.cuda.runtime.getDevice(), cp.cuda.get_current_stream().ptr
    actual = staged.qsa_staged_gpu(*inputs, n, scale)
    consumer = actual.copy()  # In-flight same-stream consumer, before host sync.
    expected = launch_source(cp, old, inputs, n, scale)
    assert cp.cuda.runtime.getDevice() == device
    assert cp.cuda.get_current_stream().ptr == stream
    a, b = cp.asnumpy(consumer), cp.asnumpy(expected)
    assert actual.dtype == cp.float32 and actual.shape == (inputs[0].size,)
    assert np.isfinite(a).all()
    assert np.array_equal(a, b)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    for base, snapshot in zip(bases, before):
        assert np.array_equal(cp.asnumpy(base).view(np.uint32), snapshot)
    return a


@pytest.mark.gpu
@pytest.mark.parametrize('n', [1, 2, 77, 128, 129, 141, 435, 1024, 2051, 8192])
def test_exact_serial_fp32_all_context_boundaries(gpu, n):
    cp, old = gpu
    with cp.cuda.Stream(non_blocking=True):
        inputs, bases = make_case(cp, n)
        check_exact(cp, old, inputs, bases, n)


@pytest.mark.gpu
@pytest.mark.parametrize('layout,n', [('strided', 129), ('reverse', 435), ('broadcast', 141)])
def test_arbitrary_strides_two_gqa_groups_nondefault_stream(gpu, layout, n):
    cp, old = gpu
    with cp.cuda.Stream(non_blocking=True):
        inputs, bases = make_case(cp, n, heads=4, kv=2, layout=layout)
        check_exact(cp, old, inputs, bases, n, .37)


@pytest.mark.gpu
@pytest.mark.parametrize('n', [1, 2, 129, 2051])
def test_no_uncommitted_cache_slot_required(gpu, n):
    cp, old = gpu
    with cp.cuda.Stream(non_blocking=True):
        inputs, bases = make_case(cp, n, heads=4, capacity=n-1)
        check_exact(cp, old, inputs, bases, n)


@pytest.mark.gpu
@pytest.mark.parametrize('n', [1, 129, 435])
def test_current_k_and_inflight_producers_with_nan_cache_sentinels(gpu, n):
    cp, old = gpu
    with cp.cuda.Stream(non_blocking=True):
        inputs, bases = make_case(cp, n, heads=4)
        q, qg, k, v, keys, values = inputs
        q[:] = 0
        q[:, 0] = 1
        qg[:, 256:] = 0
        keys[:n-1] = -100
        values[:n-1] = -17
        k[:] = 100
        v[:] = 7
        # No host read between these producers, launches, and consumer.
        actual = staged.qsa_staged_gpu(*inputs, n, 1.).copy()
        expected = launch_source(cp, old, inputs, n, 1.)
        a, b = cp.asnumpy(actual), cp.asnumpy(expected)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
        assert np.array_equal(a, np.full(a.shape, 3.5, np.float32))
        check_exact(cp, old, inputs, bases, n, 1.)


@pytest.mark.gpu
@pytest.mark.parametrize('n', [1, 129, 435, 2051])
def test_dynamic_n_source_variant_exact_and_unchanged_remainder(gpu, n):
    cp, old = gpu
    source = staged.staged_source(graph._SOURCE)
    assert source.replace(staged._STAGED_SHARED, staged._SHARED, 1).replace(
        staged._STAGED_DOT, staged._SERIAL_DOT, 1) == graph._SOURCE
    new_dynamic = cp.RawKernel(source, 'qsa_decode_graph', options=staged._OPTIONS)
    old_dynamic = cp.RawKernel(graph._SOURCE, 'qsa_decode_graph', options=staged._OPTIONS)
    with cp.cuda.Stream(non_blocking=True):
        inputs, bases = make_case(cp, n, heads=4, layout='reverse')
        before = [cp.asnumpy(a).view(np.uint32).copy() for a in bases]
        position = cp.empty((), cp.int64)
        position[...] = n-1
        a = launch_source(cp, new_dynamic, inputs, n, .0625, position).copy()
        b = launch_source(cp, old_dynamic, inputs, n, .0625, position)
        c = launch_source(cp, old, inputs, n, .0625)
        a, b, c = (cp.asnumpy(x) for x in (a, b, c))
        assert np.isfinite(a).all()
        assert np.array_equal(a, b) and np.array_equal(a, c)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
        assert np.array_equal(a.view(np.uint32), c.view(np.uint32))
        assert int(cp.asnumpy(position)) == n-1
        for base, snapshot in zip(bases, before):
            assert np.array_equal(cp.asnumpy(base).view(np.uint32), snapshot)


@pytest.mark.gpu
def test_other_current_device_is_restored(gpu):
    cp, old = gpu
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip('two GPUs required')
    original = cp.cuda.runtime.getDevice()
    target = (original + 1) % cp.cuda.runtime.getDeviceCount()
    with cp.cuda.Device(target), cp.cuda.Stream(non_blocking=True) as stream:
        inputs, bases = make_case(cp, 77, heads=4)
        expected = launch_source(cp, old, inputs, 77, .0625)
        with cp.cuda.Device(original):
            original_stream = cp.cuda.get_current_stream().ptr
            actual = staged.qsa_staged_gpu(*inputs, 77, .0625)
            assert cp.cuda.runtime.getDevice() == original
            assert cp.cuda.get_current_stream().ptr == original_stream
        assert cp.cuda.get_current_stream().ptr == stream.ptr
        assert np.array_equal(cp.asnumpy(actual).view(np.uint32), cp.asnumpy(expected).view(np.uint32))


def test_probe_requires_isolation_before_cuda_and_records_own_provenance():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'probe_0319_qsa_staged_operator.py'
    source = path.read_text()
    assert source.index('require_isolation()') < source.index('import cupy')
    assert 'triton' not in source
    assert 'qwen4exp_model' not in source
    assert 'sha256' in source and 'np.array_equal' in source
    assert 'staged._SOURCE' in source and 'serial._SOURCE' in source
    assert 'for n in (141, 435, 1024):' in source
