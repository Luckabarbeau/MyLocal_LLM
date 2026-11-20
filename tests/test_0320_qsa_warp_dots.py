"""CPU contracts; GPU exactness tests run only under the parent's 0290 gate."""

import builtins
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.ops import qsa_decode_gpu as serial
from mini_llm.ops import qsa_graph_gpu as graph
from mini_llm.ops import qsa_warp_dots_gpu as warp


@pytest.mark.parametrize('source', [serial._SOURCE, graph._SOURCE], ids=['host-n', 'device-n'])
def test_adapter_changes_only_signature_and_dot(source):
    generated = warp.warp_dots_source(source)
    assert generated.count(warp._LOGITS_DOT) == 1
    assert generated.count(warp._LOGITS_SIGNATURE) == 1
    restored = generated.replace(warp._LOGITS_SIGNATURE, warp._SIGNATURE, 1)
    restored = restored.replace(warp._LOGITS_DOT, warp._SERIAL_DOT, 1)
    assert restored == source  # Includes every byte of max/softmax/V/sigmoid.
    assert generated.count('__syncthreads();') == source.count('__syncthreads();')
    assert warp._SERIAL_DOT not in generated
    assert ('position' in generated) == ('position' in source)
    assert warp._OPTIONS == ('--std=c++14', '--fmad=false')


@pytest.mark.parametrize('fragment', warp._ANCHORS)
@pytest.mark.parametrize('drift', ['missing', 'duplicate'])
def test_adapter_fails_closed(fragment, drift):
    source = serial._SOURCE.replace(fragment, '', 1) if drift == 'missing' else serial._SOURCE + fragment
    with pytest.raises(ValueError, match='source contract changed'):
        warp.warp_dots_source(source)


def test_adapter_rejects_reapplication_and_non_source():
    with pytest.raises(ValueError, match='source contract changed'):
        warp.warp_dots_source(warp._SOURCE)
    with pytest.raises(TypeError, match='string'):
        warp.warp_dots_source(None)


@pytest.mark.parametrize('source', [serial._SOURCE, graph._SOURCE], ids=['host-n', 'device-n'])
@pytest.mark.parametrize('value_tile', [0, 32, 64, 128])
def test_finalizer_source_changes_exactly_three_value_loops(source, value_tile):
    original = warp.warp_dots_source(source)
    generated = warp.finalizer_source(source, value_tile=value_tile)
    assert original.count(warp._VALUE_LOOP) == 3
    if value_tile == 0:
        assert generated == original
        assert warp.finalizer_source(source) == original
    else:
        tiled_loop = warp._TILED_VALUE_LOOP.format(tile=value_tile)
        assert generated.count(tiled_loop) == 3
        assert warp._VALUE_LOOP not in generated
        assert generated.replace(tiled_loop, warp._VALUE_LOOP) == original
    assert generated.count('__syncthreads();') == original.count('__syncthreads();')
    assert 'for (int j = 0; j < count; ++j)' in generated


@pytest.mark.parametrize('count', [0, 1, 2, 4])
@pytest.mark.parametrize('value_tile', [0, 32, 64, 128])
def test_finalizer_requires_exactly_three_value_loops(count, value_tile):
    source = serial._SOURCE
    if count < 3:
        source = source.replace(warp._VALUE_LOOP, 'for (int d = lane; d < dim; d += 64)', 3-count)
    else:
        source += warp._VALUE_LOOP
    with pytest.raises(ValueError, match='exactly three value loops'):
        warp.finalizer_source(source, value_tile=value_tile)


@pytest.mark.parametrize('value_tile,exc', [(True, TypeError), (np.bool_(False), TypeError),
                                         (32., TypeError), (None, TypeError),
                                         (-1, ValueError), (16, ValueError), (256, ValueError)])
def test_invalid_value_tile_rejected_before_gpu_import(monkeypatch, value_tile, exc):
    monkeypatch.setitem(sys.modules, 'cupy', None)
    with pytest.raises(exc, match='value_tile|integer'):
        warp.qsa_warp_dots_gpu(*metadata(), 141, .0625, value_tile=value_tile)
    with pytest.raises(exc, match='value_tile|integer'):
        warp.finalizer_source(serial._SOURCE, value_tile=value_tile)


@pytest.mark.parametrize('value_tile', [32, 64, 128])
def test_value_tiles_partition_every_output_without_races(value_tile):
    owners = [(cta, lane, d) for cta in range((256+value_tile-1)//value_tile)
              for lane in range(128)
              for d in range(cta*value_tile+lane, min(256, (cta+1)*value_tile), 128)]
    dimensions = [d for _, _, d in owners]
    assert sorted(dimensions) == list(range(256))
    assert len(dimensions) == len(set(dimensions))
    assert 24 * ((256+value_tile-1)//value_tile) == {32: 192, 64: 96, 128: 48}[value_tile]
    assert warp._validate_value_tile(np.int64(value_tile)) == value_tile


def test_dot_source_order_full_warp_and_64bit_stride_contract():
    source = warp._DOTS_SOURCE
    assert 'threadIdx.x % 32' in source and 'threadIdx.x / 32' in source
    assert 'blockIdx.y * 4 + warp' in source
    assert 'if (cell >= n) return;' in source
    assert 'const unsigned int fullmask = 0xffffffffu;' in source
    assert 'for (int baseD = 0; baseD < 256; baseD += 32)' in source
    assert 'const float product = q[h * qs0 + d * qs1] * key;' in source
    assert 'for (int j = 0; j < 32; ++j)' in source
    assert '__shfl_sync(fullmask, product, j)' in source
    assert 'dot += ordered_product;' in source
    assert source.index('__shfl_sync') < source.index('if (lane == 0)')
    assert 'cell == n - 1 ? k[kh * k0 + d * k1]' in source
    assert 'long long qs0, long long qs1, long long k0, long long k1' in source
    assert 'long long ks0, long long ks1, long long ks2' in source
    assert [base + j for base in range(0, 256, 32) for j in range(32)] == list(range(256))
    for n in (1, 2, 77, 128, 129, 8192):
        owners = [4 * cta + w for cta in range((n + 3) // 4) for w in range(4) if 4 * cta + w < n]
        assert owners == list(range(n))
    assert warp._MAX_HEADS * warp._MAX_N * 4 < 1024**2


def test_cpu_import_does_not_import_optional_gpu_modules(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'cupy', 'triton', 'torch'} or 'qsa_graph_gpu' in name:
            raise AssertionError(f'unexpected optional/cyclic import: {name}')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    spec = importlib.util.spec_from_file_location('mini_llm.ops._warp_cpu_check', warp.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._SOURCE == warp._SOURCE and not module._KERNELS


class Array:
    """Metadata only; no allocation proportional to the pretend array size."""

    def __init__(self, shape, dtype=np.float32, device=0, strides=None):
        self.shape, self.ndim = shape, len(shape)
        self.dtype, self.itemsize = np.dtype(dtype), np.dtype(dtype).itemsize
        self.device = SimpleNamespace(id=device)
        sizes, step = [], self.itemsize
        for size in reversed(shape):
            sizes.insert(0, step)
            step *= max(1, size)
        self.strides = tuple(sizes) if strides is None else strides


def metadata(n=141, dim=256, heads=4, kv=2):
    return [Array(s) for s in ((heads, dim), (heads, 2*dim), (kv, dim), (kv, dim),
                              (n-1, kv, dim), (n-1, kv, dim))]


@pytest.mark.parametrize('n', [1, 2, 77, 128, 129, 141, 435, 1024, 2051, 8192])
def test_metadata_accepts_n_minus_one_capacity(n):
    actual_n, scale = warp._validate(SimpleNamespace(ndarray=Array), *metadata(n), np.int64(n), .0625)
    assert actual_n == n and isinstance(scale, np.float32)


@pytest.mark.parametrize('kind,exc,match', [
    ('host', TypeError, 'CuPy'), ('dtype', TypeError, 'FP32'),
    ('device', ValueError, 'same device'), ('dim', ValueError, 'only dim=256'),
    ('rank', ValueError, 'shape'), ('gate', ValueError, 'shapes'),
    ('groups', ValueError, 'grouped heads'), ('heads', ValueError, '1..24'),
    ('cache', ValueError, 'cache shapes'), ('capacity', ValueError, 'capacity'),
    ('stride', ValueError, 'whole FP32'), ('nzero', ValueError, 'n must'),
    ('nbig', ValueError, 'n must'), ('nbool', TypeError, 'integer'),
    ('nfloat', TypeError, 'integer'), ('nan', ValueError, 'scale'),
    ('inf', ValueError, 'scale'), ('negative', ValueError, 'scale'),
    ('zero', ValueError, 'scale'), ('overflow', ValueError, 'scale'),
    ('underflow', ValueError, 'scale'), ('scale_bool', TypeError, 'host scalar'),
    ('scale_array', TypeError, 'host scalar'),
])
def test_bad_metadata_rejected_before_allocation(kind, exc, match):
    arrays, n, scale = metadata(), 141, .0625
    if kind == 'host': arrays[0] = np.empty((4, 256), np.float32)
    if kind == 'dtype': arrays[0] = Array((4, 256), np.float16)
    if kind == 'device': arrays[-1].device.id = 1
    if kind == 'dim': arrays = metadata(dim=128)
    if kind == 'rank': arrays[0] = Array((1024,))
    if kind == 'gate': arrays[1] = Array((4, 256))
    if kind == 'groups': arrays[2:4] = [Array((3, 256)), Array((3, 256))]
    if kind == 'heads': arrays = metadata(heads=26)
    if kind == 'cache': arrays[-1] = Array((140, 256))
    if kind == 'capacity': arrays[-1] = Array((139, 2, 256))
    if kind == 'stride': arrays[0].strides = (1024, 1)
    if kind == 'nzero': n = 0
    if kind == 'nbig': n = 8193
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
        warp._validate(SimpleNamespace(ndarray=Array), *arrays, n, scale)


@pytest.mark.parametrize('value_tile', [0, 32, 64, 128])
def test_two_launches_bounded_allocations_int64_strides_and_current_stream(monkeypatch, value_tile):
    state = SimpleNamespace(device=7, stream=111)
    compilations, launches, allocations = [], [], []

    class Device:
        def __init__(self, device): self.device = device
        def __enter__(self):
            self.previous, state.device = state.device, self.device
        def __exit__(self, *args): state.device = self.previous

    def raw(source, name, options):
        compilations.append((state.device, source, name, options))
        return lambda grid, block, args: launches.append((name, state.device, state.stream, grid, block, args))

    def empty(size, dtype):
        allocations.append((state.device, size, dtype))
        return Array((size,), dtype=dtype, device=state.device)

    cp = SimpleNamespace(ndarray=Array, float32=np.float32, empty=empty, RawKernel=raw,
                         cuda=SimpleNamespace(Device=Device))
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    monkeypatch.setattr(warp, '_KERNELS', {})
    arrays = metadata()
    arrays[0].strides, arrays[1].strides = (-2048, 8), (0, -4)
    arrays[2].strides, arrays[4].strides = (0, -8), (2**35, -2048, 4)
    for device, stream in [(0, 111), (0, 222), (1, 333)]:
        state.stream = stream
        for a in arrays: a.device.id = device
        output = warp.qsa_warp_dots_gpu(*arrays, 141, .0625, value_tile=value_tile)
        assert output.shape == (1024,) and output.device.id == device
        assert state.device == 7 and state.stream == stream
    assert len(compilations) == 4 and len(launches) == 6
    assert [c[1:] for c in compilations[:2]] == [
        (warp._DOTS_SOURCE, 'qsa_warp_dots', warp._OPTIONS),
        (warp.finalizer_source(serial._SOURCE, value_tile=value_tile), 'qsa_decode', warp._OPTIONS)]
    assert [a[1] for a in allocations] == [4*141, 1024] * 3
    for i, (device, stream) in enumerate([(0, 111), (0, 222), (1, 333)]):
        dot, final = launches[2*i:2*i+2]
        assert dot[1:5] == (device, stream, (4, 36), (128,))
        grid = (4,) if value_tile == 0 else (4, 256//value_tile)
        assert final[1:5] == (device, stream, grid, (128,))
        assert all(isinstance(s, np.int64) for s in dot[-1][8:])
        assert tuple(dot[-1][8:]) == (-512, 2, 0, -2, 2**33, -512, 1)
        assert dot[-1][3] is final[-1][11]  # Same scratch, kept through final launch.
        assert final[-1][12] == 141 and tuple(final[-1][13:17]) == (-512, 2, 0, -1)
    with pytest.raises(ValueError, match='n must'):
        warp.qsa_warp_dots_gpu(*metadata(8193), 8193, .0625)
    assert len(allocations) == 6
    for other_tile in (0, 32, 64, 128):
        warp.qsa_warp_dots_gpu(*arrays, 141, .0625, value_tile=other_tile)
    assert set(warp._KERNELS) == {(0, value_tile)} | {(1, tile) for tile in (0, 32, 64, 128)}
    assert len(compilations) == 10  # Each device/tile cached separately, including zero.


@pytest.fixture
def gpu():
    from scripts.run_0290_isolated_benchmark import require_isolation
    require_isolation()  # Refuse before importing CUDA; parent runs these tests.
    cp = pytest.importorskip('cupy')
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip('CuPy GPU unavailable')
    return cp, cp.RawKernel(serial._SOURCE, 'qsa_decode', options=warp._OPTIONS)


def make_case(cp, n, *, heads=24, kv=2, layout='plain', capacity=None):
    rng, bases = np.random.default_rng(320 + n), []
    capacity = n + 3 if capacity is None else capacity

    def array(shape):
        step = 2 if layout in {'strided', 'reverse'} else 1
        backing = cp.asarray(rng.normal(size=tuple(step*s for s in shape)).astype(np.float32))
        bases.append(backing)
        view = backing[tuple(slice(None, None, step) for _ in shape)]
        if layout == 'reverse':
            view = view[tuple(slice(None, None, -1) for _ in shape)]
        return view

    inputs = [array(s) for s in ((heads, 256), (heads, 512), (kv, 256), (kv, 256),
                                  (capacity, kv, 256), (capacity, kv, 256))]
    inputs[1][:, :256] = cp.nan
    inputs[-2][n-1:] = cp.nan
    inputs[-1][n-1:] = cp.nan
    if layout == 'broadcast':
        # Every operand has a zero stride, including gate and cache dimensions.
        for i, a in enumerate(inputs):
            slices = (slice(None), slice(256, 257)) if i == 1 else tuple(slice(0, 1) for _ in a.shape)
            inputs[i] = cp.broadcast_to(a[slices], a.shape)
        inputs[1] = cp.broadcast_to(cp.zeros((1, 1), cp.float32), inputs[1].shape)
        bases.append(inputs[1])
    return inputs, bases


def launch_serial(cp, kernel, inputs, n, scale):
    q, _, k, _, _, _ = inputs
    output = cp.empty(q.size, cp.float32)
    strides = tuple(np.int64(s // a.itemsize) for a in inputs for s in a.strides)
    kernel((q.shape[0],), (128,), (*inputs, output, np.int32(n), np.int32(256),
                                  np.int32(q.shape[0] // k.shape[0]), np.float32(scale), *strides))
    return output


def check_exact(cp, old, inputs, bases, n, scale=.0625, *, value_tile=0):
    before = [cp.asnumpy(a).view(np.uint32).copy() for a in bases]
    device, stream = cp.cuda.runtime.getDevice(), cp.cuda.get_current_stream().ptr
    actual = warp.qsa_warp_dots_gpu(*inputs, n, scale, value_tile=value_tile)
    consumer = actual.copy()  # Same-stream consumer queued before host sync.
    expected = launch_serial(cp, old, inputs, n, scale)
    assert cp.cuda.runtime.getDevice() == device and cp.cuda.get_current_stream().ptr == stream
    a, b = cp.asnumpy(consumer), cp.asnumpy(expected)
    assert actual.dtype == cp.float32 and actual.shape == (inputs[0].size,)
    assert np.isfinite(a).all() and np.array_equal(a, b)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    for base, snapshot in zip(bases, before):
        assert np.array_equal(cp.asnumpy(base).view(np.uint32), snapshot)
    return a


@pytest.mark.gpu
@pytest.mark.parametrize('n', [1, 2, 77, 128, 129, 141, 435, 1024, 2051, 8192])
def test_exact_all_boundaries(gpu, n):
    cp, old = gpu
    with cp.cuda.Stream(non_blocking=True):
        inputs, bases = make_case(cp, n)
        check_exact(cp, old, inputs, bases, n)


@pytest.mark.gpu
@pytest.mark.parametrize('layout,n', [('strided', 129), ('reverse', 435), ('broadcast', 141)])
@pytest.mark.parametrize('heads,kv', [(4, 2), (24, 1), (4, 4)])
def test_arbitrary_strides_gqa_and_nondefault_stream(gpu, layout, n, heads, kv):
    cp, old = gpu
    with cp.cuda.Stream(non_blocking=True):
        inputs, bases = make_case(cp, n, heads=heads, kv=kv, layout=layout)
        check_exact(cp, old, inputs, bases, n, .37)


@pytest.mark.gpu
@pytest.mark.parametrize('value_tile', [32, 64, 128])
@pytest.mark.parametrize('n', [141, 435, 1024])
@pytest.mark.parametrize('layout', ['plain', 'strided', 'reverse', 'broadcast'])
def test_value_tiled_exact_across_contexts_and_views(gpu, value_tile, n, layout):
    cp, old = gpu
    with cp.cuda.Stream(non_blocking=True):
        inputs, bases = make_case(cp, n, layout=layout)
        check_exact(cp, old, inputs, bases, n, .37, value_tile=value_tile)


@pytest.mark.gpu
@pytest.mark.parametrize('value_tile', [32, 64, 128])
def test_tiled_finalizer_disjoint_writes_and_guards(gpu, value_tile):
    cp, old = gpu
    with cp.cuda.Stream(non_blocking=True):
        n, heads = 435, 24
        inputs, bases = make_case(cp, n, heads=heads, layout='reverse')
        expected = check_exact(cp, old, inputs, bases, n, value_tile=value_tile)
        q, _, k, _, keys, _ = inputs
        logits = cp.empty(heads*n, cp.float32)
        storage = cp.empty(q.size+2, cp.float32)
        output = storage[1:-1]
        strides = tuple(np.int64(s // a.itemsize) for a in inputs for s in a.strides)
        dot_strides = tuple(np.int64(s // a.itemsize) for a in (q, k, keys) for s in a.strides)
        dots = cp.RawKernel(warp._DOTS_SOURCE, 'qsa_warp_dots', options=warp._OPTIONS)
        final = cp.RawKernel(warp.finalizer_source(serial._SOURCE, value_tile=value_tile),
                             'qsa_decode', options=warp._OPTIONS)
        dots((heads, (n+3)//4), (128,), (q, k, keys, logits, np.int32(n), np.int32(12),
                                          np.float32(.0625), np.int32(n), *dot_strides))
        sentinel = np.uint32(0x7fc03200)
        for tiles in range(1, 256//value_tile+1):
            storage.view(cp.uint32)[:] = sentinel
            final((heads, tiles), (128,), (*inputs, output, np.int32(n), np.int32(256),
                                          np.int32(12), np.float32(.0625), logits, np.int32(n), *strides))
            actual = cp.asnumpy(storage).view(np.uint32)
            assert actual[0] == sentinel and actual[-1] == sentinel
            matrix = actual[1:-1].reshape(heads, 256)
            bound = tiles*value_tile
            assert np.array_equal(matrix[:, :bound], expected.view(np.uint32).reshape(heads, 256)[:, :bound])
            assert np.all(matrix[:, bound:] == sentinel)


@pytest.mark.gpu
@pytest.mark.parametrize('n', [1, 2, 129, 2051])
def test_no_uncommitted_cache_slot_required(gpu, n):
    cp, old = gpu
    with cp.cuda.Stream(non_blocking=True):
        inputs, bases = make_case(cp, n, heads=4, capacity=n-1)
        check_exact(cp, old, inputs, bases, n)


@pytest.mark.gpu
@pytest.mark.parametrize('n', [1, 129, 435])
def test_current_k_nan_tail_and_inflight_producers(gpu, n):
    cp, old = gpu
    with cp.cuda.Stream(non_blocking=True):
        inputs, bases = make_case(cp, n, heads=4)
        q, qg, k, v, keys, values = inputs
        q[:] = 0
        q[:, 0] = 1
        qg[:, 256:] = 0
        keys[:n-1], values[:n-1] = -100, -17
        k[:], v[:] = 100, 7
        # No host read between producers and two-kernel operator/consumer.
        actual = warp.qsa_warp_dots_gpu(*inputs, n, 1.).copy()
        expected = launch_serial(cp, old, inputs, n, 1.)
        a, b = cp.asnumpy(actual), cp.asnumpy(expected)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
        assert np.array_equal(a, np.full(a.shape, 3.5, np.float32))
        check_exact(cp, old, inputs, bases, n, 1.)


@pytest.mark.gpu
@pytest.mark.parametrize('case', ['huge', 'cancellation', 'fma-sensitive'])
@pytest.mark.parametrize('n', [2, 129, 435])
def test_adversarial_fp32_order(gpu, case, n):
    cp, old = gpu
    with cp.cuda.Stream(non_blocking=True):
        inputs, bases = make_case(cp, n, heads=4)
        q, qg, k, v, keys, values = inputs
        if case == 'huge':
            q *= np.float32(1e10)
            k *= np.float32(1e10)
            keys[:n-1] *= np.float32(1e10)
            v *= np.float32(1e20)
            values[:n-1] *= np.float32(1e20)
        elif case == 'cancellation':
            q[:] = 1
            pattern = cp.asarray(np.tile([2.**60, 1., -2.**60, 2.], 64).astype(np.float32))
            k[:], keys[:n-1] = pattern, pattern
            k[:, -1] = 3
        else:
            q[:] = np.float32(1 + 2.**-12)
            pattern = cp.asarray(np.tile([1 - 2.**-12, -1., 2.**-24, 0.], 64).astype(np.float32))
            k[:], keys[:n-1] = pattern, pattern
        qg[:, 256:] = cp.asarray(np.linspace(-80, 80, 256, dtype=np.float32))
        check_exact(cp, old, inputs, bases, n)
        # Test the scratch itself: softmax saturation must not hide dot errors.
        logits = cp.empty(4*n, cp.float32)
        strides = tuple(np.int64(s // a.itemsize) for a in (q, k, keys) for s in a.strides)
        dots = cp.RawKernel(warp._DOTS_SOURCE, 'qsa_warp_dots', options=warp._OPTIONS)
        dots((4, (n+3)//4), (128,), (q, k, keys, logits, np.int32(n), np.int32(2),
                                      np.float32(.0625), np.int32(n), *strides))
        host_q, host_k, host_keys = (cp.asnumpy(a) for a in (q, k, keys))
        reference = np.zeros((4, n), np.float32)
        for h in range(4):
            kh = h // 2
            for d in range(256):
                key = np.concatenate((host_keys[:n-1, kh, d], host_k[kh, d:d+1]))
                reference[h] += host_q[h, d] * key
        reference *= np.float32(.0625)
        assert np.array_equal(cp.asnumpy(logits).view(np.uint32), reference.ravel().view(np.uint32))


@pytest.mark.gpu
def test_dynamic_n_finalizer_preserves_parent_source(gpu):
    cp, old = gpu
    kernel = cp.RawKernel(warp.warp_dots_source(graph._SOURCE), 'qsa_decode_graph', options=warp._OPTIONS)
    with cp.cuda.Stream(non_blocking=True):
        n = 129
        inputs, bases = make_case(cp, n, heads=4, layout='reverse')
        expected = check_exact(cp, old, inputs, bases, n)
        # The normal API has populated neither a global state nor parent graph.
        q, _, k, _, keys, _ = inputs
        logits, output = cp.empty(4*n, cp.float32), cp.empty(4*256, cp.float32)
        position = cp.asarray(np.array(n-1, np.int64))
        strides = tuple(np.int64(s // a.itemsize) for a in inputs for s in a.strides)
        dot_strides = tuple(np.int64(s // a.itemsize) for a in (q, k, keys) for s in a.strides)
        dots = cp.RawKernel(warp._DOTS_SOURCE, 'qsa_warp_dots', options=warp._OPTIONS)
        dots((4, (n+3)//4), (128,), (q, k, keys, logits, np.int32(n), np.int32(2),
                                      np.float32(.0625), np.int32(n), *dot_strides))
        kernel((4,), (128,), (*inputs, output, position, np.int32(256), np.int32(2),
                             np.float32(.0625), logits, np.int32(n), *strides))
        assert np.array_equal(cp.asnumpy(output).view(np.uint32), expected.view(np.uint32))
        assert int(cp.asnumpy(position)) == n-1


def test_probe_isolation_provenance_and_compact_console(capsys):
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'probe_0320_qsa_warp_dots_operator.py'
    source = path.read_text()
    assert source.index('require_isolation()') < source.index('import cupy')
    assert 'triton' not in source and 'qwen4exp_model' not in source
    assert 'for n in (141, 435, 1024):' in source
    assert "default=7" in source and "default=100" in source
    assert 'begin_capture' in source and 'np.array_equal' in source
    assert 'warp._DOTS_SOURCE' in source and 'serial._SOURCE' in source
    assert "'--value-tile', type=int, choices=(0, 32, 64, 128), default=0" in source
    assert 'warp.finalizer_source(serial._SOURCE, value_tile=args.value_tile)' in source
    assert 'value_tile=args.value_tile' in source and "('warp_finalize', finalizer_source)" in source
    spec = importlib.util.spec_from_file_location('_probe0320_cpu_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = dict(accepted=True, value_tile=32, provenance={'giant_source': 'DO_NOT_PRINT' * 1000},
                  cases=[dict(n=141, accepted=True, median_ms={'kernel': {'serial': .2, 'warp': .1},
                                                              'api': {'serial': .3, 'warp': .2}})])
    module.print_summary(report, Path('result.json'))
    text = capsys.readouterr().out
    assert len(text) < 1000 and 'DO_NOT_PRINT' not in text and 'result.json' in text
    assert 'kernel' in text and 'api' in text and 'value_tile=32' in text
