"""CPU contracts and isolated GPU old-oracle/warp-selection exactness checks."""

import builtins
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops import router_decode as router

ROOT = Path(__file__).resolve().parents[1]


def load_probe():
    spec = importlib.util.spec_from_file_location('probe_0327_cpu_check',
        ROOT / 'scripts/probe_0327_router_selection.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_changes_only_selection_and_retains_original_oracle():
    assert router._SOURCE.count(router._BLOCK_SELECTION) == 1
    assert router._WARP_SOURCE.count(router._WARP_SELECTION) == 1
    assert router._BLOCK_SELECTION not in router._WARP_SOURCE
    assert router._WARP_SOURCE.replace(router._WARP_SELECTION, router._BLOCK_SELECTION, 1) == router._SOURCE
    assert router._warp_selection_source(router._SOURCE) == router._WARP_SOURCE
    assert router._OPTIONS == ('--std=c++14', '--fmad=false')
    # Old loop: publication + seven reduction barriers + winner publication.
    assert router._BLOCK_SELECTION.count('__syncthreads();') == 2
    assert router._WARP_SELECTION.count('__syncthreads();') == 1
    assert 'for (int stride = 64; stride; stride >>= 1)' in router._BLOCK_SELECTION
    assert 'for (int stride = 16; stride; stride >>= 1)' in router._WARP_SELECTION
    assert 'for (int stride = 2; stride; stride >>= 1)' in router._WARP_SELECTION
    assert router._WARP_SELECTION.index('__shfl_down_sync') < router._WARP_SELECTION.index('if (lane == 0)')
    assert 'if (warp == 0)' in router._WARP_SELECTION
    assert (1 + 7 + 1) * 10 == 90 and (1 + 1) * 10 == 20


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'reapplied', 'type'])
def test_source_adapter_fails_closed(kind):
    source = {'missing': router._SOURCE.replace(router._BLOCK_SELECTION, '', 1),
              'duplicate': router._SOURCE + router._BLOCK_SELECTION,
              'reapplied': router._WARP_SOURCE, 'type': None}[kind]
    with pytest.raises(TypeError if kind == 'type' else ValueError, match='string|source contract changed'):
        router._warp_selection_source(source)


class Array:
    def __init__(self, shape, dtype=np.float32, *, device=0, contiguous=True):
        self.shape, self.ndim, self.size = shape, len(shape), math.prod(shape)
        self.dtype = np.dtype(dtype)
        self.flags = SimpleNamespace(c_contiguous=contiguous)
        self.device = SimpleNamespace(id=device)


@pytest.fixture
def fake_gpu(monkeypatch):
    state = SimpleNamespace(device=0, stream=17, compilations=[], launches=[], allocations=[])

    def raw(source, name, options):
        state.compilations.append((state.device, source, name, options))

        def launch(grid, block, args):
            state.launches.append((state.device, state.stream, source, grid, block, args))
        return launch

    def empty(shape, dtype):
        state.allocations.append((shape, np.dtype(dtype)))
        return Array(shape, dtype, device=state.device)

    xp = SimpleNamespace(ndarray=Array, float32=np.float32, int64=np.int64, empty=empty,
                         RawKernel=raw, cuda=SimpleNamespace(runtime=SimpleNamespace(
                             getDevice=lambda: state.device)))
    monkeypatch.setattr(router, 'BACKEND_NAME', 'cupy')
    monkeypatch.setattr(router, 'xp', xp)
    monkeypatch.setattr(router, '_KERNELS', {})
    monkeypatch.delenv('MINI_LLM_ROUTER_WARP_SELECT', raising=False)
    return state


def test_dispatch_default_off_forced_oracle_and_per_device_cache(fake_gpu, monkeypatch):
    state = fake_gpu
    inputs = Array((1, 1, 512))
    router.try_normalized_topk(inputs, 10, 2**-14)
    monkeypatch.setenv('MINI_LLM_ROUTER_WARP_SELECT', '1')
    router.try_normalized_topk(inputs, 10, 2**-14)
    router.try_normalized_topk(inputs, 10, 2**-14, warp_select=False)
    monkeypatch.setenv('MINI_LLM_ROUTER_WARP_SELECT', '0')
    router.try_normalized_topk(inputs, 10, 2**-14, warp_select=True)
    state.device, state.stream, inputs.device.id = 1, 29, 1
    for mode in (False, True):
        weights, ids = router.try_normalized_topk(inputs, 10, 2**-14, warp_select=mode)
        assert weights.shape == ids.shape == (1, 1, 10)
        assert weights.dtype == np.float32 and ids.dtype == np.int64
    assert set(router._KERNELS) == {(0, False), (0, True), (1, False), (1, True)}
    assert [c[1] for c in state.compilations] == [router._SOURCE, router._WARP_SOURCE] * 2
    assert [c[2:] for c in state.compilations] == [('normalized_topk', router._OPTIONS)] * 4
    assert [row[2] for row in state.launches[:4]] == [
        router._SOURCE, router._WARP_SOURCE, router._SOURCE, router._WARP_SOURCE]
    for device, stream, _, grid, block, args in state.launches:
        assert (device, stream) in ((0, 17), (1, 29))
        assert grid == (1,) and block == (128,)
        assert isinstance(args[3], np.int32) and isinstance(args[4], np.int32)
        assert isinstance(args[5], np.float32)
    assert len(state.allocations) == 12


@pytest.mark.parametrize('value', ['0', 'false', 'true', 'yes', '2', ''])
def test_only_one_enables_environment_flag(fake_gpu, monkeypatch, value):
    monkeypatch.setenv('MINI_LLM_ROUTER_WARP_SELECT', value)
    router.try_normalized_topk(Array((512,)), 10, 2**-14)
    assert fake_gpu.compilations[0][1] == router._SOURCE


@pytest.mark.parametrize('value', [0, 1, '1', np.bool_(True)])
def test_invalid_explicit_override_rejected_without_allocation(fake_gpu, value):
    with pytest.raises(TypeError, match='warp_select'):
        router.try_normalized_topk(Array((512,)), 10, 2**-14, warp_select=value)
    assert not fake_gpu.compilations and not fake_gpu.allocations


@pytest.mark.parametrize('mode', [False, True])
@pytest.mark.parametrize('kind', ['dtype', 'layout', 'scalar', 'batch', 'empty', 'large',
                                 'kzero', 'klarge', 'kover_n', 'floor', 'nan', 'inf', 'device'])
def test_unsupported_metadata_still_returns_none(fake_gpu, mode, kind):
    x, k, floor = Array((1, 512)), 10, 2**-14
    if kind == 'dtype': x.dtype = np.dtype(np.float64)
    if kind == 'layout': x.flags.c_contiguous = False
    if kind == 'scalar': x = Array(())
    if kind == 'batch': x = Array((2, 512))
    if kind == 'empty': x = Array((0,))
    if kind == 'large': x = Array((4097,))
    if kind == 'kzero': k = 0
    if kind == 'klarge': k = 33
    if kind == 'kover_n': x, k = Array((3,)), 4
    if kind == 'floor': floor = np.finfo(np.float32).tiny / 2
    if kind == 'nan': floor = float('nan')
    if kind == 'inf': floor = float('inf')
    if kind == 'device': x.device.id = 1
    assert router.try_normalized_topk(x, k, floor, warp_select=mode) is None
    assert not fake_gpu.compilations and not fake_gpu.allocations and not fake_gpu.launches


def _better(a, b, n):
    if a[1] == n: return False
    if b[1] == n: return True
    if np.isnan(a[0]): return np.isnan(b[0]) and a[1] < b[1]
    if np.isnan(b[0]): return True
    return a[0] > b[0] or (a[0] == b[0] and a[1] < b[1])


def _shuffle_reduce(pairs, width, n):
    pairs = list(pairs)
    step = width // 2
    while step:
        previous = pairs[:]
        for lane in range(width - step):
            if _better(previous[lane + step], previous[lane], n):
                pairs[lane] = previous[lane + step]
        step //= 2
    return pairs[0]


@pytest.mark.parametrize('n', [1, 3, 31, 32, 33, 127, 128, 129, 511, 512, 513, 4096])
def test_cpu_comparator_network_handles_sentinels_ties_and_nonfinite(n):
    values = np.random.default_rng(n).integers(-5, 6, n).astype(np.float32)
    if n > 3: values[:4] = [np.nan, np.inf, -np.inf, -0.0]
    expected = sorted(range(n), key=lambda i: (bool(np.isnan(values[i])),
                      -float(values[i]) if not np.isnan(values[i]) else 0., i))[:min(n, 32)]
    used, actual = set(), []
    for _ in expected:
        local = []
        for tid in range(128):
            best = (-np.inf, n)
            for i in range(tid, n, 128):
                if i not in used and _better((values[i], i), best, n): best = (values[i], i)
            local.append(best)
        winners = [_shuffle_reduce(local[w*32:(w+1)*32], 32, n) for w in range(4)]
        chosen = _shuffle_reduce(winners, 4, n)[1]
        used.add(chosen)
        actual.append(chosen)
    assert actual == expected


def test_probe_import_is_standard_library_only_and_gate_precedes_gpu(monkeypatch):
    from scripts import run_0290_isolated_benchmark as gate
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'numpy', 'cupy', 'torch', 'triton', 'mini_llm'}:
            raise AssertionError(f'premature optional/backend import: {name}')
        return original(name, *args, **kwargs)

    def refuse():
        raise RuntimeError('isolation refused')

    monkeypatch.setattr(gate, 'require_isolation', refuse)
    monkeypatch.setattr(builtins, '__import__', guarded)
    probe = load_probe()
    assert probe.BATCH_CALLS == 48
    with pytest.raises(RuntimeError, match='isolation refused'):
        probe.run(SimpleNamespace())


@pytest.mark.parametrize('option,value', [('--n', '0'), ('--n', '4097'), ('--k', '33'),
    ('--k', '0'), ('--floor', 'nan'), ('--floor', 'inf'), ('--floor', '1e-50'),
    ('--warmups', '0'), ('--replays', '0'), ('--trials', '2'), ('--seed', '-1')])
def test_probe_argument_limits_without_gpu(tmp_path, option, value):
    with pytest.raises(SystemExit):
        load_probe().parse_args(['--output', str(tmp_path / 'new.json'), option, value])


def test_probe_exclusive_json_and_source_hashes_without_gpu(tmp_path, monkeypatch):
    probe = load_probe()
    output = tmp_path / 'new.json'
    args = probe.parse_args(['--output', str(output)])
    assert (args.n, args.k, args.warmups, args.replays, args.trials) == (512, 10, 10, 20, 9)
    manifest = probe.provenance(router)
    assert manifest['kernels']['old']['sha256'] == probe.digest(router._SOURCE.encode())
    assert manifest['kernels']['warp']['sha256'] == probe.digest(router._WARP_SOURCE.encode())
    for path, record in manifest['files'].items():
        assert record['sha256'] == probe.digest((ROOT / path).read_bytes())
    result = dict(accepted=True, median_call_us={'old': 14., 'warp': 7.}, median_old_over_warp=2.)
    monkeypatch.setattr(probe, 'run', lambda args: result)
    monkeypatch.setattr(sys, 'path', sys.path[:])
    probe.main(['--output', str(output)])
    before = output.read_bytes()
    assert json.loads(before) == result
    with pytest.raises(SystemExit):
        probe.main(['--output', str(output)])
    assert output.read_bytes() == before
    with pytest.raises(SystemExit):
        probe.parse_args(['--output', str(tmp_path / 'missing' / 'new.json')])


@pytest.fixture
def gpu():
    if BACKEND_NAME != 'cupy':
        pytest.skip('Requires MINI_LLM_BACKEND=cupy')
    from scripts.run_0290_isolated_benchmark import require_isolation
    require_isolation()
    cp = pytest.importorskip('cupy')
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CuPy GPU unavailable')
    if not count:
        pytest.skip('CuPy GPU unavailable')
    return cp


def assert_outputs(cp, old, candidate, *, nonfinite=False):
    a, ia = (cp.asnumpy(x) for x in old)
    b, ib = (cp.asnumpy(x) for x in candidate)
    assert a.dtype == b.dtype == np.float32 and ia.dtype == ib.dtype == np.int64
    assert a.shape == b.shape == ia.shape == ib.shape
    assert a.flags.c_contiguous and b.flags.c_contiguous
    np.testing.assert_array_equal(ia, ib)
    if nonfinite:
        np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True)
    else:
        assert np.isfinite(a).all() and np.isfinite(b).all()
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
    return ia


@pytest.mark.gpu
@pytest.mark.parametrize('n,k', [(1, 1), (3, 3), (31, 31), (32, 32), (33, 10),
    (127, 32), (128, 32), (129, 10), (511, 10), (512, 10), (513, 32), (4096, 32)])
@pytest.mark.parametrize('floor', [2**-14, .9, 2., float(np.finfo(np.float32).tiny),
                                 float(np.finfo(np.float32).max)])
def test_gpu_finite_bitwise_parity_shape_dtype_order_and_floor(gpu, n, k, floor, monkeypatch):
    cp = gpu
    host = np.random.default_rng(327+n).normal(0, 5, (1, 1, n)).astype(np.float32)
    with cp.cuda.Stream(non_blocking=True) as stream:
        logits = cp.asarray(host)
        monkeypatch.setenv('MINI_LLM_ROUTER_WARP_SELECT', '1')
        old = router.try_normalized_topk(logits, k, floor, warp_select=False)
        candidate = router.try_normalized_topk(logits, k, floor)
        ids = assert_outputs(cp, old, candidate)
        assert old[0].shape == candidate[0].shape == (1, 1, k)
        expected = np.lexsort((np.arange(n), -host.reshape(-1)))[:k]
        np.testing.assert_array_equal(ids.reshape(-1), expected)
        np.testing.assert_array_equal(cp.asnumpy(logits).view(np.uint32), host.view(np.uint32))
        assert cp.cuda.get_current_stream().ptr == stream.ptr


@pytest.mark.gpu
@pytest.mark.parametrize('values,k', [([0.] * 512, 10), ([-0., 0.] * 256, 10),
    ([1., np.nextafter(np.float32(1), np.float32(2)), 1., -10.], 4),
    ([np.finfo(np.float32).max, -np.finfo(np.float32).max, 0., 1.], 4),
    ([10000., -10000., 0., 1.], 4), ([np.nan, 1., -np.inf, 1., np.nan], 5),
    ([np.inf, 2., np.inf, -np.inf], 4), ([-np.inf] * 512, 10), ([np.nan] * 33, 32)])
@pytest.mark.parametrize('leading', [(), (1,), (1, 1, 1)])
def test_gpu_ties_signed_zero_extremes_and_nonfinite(gpu, values, k, leading):
    cp = gpu
    host = np.asarray(values, np.float32).reshape(leading + (len(values),))
    logits = cp.asarray(host)
    old = router.try_normalized_topk(logits, k, 2**-14, warp_select=False)
    candidate = router.try_normalized_topk(logits, k, 2**-14, warp_select=True)
    ids = assert_outputs(cp, old, candidate, nonfinite=not np.isfinite(host).all())
    expected = sorted(range(len(values)), key=lambda i: (bool(np.isnan(values[i])),
                      -float(values[i]) if not np.isnan(values[i]) else 0., i))[:k]
    np.testing.assert_array_equal(ids.reshape(-1), expected)
    assert candidate[0].shape == leading + (k,)
    np.testing.assert_array_equal(cp.asnumpy(logits).view(np.uint32), host.view(np.uint32))


@pytest.mark.gpu
@pytest.mark.parametrize('mode', [False, True])
def test_gpu_unsupported_layout_dtype_batch_and_k(gpu, mode):
    cp = gpu
    for x, k in [(cp.zeros(512, cp.float16), 10), (cp.zeros(512, cp.float64), 10),
                 (cp.zeros(1024, cp.float32)[::2], 10), (cp.zeros((2, 512), cp.float32), 10),
                 (cp.zeros(0, cp.float32), 1), (cp.zeros(4097, cp.float32), 10),
                 (cp.zeros(512, cp.float32), 33), (cp.zeros(3, cp.float32), 4)]:
        assert router.try_normalized_topk(x, k, 2**-14, warp_select=mode) is None


@pytest.mark.gpu
@pytest.mark.parametrize('n,k', [(3, 3), (512, 10), (4096, 32)])
def test_gpu_preallocated_graph_replay_updated_inputs_and_caller_unchanged(gpu, n, k):
    cp = gpu
    graphs = {}
    with cp.cuda.Stream(non_blocking=True) as stream:
        logits = cp.zeros(n, cp.float32)
        outputs = {mode: (cp.empty(k, cp.float32), cp.empty(k, cp.int64)) for mode in (False, True)}
        arguments = {mode: (logits, *out, np.int32(n), np.int32(k), np.float32(2**-14))
                     for mode, out in outputs.items()}
        for mode in outputs:
            kernel = router._get_kernel(cp.cuda.runtime.getDevice(), warp_select=mode)
            kernel((1,), (128,), arguments[mode])
            stream.synchronize()
            stream.begin_capture()
            for _ in range(48):
                kernel((1,), (128,), arguments[mode])
            graphs[mode] = stream.end_capture()
        cases = [np.random.default_rng(n).normal(size=n).astype(np.float32),
                 np.resize(np.array([-0., 0.], np.float32), n), np.full(n, np.nan, np.float32)]
        for host in cases:
            logits.set(host, stream=stream)
            direct = router.try_normalized_topk(logits, k, 2**-14, warp_select=False)
            for mode in (True, False, False, True):
                graphs[mode].launch(stream)
            assert_outputs(cp, direct, outputs[False], nonfinite=not np.isfinite(host).all())
            assert_outputs(cp, direct, outputs[True], nonfinite=not np.isfinite(host).all())
            np.testing.assert_array_equal(cp.asnumpy(logits).view(np.uint32), host.view(np.uint32))
            assert cp.cuda.get_current_stream().ptr == stream.ptr
        stream.synchronize()
        graphs.clear()
