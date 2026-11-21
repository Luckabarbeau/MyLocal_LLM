"""Exact-tree reduction contracts; GPU checks require the parent's 0290 gate."""

import hashlib

import numpy as np
import pytest

from mini_llm.ops import router_decode as router
from test_0327_router_warp_selection import Array, assert_outputs, fake_gpu, gpu, load_probe


ORIGINAL_SHA256 = 'e5850df19e3e8de75c76f7bef03e9b730e93e681f3e11dcbad070f877e147a50'
SELECTION_SHA256 = '435ee2c069d4fcd0c24974673d77ca23dc8203f75a30b6fa5a37adc30df11e4f'


def test_existing_kernel_sources_remain_byte_identical():
    assert hashlib.sha256(router._SOURCE.encode()).hexdigest() == ORIGINAL_SHA256
    assert hashlib.sha256(router._WARP_SOURCE.encode()).hexdigest() == SELECTION_SHA256
    assert router._WARP_SOURCE.replace(router._WARP_SELECTION, router._BLOCK_SELECTION, 1) == router._SOURCE
    assert router._OPTIONS == ('--std=c++14', '--fmad=false')


def test_generated_full_source_changes_only_two_reduction_fragments():
    full = router._REDUCE_SOURCE
    assert router._warp_reduction_source(router._WARP_SOURCE) == full
    restored = full
    for before, after in ((router._MAX_REDUCTION, router._WARP_MAX_REDUCTION),
                          (router._SUM_REDUCTION, router._WARP_SUM_REDUCTION)):
        assert router._WARP_SOURCE.count(before) == full.count(after) == 1
        restored = restored.replace(after, before, 1)
        # Both shared-memory combines and their barriers are literally retained.
        shared = after.split('    if (tid < 32) {', 1)[0]
        assert shared == before.replace('for (int stride = 64; stride; stride >>= 1)',
                                       'for (int stride = 64; stride >= 32; stride >>= 1)', 1)
        assert 'int lane = tid;' in after
        assert 'for (int stride = 16; stride; stride >>= 1)' in after
        assert '__shfl_down_sync(0xffffffffu,' in after
        assert after.endswith('    __syncthreads();\n')
    assert restored == router._WARP_SOURCE  # All exp/div/mass/floor/selection bytes unchanged.
    assert router._WARP_MAX_REDUCTION + '    maximum = partial[0];' in full
    assert router._WARP_SUM_REDUCTION + '    denominator = partial[0];' in full
    assert 'if (lane < stride) a = isnan(a) || isnan(b) ? NAN : fmaxf(a, b);' in full
    assert 'if (lane < stride) sum += b;' in full
    assert '__shfl_xor_sync' not in full
    # Seven old stage barriers -> two shared stages plus one publication barrier.
    assert 2 * (7 - (2 + 1)) == 8


@pytest.mark.parametrize('fragment', ['max', 'sum', 'selection'])
@pytest.mark.parametrize('drift', ['missing', 'duplicate'])
def test_reduction_adapter_fails_closed(fragment, drift):
    anchor = {'max': router._MAX_REDUCTION, 'sum': router._SUM_REDUCTION,
              'selection': router._WARP_SELECTION}[fragment]
    source = (router._WARP_SOURCE.replace(anchor, '', 1) if drift == 'missing'
              else router._WARP_SOURCE + anchor)
    with pytest.raises(ValueError, match='reduction source contract changed'):
        router._warp_reduction_source(source)


@pytest.mark.parametrize('source', [router._SOURCE, router._REDUCE_SOURCE, None])
def test_adapter_requires_selection_only_base_and_single_application(source):
    with pytest.raises(TypeError if source is None else ValueError, match='string|source contract changed'):
        router._warp_reduction_source(source)


@pytest.mark.parametrize('operator', ['max', 'add'])
def test_symbolic_tree_preserves_every_left_and_right_operand(operator):
    initial = list(range(128))

    def stages(values, strides):
        values, operations = list(values), []
        for stride in strides:
            previous = values[:]
            for lane in range(stride):
                node = (operator, previous[lane], previous[lane + stride])
                values[lane] = node
                operations.append((stride, lane, node))
        return values, operations

    original, old_operations = stages(initial, (64, 32, 16, 8, 4, 2, 1))
    shared, shared_operations = stages(initial, (64, 32))
    warp, warp_operations = stages(shared[:32], (16, 8, 4, 2, 1))
    assert original[0] == warp[0]
    assert old_operations == shared_operations + warp_operations
    assert len(old_operations) == 127


@pytest.fixture
def modes_gpu(fake_gpu, monkeypatch):
    monkeypatch.delenv('MINI_LLM_ROUTER_WARP_REDUCE', raising=False)
    return fake_gpu


@pytest.mark.parametrize('select_env', ['0', '1'])
@pytest.mark.parametrize('reduce_env', ['0', '1'])
@pytest.mark.parametrize('warp_select', [None, False, True])
@pytest.mark.parametrize('warp_reduce', [None, False, True])
def test_public_override_interactions(modes_gpu, monkeypatch, select_env, reduce_env,
                                      warp_select, warp_reduce):
    monkeypatch.setenv('MINI_LLM_ROUTER_WARP_SELECT', select_env)
    monkeypatch.setenv('MINI_LLM_ROUTER_WARP_REDUCE', reduce_env)
    if warp_select is False and warp_reduce is True:
        with pytest.raises(ValueError, match='conflicts'):
            router.try_normalized_topk(Array((512,)), 10, 2**-14,
                                      warp_select=warp_select, warp_reduce=warp_reduce)
        assert not modes_gpu.allocations and not modes_gpu.compilations
        return
    full = warp_reduce is True or (warp_reduce is None and warp_select is None and reduce_env == '1')
    selected = warp_select if warp_select is not None else select_env == '1'
    expected = router._REDUCE_SOURCE if full else router._WARP_SOURCE if selected else router._SOURCE
    router.try_normalized_topk(Array((512,)), 10, 2**-14,
                              warp_select=warp_select, warp_reduce=warp_reduce)
    assert modes_gpu.compilations[0][1] == expected


def test_raw_three_way_cache_is_device_specific_and_ignores_flags(modes_gpu, monkeypatch):
    monkeypatch.setenv('MINI_LLM_ROUTER_WARP_SELECT', '1')
    monkeypatch.setenv('MINI_LLM_ROUTER_WARP_REDUCE', '1')
    for device in (0, 1):
        modes_gpu.device = device
        old = router._get_kernel(device)
        selection = router._get_kernel(device, warp_select=True)
        full = router._get_kernel(device, warp_reduce=True)
        assert router._get_kernel(device, warp_select=False) is old
        assert router._get_kernel(device, warp_select=True, warp_reduce=False) is selection
        assert router._get_kernel(device, warp_select=True, warp_reduce=True) is full
        assert router._get_kernel(device, warp_select=None, warp_reduce=None) is old
        assert old is not selection and selection is not full and full is not old
    assert set(router._KERNELS) == {(d, v) for d in (0, 1) for v in (False, True, 'reduce')}
    assert [row[1] for row in modes_gpu.compilations] == [
        router._SOURCE, router._WARP_SOURCE, router._REDUCE_SOURCE] * 2
    assert all(row[2:] == ('normalized_topk', router._OPTIONS) for row in modes_gpu.compilations)
    with pytest.raises(ValueError, match='conflicts'):
        router._get_kernel(0, warp_select=False, warp_reduce=True)


@pytest.mark.parametrize('value', [0, 1, '1', np.bool_(True)])
def test_invalid_reduce_override_rejected_before_gpu_work(modes_gpu, value):
    with pytest.raises(TypeError, match='warp_reduce'):
        router.try_normalized_topk(Array((512,)), 10, 2**-14, warp_reduce=value)
    with pytest.raises(TypeError, match='warp_reduce'):
        router._get_kernel(0, warp_reduce=value)
    assert not modes_gpu.compilations and not modes_gpu.allocations


@pytest.mark.parametrize('value', ['false', 'true', 'yes', '2', ''])
def test_reduce_environment_requires_literal_one(modes_gpu, monkeypatch, value):
    monkeypatch.setenv('MINI_LLM_ROUTER_WARP_REDUCE', value)
    router.try_normalized_topk(Array((512,)), 10, 2**-14)
    assert modes_gpu.compilations[0][1] == router._SOURCE


@pytest.mark.parametrize('full', [False, True])
def test_probe_opt_in_selected_source_provenance_and_legacy_contract(tmp_path, full):
    probe = load_probe()
    argv = ['--output', str(tmp_path / 'new.json')] + (['--warp-reductions'] if full else [])
    args = probe.parse_args(argv)
    assert args.warp_reductions is full
    selected, unchanged = probe.candidate_source(router, warp_reductions=full)
    assert selected == (router._REDUCE_SOURCE if full else router._WARP_SOURCE)
    assert unchanged
    manifest = probe.provenance(router, warp_reductions=full)
    assert manifest['selected_candidate']['source'] == selected
    assert manifest['selected_candidate']['sha256'] == probe.digest(selected.encode())
    assert manifest['selected_candidate']['remainder_byte_identical']
    assert manifest['timed_kernel_sources'] == {'old': 'old', 'warp': 'full' if full else 'warp'}
    assert manifest['kernels']['old']['sha256'] == ORIGINAL_SHA256
    assert manifest['kernels']['warp']['sha256'] == SELECTION_SHA256
    assert manifest['kernels']['full']['source'] == router._REDUCE_SOURCE
    assert router._WARP_SOURCE.replace(router._WARP_SELECTION, router._BLOCK_SELECTION, 1) == router._SOURCE
    assert probe.BATCH_CALLS == 48


@pytest.mark.gpu
@pytest.mark.parametrize('n,k', [(1, 1), (3, 3), (31, 31), (32, 32), (33, 10),
    (127, 32), (128, 32), (129, 10), (511, 10), (512, 10), (513, 32), (4096, 32)])
@pytest.mark.parametrize('floor', [2**-14, .9, 2., float(np.finfo(np.float32).tiny),
                                 float(np.finfo(np.float32).max)])
def test_gpu_full_finite_bitwise_vs_original_and_selection(gpu, monkeypatch, n, k, floor):
    cp = gpu
    monkeypatch.setenv('MINI_LLM_ROUTER_WARP_SELECT', '0')
    monkeypatch.setenv('MINI_LLM_ROUTER_WARP_REDUCE', '1')
    host = np.random.default_rng(329+n).normal(0, 5, (1, 1, n)).astype(np.float32)
    with cp.cuda.Stream(non_blocking=True) as stream:
        logits = cp.asarray(host)
        old = router.try_normalized_topk(logits, k, floor, warp_select=False)
        selection = router.try_normalized_topk(logits, k, floor, warp_select=True)
        full = router.try_normalized_topk(logits, k, floor, warp_reduce=True)
        automatic = router.try_normalized_topk(logits, k, floor)
        for result in (selection, full, automatic):
            assert_outputs(cp, old, result)
            assert result[0].shape == result[1].shape == (1, 1, k)
            assert result[0].flags.c_contiguous and result[1].flags.c_contiguous
        np.testing.assert_array_equal(cp.asnumpy(full[1]).reshape(-1),
                                      np.lexsort((np.arange(n), -host.reshape(-1)))[:k])
        np.testing.assert_array_equal(cp.asnumpy(logits).view(np.uint32), host.view(np.uint32))
        assert cp.cuda.get_current_stream().ptr == stream.ptr


@pytest.mark.gpu
@pytest.mark.parametrize('values,k', [([0.] * 512, 10), ([-0., 0.] * 256, 10),
    ([1., np.nextafter(np.float32(1), np.float32(2)), 1., -10.], 4),
    ([np.finfo(np.float32).max, -np.finfo(np.float32).max, 0., 1.], 4),
    ([10000., -10000., 0., 1.], 4), ([np.nan, 1., -np.inf, 1., np.nan], 5),
    ([np.inf, 2., np.inf, -np.inf], 4), ([-np.inf] * 512, 10), ([np.nan] * 33, 32)])
@pytest.mark.parametrize('leading', [(), (1,), (1, 1, 1)])
def test_gpu_full_ties_zeros_extremes_nonfinite_and_layout(gpu, values, k, leading):
    cp = gpu
    host = np.asarray(values, np.float32).reshape(leading + (len(values),))
    logits = cp.asarray(host)
    old = router.try_normalized_topk(logits, k, 2**-14, warp_select=False, warp_reduce=False)
    full = router.try_normalized_topk(logits, k, 2**-14, warp_select=True, warp_reduce=True)
    ids = assert_outputs(cp, old, full, nonfinite=not np.isfinite(host).all())
    expected = sorted(range(len(values)), key=lambda i: (bool(np.isnan(values[i])),
                      -float(values[i]) if not np.isnan(values[i]) else 0., i))[:k]
    np.testing.assert_array_equal(ids.reshape(-1), expected)
    assert full[0].shape == full[1].shape == leading + (k,)
    np.testing.assert_array_equal(cp.asnumpy(logits).view(np.uint32), host.view(np.uint32))


@pytest.mark.gpu
def test_gpu_full_unsupported_metadata_rejected(gpu):
    cp = gpu
    for x, k, floor in [(cp.zeros(512, cp.float16), 10, 2**-14),
                       (cp.zeros(512, cp.float64), 10, 2**-14),
                       (cp.zeros(1024, cp.float32)[::2], 10, 2**-14),
                       (cp.zeros((2, 512), cp.float32), 10, 2**-14),
                       (cp.zeros(0, cp.float32), 1, 2**-14),
                       (cp.zeros(4097, cp.float32), 10, 2**-14),
                       (cp.zeros(512, cp.float32), 33, 2**-14),
                       (cp.zeros(3, cp.float32), 4, 2**-14),
                       (cp.zeros(512, cp.float32), 10, np.finfo(np.float32).tiny / 2),
                       (cp.zeros(512, cp.float32), 10, float('nan'))]:
        assert router.try_normalized_topk(x, k, floor, warp_reduce=True) is None


@pytest.mark.gpu
@pytest.mark.parametrize('n,k', [(3, 3), (512, 10), (4096, 32)])
def test_gpu_full_graph_replay_updated_inputs_and_unmodified_callers(gpu, n, k):
    cp = gpu
    graphs = {}
    modes = {'old': (False, False), 'selection': (True, False), 'full': (True, True)}
    with cp.cuda.Stream(non_blocking=True) as stream:
        logits = cp.zeros(n, cp.float32)
        outputs = {name: (cp.empty(k, cp.float32), cp.empty(k, cp.int64)) for name in modes}
        arguments = {name: (logits, *out, np.int32(n), np.int32(k), np.float32(2**-14))
                     for name, out in outputs.items()}
        for name, (select, reduce) in modes.items():
            kernel = router._get_kernel(cp.cuda.runtime.getDevice(), warp_select=select, warp_reduce=reduce)
            kernel((1,), (128,), arguments[name])
            stream.synchronize()
            stream.begin_capture()
            for _ in range(48):
                kernel((1,), (128,), arguments[name])
            graphs[name] = stream.end_capture()
        cases = [np.random.default_rng(n).normal(size=n).astype(np.float32),
                 np.resize(np.array([-0., 0.], np.float32), n),
                 np.full(n, np.nan, np.float32), np.full(n, -np.inf, np.float32)]
        for host in cases:
            logits.set(host, stream=stream)
            direct = router.try_normalized_topk(logits, k, 2**-14, warp_select=False, warp_reduce=False)
            for name in ('full', 'old', 'selection', 'selection', 'old', 'full'):
                graphs[name].launch(stream)
            for result in outputs.values():
                assert_outputs(cp, direct, result, nonfinite=not np.isfinite(host).all())
            np.testing.assert_array_equal(cp.asnumpy(logits).view(np.uint32), host.view(np.uint32))
            assert cp.cuda.get_current_stream().ptr == stream.ptr
        stream.synchronize()
        graphs.clear()
