"""Synthetic caller/stager/issuer tests only; never load or run a model.

Every GPU test is gated to the physical GPU0 UUID, including concurrency tests.
The independent scalar CUDA oracle consumes original route order, not the
pipeline's grouping or pointer tables. CPU collection does not import CuPy.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mini_llm.ops import native_prefill_cuda as native
from mini_llm.ops.native_prefill_pipeline import NativePrefillPipeline
from test_triton_native_prefill import decode_numpy, make_weights, oracle, exact

UUID = 'GPU-bea264b4-9e76-3be5-5883-7bde3fe9f82d'


def test_import_is_lazy_and_short_path_is_explicit():
    subprocess.run([sys.executable, '-c',
        'import sys; from mini_llm.ops.native_prefill_pipeline import NativePrefillPipeline; '
        'assert "cupy" not in sys.modules'], check=True,
        env=dict(os.environ, MINI_LLM_BACKEND='numpy'))
    p = NativePrefillPipeline()
    assert p.device_ring_bytes == 128 * 1024**2 and p.ring_slots == 128
    assert p.host_depth == 16 and 2 <= p.workers <= 4
    with pytest.raises(ValueError, match='CPU/MMQ'):
        p.begin_chunk(None, 3071)
    with pytest.raises(RuntimeError, match='begin_chunk'):
        p.run(None, None, None, None)
    p.end_chunk()
    p.close()
    with pytest.raises(RuntimeError, match='closed'):
        p.begin_chunk(None, 3072)
    source = Path(sys.modules[NativePrefillPipeline.__module__].__file__).read_text()
    assert 'asnumpy' not in source and '.get(' not in source.replace('snapshot.get(', '')
    assert 'fmaf(weights[j], dm[slot[j] * D + d], v)' in source


@pytest.mark.parametrize('kwargs', [dict(ring_slots=1), dict(workers=0),
    dict(host_depth=True), dict(device_ring_bytes=-1), dict(stream_min=0)])
def test_configuration_contract(kwargs):
    with pytest.raises(ValueError):
        NativePrefillPipeline(**kwargs)


@pytest.fixture(scope='module')
def gpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != UUID:
        pytest.skip('GPU tests require CUDA_VISIBLE_DEVICES='+UUID)
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() != 1:
        pytest.skip('exactly the selected GPU0 required')
    cp.cuda.Device(0).use()
    return cp


class Executor:
    def __init__(self):
        self.model_id = 'synthetic-native-pipeline'
        self.cache = NS(_admission_limit=0, resident={})
        self.protected = set()
        self.sources, self.calls, self.scopes = {}, [], []

    @contextmanager
    def dispatch_scope(self, layer_id, ids):
        ids = tuple(ids)
        self.scopes.append((layer_id, ids))
        keys = {(layer_id, e) for e in ids}
        self.protected.update(keys)
        try:
            yield
        finally:
            self.protected.difference_update(keys)

    def cpu_source_weights(self, key):
        assert (key.layer_id, key.expert_id) in self.protected
        self.calls.append((threading.get_ident(), key))
        return self.sources[key]


def model_fixture(cp, *, E=8, layers=2, gu='IQ2_S', down='IQ4_NL', resident=(1, 4)):
    from mini_llm.ops.expert_residency import ExpertKey
    executor = Executor()
    blocks, raw = [], []
    for layer_id in range(layers):
        ex = NS(d_model=256, d_ff=128, n_experts=E, layer_id=layer_id,
                expert_executor=executor)
        blocks.append(NS(moe=NS(experts=ex, spec=NS(top_k=3))))
        all_weights = []
        for e in range(E):
            weights = tuple(NS(data=make_weights(kind, n, k, 911+layer_id*101+e*3+j),
                               shape=(n, k), quant_type=kind)
                for j, (kind, n, k) in enumerate(((gu, 128, 256), (gu, 128, 256), (down, 256, 128))))
            key = ExpertKey(executor.model_id, layer_id, e)
            all_weights.append(weights)
            if e in resident:
                dw = [NS(data=cp.asarray(w.data), shape=w.shape, quant_type=w.quant_type) for w in weights]
                handle = NS(gate=dw[0], up=dw[1], down=dw[2])
                executor.cache.resident[key] = (handle, sum(w.data.nbytes for w in weights))
            else:
                executor.sources[key] = weights
        raw.append(all_weights)
    return NS(blocks=blocks), executor, raw


def inputs(cp, E, T=5):
    rng = np.random.default_rng(765)
    x = cp.asarray(rng.normal(0, .2, (T, 256)).astype(np.float32))
    # Deliberately duplicate IDs, leave many unrouted, and use non-ID rank order.
    ids_host = np.resize(np.array([[E-1, 0, 3], [3, 3, 0], [0, E-1, 3]], np.int32), (T, 3))
    weights = cp.asarray(rng.uniform(-.3, .8, (T, 3)).astype(np.float32))
    return x, weights, cp.asarray(ids_host), ids_host


_ORDERED_ORACLE = r'''
extern "C" __global__ void ordered(const float* d, const float* w, float* y, int K) {
    int t=blockIdx.x, col=threadIdx.x;
    float a=0.f;
    for(int j=0;j<K;++j) a=fmaf(w[t*K+j], d[(t*K+j)*256+col], a);
    y[t*256+col]=a;
}
'''


def scalar_reference(cp, x, weights, ids_host, raw):
    """Independent scalar arithmetic, fully selected experts, original routes."""
    T, K = ids_host.shape
    R, D, F = T*K, 256, 128
    rows = cp.asarray(np.repeat(np.arange(T, dtype=np.int32), K))
    xr = x[rows]
    aq, ax = cp.empty((R, D), cp.int8), cp.empty((R, D//32), cp.float32)
    packed = cp.empty((R, D//64, 80), cp.uint8)
    oracle(cp, 'quant')((R, D//32), (1,), (xr, xr, aq, ax, packed, np.int32(D), np.int32(0)))
    route_d = cp.empty((R, D), cp.float32)
    for e in np.unique(ids_host):
        idx = cp.asarray(np.flatnonzero(ids_host.reshape(-1) == e).astype(np.int32))
        n = idx.size
        G, U = cp.empty((n, F), cp.float32), cp.empty((n, F), cp.float32)
        for w, dest in zip(raw[e][:2], (G, U)):
            c, s = decode_numpy(w.data, w.quant_type, F, D)
            oracle(cp, 'linear')((n, F), (1,), (aq[idx], ax[idx], cp.asarray(c), cp.asarray(s), dest,
                np.int32(n), np.int32(F), np.int32(D), np.int32(w.quant_type == 'IQ2_S')))
        hq, hs = cp.empty((n, F), cp.int8), cp.empty((n, F//32), cp.float32)
        hp = cp.empty((n, F//64, 80), cp.uint8)
        oracle(cp, 'quant')((n, F//32), (1,), (G, U, hq, hs, hp, np.int32(F), np.int32(1)))
        w = raw[e][2]
        c, s = decode_numpy(w.data, w.quant_type, D, F)
        de = cp.empty((n, D), cp.float32)
        oracle(cp, 'linear')((n, D), (1,), (hq, hs, cp.asarray(c), cp.asarray(s), de,
            np.int32(n), np.int32(D), np.int32(F), np.int32(0)))
        route_d[idx] = de
    out = cp.empty((T, D), cp.float32)
    cp.RawKernel(_ORDERED_ORACLE, 'ordered')((T,), (D,), (route_d, weights, out, np.int32(K)))
    return out


@pytest.mark.parametrize('gu', native.GU_FORMATS)
@pytest.mark.parametrize('down', native.DOWN_FORMATS)
def test_scalar_oracle_ring_wrap_and_no_route_readback(gpu, monkeypatch, gu, down):
    cp = gpu
    model, executor, raw = model_fixture(cp, gu=gu, down=down)
    x, weights, ids, ids_host = inputs(cp, 8)
    expected = [scalar_reference(cp, x, weights, ids_host, layer) for layer in raw]
    p = NativePrefillPipeline(stream_min=1, ring_slots=2, host_depth=2, workers=2)
    outputs = []
    launches = []
    execute = native.execute_batch

    def tracked(e0, e1, R, E, *args, **kwargs):
        assert (R, E) == (15, 8)
        launches.append((e0, e1))
        return execute(e0, e1, R, E, *args, **kwargs)

    def no_readback(*args, **kwargs):
        raise AssertionError('host route/weight readback')

    try:
        with monkeypatch.context() as patch:
            patch.setattr(cp, 'asnumpy', no_readback)
            patch.setattr(native, 'execute_batch', tracked)
            p.begin_chunk(model, 5)  # derives K=3, all sources already captured
            assert len(executor.calls) == 12
            assert {thread for thread, _ in executor.calls} == {threading.get_ident()}
            assert len(executor.protected) == 16
            for layer in p._layers:
                for e in (1, 4):
                    assert layer.sequences[e] is None
                    assert layer.host_ptrs[e, 0] == layer.weights[e][0].data.data.ptr
            assert p._device.nbytes <= p.device_ring_bytes
            for b in model.blocks:
                outputs.append(p.run(b.moe.experts, x, weights, ids))
            p.end_chunk()
        for out, ref in zip(outputs, expected):
            exact(cp, out, ref)
        assert p.counters['experts_streamed'] == 12  # includes ALL unrouted experts
        assert p.counters['resident_reads'] == 4
        total = sum(w.data.nbytes for layer in raw for trio in layer for w in trio)
        assert p.counters['weight_bytes_streamed'] + p.counters['resident_weight_bytes'] == total
        assert p.counters['layers'] == 2 and p.counters['batches'] == len(launches)
        assert not executor.protected and not p._threads
        # Ring, pinned owners and same-T workspace survive chunk retirement.
        retained = (p._device, p._host_owner, p._workspace)
        p.begin_chunk(model, 5, 3)
        for b in model.blocks:
            p.run(b.moe.experts, x, weights, ids)
        p.end_chunk()
        assert p._device is retained[0] and p._host_owner is retained[1] and p._workspace is retained[2]
    finally:
        p.close()


@pytest.mark.parametrize('resident', [(), tuple(range(16)), (0, 2, 5, 9, 15)])
def test_cross_layer_lookahead_and_interleaved_input(gpu, resident):
    cp = gpu
    model, executor, raw = model_fixture(cp, E=16, layers=3, resident=resident)
    p = NativePrefillPipeline(stream_min=1, ring_slots=128, device_ring_bytes=2*1024**2,
                              host_depth=2, workers=2)
    staged = []
    stage_one = p._stage_one

    def delayed(idx):
        time.sleep(.001 if idx % 2 else .003)
        stage_one(idx)
        staged.append((idx, threading.current_thread().name))

    p._stage_one = delayed
    x, weights, ids, _ = inputs(cp, 16)
    saved = []
    try:
        p.begin_chunk(model, 5, 3)
        if p._seq:
            # Proves copies issue before attention/run, not inline at first MoE.
            with p._cv:
                assert p._cv.wait_for(lambda: p._issued >= min(len(p._seq), p.capacity), timeout=10)
            if len(resident) < 16:
                first_layer_streamed = 16 - len(resident)
                assert p._issued > first_layer_streamed  # cross-layer lookahead
        for i, b in enumerate(model.blocks):
            # New input is produced between calls, as with real attention.
            x = x * np.float32(.75) + np.float32(.01 * i)
            saved.append(p.run(b.moe.experts, x, weights, ids))
        p.end_chunk()
        assert all(np.isfinite(cp.asnumpy(y)).all() for y in saved)
        assert p.counters['layers'] == 3
        assert p.counters['experts_streamed'] == 3*(16-len(resident))
        assert all(name.startswith('native-stage-') for _, name in staged)
        assert sorted(idx for idx, _ in staged) == list(range(len(staged)))
        assert not executor.protected
    finally:
        p.close()


def test_background_failure_propagates_drains_and_can_reuse(gpu):
    cp = gpu
    model, executor, _ = model_fixture(cp, resident=())
    x, w, ids, _ = inputs(cp, 8)
    p = NativePrefillPipeline(stream_min=1, ring_slots=2, host_depth=2, workers=2)
    original = p._stage_one

    def fail(idx):
        if idx == 4:
            raise OSError('synthetic RAM source failure')
        original(idx)

    p._stage_one = fail
    p.begin_chunk(model, 5, 3)
    threads = list(p._threads)
    with pytest.raises(RuntimeError, match='background worker') as error:
        p.run(model.blocks[0].moe.experts, x, w, ids)
    assert isinstance(error.value.__cause__, OSError)
    with pytest.raises(RuntimeError, match='background worker'):
        p.end_chunk()
    assert not executor.protected and not p._active and all(not t.is_alive() for t in threads)
    p._stage_one = original
    p.begin_chunk(model, 5, 3)
    for b in model.blocks:
        p.run(b.moe.experts, x, w, ids)
    p.end_chunk()
    p.close()
    assert p._device is None and p._host_owner is None


def test_frozen_contracts_order_validation_and_partial_close(gpu):
    cp = gpu
    model, executor, _ = model_fixture(cp)
    x, w, ids, _ = inputs(cp, 8)
    p = NativePrefillPipeline(stream_min=1, ring_slots=2)
    executor.cache._admission_limit = None
    with pytest.raises(ValueError, match='frozen admission'):
        p.begin_chunk(model, 5, 3)
    assert not p._active and not executor.protected
    executor.cache._admission_limit = 0
    p.begin_chunk(model, 5, 3)
    threads = list(p._threads)
    with pytest.raises(ValueError, match='layer order'):
        p.run(model.blocks[1].moe.experts, x, w, ids)
    with pytest.raises(TypeError, match='float32'):
        p.run(model.blocks[0].moe.experts, x.astype(cp.float16), w, ids)
    with pytest.raises(ValueError, match='shape'):
        p.run(model.blocks[0].moe.experts, x, w[:, :2].copy(), ids)
    executor.cache.resident = dict(executor.cache.resident)
    key = next(iter(executor.cache.resident))
    old = executor.cache.resident.pop(key)
    with pytest.raises(RuntimeError, match='snapshot changed'):
        p.run(model.blocks[0].moe.experts, x, w, ids)
    executor.cache.resident[key] = old
    p.close()  # partial chunks must wake the ring-blocked issuer and stagers
    assert not executor.protected and all(not t.is_alive() for t in threads)


@pytest.mark.parametrize('defect', ['bytes', 'format', 'geometry', 'budget'])
def test_invalid_weight_preparation_is_transactional(gpu, defect):
    cp = gpu
    model, executor, _ = model_fixture(cp, resident=())
    first = next(iter(executor.sources.values()))
    options = {}
    if defect == 'bytes':
        first[0].data = first[0].data[:-1]
    elif defect == 'format':
        first[0].quant_type = 'IQ4_NL'
    elif defect == 'geometry':
        model.blocks[0].moe.experts.d_model = 128
    else:
        options['device_ring_bytes'] = 1
    p = NativePrefillPipeline(stream_min=1, **options)
    with pytest.raises(ValueError):
        p.begin_chunk(model, 5, 3)
    assert not p._active and not executor.protected and not p._threads
    p.close()
