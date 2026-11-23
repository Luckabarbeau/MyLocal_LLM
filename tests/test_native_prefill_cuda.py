"""Pinned physical-kernel port: CPU collection and opt-in GPU0 scalar oracles.

No model runs. Oracle decoding/arithmetic is independently defined in the
existing native-prefill tests, not imported from any Strata runtime.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from mini_llm.ops import native_prefill_cuda as native
from test_triton_native_prefill import decode_numpy, make_weights, oracle, exact

UUID = 'GPU-bea264b4-9e76-3be5-5883-7bde3fe9f82d'


@pytest.mark.parametrize('n,e,ww', [(0, 5, 4), (280, 5, 4), (281, 5, 2),
                                    (560, 5, 2), (561, 5, 4), (354, 5, 2)])
def test_source_tile_selection(n, e, ww):
    assert native.pick_ww(n, e) == ww
    assert native.shared_bytes('IQ2_S', ww) == (78080 if ww == 4 else 74240)


def test_import_and_owned_source_contract():
    source = Path(native.__file__).with_suffix('.cu').read_text()
    assert native.REVISION in source
    assert 'Copyright (c) 2026 Niko1221 and the Strata contributors' in source
    for instruction in ('cp.async.cg.shared.global', 'cp.async.wait_group',
                        'ldmatrix.sync.aligned.m8n8.x4', 'mma.sync.aligned.m16n8k32',
                        'mma.sync.aligned.m16n8k16'):
        assert instruction in source
    assert 'constexpr int THREADS = 512;' in source
    assert 'constexpr int ASTAGES = 4;' in source
    assert 'w += gridDim.x' in source
    assert '#include "strata/' not in source
    assert '#include "ggml' not in source
    assert native.group_bytes(0, 1) == 32+8
    with pytest.raises(ValueError):
        native.pick_ww(1, 0)


@pytest.fixture(scope='module')
def gpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != UUID:
        pytest.skip('GPU tests require CUDA_VISIBLE_DEVICES='+UUID)
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() != 1:
            pytest.skip('exactly the selected GPU0 required')
        cp.cuda.Device(0).use()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))
    return cp


@pytest.mark.parametrize('kind', native.FORMATS)
def test_production_decode(gpu, kind):
    cp = gpu
    raw = make_weights(kind, 5, 512, 109)
    c, s = native.decode_weights(cp.asarray(raw), kind, 5, 512)
    ec, es = decode_numpy(raw, kind, 5, 512)
    exact(cp, c, cp.asarray(ec))
    exact(cp, s, cp.asarray(es))


def test_quantization(gpu):
    cp = gpu
    x = np.random.default_rng(762).normal(0, 4, (7, 512)).astype(np.float32)
    x[0] = 0
    x[1, :32] = np.resize(np.array([127, -127, .5, 1.5, 2.5, -.5, -1.5, -2.5], np.float32), 32)
    x[2, :32] = np.linspace(-100, 100, 32, dtype=np.float32)
    x[3, :32] *= np.float32(1e-30)
    x = cp.asarray(x)
    packed = cp.empty((7, 8, 80), cp.uint8)
    assert native.quantize(x, out=packed) is packed
    q, s, p = cp.empty((7, 512), cp.int8), cp.empty((7, 16), cp.float32), cp.empty_like(packed)
    oracle(cp, 'quant')((7, 16), (1,), (x, x, q, s, p, np.int32(512), np.int32(0)))
    exact(cp, packed, p)


@pytest.mark.parametrize('counts', [[0, 1, 7, 65, 129], [0, 1, 7, 129, 217]])
def test_grouping(gpu, counts):
    cp = gpu
    ids = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    np.random.default_rng(32).shuffle(ids)
    ids = cp.asarray(ids.reshape(-1, 2))
    tb = native.allocate_group(ids.size, len(counts), 2)
    assert native.group(ids, len(counts), out=tb) is tb
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int32)
    np.testing.assert_array_equal(cp.asnumpy(tb.counts), counts)
    np.testing.assert_array_equal(cp.asnumpy(tb.offsets), offsets)
    slot, src = cp.asnumpy(tb.slot), cp.asnumpy(tb.src)
    flat = cp.asnumpy(ids).reshape(-1)
    np.testing.assert_array_equal(np.sort(slot), np.arange(ids.size))
    np.testing.assert_array_equal(src[slot], np.arange(ids.size)//2)
    for e in range(len(counts)):
        assert np.all(flat[np.argsort(slot)[offsets[e]:offsets[e+1]]] == e)
    ts = np.r_[0, np.cumsum((np.array(counts)+63)//64)]
    np.testing.assert_array_equal(cp.asnumpy(tb.tile_offsets), ts)
    expected_tiles = [(e, int(offsets[e])+r) for e, c in enumerate(counts) for r in range(0, c, 64)]
    np.testing.assert_array_equal(cp.asnumpy(tb.tiles[:ts[-1]]), expected_tiles)


@pytest.mark.parametrize('gu_kind', native.GU_FORMATS)
@pytest.mark.parametrize('down_kind', native.DOWN_FORMATS)
@pytest.mark.parametrize('counts', [[0, 1, 7, 65, 129], [0, 1, 7, 129, 217]])
@pytest.mark.parametrize('geometry', [(256, 128), (768, 384)])
def test_fused_intermediates_and_waves(gpu, gu_kind, down_kind, counts, geometry):
    cp = gpu
    embd, ff = geometry
    E = len(counts)
    n = sum(counts)
    rng = np.random.default_rng(65)
    ids = np.repeat(np.arange(E, dtype=np.int32), counts)
    rng.shuffle(ids)
    ids = cp.asarray(ids.reshape(-1, 2))
    tb = native.group(ids, E)
    x = cp.asarray(rng.normal(0, .7, (n//2, embd)).astype(np.float32))
    xa = native.quantize(x)
    # Separate allocations and an intentionally 2-byte-offset weight base verify
    # pointer-table addressing and GGUF's only-2-byte alignment requirement.
    weights, raw_weights = [], []
    for e in range(E):
        raw = [make_weights(gu_kind, ff, embd, 100+3*e),
               make_weights(gu_kind, ff, embd, 101+3*e),
               make_weights(down_kind, embd, ff, 102+3*e)]
        raw_weights.append(raw)
        weights.append([cp.asarray(np.r_[np.zeros(2, np.uint8), w])[2:] for w in raw])
    ptrs = cp.asarray([[w.data.ptr for w in trio] for trio in weights], dtype=cp.int64)
    H = cp.full((n, ff//64, 80), 0xA5, cp.uint8)
    Dm = cp.full((n, embd), np.nan, cp.float32)
    stream = cp.cuda.Stream(non_blocking=True)
    stream.wait_event(cp.cuda.get_current_stream().record())
    with stream:
        for e0, e1 in ((0, 2), (2, E)):
            h, d = native.execute_batch(e0, e1, n, E, ptrs, xa, tb,
                embd=embd, ff=ff, gu_kind=gu_kind, down_kind=down_kind, H=H, Dm=Dm)
            assert h is H and d is Dm
    cp.cuda.get_current_stream().wait_event(stream.record())
    src = cp.asnumpy(tb.src)
    offsets = np.r_[0, np.cumsum(counts)]
    aq = xa[:, :, :64].copy().view(cp.int8).reshape(n//2, embd)
    ax = xa[:, :, 64:72].copy().view(cp.float32).reshape(n//2, embd//32)
    G, U = cp.empty((n, ff), cp.float32), cp.empty((n, ff), cp.float32)
    for e, c in enumerate(counts):
        if not c:
            continue
        start = offsets[e]
        rows = cp.asarray(src[start:start+c])
        for j, output in ((0, G), (1, U)):
            wc, ws = decode_numpy(raw_weights[e][j], gu_kind, ff, embd)
            oracle(cp, 'linear')((c, ff), (1,), (aq[rows], ax[rows], cp.asarray(wc), cp.asarray(ws),
                output[start:start+c], np.int32(c), np.int32(ff), np.int32(embd), np.int32(gu_kind == 'IQ2_S')))
    hq, hs = cp.empty((n, ff), cp.int8), cp.empty((n, ff//32), cp.float32)
    expected_h = cp.empty_like(H)
    oracle(cp, 'quant')((n, ff//32), (1,), (G, U, hq, hs, expected_h, np.int32(ff), np.int32(1)))
    exact(cp, H[:, :, :72], expected_h[:, :, :72])
    # Source leaves H's code-sum padding alone.
    np.testing.assert_array_equal(cp.asnumpy(H[:, :, 72:]), 0xA5)
    expected_d = cp.empty_like(Dm)
    for e, c in enumerate(counts):
        if not c:
            continue
        start = offsets[e]
        wc, ws = decode_numpy(raw_weights[e][2], down_kind, embd, ff)
        oracle(cp, 'linear')((c, embd), (1,), (hq[start:start+c], hs[start:start+c],
            cp.asarray(wc), cp.asarray(ws), expected_d[start:start+c],
            np.int32(c), np.int32(embd), np.int32(ff), np.int32(0)))
    exact(cp, Dm, expected_d)


def test_source_geometry_pipeline(gpu):
    # Source geometry: 40 K stages in GU, 10 in down. Synthetic, not a model.
    test_fused_intermediates_and_waves(gpu, 'IQ2_S', 'IQ4_NL', [0, 2], (2560, 640))


def test_empty_and_contracts(gpu):
    cp = gpu
    tb = native.group(cp.empty((0, 2), cp.int32), 5)
    np.testing.assert_array_equal(cp.asnumpy(tb.offsets), 0)
    q = native.quantize(cp.empty((0, 256), cp.float32))
    H, Dm = native.execute_batch(0, 5, 0, 5, cp.zeros((5, 3), cp.int64), q, tb,
        embd=256, ff=128, gu_kind='IQ2_S', down_kind='IQ4_NL')
    assert H.shape == (0, 2, 80) and Dm.shape == (0, 256)
    with pytest.raises(TypeError, match='float32'):
        native.quantize(cp.empty((1, 64), cp.float16))
    with pytest.raises(ValueError, match='64'):
        native.quantize(cp.empty((1, 32), cp.float32))
    with pytest.raises(ValueError, match='agree'):
        native.group(cp.zeros((2, 2), cp.int32), 5, top_k=3)


def test_preallocated_graph_replay(gpu, monkeypatch):
    cp = gpu
    ids = cp.asarray([[0, 1], [1, 0]], dtype=cp.int32)
    x = cp.asarray(np.random.default_rng(7).normal(size=(2, 256)).astype(np.float32))
    tb = native.allocate_group(4, 2, 2)
    xa = cp.empty((2, 4, 80), cp.uint8)
    H = cp.empty((4, 2, 80), cp.uint8)
    Dm = cp.empty((4, 256), cp.float32)
    weights = [[cp.asarray(make_weights(kind, n, k, 20+3*e+j))
                for j, (kind, n, k) in enumerate((('IQ3_S', 128, 256),
                    ('IQ3_S', 128, 256), ('Q2_0', 256, 128)))] for e in range(2)]
    ptrs = cp.asarray([[w.data.ptr for w in trio] for trio in weights], dtype=cp.int64)

    def launch():
        native.quantize(x, out=xa)
        native.group(ids, 2, out=tb)
        native.execute_batch(0, 2, 4, 2, ptrs, xa, tb, embd=256, ff=128,
                             gu_kind='IQ3_S', down_kind='Q2_0', H=H, Dm=Dm)

    launch()  # Warm compilation, globals and all lazy CuPy kernels before capture.
    expected = Dm[tb.slot].copy()
    stream = cp.cuda.Stream(non_blocking=True)
    stream.wait_event(cp.cuda.get_current_stream().record())

    def no_readback(*args, **kwargs):
        raise AssertionError('host readback in native pipeline')

    with monkeypatch.context() as patch:
        patch.setattr(cp, 'asnumpy', no_readback)
        with stream:
            stream.begin_capture()
            launch()
            graph = stream.end_capture()
            Dm.fill(np.nan)
            graph.launch(stream)
    cp.cuda.get_current_stream().wait_event(stream.record())
    exact(cp, Dm[tb.slot], expected)
