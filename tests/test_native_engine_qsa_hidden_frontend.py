"""Bounded hidden->packed matvec->unchanged QSA frontend; no model execution."""
from contextlib import ExitStack
from dataclasses import asdict
import ctypes as C
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mini_llm.native import Runtime, NativeError
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from test_native_engine_qsa import inputs
from test_native_engine_qsa_projection import weight_upload
from test_native_engine_qsa_frontend import arrays, reference, compare, read, BOUNDARIES

pytestmark = pytest.mark.gpu
KINDS = {0: 'F32', 30: 'BF16', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 23: 'IQ4_XS'}
ROWS = [12288, 512, 512, 512, 128]  # public weight order Q,K,V,IQ,IK
PSHAPES = [(24, 512), (2, 256), (2, 256), (128,), (4, 128)]
STEMS = ['attn_q.weight', 'attn_k.weight', 'attn_v.weight',
         'indexer.q_proj.weight', 'indexer.k_proj.weight']
NORMS = ['attn_q_norm.weight', 'attn_k_norm.weight',
         'indexer.q_norm.weight', 'indexer.k_norm.weight']


def data():
    a = arrays()
    h = np.random.default_rng(2026112304).normal(size=2560).astype(np.float32)
    return a, dict(hidden=h, **{k: a[k] for k in
        ['gamma_q', 'gamma_k', 'gamma_iq', 'gamma_ik', 'first_raw', 'partial']})


def call(w, u, weights, n):
    w.from_hidden(u[0], *weights, *u[1:], n)


def projections(w):
    p = [np.empty(s, np.float32) for s in PSHAPES]
    w.hidden_readback_for_test(*p)
    return p


def decode(raw, kind, rows):
    if kind == 0:
        return np.frombuffer(raw, '<f4').reshape(rows, 2560)
    if kind == 30:
        return (np.frombuffer(raw, '<u2').astype(np.uint32) << 16).view(np.float32).reshape(rows, 2560)
    return PackedGGUFWeight(np.frombuffer(raw, np.uint8), (rows, 2560),
                            KINDS[kind]).dequantize_reference()


def synthetic(kind, rows):
    rng = np.random.default_rng(2026112304 + kind + rows)
    if kind in (0, 30):
        a = (rng.normal(size=(rows, 2560)) * .02).astype(np.float32)
        return a.tobytes() if kind == 0 else (a.view(np.uint32) >> 16).astype('<u2').tobytes()
    size = {12: 144, 13: 176, 14: 210, 23: 136}[kind]
    b = rng.integers(0, 256, (rows*10, size), dtype=np.uint8)
    off = 208 if kind == 14 else 0
    # Negative/zero/subnormal scales and arbitrary packed bit planes.
    scale = np.array([0, -0., 2**-24, -2**-14, .0005, -.001, .002], '<f2')
    b[:, off:off+2] = np.resize(scale.view(np.uint8).reshape(-1, 2), (len(b), 2))
    if kind in (12, 13):
        b[:, 2:4] = np.frombuffer(np.float16(.0002).tobytes(), np.uint8)
    return b.tobytes()


def expected(raws, kinds, h):
    result, bounds = [], []
    for raw, kind, rows in zip(raws, kinds, ROWS):
        rb = len(raw)//rows
        y, b = np.empty(rows, np.float32), np.empty(rows, np.float32)
        for start in range(0, rows, 32):
            count = min(32, rows-start)
            tile = decode(raw[start*rb:(start+count)*rb], kind, count)
            y[start:start+count] = tile @ h
            b[start:start+count] = 2e-6*(np.abs(tile) @ np.abs(h)) + 2e-4
        result.append(y)
        bounds.append(b)
    # scratch/readback order Q,K,V,IK,IQ
    return [result[j].reshape(PSHAPES[i]) for i, j in enumerate([0, 1, 2, 4, 3])], [bounds[j].reshape(PSHAPES[i]) for i, j in enumerate([0, 1, 2, 4, 3])]


def check(w, a, want, bounds, n):
    got = projections(w)
    max_projection = max(float(np.abs(g-x).max()) for g, x in zip(got, want))
    for g, x, b in zip(got, want, bounds):
        error = np.abs(g-x)
        assert np.all(error <= b), (float(error.max()), float((error/b).max()))
        np.testing.assert_allclose(g, x, rtol=5e-5, atol=2e-4)
    projected = dict(a, qg=got[0], k=got[1], raw=got[3], iq=got[4])
    # Isolate the unchanged frontend math from matvec reduction error.
    component_error = compare(w, projected, n)
    full = dict(a, qg=want[0], k=want[1], raw=want[3], iq=want[4])
    actual, device_n, ready = read(w)
    assert device_n == n and ready == (n % 4 == 0)
    max_end, max_ratio = 0., 0.
    for j, (g, x) in enumerate(zip(actual, reference(full, n, 1e-6))):
        if x is None:
            continue
        # End-to-end allowance includes projection and RMS reduction differences.
        bound = 5e-5*(1+np.abs(x))
        if j not in (2, 4):
            position = n-4 if j == 5 else n-1
            f = np.power(np.float32(1e7), -np.arange(32, dtype=np.float32)/32)
            pair = np.abs(x[..., :32])+np.abs(x[..., 32:64])
            angle = 8*np.finfo(np.float32).eps*position*f*pair
            bound[..., :32] += angle
            bound[..., 32:64] += angle
        error = np.abs(g-x)
        assert np.all(error <= bound), (j, float(error.max()), float((error/bound).max()))
        max_end = max(max_end, float(error.max()))
        max_ratio = max(max_ratio, float((error/bound).max()))
    return dict(projection_max_abs=max_projection, frontend_component_max_abs=component_error,
                end_to_end_max_abs=max_end, end_to_end_bound_ratio=max_ratio)


@pytest.mark.parametrize('kind', KINDS)
def test_synthetic_formats_boundaries_and_ownership(tmp_path, kind):
    r = Runtime()
    a, source = data()
    raws = [synthetic(kind, rows) for rows in ROWS]
    want, bounds = expected(raws, [kind]*5, source['hidden'])
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=256*2**20))
        u = stack.enter_context(inputs(tmp_path, r, e, source))
        weights = [weight_upload(stack, tmp_path, r, e, raw, kind, rows, 2560)
                   for raw, rows in zip(raws, ROWS)]
        w = stack.enter_context(e.qsa_frontend())
        before = e.stats
        with pytest.raises(NativeError, match='unavailable'):
            projections(w)
        call(w, u, weights, 1)
        assert e.stats.device_bytes-before.device_bytes == 55808
        assert e.stats.pinned_bytes == before.pinned_bytes
        state = bytes(e.stats)
        for n in BOUNDARIES:
            call(w, u, weights, n)
            check(w, a, want, bounds, n)
            assert bytes(e.stats) == state
        for upload, raw in zip(weights, raws):
            copied = bytearray(len(raw))
            upload.readback_for_test(0, len(raw), copied)
            assert copied == raw
        saved = projections(w)
        for tensor in u + weights:
            tensor.close()
        e.close()
        for g, x in zip(projections(w), saved):
            np.testing.assert_array_equal(g, x)
        stale = w._handle
        w.close()
        assert r._lib.ml_qsa_frontend_destroy(stale) == 1


@pytest.mark.parametrize('defect', ['format', 'shape', 'engine', 'stale', 'hidden_nan',
                                  'v_nan', 'n', 'epsilon', 'budget'])
def test_rejection_invalidation_and_accounting(tmp_path, defect):
    r = Runtime()
    a, source = data()
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=128*2**20))
        u = stack.enter_context(inputs(tmp_path, r, e, source))
        # BF16 keeps rejection fixtures small.
        weights = [weight_upload(stack, tmp_path, r, e, synthetic(30, rows), 30, rows, 2560)
                   for rows in ROWS]
        w = stack.enter_context(e.qsa_frontend())
        call(w, u, weights, 4)
        before = bytes(e.stats)
        bad_u, bad_weights, n, epsilon = list(u), list(weights), 4, 1e-6
        if defect in ('format', 'shape', 'engine', 'v_nan'):
            kind, rows, width, engine = 30, 512, 2560, e
            if defect == 'format': kind = 1
            if defect == 'shape': width = 1280
            if defect == 'engine': engine = stack.enter_context(r.engine(budget_bytes=2**23))
            raw = bytes(rows*width*2)
            if defect == 'v_nan':
                raw = bytearray(raw); raw[-2:] = bytes([128, 127])
            bad_weights[2] = weight_upload(stack, tmp_path, r, engine, raw, kind, rows, width)
        elif defect == 'hidden_nan':
            changed = dict(source); changed['hidden'] = source['hidden'].copy()
            changed['hidden'][-1] = np.nan
            bad_u = stack.enter_context(inputs(tmp_path, r, e, changed))
        elif defect == 'stale':
            bad_weights[0].close()
        elif defect == 'n': n = 0
        elif defect == 'epsilon': epsilon = 0
        elif defect == 'budget':
            # Admission is atomic and must include the lazy scratch.
            used = e.stats.device_bytes + e.stats.pinned_bytes - 55808
            tight = stack.enter_context(r.engine(budget_bytes=used+55807))
            bad_u = stack.enter_context(inputs(tmp_path, r, tight, source))
            bad_weights = [weight_upload(stack, tmp_path, r, tight, synthetic(30, rows), 30, rows, 2560)
                           for rows in ROWS]
            w = stack.enter_context(tight.qsa_frontend())
            before = bytes(tight.stats)
        message = {'format': 'unsupported', 'shape': 'shapes', 'engine': 'cross-engine',
                   'stale': 'closed', 'hidden_nan': 'nonfinite', 'v_nan': 'nonfinite',
                   'n': 'n/epsilon', 'epsilon': 'n/epsilon', 'budget': 'budget'}[defect]
        with pytest.raises(NativeError, match=message):
            w.from_hidden(bad_u[0], *bad_weights, *bad_u[1:], n, epsilon=epsilon)
        # Python closed-handle rejection does not enter native code.
        if defect != 'stale':
            with pytest.raises(NativeError, match='unavailable'):
                projections(w)
        if defect == 'budget': assert bytes(tight.stats) == before
        if defect in ('n', 'epsilon'): assert bytes(e.stats) == before


def test_hidden_snapshot_reset_and_readback_contract(tmp_path):
    r = Runtime()
    _, source = data()
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=128*2**20))
        u = stack.enter_context(inputs(tmp_path, r, e, source))
        weights = [weight_upload(stack, tmp_path, r, e, synthetic(30, rows), 30, rows, 2560)
                   for rows in ROWS]
        w = stack.enter_context(e.qsa_frontend())
        call(w, u, weights, 4)
        p = projections(w)
        with pytest.raises(ValueError, match='F32'):
            w.hidden_readback_for_test(p[0].astype(np.float64), *p[1:])
        short = np.full(1, -99, np.float32)
        with pytest.raises(NativeError, match='capacity'):
            w.hidden_readback_for_test(short, *p[1:])
        assert short[0] == -99
        state = bytes(e.stats)
        for action in (lambda: w.set_n(3), w.reset):
            action()
            with pytest.raises(NativeError, match='unavailable'): projections(w)
            call(w, u, weights, 4)
            assert bytes(e.stats) == state
        a = arrays()
        with inputs(tmp_path, r, e, a) as projected:
            w.run(*projected)
            with pytest.raises(NativeError, match='unavailable'): projections(w)
        stale = weights[0]._handle
        weights[0].close()
        handles = [u[0]._handle, stale] + [x._handle for x in weights[1:]] + [x._handle for x in u[1:]]
        assert r._lib.ml_qsa_frontend_from_hidden(w._handle, *handles, 4, 1e-6) == 1
        with pytest.raises(NativeError, match='unavailable'): read(w)


@pytest.mark.skipif(os.environ.get('MINI_LLM_QSA_HIDDEN_CHECKPOINT') not in ('1', 'all'),
                    reason='optional bounded actual checkpoint hidden frontend')
def test_actual_checkpoint(tmp_path):
    inventory = json.loads(Path('docs/native_engine/WP2_GGUF_INVENTORY.json').read_text())
    r = Runtime()
    results = []
    layers = range(3, 48, 4) if os.environ['MINI_LLM_QSA_HIDDEN_CHECKPOINT'] == 'all' else [3]
    with r.gguf(inventory['shards'][0]['path']) as store:
        # Always verify all 60 actual projection descriptors, even in one-layer mode.
        descriptors = [asdict(store.tensors[f'blk.{layer}.{stem}'])
                       for layer in range(3, 48, 4) for stem in STEMS]
        for d in descriptors:
            assert d['ggml_shape'][0] == 2560 and d['type_id'] in KINDS
        for layer in layers:
            a, source = data()
            prefix = f'blk.{layer}.'
            with ExitStack() as stack:
                e = stack.enter_context(r.engine(budget_bytes=128*2**20))
                for key, stem in zip(['gamma_q', 'gamma_k', 'gamma_iq', 'gamma_ik'], NORMS):
                    d = store.tensors[prefix+stem]
                    assert d.type_id == 0
                    with store.lease(prefix+stem) as lease:
                        gamma = np.frombuffer(lease.read(0, d.nbytes), '<f4').copy()
                    source[key] = a[key] = gamma
                u = stack.enter_context(inputs(tmp_path, r, e, source))
                weights, raws, kinds = [], [], []
                for stem in STEMS:
                    name = prefix+stem
                    d = store.tensors[name]
                    lease = stack.enter_context(store.lease(name))
                    weights.append(stack.enter_context(e.upload(lease)))
                    raws.append(lease.read(0, d.nbytes))
                    kinds.append(d.type_id)
                want, bounds = expected(raws, kinds, source['hidden'])
                w = stack.enter_context(e.qsa_frontend())
                checks = []
                for n in BOUNDARIES:
                    call(w, u, weights, n)
                    checks.append(dict(n=n, **check(w, a, want, bounds, n)))
                state = bytes(e.stats)
                call(w, u, weights, 4)
                assert bytes(e.stats) == state
                assert e.stats.peak_bytes < 512*2**20
                results.append(dict(layer=layer, kinds=kinds, checks=checks,
                    device_bytes=e.stats.device_bytes, pinned_bytes=e.stats.pinned_bytes,
                    peak_bytes=e.stats.peak_bytes))
    print('\nWP3_HIDDEN_RESULTS='+json.dumps(dict(descriptors=descriptors, results=results)))
