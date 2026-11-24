"""WP3 selected attention: explicit NumPy graph, lifecycle and CUDA timings.

Run with MINI_LLM_BACKEND=numpy and the built native library. No checkpoint.
"""
import ctypes as C
import json
import os
from contextlib import ExitStack

import numpy as np
import pytest

from mini_llm.native import Runtime, NativeError
from test_native_engine_qsa import (BOUNDARIES, arrays, inputs, reference,
                                    selected, read)

pytestmark = pytest.mark.gpu


def fixture(capacity, mode='controlled'):
    a = arrays(capacity, mode)
    rng = np.random.default_rng(363386)
    a.update(qmain=rng.normal(size=(24, 256)).astype(np.float32),
             gate=rng.normal(size=(24, 256)).astype(np.float32),
             k=rng.normal(size=(2, 256)).astype(np.float32),
             v=rng.normal(size=(2, 256)).astype(np.float32),
             keys=rng.normal(size=(capacity, 2, 256)).astype(np.float32),
             values=rng.normal(size=(capacity, 2, 256)).astype(np.float32))
    a['gate'].flat[:9] = [-1000, -100, -20, -1, 0, 1, 20, 100, 1000]
    return a


def numpy_graph(a, ids, n, scale=1/16):
    # The reference step363-386 graph, including the uncommitted current token.
    keys, values = a['keys'][ids].copy(), a['values'][ids].copy()
    keys[ids == n-1], values[ids == n-1] = a['k'], a['v']
    contexts = np.empty((24, 256), np.float32)
    for h in range(24):
        logits = (keys[:, h//12] @ a['qmain'][h]) * np.float32(scale)
        probability = np.exp(logits - np.max(logits))
        probability /= np.sum(probability, dtype=np.float32)
        contexts[h] = probability @ values[:, h//12]
    z = np.exp(-np.abs(a['gate']))
    sigmoid = np.where(a['gate'] >= 0, 1/(1+z), z/(1+z))
    return contexts, contexts*sigmoid


def outputs(attention):
    c, g = np.empty((24, 256), np.float32), np.empty((24, 256), np.float32)
    attention.readback_for_test(c, g)
    return c, g


def compare(attention, a, ids, n, scale=1/16):
    actual, expected = outputs(attention), numpy_graph(a, ids, n, scale)
    for got, want in zip(actual, expected):
        # BLAS reduction and online tile rescaling differ from the reference.
        # Absolute tolerance protects near-zero contexts, not selector membership.
        np.testing.assert_allclose(got, want, rtol=4e-5, atol=3e-6)
    return max(float(np.max(np.abs(got-want))) for got, want in zip(actual, expected))


@pytest.mark.parametrize('n', BOUNDARIES)
@pytest.mark.parametrize('policy', ['ggml_cpu', 'strata'])
@pytest.mark.parametrize('mode', ['controlled', 'flat'])
def test_selected_attention_boundaries(tmp_path, n, policy, mode, monkeypatch):
    monkeypatch.setenv('MINI_LLM_QSA_NATIVE_SELECT', '0')
    r, capacity = Runtime(), (n+3)//4*4
    a = fixture(capacity, mode)
    # The current slot and future capacity are uncommitted and must not be read.
    a['keys'][n-1:] = np.nan
    a['values'][n-1:] = np.nan
    with r.engine(budget_bytes=2**28) as e:
        with inputs(tmp_path, r, e, a) as u, e.qsa_selection(capacity) as s, s.attention() as att:
            s.set_n(n)
            before = bytes(e.stats)
            assert att.run(*u[3:], selection_inputs=u[:3], policy=policy) > 0
            ids = selected(reference(a, n), policy)
            np.testing.assert_array_equal(read(s, n)[1], ids)
            compare(att, a, ids, n)
            assert att.run(*u[3:], repeats=2) > 0
            compare(att, a, ids, n)
            assert bytes(e.stats) == before
            # Exact upload bytes are immutable, including history/current inputs.
            for tensor, source in zip(u[3:], list(a.values())[3:]):
                copied = bytearray(source.nbytes)
                assert tensor.readback_for_test(0, source.nbytes, copied) == source.nbytes
                assert copied == source.tobytes()


@pytest.mark.parametrize('policy', ['ggml_cpu', 'strata'])
@pytest.mark.parametrize('has_current', [False, True])
def test_current_membership_and_large_logits(tmp_path, policy, has_current):
    r, n, capacity = Runtime(), 4096, 4096
    a = fixture(capacity, 'flat')
    a['q'].fill(1)
    a['p'][:700].fill(1)
    a['p'][-1].fill(100 if has_current else -100)
    # Exactly representable large positive logits: naive exp(logits) overflows.
    a['qmain'].fill(0)
    a['qmain'][:, 0] = 16
    a['keys'][:, :, 0] = 10000 + np.arange(capacity)[:, None] % 5
    a['k'][:, 0] = 10006
    a['v'].fill(37)
    a['values'][n-1].fill(-99)
    ids = selected(reference(a, n), policy)
    assert bool(n-1 in ids) == has_current
    with r.engine(budget_bytes=2**28) as e:
        with inputs(tmp_path, r, e, a) as u, e.qsa_selection(capacity) as s, s.attention() as att:
            s.set_n(n)
            att.run(*u[3:], selection_inputs=u[:3], policy=policy)
            compare(att, a, ids, n)
            assert np.isfinite(outputs(att)).all()


def test_lifecycle_budget_and_dynamic_n(tmp_path):
    r, capacity = Runtime(), 8
    a = fixture(capacity)
    with r.engine(budget_bytes=2**24) as e:
        with inputs(tmp_path, r, e, a) as u:
            s = e.qsa_selection(capacity)
            before = e.stats
            att = s.attention()
            current = e.stats
            assert current.device_bytes-before.device_bytes == 49156
            assert current.pinned_bytes-before.pinned_bytes == 8
            with pytest.raises(NativeError, match='unavailable'):
                outputs(att)
            for n in [1, 5, 4, 8, 2]:
                s.set_n(n)
                with pytest.raises(NativeError, match='selection'):
                    att.run(*u[3:])
                with pytest.raises(NativeError, match='unavailable'):
                    outputs(att)
                att.run(*u[3:], selection_inputs=u[:3], repeats=2)
                compare(att, a, np.arange(n), n)
                assert bytes(e.stats) == bytes(current)
                s.reset()
                # Outputs are snapshots; resetting selection does not erase them.
                compare(att, a, np.arange(n), n)
            s.set_n(3)
            att.run(*u[3:], selection_inputs=u[:3])
            s.close()
            for tensor in u:
                tensor.close()
            e.close()
            compare(att, a, np.arange(3), 3)
            stale = att._handle
            att.close()
            assert r._lib.ml_qsa_attention_destroy(stale) == 1
    selection_bytes = 24+8*8+4*8
    for spare in [49163, 49164]:
        with r.engine(budget_bytes=selection_bytes+spare) as e, e.qsa_selection(8) as s:
            before = bytes(e.stats)
            if spare == 49163:
                with pytest.raises(NativeError, match='budget'):
                    s.attention()
                assert bytes(e.stats) == before
            else:
                with s.attention():
                    assert e.stats.device_bytes+e.stats.pinned_bytes == selection_bytes+spare
                    with pytest.raises(NativeError, match='budget'):
                        e.session()
                assert e.stats.device_bytes+e.stats.pinned_bytes == selection_bytes
            assert e.stats.sessions == 0
    assert r._lib.ml_qsa_attention_destroy(0) == 0


@pytest.mark.parametrize('name', ['qmain', 'gate', 'k', 'v', 'keys', 'values'])
@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_nonfinite_sources(tmp_path, name, value):
    r, a = Runtime(), fixture(8)
    a[name].flat[0] = value
    with r.engine(budget_bytes=2**24) as e:
        with inputs(tmp_path, r, e, a) as u, e.qsa_selection(8) as s, s.attention() as att:
            s.set_n(5)
            before = bytes(e.stats)
            with pytest.raises(NativeError, match='nonfinite'):
                att.run(*u[3:], selection_inputs=u[:3])
            with pytest.raises(NativeError, match='unavailable'):
                outputs(att)
            assert bytes(e.stats) == before


@pytest.mark.parametrize('failure', ['dot_overflow', 'numerator_overflow', 'selection'])
def test_nonfinite_arithmetic(tmp_path, failure):
    r, a = Runtime(), fixture(8)
    if failure == 'dot_overflow':
        a['qmain'].fill(np.finfo(np.float32).max)
        a['k'].fill(2)
    elif failure == 'numerator_overflow':
        a['qmain'].fill(0)
        a['v'].fill(np.finfo(np.float32).max)
        a['values'].fill(np.finfo(np.float32).max)
    else:
        a['q'].fill(np.nan)
    with r.engine(budget_bytes=2**24) as e:
        with inputs(tmp_path, r, e, a) as u, e.qsa_selection(8) as s, s.attention() as att:
            s.set_n(5)
            with pytest.raises(NativeError, match='nonfinite'):
                att.run(*u[3:], selection_inputs=u[:3])
            with pytest.raises(NativeError, match='unavailable'):
                outputs(att)


def test_invalid_contracts(tmp_path):
    r, a = Runtime(), fixture(8)
    a.update(wrong=np.zeros((256, 24), np.float32), half=np.zeros((24, 256), np.float16))
    with r.engine(budget_bytes=2**24) as e, r.engine(budget_bytes=2**24) as other:
        with inputs(tmp_path, r, e, a) as u, e.qsa_selection(8) as s, s.attention() as att:
            s.set_n(5)
            args = u[3:9]
            att.run(*args, selection_inputs=u[:3])
            for scale in [0, -1, np.nan, np.inf]:
                with pytest.raises(NativeError):
                    att.run(*args, scale=scale)
            for repeats in [0, -1, True, 10001]:
                with pytest.raises((NativeError, ValueError)):
                    att.run(*args, repeats=repeats)
            for j in range(6):
                wrong = args.copy()
                wrong[j] = u[9]
                with pytest.raises(NativeError, match='shapes'):
                    att.run(*wrong)
            with pytest.raises(NativeError, match='shapes'):
                att.run(u[10], *args[1:])
            with other.qsa_selection(8) as cross, cross.attention() as ca:
                cross.set_n(5)
                with pytest.raises(NativeError, match='cross-engine'):
                    ca.run(*args, selection_inputs=u[:3])
            with pytest.raises(ValueError):
                att.run(None, *args[1:])
            with pytest.raises(ValueError):
                att.run(*args, policy='unknown')
            with pytest.raises(ValueError):
                att.run(*args, selection_inputs=u[:2])
            att.run(*args, selection_inputs=u[:3], scale=0.125)
            compare(att, a, np.arange(5), 5, 0.125)
            small = np.full(6143, 123, np.float32)
            full = np.full(6144, 123, np.float32)
            with pytest.raises(NativeError, match='capacity'):
                att.readback_for_test(small, full)
            assert np.all(small == 123) and np.all(full == 123)
            for buffer in [np.empty(6144, np.float64), full[::2], full.tobytes()]:
                with pytest.raises(ValueError):
                    att.readback_for_test(buffer, full)
            lib, out = r._lib, C.c_uint64(123)
            assert lib.ml_qsa_attention_create(0, C.byref(out)) == 1 and out.value == 0
            assert lib.ml_qsa_attention_create(s._handle, None) == 1
            assert lib.ml_qsa_attention_readback_for_test(att._handle, None, 6144, None, 6144) == 1
            handles = [x._handle for x in args]+[0, 0, 0]
            elapsed = C.c_float(123)
            for bad in [0, e._handle, s._handle, args[0]._handle]:
                assert lib.ml_qsa_attention_run(bad, *handles, 0, 1/16, 1, C.byref(elapsed)) == 1
                assert elapsed.value == 0
            assert lib.ml_qsa_attention_run(att._handle, *handles, 0, 1/16, 1, None) == 1
            assert lib.ml_qsa_attention_run(att._handle, *handles, 2, 1/16, 1, C.byref(elapsed)) == 1
            stale = args[0]._handle
            args[0].close()
            handles[0] = stale
            assert lib.ml_qsa_attention_run(att._handle, *handles, 0, 1/16, 1, C.byref(elapsed)) == 1
            with pytest.raises(NativeError, match='closed'):
                att.run(*args)


@pytest.mark.skipif(os.environ.get('MINI_LLM_QSA_ATTENTION_BENCH') != '1', reason='opt-in CUDA timings')
def test_cuda_timings(tmp_path):
    r, a = Runtime(), fixture(16384)
    result = dict(capacity=16384, repeats=10, samples=3, timings=[])
    with r.engine(budget_bytes=2**28) as e:
        with inputs(tmp_path, r, e, a) as u, e.qsa_selection(16384) as s, s.attention() as att:
            before = bytes(e.stats)
            for n in [1, 2051, 2053, 4096, 8192, 16384]:
                s.set_n(n)
                for policy in ['ggml_cpu', 'strata']:
                    att.run(*u[3:], selection_inputs=u[:3], policy=policy)
                    for combined in [False, True]:
                        times = [att.run(*u[3:], selection_inputs=u[:3] if combined else None,
                                         policy=policy, repeats=10) for _ in range(3)]
                        result['timings'].append(dict(n=n, policy=policy, combined=combined,
                                                      samples_ms=times, median_ms=float(np.median(times))))
            assert bytes(e.stats) == before
            result['device_bytes'], result['pinned_bytes'] = e.stats.device_bytes, e.stats.pinned_bytes
    print('\nQSA_ATTENTION_CUDA_TIMINGS=' + json.dumps(result, sort_keys=True))
