"""WP3 GPU-only scoring/selection, using exact uploaded synthetic F32 GGUFs."""
from contextlib import ExitStack, contextmanager
import ctypes as C

import numpy as np
import pytest

from mini_llm.native import Runtime, NativeError
from mini_llm.backend import xp, asnumpy
from mini_llm.external.qwen4exp_attention import select_cells
from test_0201_gguf_store import write_gguf

pytestmark = pytest.mark.gpu
BOUNDARIES = [1, 2, 3, 4, 5, 127, 1024, 2048, 2051, 2052, 2053, 4096, 8192, 16384]


@contextmanager
def inputs(tmp_path, runtime, engine, arrays):
    tensors, offset = [], 0
    for name, array in arrays.items():
        tensors.append((name, array.shape[::-1], 0 if array.dtype == np.float32 else 1,
                        offset, array.tobytes()))
        offset += (array.nbytes + 31) // 32 * 32
    path = tmp_path / "qsa.gguf"
    write_gguf(path, tensors)
    with ExitStack() as stack:
        store = stack.enter_context(runtime.gguf(path))
        result = []
        for name in arrays:
            lease = stack.enter_context(store.lease(name))
            result.append(stack.enter_context(engine.upload(lease)))
        yield result


def arrays(capacity, mode):
    rng = np.random.default_rng(20261123)
    if mode == "flat":
        return dict(q=np.zeros((4, 128), np.float32),
                    p=np.zeros((capacity//4, 128), np.float32),
                    d=np.zeros(128, np.float32))
    if mode == "controlled":
        # Small integer products and sums are exactly representable in F32.
        # Random block scores, including genuine repeat/tie groups.
        return dict(q=rng.integers(-3, 4, (4, 128)).astype(np.float32),
                    p=rng.integers(-3, 4, (capacity//4, 128)).astype(np.float32),
                    d=rng.integers(-3, 4, 128).astype(np.float32))
    return dict(q=rng.normal(size=(4, 128)).astype(np.float32),
                p=rng.normal(size=(capacity//4, 128)).astype(np.float32),
                d=rng.normal(size=128).astype(np.float32))


def reference(a, n):
    keys = a['p'][:n//4]
    if n % 4:
        keys = np.concatenate((keys, a['d'][None]))
    dots = np.maximum(a['q'] @ keys.T, np.float32(0))
    block = dots[0].copy()
    for h in range(1, 4):
        block += dots[h]
    scores = np.repeat(block, 4)[:n]
    scores[n//4*4:] += np.float32(1e9)
    return scores


def selected(scores, policy):
    return asnumpy(select_cells(xp.asarray(scores), min(len(scores), 2051), policy))


def read(workspace, n):
    scores = np.empty(n, np.float32)
    ids = np.empty(min(n, 2051), np.int64)
    assert workspace.readback_for_test(scores, ids) == n
    return scores, ids


@pytest.mark.parametrize('n', BOUNDARIES)
@pytest.mark.parametrize('policy', ['ggml_cpu', 'strata'])
@pytest.mark.parametrize('mode', ['flat', 'controlled'])
def test_boundaries_and_exact_membership(tmp_path, n, policy, mode, monkeypatch):
    monkeypatch.setenv('MINI_LLM_QSA_NATIVE_SELECT', '0')
    runtime = Runtime()
    capacity = (n+3)//4*4
    a = arrays(capacity, mode)
    with runtime.engine(budget_bytes=2**24) as engine:
        with inputs(tmp_path, runtime, engine, a) as uploaded, engine.qsa_selection(capacity) as w:
            w.set_n(n)
            assert w.run(*uploaded, policy=policy) > 0
            scores, ids = read(w, n)
            expected = reference(a, n)
            np.testing.assert_array_equal(scores, expected)
            np.testing.assert_array_equal(ids, selected(expected, policy))
            assert np.all(ids[1:] > ids[:-1])
            assert np.all(np.isin(np.arange(n//4*4, n), ids))


@pytest.mark.parametrize('policy', ['ggml_cpu', 'strata'])
def test_random_scores_reduction_order(tmp_path, policy, monkeypatch):
    monkeypatch.setenv('MINI_LLM_QSA_NATIVE_SELECT', '0')
    runtime, capacity, n = Runtime(), 16384, 16383
    a = arrays(capacity, 'normal')
    with runtime.engine(budget_bytes=2**24) as engine:
        with inputs(tmp_path, runtime, engine, a) as uploaded, engine.qsa_selection(capacity) as w:
            w.set_n(n)
            w.run(*uploaded, policy=policy)
            scores, ids = read(w, n)
            expected = reference(a, n)
            # Different dot reduction order: gamma_128 * sum(abs(q*k)) bound.
            keys = np.concatenate((a['p'][:n//4], a['d'][None]))
            gamma = 128*np.finfo(np.float32).eps/(1-128*np.finfo(np.float32).eps)
            bound = 2*gamma*np.sum(np.abs(a['q']) @ np.abs(keys).T, axis=0)
            bound = np.repeat(bound, 4)[:n] + 1e-5
            assert np.all(np.abs(scores[:-3]-expected[:-3]) <= bound[:-3])
            np.testing.assert_allclose(scores, expected, rtol=3e-5, atol=3e-5)
            # Exact selector against its actual FP32 scores, plus reference scores:
            # the fixed random fixture's cutoff gap must dominate arithmetic error.
            np.testing.assert_array_equal(ids, selected(scores, policy))
            np.testing.assert_array_equal(ids, selected(expected, policy))


def test_update_reset_budget_ownership(tmp_path):
    runtime, capacity = Runtime(), 8192
    a = arrays(capacity, 'controlled')
    device_bytes = 8 + 8*2051 + 4*capacity + 4
    with runtime.engine(budget_bytes=2**24) as engine:
        with inputs(tmp_path, runtime, engine, a) as uploaded:
            before = engine.stats
            w = engine.qsa_selection(capacity)
            current = engine.stats
            assert current.device_bytes-before.device_bytes == device_bytes
            assert current.pinned_bytes-before.pinned_bytes == 12
            for n in [5, 4, 2053, 1, 8192]:
                w.set_n(n)
                with pytest.raises(NativeError, match='unavailable'):
                    read(w, n)
                w.run(*uploaded, policy='ggml_cpu', repeats=2)
                np.testing.assert_array_equal(read(w, n)[0], reference(a, n))
                assert bytes(engine.stats) == bytes(current)
                w.reset()
                with pytest.raises(NativeError, match='invalid QSA'):
                    w.run(*uploaded, policy='strata')
                assert bytes(engine.stats) == bytes(current)
            # Public input/engine handles may close after a successful run.
            w.set_n(4)
            w.run(*uploaded, policy='strata')
            for u in uploaded:
                u.close()
            engine.close()
            np.testing.assert_array_equal(read(w, 4)[1], np.arange(4))
            stale = w._handle
            w.close()
            assert runtime._lib.ml_qsa_reset(stale) == 1
            assert runtime._lib.ml_qsa_destroy(stale) == 1
    for budget in [device_bytes+11, device_bytes+12]:
        with runtime.engine(budget_bytes=budget) as engine:
            if budget == device_bytes+11:
                with pytest.raises(NativeError, match='budget'):
                    engine.qsa_selection(capacity)
                assert engine.stats.peak_bytes == 0
            else:
                with engine.qsa_selection(capacity):
                    assert engine.stats.peak_bytes == budget
                    with pytest.raises(NativeError, match='budget'):
                        engine.session()
                assert engine.stats.device_bytes == engine.stats.pinned_bytes == 0


def test_invalid_contracts_and_c_boundary(tmp_path):
    runtime = Runtime()
    lib = runtime._lib
    with runtime.engine(budget_bytes=2**24) as e, runtime.engine(budget_bytes=2**24) as other:
        for capacity in [0, 1, 5, 2**31]:
            with pytest.raises((ValueError, NativeError)):
                e.qsa_selection(capacity)
        a = arrays(8, 'controlled')
        a.update(bad_q=np.zeros((128, 4), np.float32),
                 bad_p=np.zeros((1, 128), np.float32),
                 bad_d=np.zeros((1, 128), np.float32),
                 half=np.zeros(128, np.float16),
                 nan=np.full((4, 128), np.nan, np.float32),
                 inf=np.full((4, 128), np.inf, np.float32))
        with inputs(tmp_path, runtime, e, a) as u, e.qsa_selection(8) as w:
            q, p, d, bq, bp, bd, half, nan, inf = u
            for n in [0, 9, -1, True, 1.5, 2**64]:
                with pytest.raises((ValueError, NativeError)):
                    w.set_n(n)
            w.set_n(5)
            for args in [(bq,p,d), (q,bp,d), (q,p,bd), (q,p,half), (nan,p,d), (inf,p,d)]:
                with pytest.raises(NativeError):
                    w.run(*args, policy='ggml_cpu')
            for policy in ['bad', None, 0]:
                with pytest.raises(ValueError):
                    w.run(q,p,d,policy=policy)
            for repeats in [0, -1, True, 10001]:
                with pytest.raises((ValueError, NativeError)):
                    w.run(q,p,d,policy='strata',repeats=repeats)
            with pytest.raises(ValueError):
                w.run(None,p,d,policy='strata')
            w.run(q,p,d,policy='strata')
            with pytest.raises(NativeError, match='capacity'):
                read(w, 4)
            with pytest.raises(ValueError):
                w.readback_for_test(np.empty(5, np.float64), np.empty(5, np.int64))
            out = C.c_uint64(123)
            assert lib.ml_qsa_create(0, 8, C.byref(out)) == 1 and out.value == 0
            assert lib.ml_qsa_create(e._handle, 8, None) == 1
            elapsed = C.c_float(123)
            assert lib.ml_qsa_run(w._handle,q._handle,p._handle,d._handle,2,1,C.byref(elapsed)) == 1
            assert elapsed.value == 0
            assert lib.ml_qsa_run(w._handle,q._handle,p._handle,d._handle,0,1,None) == 1
            count = C.c_uint64(123)
            assert lib.ml_qsa_readback_for_test(w._handle,None,5,None,5,C.byref(count)) == 1
            assert count.value == 0
            assert lib.ml_qsa_readback_for_test(w._handle,None,5,None,5,None) == 1
            for invalid in [0, e._handle, q._handle]:
                assert lib.ml_qsa_set_n(invalid,1) == 1
                assert lib.ml_qsa_reset(invalid) == 1
                assert lib.ml_qsa_run(invalid,q._handle,p._handle,d._handle,0,1,C.byref(elapsed)) == 1
            with other.qsa_selection(8) as cross:
                cross.set_n(4)
                with pytest.raises(NativeError, match='cross-engine'):
                    cross.run(q,p,d,policy='strata')
            stale = q._handle
            q.close()
            assert lib.ml_qsa_run(w._handle,stale,p._handle,d._handle,0,1,C.byref(elapsed)) == 1
            with pytest.raises(NativeError, match='closed'):
                w.run(q,p,d,policy='strata')
    assert lib.ml_qsa_destroy(0) == 0
