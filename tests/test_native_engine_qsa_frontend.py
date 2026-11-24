"""Projected-input native frontend, compared to unchanged Qwen reference ops.

No checkpoint, packed projections, session commits, or full attention integration.
Run with MINI_LLM_BACKEND=numpy and the built native library; CUDA is required.
"""
import ctypes as C

import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.external.qwen4exp_attention import QSAConfig, Qwen4ExpAttention
from mini_llm.native import Runtime, NativeError
from test_native_engine_qsa import inputs

pytestmark = pytest.mark.gpu
BOUNDARIES = [1, 2, 3, 4, 5, 127, 2048, 2051, 2052, 2053, 4096, 16384]
SHAPES = [(24, 256), (2, 256), (24, 256), (4, 128), (128,), (128,)]
NAMES = ['qg', 'k', 'raw', 'iq', 'first_raw', 'partial',
         'gamma_q', 'gamma_k', 'gamma_iq', 'gamma_ik']


def arrays():
    rng = np.random.default_rng(2026112303)
    shapes = [(24, 512), (2, 256), (128,), (4, 128), (128,), (3, 128),
              (256,), (256,), (128,), (128,)]
    a = {name: rng.normal(size=shape).astype(np.float32)
         for name, shape in zip(NAMES, shapes)}
    # Deliberately not near 1: catches accidental 1+gamma conversion.
    a['gamma_q'][::5] = 0
    a['qg'][:, 256:][:, :5] = [-1000, -1, 0, 1, 1000]
    return a


def reference(a, n, epsilon):
    def unused(_):
        raise AssertionError('projected-input test must not call projections')
    ref = Qwen4ExpAttention(
        QSAConfig(d_model=1, epsilon=epsilon),
        **dict.fromkeys(('query_gate', 'key', 'value', 'output', 'index_query', 'index_key'), unused),
        gamma_q=xp.asarray(a['gamma_q']), gamma_k=xp.asarray(a['gamma_k']),
        gamma_index_q=xp.asarray(a['gamma_iq']), gamma_index_k=xp.asarray(a['gamma_ik']))
    def norm(j, x):
        return ref.norms[j].forward(xp.asarray(x))
    q = ref._rotate(norm(0, a['qg'][:, :256]), n-1)
    k = ref._rotate(norm(1, a['k']), n-1)
    iq = ref._rotate(norm(2, a['iq']), n-1)
    dead = norm(3, (a['raw'] if n == 1 else a['first_raw'])[None])[0]
    pool = None
    if n % 4 == 0:
        pool = a['partial'][0].copy()
        pool += a['partial'][1]
        pool += a['partial'][2]
        pool += a['raw']
        pool *= np.float32(0.25)
        pool = ref._rotate(norm(3, pool[None]), n-4)[0]
    return [asnumpy(x) if x is not None else None
            for x in (q, k, a['qg'][:, 256:], iq, dead, pool)]


def read(w):
    out = [np.empty(shape, np.float32) for shape in SHAPES]
    n, ready = w.readback_for_test(*out)
    return out, n, ready


def compare(w, a, n, epsilon=1e-6):
    got, device_n, ready = read(w)
    assert device_n == n  # actual device parameter, not the host shadow
    assert ready == (n % 4 == 0)
    want = reference(a, n, epsilon)
    maximum = 0.0
    for j, (actual, expected) in enumerate(zip(got, want)):
        if expected is None:
            continue
        assert np.isfinite(actual).all()
        if j == 2:
            np.testing.assert_array_equal(actual, expected)  # copied logits, no sigmoid
            continue
        # RMS tree vs NumPy reduction: a small FP32 relative/absolute allowance.
        bound = 3e-6 * (1 + np.abs(expected))
        if j != 4:
            # powf/NumPy power may differ by a few ulps. Angle error grows with
            # position; bound it per frequency/pair rather than a blanket atol.
            position = n-4 if j == 5 else n-1
            frequency = np.power(np.float32(1e7), -np.arange(32, dtype=np.float32)/32)
            pair_size = np.abs(expected[..., :32]) + np.abs(expected[..., 32:64])
            angle_bound = 8*np.finfo(np.float32).eps*position*frequency*pair_size
            bound[..., :32] += angle_bound
            bound[..., 32:64] += angle_bound
        error = np.abs(actual-expected)
        assert np.all(error <= bound), (j, float(error.max()), float((error/bound).max()))
        maximum = max(maximum, float(error.max()))
    return maximum


@pytest.mark.parametrize('n', BOUNDARIES)
@pytest.mark.parametrize('epsilon', [1e-6, 0.125])
def test_reference_boundaries(tmp_path, n, epsilon):
    r, a = Runtime(), arrays()
    with r.engine(budget_bytes=2**22) as e:
        with inputs(tmp_path, r, e, a) as u, e.qsa_frontend() as w:
            w.set_n(n)
            before = bytes(e.stats)
            w.run(*u, epsilon=epsilon)
            maximum = compare(w, a, n, epsilon)
            print(f'frontend n={n} epsilon={epsilon}: max_abs={maximum:.9g}')
            w.run(*u, epsilon=epsilon)
            compare(w, a, n, epsilon)
            assert bytes(e.stats) == before
            for tensor, source in zip(u, a.values()):
                copied = bytearray(source.nbytes)
                tensor.readback_for_test(0, source.nbytes, copied)
                assert copied == source.tobytes()


def test_lifecycle_reset_budget_stable_n(tmp_path):
    r, a = Runtime(), arrays()
    with r.engine(budget_bytes=2**22) as e:
        with inputs(tmp_path, r, e, a) as u:
            before = e.stats
            w = e.qsa_frontend()
            current = e.stats
            assert current.device_bytes-before.device_bytes == 54288
            assert current.pinned_bytes-before.pinned_bytes == 16
            with pytest.raises(NativeError, match='unavailable'):
                read(w)
            with pytest.raises(NativeError, match='n/epsilon'):
                w.run(*u)
            previous_pool = None
            for n in [4, 1, 2052, 3, 16384, 5, 2]:
                w.set_n(n)
                with pytest.raises(NativeError, match='unavailable'):
                    read(w)
                w.run(*u)
                compare(w, a, n)
                out, device_n, ready = read(w)
                assert device_n == n
                if ready:
                    previous_pool = out[-1].copy()
                else:
                    np.testing.assert_array_equal(out[-1], previous_pool)
                w.reset()
                with pytest.raises(NativeError, match='unavailable'):
                    read(w)
                with pytest.raises(NativeError, match='n/epsilon'):
                    w.run(*u)
                assert bytes(e.stats) == bytes(current)
            # uint64 n has no n+1 overflow in this projected-only graph.
            w.set_n(2**64-1)
            w.run(*u)
            out, device_n, ready = read(w)
            assert device_n == 2**64-1 and not ready
            assert all(np.isfinite(x).all() for x in out)
            w.set_n(4)
            w.run(*u)
            for tensor in u:
                tensor.close()
            e.close()
            compare(w, a, 4)  # workspace independently retains engine/results
            stale = w._handle
            w.close()
            assert r._lib.ml_qsa_frontend_destroy(stale) == 1
            assert r._lib.ml_qsa_frontend_destroy(0) == 0
    for budget in [54303, 54304]:
        with r.engine(budget_bytes=budget) as e:
            if budget == 54303:
                before = bytes(e.stats)
                with pytest.raises(NativeError, match='budget'):
                    e.qsa_frontend()
                assert bytes(e.stats) == before
            else:
                with e.qsa_frontend():
                    assert e.stats.device_bytes + e.stats.pinned_bytes == budget
                assert e.stats.device_bytes == e.stats.pinned_bytes == 0


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_consumed_invalidates(tmp_path, name, bad):
    r, a = Runtime(), arrays()
    with r.engine(budget_bytes=2**22) as e, e.qsa_frontend() as w:
        with inputs(tmp_path, r, e, a) as good:
            w.set_n(4)
            w.run(*good)
            changed = {key: value.copy() for key, value in a.items()}
            changed[name].flat[0] = bad
            with inputs(tmp_path, r, e, changed) as u:
                with pytest.raises(NativeError, match='nonfinite'):
                    w.run(*u)
                with pytest.raises(NativeError, match='unavailable'):
                    read(w)
            w.run(*good)
            compare(w, a, 4)


@pytest.mark.parametrize('kind', ['square', 'pool_sum', 'normalized'])
def test_finite_input_overflow(tmp_path, kind):
    a, r = arrays(), Runtime()
    if kind == 'square':
        a['k'].fill(np.finfo(np.float32).max)
    elif kind == 'pool_sum':
        a['partial'].fill(np.finfo(np.float32).max)
    else:
        a['iq'].fill(0)
        a['iq'][:, 0] = 1
        a['gamma_iq'].fill(np.finfo(np.float32).max)
    with r.engine(budget_bytes=2**22) as e:
        with inputs(tmp_path, r, e, a) as u, e.qsa_frontend() as w:
            w.set_n(4)
            with pytest.raises(NativeError, match='nonfinite'):
                w.run(*u)
            with pytest.raises(NativeError, match='unavailable'):
                read(w)


def test_unused_history_is_not_consumed(tmp_path):
    r, a = Runtime(), arrays()
    a['first_raw'].fill(np.nan)
    a['partial'].fill(np.inf)
    with r.engine(budget_bytes=2**22) as e:
        with inputs(tmp_path, r, e, a) as u, e.qsa_frontend() as w:
            w.set_n(1)
            w.run(*u)
            compare(w, a, 1)
            w.set_n(2)
            with pytest.raises(NativeError, match='nonfinite'):
                w.run(*u)
    a['first_raw'].fill(1)
    with r.engine(budget_bytes=2**22) as e:
        with inputs(tmp_path, r, e, a) as u, e.qsa_frontend() as w:
            w.set_n(3)
            w.run(*u)
            compare(w, a, 3)
            w.set_n(4)
            with pytest.raises(NativeError, match='nonfinite'):
                w.run(*u)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('defect', ['shape', 'type'])
def test_exact_input_contract(tmp_path, name, defect):
    r, a = Runtime(), arrays()
    with r.engine(budget_bytes=2**22) as e, e.qsa_frontend() as w:
        with inputs(tmp_path, r, e, a) as good:
            w.set_n(4)
            w.run(*good)
            changed = dict(a)
            changed[name] = (a[name].reshape(-1, 1) if defect == 'shape'
                             else a[name].astype(np.float16))
            with inputs(tmp_path, r, e, changed) as bad:
                with pytest.raises(NativeError, match='F32 input shapes'):
                    w.run(*bad)
                with pytest.raises(NativeError, match='unavailable'):
                    read(w)


def test_engine_stale_binding_and_diagnostic_contract(tmp_path):
    r, a = Runtime(), arrays()
    with r.engine(budget_bytes=2**22) as e, r.engine(budget_bytes=2**22) as other:
        with inputs(tmp_path, r, e, a) as good, inputs(tmp_path, r, other, a) as foreign:
            with e.qsa_frontend() as w:
                w.set_n(4)
                w.run(*good)
                with pytest.raises(NativeError, match='cross-engine'):
                    w.run(*foreign)
                with pytest.raises(NativeError, match='unavailable'):
                    read(w)
                for epsilon in [0, -1, np.nan, np.inf, 1e100, 1e-100]:
                    w.run(*good)
                    with pytest.raises(NativeError, match='epsilon'):
                        w.run(*good, epsilon=epsilon)
                    with pytest.raises(NativeError, match='unavailable'):
                        read(w)
                for n in [True, -1, 1.5, 2**64]:
                    with pytest.raises(ValueError):
                        w.set_n(n)
                with pytest.raises(NativeError, match='positive'):
                    w.set_n(0)
                w.set_n(4)
                w.run(*good)
                out, _, _ = read(w)
                for bad in [np.empty(6144, np.float64), np.empty(12288, np.float32)[::2],
                            bytes(24576)]:
                    with pytest.raises(ValueError, match='F32'):
                        w.readback_for_test(bad, *out[1:])
                short = np.full(1, -99, np.float32)
                with pytest.raises(NativeError, match='capacity'):
                    w.readback_for_test(short, *out[1:])
                assert short[0] == -99
                with pytest.raises(ValueError, match='UploadedTensor'):
                    w.run(np.zeros((24, 512), np.float32), *good[1:])
                other_runtime = Runtime()
                with other_runtime.engine(budget_bytes=2**22) as oe:
                    with inputs(tmp_path, other_runtime, oe, a) as ou:
                        with pytest.raises(ValueError, match='same Runtime'):
                            w.run(*ou)
                stale = good[0]._handle
                good[0].close()
                with pytest.raises(NativeError, match='closed'):
                    w.run(*good)
                # Exercise native stale-input rejection and snapshot invalidation.
                handles = [stale] + [x._handle for x in good[1:]]
                assert r._lib.ml_qsa_frontend_run(w._handle, *handles, 1e-6) == 1
                with pytest.raises(NativeError, match='unavailable'):
                    read(w)
                stale_workspace = w._handle
            assert r._lib.ml_qsa_frontend_set_n(stale_workspace, 1) == 1
            assert r._lib.ml_qsa_frontend_reset(stale_workspace) == 1
