"""WP6 first resident FP32 paging slice; no checkpoint or production dispatch."""
import ctypes as C

import numpy as np
import pytest

from mini_llm.native import Runtime, NativeError
from test_native_engine_qsa import inputs, read
from test_native_engine_qsa_attention import fixture, outputs

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize('page_size,n', [(1, 1), (3, 2), (4, 4), (4, 5),
                                        (7, 8), (16, 33), (32, 2053), (64, 4096)])
@pytest.mark.parametrize('capacity', [4096, 262144])
def test_history_attention_reset_remap(tmp_path, page_size, n, capacity):
    r = Runtime()
    active_capacity = (n+3)//4*4
    a = fixture(active_capacity)
    history = dict(hk=a['keys'][:n-1].copy(), hv=a['values'][:n-1].copy())
    # Current and unused slots must never be consumed as committed history.
    a['keys'][n-1:] = np.nan
    a['values'][n-1:] = np.nan
    physical_pages = max(2, (n+page_size-1)//page_size)
    with r.engine(budget_bytes=2**27) as e:
        with inputs(tmp_path, r, e, dict(**a, **history) if n>1 else a) as u:
            with e.qsa_selection(active_capacity) as s, s.attention() as att:
                s.set_n(n)
                s.run(*u[:3], policy='ggml_cpu')
                att.run(*u[3:9])
                baseline = outputs(att)
                ids_before = read(s, n)[1]
                before = e.stats.device_bytes
                w = e.kv_pages(capacity, page_size=page_size, physical_pages=physical_pages)
                expected = ((capacity+page_size-1)//page_size*8 +
                            physical_pages*page_size*4096 + 2051*4096 + 8 + 2051*8 + page_size*4096)
                assert e.stats.device_bytes-before == expected
                assert w.state == (0, 1)
                if n>1:
                    w.append(*u[9:], generation=1)
                assert w.state == (n-1, 1)
                got = np.empty((n-1, 2, 512), np.float32)
                w.readback_for_test(got, generation=1)
                if n>1:
                    np.testing.assert_array_equal(got[:, 0], history['hk'].reshape(n-1, 512))
                    np.testing.assert_array_equal(got[:, 1], history['hv'].reshape(n-1, 512))
                stable = bytes(e.stats)
                for remap in [False, True]:
                    if remap and n>1:
                        w.remap(0, physical_pages-1, generation=1)
                        again = np.empty_like(got)
                        w.readback_for_test(again, generation=1)
                        np.testing.assert_array_equal(again, got)
                    w.attention(att, *u[3:7], generation=1)
                    for actual, want in zip(outputs(att), baseline):
                        np.testing.assert_array_equal(actual, want)
                    np.testing.assert_array_equal(read(s, n)[1], ids_before)
                    assert bytes(e.stats) == stable
                w.reset()
                assert w.state == (0, 2)
                with pytest.raises(NativeError, match='stale generation'):
                    w.attention(att, *u[3:7], generation=1)
                if n>1:
                    w.append(*u[9:], generation=2)
                w.attention(att, *u[3:7], generation=2)
                for actual, want in zip(outputs(att), baseline):
                    np.testing.assert_array_equal(actual, want)
                stale = w._handle
                w.close()
                assert r._lib.ml_kv_reset(stale) == 1
                assert e.stats.device_bytes == before
                # Completed snapshot does not borrow pages after close.
                for actual, want in zip(outputs(att), baseline):
                    np.testing.assert_array_equal(actual, want)


@pytest.mark.parametrize('has_current', [False, True])
def test_sparse_current_membership(tmp_path, has_current):
    r, n = Runtime(), 4096
    a = fixture(n, 'flat')
    a['q'].fill(1)
    a['p'][:700].fill(1)
    a['p'][-1].fill(100 if has_current else -100)
    with r.engine(budget_bytes=2**27) as e:
        with inputs(tmp_path, r, e, dict(**a, hk=a['keys'][:-1].copy(), hv=a['values'][:-1].copy())) as u:
            with e.qsa_selection(n) as s, s.attention() as att, e.kv_pages(262144, page_size=17, physical_pages=241) as w:
                s.set_n(n)
                s.run(*u[:3], policy='strata')
                att.run(*u[3:9])
                baseline = outputs(att)
                w.append(*u[9:], generation=1)
                # No selected-ID readback is needed to execute the paged path.
                w.attention(att, *u[3:7], generation=1)
                for actual, want in zip(outputs(att), baseline):
                    np.testing.assert_array_equal(actual, want)
                assert bool(n-1 in read(s, n)[1]) == has_current


def test_errors_budget_isolation_lifetime(tmp_path):
    r = Runtime()
    a = fixture(8)
    sources = dict(k=a['keys'][:4].copy(), v=a['values'][:4].copy(),
                   bad=a['keys'][:4].copy(), q=a['qmain'], g=a['gate'],
                   ck=a['k'], cv=a['v'])
    sources['bad'][0, 0, 0] = np.nan
    with r.engine(budget_bytes=2**26) as e, r.engine(budget_bytes=2**26) as other:
        with inputs(tmp_path, r, e, sources) as u, e.kv_pages(262144, page_size=4, physical_pages=1) as w:
            initial = bytes(e.stats)
            with pytest.raises(NativeError, match='nonfinite'):
                w.append(u[2], u[1], generation=1)
            assert w.state == (0, 1)
            with pytest.raises(NativeError, match='F32'):
                w.append(u[3], u[1], generation=1)
            w.append(*u[:2], generation=1)
            with pytest.raises(NativeError, match='physical page budget'):
                w.append(*u[:2], generation=1)
            assert w.state == (4, 1)
            with pytest.raises(NativeError, match='remap'):
                w.remap(1, 0, generation=1)
            with e.kv_pages(8, page_size=4, physical_pages=1) as independent:
                assert independent.state == (0, 1)
                independent.reset()
                assert w.state == (4, 1)
            with other.kv_pages(8, page_size=4, physical_pages=1) as foreign:
                with pytest.raises(NativeError, match='same-engine'):
                    foreign.append(*u[:2], generation=1)
            with e.qsa_selection(8) as s, s.attention() as att:
                with pytest.raises(NativeError, match='selection'):
                    w.attention(att, *u[3:7], generation=1)
            assert bytes(e.stats)[:16] == initial[:16]
            for value in u:
                value.close()
            # Store owns payload and engine, not source-upload handles.
            expected = np.stack((sources['k'].reshape(4, 512), sources['v'].reshape(4, 512)), axis=1)
            e.close()
            got = np.empty_like(expected)
            w.readback_for_test(got, generation=1)
            np.testing.assert_array_equal(got, expected)
    size = 8*65536 + 4*4096 + 2051*4096 + 8 + 2051*8 + 4*4096
    for budget in [size-1, size]:
        with r.engine(budget_bytes=budget) as e:
            if budget<size:
                with pytest.raises(NativeError, match='budget'):
                    e.kv_pages(262144, page_size=4, physical_pages=1)
                assert e.stats.device_bytes == 0
            else:
                with e.kv_pages(262144, page_size=4, physical_pages=1):
                    assert e.stats.device_bytes == size
                assert e.stats.device_bytes == 0
    for capacity, page, pages in [(0, 4, 1), (262145, 4, 1), (8, 0, 1), (8, 4, 0), (8, 4, 3)]:
        with r.engine(budget_bytes=2**24) as e:
            error = ValueError if capacity>262144 else NativeError
            with pytest.raises(error):
                e.kv_pages(capacity, page_size=page, physical_pages=pages)
    out = C.c_uint64(99)
    assert r._lib.ml_kv_create(0, 8, 4, 1, C.byref(out)) == 1
    assert out.value == 0


def test_incremental_page_tail_and_failed_current(tmp_path):
    r, capacity = Runtime(), 32
    a = fixture(capacity)
    row_inputs = {}
    for token in range(17):
        row_inputs[f'k{token}'] = a['keys'][token].copy()
        row_inputs[f'v{token}'] = a['values'][token].copy()
        row_inputs[f'hk{token}'] = a['keys'][token:token+1].copy()
        row_inputs[f'hv{token}'] = a['values'][token:token+1].copy()
    bad = a['k'].copy()
    bad[0, 0] = np.nan
    with r.engine(budget_bytes=2**26) as e:
        with inputs(tmp_path, r, e, dict(**a, **row_inputs, bad=bad)) as u:
            with e.qsa_selection(capacity) as s, s.attention() as att, e.kv_pages(262144, page_size=5, physical_pages=4) as w:
                stable = bytes(e.stats)
                for token in range(17):
                    current = u[9+4*token:11+4*token]
                    s.set_n(token+1)
                    s.run(*u[:3], policy='strata')
                    att.run(u[3], u[4], *current, u[7], u[8])
                    baseline = outputs(att)
                    w.attention(att, u[3], u[4], *current, generation=1)
                    for got, want in zip(outputs(att), baseline):
                        np.testing.assert_array_equal(got, want)
                    if token in (0, 5, 16):
                        with pytest.raises(NativeError, match='nonfinite'):
                            w.attention(att, u[3], u[4], u[-1], current[1], generation=1)
                        with pytest.raises(NativeError, match='unavailable'):
                            outputs(att)
                        assert w.state == (token, 1)
                    w.append(*u[11+4*token:13+4*token], generation=1)
                    assert w.state == (token+1, 1)
                    assert bytes(e.stats) == stable
                got = np.empty((17, 2, 512), np.float32)
                w.readback_for_test(got, generation=1)
                np.testing.assert_array_equal(got[:, 0], a['keys'][:17].reshape(17, 512))
                np.testing.assert_array_equal(got[:, 1], a['values'][:17].reshape(17, 512))
