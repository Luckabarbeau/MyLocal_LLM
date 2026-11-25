"""Explicit INT8/pageable-host policy; bounded actual populated-context tests."""
import ctypes as C
import json

import numpy as np
import pytest

from mini_llm.native import Runtime, NativeError
from test_native_engine_qsa import inputs, read
from test_native_engine_qsa_attention import fixture, numpy_graph, outputs

pytestmark = pytest.mark.gpu
POLICY = 'strata_int8_per64_fp16'
ROW = 1056


def quant_reference(k, v):
    groups = np.concatenate((k.reshape(-1, 512), v.reshape(-1, 512)), axis=1).reshape(-1, 16, 64)
    maximum = np.max(np.abs(groups), axis=-1)
    with np.errstate(over='ignore'):
        scales = (maximum / np.float32(127)).astype(np.float16)
    if not np.all(np.isfinite(scales)):
        raise ValueError('nonrepresentable scale')
    sf = scales.astype(np.float32)
    divided = np.divide(groups, sf[..., None], out=np.zeros_like(groups), where=sf[..., None]>0)
    q = np.clip(np.rint(divided), -127, 127).astype(np.int8)
    packed = np.empty((*q.shape[:-1], 66), np.uint8)
    packed[..., :64] = q.view(np.uint8)
    packed[..., 64:] = scales[..., None].view(np.uint8)
    decoded = (q.astype(np.float32) * sf[..., None]).reshape(-1, 2, 512)
    return packed.reshape(-1, ROW), decoded, scales


def create(engine, capacity, page, slots, host_budget=2**29, policy=POLICY):
    return engine.quantized_streamed_kv(capacity, page_size=page, physical_pages=slots,
                                       host_budget_bytes=host_budget, policy=policy)


def packed_read(w, start, rows, generation=1):
    result = np.empty((rows, ROW), np.uint8)
    w.readback_for_test(result, start=start, rows=rows, generation=generation)
    return result


def append_chunks(tmp_path, r, e, w, keys, values, page, generation=1):
    tmp_path.mkdir(exist_ok=True)
    for start in range(0, len(keys), page):
        with inputs(tmp_path, r, e, dict(k=keys[start:start+page], v=values[start:start+page])) as u:
            w.append(*u, generation=generation)


def test_quantization_exact_bytes_scales_and_error(tmp_path):
    r = Runtime()
    rng = np.random.default_rng(604)
    k = rng.normal(size=(12, 2, 256)).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    k[0].fill(0)
    g = k.reshape(-1, 64)
    g[8].fill(0)
    g[8, :9] = [127, .5, 1.5, 2.5, -.5, -1.5, -2.5, -127, 126.5]
    g[9].fill(np.float32(1e-8))  # rounded FP16 scale is zero
    g[10].fill(np.float32(127 * 2**-24))  # smallest positive FP16 scale
    g[11].fill(np.float32(127 * 65504))
    g[12].fill(np.float32(127 * 1.49 * 2**-24))  # scale rounds down; +127 clipping is required
    g[13].fill(np.float32(-127 * 1.49 * 2**-24))  # symmetric -127 clipping
    packed, decoded, scales = quant_reference(k, v)
    with r.engine(budget_bytes=2**25) as e:
        with inputs(tmp_path, r, e, dict(k=k, v=v)) as u, create(e, 64, 16, 1) as w:
            w.append(*u, generation=1)
            actual = packed_read(w, 0, len(k))
            np.testing.assert_array_equal(actual, packed)
            actual_scales = actual.reshape(-1, 16, 66)[..., 64:].copy().view(np.uint16).reshape(-1, 16)
            np.testing.assert_array_equal(actual_scales, scales.view(np.uint16))
            source = np.concatenate((k.reshape(-1, 512), v.reshape(-1, 512)), axis=1).reshape(-1, 16, 64)
            error = np.abs(decoded.reshape(-1, 16, 64) - source)
            sf = scales.astype(np.float32)
            maximum = np.max(np.abs(source), axis=-1)
            bound = np.maximum(.50001*sf, np.maximum(maximum-127*sf, 0))
            assert np.all(error <= bound[..., None] + 2*np.finfo(np.float32).eps*np.abs(source))
            print('quantization_exact_bytes_scales=True max_absolute_error=', float(error.max()))


@pytest.mark.parametrize('n,page,slots', [(1, 7, 1), (9, 3, 3), (33, 8, 4), (2053, 1024, 3)])
def test_decoded_attention_current_reset_lifetime(tmp_path, n, page, slots):
    r = Runtime()
    capacity = (max(n, page)+3)//4*4
    a = fixture(capacity)
    packed, decoded, _ = quant_reference(a['keys'][:n-1], a['values'][:n-1])
    _, current, _ = quant_reference(a['k'][None], a['v'][None])
    reference = dict(a)
    reference['keys'], reference['values'] = a['keys'].copy(), a['values'].copy()
    reference['keys'][:n-1] = decoded[:, 0].reshape(-1, 2, 256)
    reference['values'][:n-1] = decoded[:, 1].reshape(-1, 2, 256)
    reference['k'], reference['v'] = current[0, 0].reshape(2, 256), current[0, 1].reshape(2, 256)
    with r.engine(budget_bytes=2**27) as e:
        with inputs(tmp_path, r, e, a) as u, e.qsa_selection(capacity) as s, s.attention() as att:
            w = create(e, capacity, page, slots)
            append_chunks(tmp_path/'batch', r, e, w, a['keys'][:n-1], a['values'][:n-1], page)
            s.set_n(n); s.run(*u[:3], policy='strata')
            w.attention(att, *u[3:7], generation=1)
            ids = read(s, n)[1]  # Diagnostic only, after native paging.
            expected = numpy_graph(reference, ids, n)
            actual = outputs(att)
            for got, want in zip(actual, expected):
                np.testing.assert_allclose(got, want, rtol=4e-5, atol=3e-6)
            fp32 = numpy_graph(a, ids, n)
            discrepancy = max(float(np.max(np.abs(x-y))) for x, y in zip(actual, fp32))
            print(json.dumps(dict(n=n, decoded_reference_maxabs=max(float(np.max(np.abs(x-y))) for x,y in zip(actual, expected)),
                                  fp32_policy_discrepancy_maxabs=discrepancy)))
            before = w.stats
            w.attention(att, *u[3:7], generation=1)
            assert w.stats.misses == before.misses
            assert w.stats.hits-before.hits == before.resident_pages
            assert w.stats.h2d_bytes == before.h2d_bytes
            assert w.stats.pending == 0
            w.reset()
            assert w.state == (0, 2)
            assert w.stats.resident_pages == w.stats.hits == w.stats.misses == 0
            with pytest.raises(NativeError, match='stale generation'):
                w.attention(att, *u[3:7], generation=1)
            append_chunks(tmp_path/'batch', r, e, w, a['keys'][:n-1], a['values'][:n-1], page, generation=2)
            w.attention(att, *u[3:7], generation=2)
            np.testing.assert_array_equal(packed_read(w, 0, n-1, 2), packed)
            snapshot = outputs(att)
            e.close()
            for item in u: item.close()
            np.testing.assert_array_equal(packed_read(w, 0, n-1, 2), packed)
            stale = w._handle
            w.close()
            for got, want in zip(outputs(att), snapshot): np.testing.assert_array_equal(got, want)
            assert r._lib.ml_kvi8_reset(stale) == 1


def test_hits_misses_eviction_remap_and_working_set_refusal(tmp_path):
    r, n, page = Runtime(), 4096, 1024
    a = fixture(n, 'flat')
    a['q'].fill(1)
    alternate = a['p'].copy()
    alternate[256:].fill(1)
    with r.engine(budget_bytes=2**27) as e:
        with inputs(tmp_path, r, e, dict(**a, alternate=alternate)) as u:
            with e.qsa_selection(n) as s, s.attention() as att, create(e, n, page, 3) as w:
                append_chunks(tmp_path/'batch', r, e, w, a['keys'][:-1], a['values'][:-1], page)
                stable = e.stats.device_bytes
                s.set_n(n); s.run(*u[:3], policy='strata')
                w.attention(att, *u[3:7], generation=1)
                assert w.stats.misses == 3 and w.stats.hits == 0
                w.attention(att, *u[3:7], generation=1)
                assert w.stats.misses == 3 and w.stats.hits == 3
                s.run(u[0], u[-1], u[2], policy='strata')
                before = w.stats
                w.attention(att, *u[3:7], generation=1)
                assert w.stats.misses-before.misses == 1
                assert w.stats.hits-before.hits == 2
                snapshot = outputs(att)
                w.remap(0, 1, generation=1)  # Explicitly stage a nonselected host page, evict a slot.
                w.attention(att, *u[3:7], generation=1)
                for got, want in zip(outputs(att), snapshot): np.testing.assert_array_equal(got, want)
                assert e.stats.device_bytes == stable
                print('resident_hits_preserved=True staged_only_missing_pages=True')
            with create(e, n, page, 2) as small, e.qsa_selection(n) as s, s.attention() as att:
                s.set_n(n); s.run(u[0], u[-1], u[2], policy='strata')
                append_chunks(tmp_path/'batch', r, e, small, a['keys'][:-1], a['values'][:-1], page)
                before = small.stats
                with pytest.raises(NativeError, match='working set'):
                    small.attention(att, *u[3:7], generation=1)
                assert small.stats.resident_pages == before.resident_pages == 0
                assert small.stats.h2d_bytes == before.h2d_bytes == 0
                with pytest.raises(NativeError, match='unavailable'): outputs(att)
                assert small.state == (4095, 1)


def test_budgets_errors_partial_append_and_isolation(tmp_path):
    r = Runtime()
    a = fixture(8)
    bad = a['keys'][:2].copy(); bad[0, 0, 0] = np.nan
    huge = a['keys'][:2].copy(); huge.fill(1e10)
    with r.engine(budget_bytes=2**25) as e, r.engine(budget_bytes=2**25) as other:
        with inputs(tmp_path, r, e, dict(k=a['keys'][:2], v=a['values'][:2], bad=bad, huge=huge,
                                        q=a['qmain'], g=a['gate'], ck=a['k'], cv=a['v'],
                                        iq=a['q'], p=a['p'], dead=a['d'])) as u:
            with create(e, 8, 3, 2) as w, create(other, 8, 3, 2) as foreign:
                for invalid in (u[2], u[3]):
                    with pytest.raises(NativeError, match='nonfinite|nonrepresentable'):
                        w.append(invalid, u[1], generation=1)
                    assert w.state == (0, 1)
                with pytest.raises(NativeError, match='same-engine'):
                    foreign.append(*u[:2], generation=1)
                w.append(*u[:2], generation=1)
                w.append(*u[:2], generation=1)  # Cross an already partially filled page.
                assert w.state == (4, 1) and foreign.state == (0, 1)
                expected, _, _ = quant_reference(np.tile(a['keys'][:2], (2,1,1)), np.tile(a['values'][:2], (2,1,1)))
                np.testing.assert_array_equal(packed_read(w, 0, 4), expected)
                with e.qsa_selection(8) as s, s.attention() as att:
                    s.set_n(5); s.run(*u[8:11], policy='strata')
                    w.attention(att, *u[4:8], generation=1)
                    before = w.stats
                    with pytest.raises(NativeError, match='F32'):
                        w.attention(att, u[0], *u[5:8], generation=1)
                    assert w.stats.h2d_bytes == before.h2d_bytes
                    with pytest.raises(NativeError, match='unavailable'): outputs(att)
                with pytest.raises(NativeError, match='bounds'):
                    packed_read(w, 4, 1)
                with pytest.raises(NativeError, match='remap'): w.remap(7, 0, generation=1)
                device, host = w.stats.device_bytes, w.stats.host_bytes
        before = e.stats.device_bytes
        with pytest.raises(NativeError, match='ordinary-RAM budget'):
            create(e, 8, 3, 2, host_budget=host-1)
        assert e.stats.device_bytes == before
        with create(e, 8, 3, 2, host_budget=host) as w:
            assert w.stats.host_bytes == host
        with pytest.raises(NativeError, match='policy'):
            create(e, 8, 3, 2, policy='int8')
    for budget in (device-1, device):
        with r.engine(budget_bytes=budget) as e:
            if budget<device:
                with pytest.raises(NativeError, match='engine allocation budget'): create(e, 8, 3, 2)
                assert e.stats.device_bytes == 0
            else:
                with create(e, 8, 3, 2) as w: assert e.stats.device_bytes == device
                assert e.stats.device_bytes == 0


def test_cached_tail_invalidation_bad_current_and_full_capacity(tmp_path):
    r = Runtime()
    a = fixture(8)
    bad = a['k'].copy(); bad[0, 0] = np.nan
    sources = dict(hk=a['keys'][:2], hv=a['values'][:2], q=a['qmain'], g=a['gate'],
                   k=a['k'], v=a['v'], iq=a['q'], p=a['p'], dead=a['d'], bad=bad)
    with r.engine(budget_bytes=2**25) as e:
        with inputs(tmp_path, r, e, sources) as u, e.qsa_selection(8) as s, s.attention() as att:
            with create(e, 8, 3, 2) as w:
                w.append(*u[:2], generation=1)
                s.set_n(3); s.run(*u[6:9], policy='ggml_cpu')
                w.attention(att, *u[2:6], generation=1)
                assert w.stats.resident_pages == 1
                w.append(*u[:2], generation=1)  # Updates cached page0 and begins page1.
                assert w.stats.resident_pages == 0
                s.set_n(5); s.run(*u[6:9], policy='ggml_cpu')
                w.attention(att, *u[2:6], generation=1)
                assert w.stats.resident_pages == 2
                w.append(*u[:2], generation=1)  # Only page1 changes; page0 is a retained hit.
                assert w.stats.resident_pages == 1
                s.set_n(7); s.run(*u[6:9], policy='ggml_cpu')
                before = w.stats
                with pytest.raises(NativeError, match='nonfinite'):
                    w.attention(att, u[2], u[3], u[-1], u[5], generation=1)
                assert w.state == (6, 1)
                assert w.stats.h2d_bytes == before.h2d_bytes
                with pytest.raises(NativeError, match='unavailable'): outputs(att)
                w.attention(att, *u[2:6], generation=1)
                assert w.stats.hits-before.hits == 1
                assert w.stats.misses-before.misses == 1
                w.append(*u[:2], generation=1)
                assert w.state == (8, 1)
                with pytest.raises(NativeError, match='logical capacity'):
                    w.append(*u[:2], generation=1)
                w.reset()
                for operation in (lambda: w.append(*u[:2], generation=1),
                                  lambda: w.remap(0, 0, generation=1),
                                  lambda: packed_read(w, 0, 1)):
                    with pytest.raises(NativeError, match='stale generation'): operation()
                assert w.state == (0, 2)


def test_actual_progressive_64k_128k_262k_bounded_store(tmp_path):
    r, capacity, page = Runtime(), 262144, 1024
    batch_dir = tmp_path/'batch'; batch_dir.mkdir()
    rng = np.random.default_rng(262144)
    # One bounded FP32 chunk, reused for all positions. No full FP32 history exists.
    keys = rng.normal(size=(page, 2, 256)).astype(np.float32)
    values = rng.normal(size=keys.shape).astype(np.float32)
    packed, decoded, _ = quant_reference(keys, values)
    controls = dict(q=np.ones((4, 128), np.float32),
                    p=np.zeros((capacity//4, 128), np.float32), d=np.zeros(128, np.float32),
                    qm=rng.normal(size=(24, 256)).astype(np.float32),
                    g=rng.normal(size=(24, 256)).astype(np.float32), k=keys[-1], v=values[-1],
                    ranked=np.broadcast_to(np.arange(capacity//4, dtype=np.float32)[:, None],
                                           (capacity//4, 128)).copy())
    records = []
    with r.engine(budget_bytes=2**27) as e:
        with inputs(tmp_path, r, e, controls) as u, e.qsa_selection(capacity) as s, s.attention() as att:
            with create(e, capacity, page, 4, host_budget=280_000_000) as w:
                initial = w.stats
                steady = e.stats.device_bytes
                position = 0
                for milestone in (65536, 131072, 262144):
                    # Reach n-1, evaluate current, then commit the last row to populate all n rows.
                    while position<milestone-1:
                        count = min(page-position%page, milestone-1-position)
                        offset = position%page
                        with inputs(batch_dir, r, e, dict(k=keys[offset:offset+count], v=values[offset:offset+count])) as batch:
                            w.append(*batch, generation=1)
                        got = packed_read(w, position, count)
                        np.testing.assert_array_equal(got, packed[offset:offset+count])
                        position += count
                    s.set_n(milestone)
                    for tail in (False, True):
                        s.run(u[0], u[-1] if tail else u[1], u[2], policy='strata')
                        before = w.stats
                        w.attention(att, *u[3:7], generation=1)
                        first = w.stats
                        if tail: assert first.misses > before.misses
                        result = outputs(att)
                        w.attention(att, *u[3:7], generation=1)
                        second = w.stats
                        assert second.misses == first.misses and second.h2d_bytes == first.h2d_bytes
                        # Diagnostics occur only after native page selection/staging.
                        ids = read(s, milestone)[1]
                        if tail:
                            assert ids.min() >= milestone-2052 and ids[-1] == milestone-1
                        else:
                            np.testing.assert_array_equal(ids, np.arange(2051, dtype=np.int64))
                        history = decoded[ids%page]
                        for h in range(24):
                            logits = (history[:, 0].reshape(-1, 2, 256)[:, h//12] @ controls['qm'][h])*np.float32(1/16)
                            p = np.exp(logits-logits.max()); p /= p.sum(dtype=np.float32)
                            expected = p @ history[:, 1].reshape(-1, 2, 256)[:, h//12]
                            np.testing.assert_allclose(result[0][h], expected, rtol=4e-5, atol=3e-6)
                    with inputs(batch_dir, r, e, dict(k=keys[-1:], v=values[-1:])) as batch:
                        w.append(*batch, generation=1)
                    position += 1
                    assert w.state == (milestone, 1)
                    np.testing.assert_array_equal(packed_read(w, milestone-1, 1), packed[-1:])
                    stats = w.stats
                    assert stats.device_bytes == initial.device_bytes
                    assert stats.host_bytes == initial.host_bytes <= 280_000_000
                    assert stats.populated_bytes == milestone*ROW
                    assert stats.resident_pages <= 4 and stats.pending == 0
                    assert e.stats.device_bytes == steady
                    records.append({name: int(getattr(stats, name)) for name, _ in stats._fields_})
    print('ACTUAL_POPULATED_CONTEXT_MEMORY=' + json.dumps(records, sort_keys=True))
