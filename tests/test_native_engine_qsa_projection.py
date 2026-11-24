"""Bounded WP3 reference_fp32 projection, no model construction.

Optional actual checkpoint: MINI_LLM_QSA_PROJECTION_CHECKPOINT=1.
"""
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
from test_0201_gguf_store import write_gguf
from test_native_engine_qsa import inputs
from test_native_engine_qsa_attention import fixture, outputs

pytestmark = pytest.mark.gpu
RTOL, ATOL = 5e-5, 2e-4


def weight_upload(stack, tmp_path, r, e, raw, kind, rows, width=6144):
    path = tmp_path / 'projection.gguf'
    write_gguf(path, [('w', (width, rows), kind, 0, raw)])
    store = stack.enter_context(r.gguf(path))
    lease = stack.enter_context(store.lease('w'))
    return stack.enter_context(e.upload(lease))


def attention(stack, tmp_path, r, e):
    # n=1 gives deterministic current V repeated across query groups. gate=0
    # makes sigmoid exactly 0.5, isolating projection arithmetic from softmax.
    a = fixture(4)
    a['gate'].fill(0)
    u = stack.enter_context(inputs(tmp_path, r, e, a))
    s = stack.enter_context(e.qsa_selection(4))
    att = stack.enter_context(s.attention())
    s.set_n(1)
    att.run(*u[3:], selection_inputs=u[:3])
    x = np.repeat(a['v'], 12, axis=0).ravel()*np.float32(.5)
    return att, x, s, u


def synthetic(kind, rows=7):
    rng = np.random.default_rng(61442560+kind)
    if kind in (0, 30):
        w = rng.normal(size=(rows, 6144)).astype(np.float32)
        w[0] = 0
        if kind == 30:
            bits = (w.view(np.uint32)>>16).astype('<u2')
            return bits.tobytes(), (bits.astype(np.uint32)<<16).view(np.float32)
        return w.tobytes(), w
    qb = 144 if kind == 12 else 210
    b = rng.integers(0, 256, (rows*24, qb), dtype=np.uint8)
    # Exercise zero/negative/subnormal/max finite half scales, all packed bits,
    # signed Q6 scales including -128 and 127, group boundaries and minima.
    scales = np.array([0, -0., 2**-24, -2**-14, .03125, -1, 65504], '<f2')
    off = 0 if kind == 12 else 208
    b[:, off:off+2] = np.resize(scales.view(np.uint8).reshape(-1, 2), (len(b), 2))
    if kind == 12:
        b[:, 2:4] = np.frombuffer(np.float16(.015625).tobytes(), np.uint8)
        b[0, 4:16] = 0
        b[1, 4:16] = 255
    else:
        b[0, 192:208] = 128
        b[1, 192:208] = 127
    raw = b.tobytes()
    w = PackedGGUFWeight(np.frombuffer(raw, np.uint8), (rows, 6144),
                         'Q4_K' if kind == 12 else 'Q6_K').dequantize_reference()
    return raw, w


@pytest.mark.parametrize('kind', [0, 30, 12, 14])
def test_reference_formats(tmp_path, kind):
    r = Runtime()
    raw, w = synthetic(kind)
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=2**26))
        att, x, s, u = attention(stack, tmp_path, r, e)
        upload = weight_upload(stack, tmp_path, r, e, raw, kind, 7)
        before = e.stats
        p = stack.enter_context(att.projection(upload, arithmetic='reference_fp32'))
        assert e.stats.device_bytes-before.device_bytes == 7*4+4
        assert e.stats.pinned_bytes-before.pinned_bytes == 4
        got = np.empty(7, np.float32)
        with pytest.raises(NativeError, match='unavailable'):
            p.readback_for_test(got)
        state = bytes(e.stats)
        assert p.run(repeats=3) > 0
        p.readback_for_test(got)
        # Half max scales produce large cancellation-sensitive sums. Compare to
        # FP32 matvec using a forward-error bound as well as ordinary tolerance.
        want = w @ x
        bound = 2e-6 * (np.abs(w) @ np.abs(x)) + 2e-4
        assert np.all(np.abs(got-want) <= bound), (got, want, bound)
        if kind in (0, 30):
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
        assert bytes(e.stats) == state
        # No hidden mutation of packed bytes; ownership outlives public handles.
        copied = bytearray(len(raw))
        upload.readback_for_test(0, len(raw), copied)
        assert copied == raw
        upload.close(); att.close(); s.close()
        for tensor in u:
            tensor.close()
        e.close()
        p.run()
        again = np.empty_like(got); p.readback_for_test(again)
        np.testing.assert_array_equal(again, got)
        stale = p._handle
        p.close()
        assert r._lib.ml_qsa_projection_destroy(stale) == 1


def test_invalid_and_snapshots(tmp_path):
    r = Runtime()
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=2**25))
        att, x, s, u = attention(stack, tmp_path, r, e)
        raw, _ = synthetic(0)
        w = weight_upload(stack, tmp_path, r, e, raw, 0, 7)
        p = stack.enter_context(att.projection(w))
        for mode in ['q8', 'ggml_cpu', None]:
            with pytest.raises(ValueError, match='reference_fp32'):
                att.projection(w, arithmetic=mode)
        assert r._lib.ml_qsa_projection_create(att._handle, w._handle, None) == 1
        p.run()
        got = np.empty(7, np.float32); p.readback_for_test(got)
        short = np.full(6, 123, np.float32)
        with pytest.raises(NativeError, match='capacity'):
            p.readback_for_test(short)
        assert np.all(short == 123)
        for bad in [got.astype(np.float64), got[::2], bytes(28)]:
            with pytest.raises(ValueError):
                p.readback_for_test(bad)
        # Reset of selection does not erase attention/projection snapshots.
        s.reset(); p.run()
        for repeats in [0, 10001]:
            elapsed = C.c_float(-1)
            assert r._lib.ml_qsa_projection_run(p._handle, repeats, C.byref(elapsed)) == 1
            assert elapsed.value == 0
            with pytest.raises(NativeError, match='unavailable'):
                p.readback_for_test(got)
            p.run()
        # Failed attention invalidates its snapshot; projection cannot use it.
        with pytest.raises(NativeError):
            att.run(*u[3:])
        with pytest.raises(NativeError, match='unavailable'):
            p.run()
        with pytest.raises(NativeError, match='unavailable'):
            p.readback_for_test(got)


@pytest.mark.parametrize('kind', [0, 30, 12, 14])
def test_nonfinite_weights(tmp_path, kind):
    r = Runtime()
    raw, _ = synthetic(kind)
    raw = bytearray(raw)
    if kind == 0:
        raw[-4:] = np.float32(np.nan).tobytes()
    elif kind == 30:
        raw[-2:] = bytes([128, 127])
    elif kind == 12:
        raw[-144:-142] = np.float16(np.inf).tobytes()
    else:
        raw[-2:] = np.float16(np.nan).tobytes()
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=2**25))
        att, _, _, _ = attention(stack, tmp_path, r, e)
        w = weight_upload(stack, tmp_path, r, e, raw, kind, 7)
        p = stack.enter_context(att.projection(w))
        before = bytes(e.stats)
        with pytest.raises(NativeError, match='nonfinite'):
            p.run()
        with pytest.raises(NativeError, match='unavailable'):
            p.readback_for_test(np.empty(7, np.float32))
        assert bytes(e.stats) == before


def test_shape_engine_and_budget(tmp_path):
    r = Runtime()
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=2**27))
        att, _, _, _ = attention(stack, tmp_path, r, e)
        for kind, width, rows in [(1, 6144, 7), (0, 256, 7), (0, 6144, 2561)]:
            raw = bytes(width*rows*(2 if kind == 1 else 4))
            w = weight_upload(stack, tmp_path, r, e, raw, kind, rows, width)
            before = bytes(e.stats)
            with pytest.raises(NativeError, match='requires'):
                att.projection(w)
            assert bytes(e.stats) == before
            w.close()
        other = stack.enter_context(r.engine(budget_bytes=2**20))
        raw, _ = synthetic(0)
        w = weight_upload(stack, tmp_path, r, other, raw, 0, 7)
        with pytest.raises(NativeError, match='cross-engine'):
            att.projection(w)
        # Budget includes projected output and pinned status, not just weights.
        used = e.stats.device_bytes+e.stats.pinned_bytes
        tight = stack.enter_context(r.engine(budget_bytes=used+len(raw)+35))
        att2, _, _, _ = attention(stack, tmp_path, r, tight)
        w2 = weight_upload(stack, tmp_path, r, tight, raw, 0, 7)
        before = bytes(tight.stats)
        with pytest.raises(NativeError, match='budget'):
            att2.projection(w2)
        assert bytes(tight.stats) == before


@pytest.mark.skipif(os.environ.get('MINI_LLM_QSA_PROJECTION_CHECKPOINT') != '1',
                    reason='optional bounded actual checkpoint validation')
def test_actual_checkpoint(tmp_path):
    inventory = json.loads(Path('docs/native_engine/WP2_GGUF_INVENTORY.json').read_text())
    r = Runtime()
    results = []
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=64*2**20))
        att, x, _, _ = attention(stack, tmp_path, r, e)
        store = stack.enter_context(r.gguf(inventory['shards'][0]['path']))
        names = [f'blk.{layer}.attn_output.weight' for layer in range(3, 48, 4)]
        for name in names:
            d = store.tensors[name]
            assert d.ggml_shape == (6144, 2560)
            assert d.type_id == (12 if name == names[0] else 14)
            with store.lease(name) as lease, e.upload(lease) as w, att.projection(w) as p:
                want = np.empty(2560, np.float32)
                # Only 32 decoded host rows at once (786432 bytes). No full
                # decoded host/device matrix, model, or activation quantization.
                for start in range(0, 2560, 32):
                    raw = lease.read(start*d.row_bytes, 32*d.row_bytes)
                    block = PackedGGUFWeight(np.frombuffer(raw, np.uint8),
                        (32, 6144), 'Q4_K' if d.type_id == 12 else 'Q6_K')
                    want[start:start+32] = block.dequantize_reference() @ x
                p.run()
                samples = [p.run(repeats=20) for _ in range(3)]
                got = np.empty_like(want); p.readback_for_test(got)
                np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
                err = np.abs(got-want)
                results.append(dict(descriptor=asdict(d), max_abs_error=float(err.max()),
                    max_relative_error=float((err/np.maximum(np.abs(want), 1e-30)).max()),
                    normalized_tolerance=float((err/(ATOL+RTOL*np.abs(want))).max()),
                    event_ms=samples, device_bytes=e.stats.device_bytes,
                    pinned_bytes=e.stats.pinned_bytes, peak_bytes=e.stats.peak_bytes))
    print('\nWP3_PROJECTION_RESULTS='+json.dumps(dict(rtol=RTOL, atol=ATOL,
        seed=363386, rows=2560, tile_rows=32, results=results)))
