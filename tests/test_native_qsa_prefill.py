"""Bounded component tests only. No model, server or selection approximation."""
import os
import hashlib
from pathlib import Path
import numpy as np
import pytest
from mini_llm.ops import native_qsa_prefill as native

UUID = 'GPU-bea264b4-9e76-3be5-5883-7bde3fe9f82d'


@pytest.fixture(scope='module')
def gpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != UUID:
        pytest.skip('GPU tests require CUDA_VISIBLE_DEVICES='+UUID)
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() != 1:
        pytest.skip('exactly the selected GPU0 required')
    cp.cuda.Device(0).use()
    return cp


def quantize(x):
    """Source append equation, including rounding scale BEFORE quantization."""
    grouped = x.reshape(*x.shape[:-1], 4, 64)
    scales = (np.max(np.abs(grouped), axis=-1)/np.float32(127)).astype(np.float16)
    divided = np.divide(grouped, scales.astype(np.float32)[..., None],
                        out=np.zeros_like(grouped), where=scales[..., None] > 0)
    codes = np.clip(np.rint(divided), -127, 127).astype(np.int8).reshape(x.shape)
    return codes, scales


def unpage(cp, cache, name, count):
    a = cp.asnumpy(getattr(cache, name))
    table = cp.asnumpy(cache.page_table)
    pos = np.arange(count)
    return a[table[pos//cache.page_size], :, pos % cache.page_size]


def dense_reference(q, k, v, ids, widths):
    """FP64 oracle on the SAME quantized cache/selection, not a QSA selector."""
    out = np.empty_like(q)
    for i, width in enumerate(widths):
        cells = ids[i, :width]
        for h in range(24):
            kh = h//12
            scores = k[cells, kh].astype(np.float64) @ q[i, h].astype(np.float64)/16
            prob = np.exp(scores-scores.max())
            prob /= prob.sum()
            out[i, h] = prob @ v[cells, kh].astype(np.float64)
    return out


def test_pinned_kernel_body_digests():
    # Digests computed from the GitHub API blob, independently of this port.
    text = Path(native.__file__).with_suffix('.cu').read_text()
    start = text.index('    Smem3& S = *reinterpret_cast<Smem3*>(smem_raw);')
    stop = text.index('\nextern "C" __global__ void kv_append_kernel')
    assert hashlib.sha256(text[start:stop].rstrip().encode()).hexdigest() == (
        'ed1418fb8fd4a70ecfb7cbeac9f97a3bd39bb40ce95a9b8937a4218ba197fa17')
    start = text.index('    const int64_t t = blockIdx.x;', stop)
    stop = text.index('\nextern "C" __global__ void gate_attn_kernel')
    assert hashlib.sha256(text[start:stop].rstrip().encode()).hexdigest() == (
        '366a75c789a594b83181217c41ca8281c5c02b51b1ad3a1ebe8a355197f4adc1')
    assert 'Copyright (c) 2026 Niko1221 and the Strata contributors' in text


def test_steps_cpu_contract():
    np.testing.assert_array_equal(native.make_steps(3, 3),
                                  [[3, 4, 1, 4], [4, 5, 1, 5], [5, 6, 1, 6]])
    assert native.make_steps(2050, 3)[:, 3].tolist() == [2051]*3
    assert native.make_steps(0, 0).shape == (0, 4)
    with pytest.raises(ValueError):
        native.make_steps(-1, 3)
    with pytest.raises(ValueError):
        native.make_steps(2147483647, 1)
    assert len(native.SOURCE_SHA256) == 9


def test_append_bytes_scales_paging_and_zero(gpu):
    cp = gpu
    rng = np.random.default_rng(15)
    k = rng.standard_normal((11, 2, 256), dtype=np.float32)
    v = rng.standard_normal(k.shape, dtype=np.float32)*3
    k[0] = 0
    k[1, 0, :64] = np.float32(1e-9)  # FP16 scale underflow -> zero codes
    # A representable scale, with nearest-even ties inside the group.
    v[2, 0, :64] = 0
    v[2, 0, :5] = [127, 0.5, 1.5, -0.5, -1.5]
    cache = native.allocate_cache(4, 4, page_table=cp.asarray([2, 0, 3, 1], dtype=cp.int32))
    cache.k.fill(-128)
    native.kv_append(cp.asarray(k[:5]), cp.asarray(v[:5]), 0, cache)
    native.kv_append(cp.asarray(k[5:]), cp.asarray(v[5:]), 5, cache)
    for x, key in [(k, 'k'), (v, 'v')]:
        codes, scales = quantize(x)
        np.testing.assert_array_equal(unpage(cp, cache, key, 11), codes)
        np.testing.assert_array_equal(unpage(cp, cache, key+'_scale', 11).view(np.uint16),
                                      scales.view(np.uint16))
    assert cp.asnumpy(cache.k)[1, 0, 3, 0] == -128  # outside appended range


@pytest.mark.parametrize('prefix,count', [(0, 1), (3, 5), (63, 7)])
def test_causal_chunks_batched_attention_and_gate(gpu, prefix, count):
    cp = gpu
    rng = np.random.default_rng(80+prefix)
    end = prefix+count
    k = rng.standard_normal((end, 2, 256), dtype=np.float32)*0.7
    v = rng.standard_normal(k.shape, dtype=np.float32)
    q = rng.standard_normal((count, 24, 256), dtype=np.float32)
    q[0, 0, :64] = 0
    pages = (end+3)//4
    cache = native.allocate_cache(pages, page_table=cp.asarray(np.arange(pages)[::-1].copy(), dtype=cp.int32))
    if prefix:
        native.kv_append(cp.asarray(k[:prefix]), cp.asarray(v[:prefix]), 0, cache)
    native.kv_append(cp.asarray(k[prefix:]), cp.asarray(v[prefix:]), prefix, cache)
    steps = native.make_steps(prefix, count)
    ids = np.broadcast_to(np.arange(end, dtype=np.int32), (count, end)).copy()
    kq, ks = quantize(k)
    vq, vs = quantize(v)
    kd = (kq.reshape(end, 2, 4, 64).astype(np.float32)*ks[..., None]).reshape(k.shape)
    vd = (vq.reshape(end, 2, 4, 64).astype(np.float32)*vs[..., None]).reshape(v.shape)
    expected = dense_reference(q, kd, vd, ids, steps[:, 3])
    dq, di, ds = cp.asarray(q), cp.asarray(ids), cp.asarray(steps)
    out = cp.empty_like(dq)
    got = native.prompt_attn_batch(dq, cache, di, ds, out=out, check=True)
    assert got is out
    np.testing.assert_allclose(cp.asnumpy(got), expected, atol=3e-6, rtol=3e-5)
    # Every batched row must be bit-identical to the same source kernel alone.
    for i in range(count):
        single = native.prompt_attn_batch(dq[i:i+1], cache, di[i:i+1], ds[i:i+1])
        np.testing.assert_array_equal(cp.asnumpy(single[0]), cp.asnumpy(got[i]))
    qf = rng.standard_normal((count, 24, 512), dtype=np.float32)
    qf[0, 0, 256:259] = [-90, 0, 90]
    padded = cp.full((count, 6152), np.float16(-17), dtype=cp.float16)
    gated = native.prompt_attn_batch_gated(dq, cache, di, ds, cp.asarray(qf),
                                          attn_out=out, out=padded)
    ref_gate = (expected.astype(np.float64)/(1+np.exp(-qf[:, :, 256:].astype(np.float64)))).astype(np.float16)
    # Upstream attention is FP32-level, not bitwise FP64. FP16 gate can cross a rounding boundary.
    np.testing.assert_allclose(cp.asnumpy(gated[:, :6144]).reshape(count, 24, 256),
                               ref_gate, atol=0.001, rtol=0.001)
    np.testing.assert_array_equal(cp.asnumpy(gated[:, 6144:]), -17)


def test_sparse_long_history_selection_is_caller_owned(gpu):
    cp = gpu
    rng = np.random.default_rng(19)
    end, count = 2073, 2
    cache = native.allocate_cache((end+3)//4)
    k = rng.standard_normal((end, 2, 256), dtype=np.float32)
    v = rng.standard_normal(k.shape, dtype=np.float32)
    native.kv_append(cp.asarray(k), cp.asarray(v), 0, cache)
    # Deliberately chosen sparse ids, NOT claimed to be upstream top-k output.
    ids = np.stack((np.arange(22, 2073, dtype=np.int32), np.arange(21, 2072, dtype=np.int32)))
    steps = native.make_steps(end-count, count)
    q = np.zeros((count, 24, 256), dtype=np.float32)
    vq, vs = quantize(v)
    vd = (vq.reshape(end, 2, 4, 64).astype(np.float32)*vs[..., None]).reshape(v.shape)
    # First row must not read future cell 2072; demonstrate explicit rejection.
    with pytest.raises(ValueError, match='causal'):
        native.prompt_attn_batch(cp.asarray(q), cache, cp.asarray(ids), cp.asarray(steps), check=True)
    ids[0] = np.arange(21, 2072, dtype=np.int32)
    got = native.prompt_attn_batch(cp.asarray(q), cache, cp.asarray(ids), cp.asarray(steps), check=True)
    ref = np.repeat(vd[ids].mean(axis=1, dtype=np.float64), 12, axis=1)
    np.testing.assert_allclose(cp.asnumpy(got), ref, atol=3e-7, rtol=3e-5)
    q = rng.standard_normal(q.shape, dtype=np.float32)*0.3
    kq, ks = quantize(k)
    kd = (kq.reshape(end, 2, 4, 64).astype(np.float32)*ks[..., None]).reshape(k.shape)
    ref = dense_reference(q, kd, vd, ids, steps[:, 3])
    got = native.prompt_attn_batch(cp.asarray(q), cache, cp.asarray(ids), cp.asarray(steps), check=True)
    np.testing.assert_allclose(cp.asnumpy(got), ref, atol=3e-6, rtol=3e-5)


def test_mapped_host_resident_mirror_and_bounded_stage(gpu):
    cp = gpu
    rng = np.random.default_rng(51)
    k = rng.standard_normal((13, 2, 256), dtype=np.float32)
    v = rng.standard_normal(k.shape, dtype=np.float32)
    resident = native.allocate_cache(2, page_table=cp.asarray([-1, 1, -1, 0], dtype=cp.int32))
    stage = native.allocate_cache(4)
    with native.HostQ8Cache(4) as host:
        for name in ('k', 'v', 'k_scale', 'v_scale'):
            getattr(host, name).fill(0)
        st = cp.cuda.Stream(non_blocking=True)
        st.wait_event(cp.cuda.get_current_stream().record())
        with st:
            dk, dv = cp.asarray(k), cp.asarray(v)
            native.kv_append(dk, dv, 0, resident, host=host, stage=stage, stream=st)
        st.synchronize()
        for x, name in [(k, 'k'), (v, 'v')]:
            codes, scales = quantize(x)
            # Identity layout is page/head/cell, not cell/head.
            for pos in range(13):
                np.testing.assert_array_equal(getattr(host, name)[pos//4, :, pos%4], codes[pos])
                np.testing.assert_array_equal(getattr(host, name+'_scale')[pos//4, :, pos%4], scales[pos])
            np.testing.assert_array_equal(unpage(cp, stage, name, 13), codes)
            for pos in (4, 5, 6, 7, 12):
                slot = [ -1, 1, -1, 0 ][pos//4]
                np.testing.assert_array_equal(cp.asnumpy(getattr(resident, name))[slot, :, pos%4], codes[pos])
        copied = native.allocate_cache(4)
        native.stage_from_host(host, copied, 13, stream=st)
        st.synchronize()
        for name in ('k', 'v', 'k_scale', 'v_scale'):
            np.testing.assert_array_equal(cp.asnumpy(getattr(copied, name)), getattr(host, name))
        with pytest.raises(ValueError, match='capacity'):
            native.stage_from_host(host, native.allocate_cache(2), 9)
        q = cp.zeros((1, 24, 256), dtype=cp.float32)
        args = (cp.arange(13, dtype=cp.int32)[None], cp.asarray(native.make_steps(12, 1)))
        a = native.prompt_attn_batch(q, stage, *args, check=True)
        b = native.prompt_attn_batch(q, copied, *args, check=True)
        np.testing.assert_array_equal(cp.asnumpy(a), cp.asnumpy(b))


def test_dtype_shape_and_selection_rejections(gpu):
    cp = gpu
    cache = native.allocate_cache(2)
    k = cp.zeros((1, 2, 256), cp.float32)
    with pytest.raises(TypeError):
        native.kv_append(k.astype(cp.float16), k, 0, cache)
    with pytest.raises(ValueError):
        native.kv_append(k, k, 8, cache)
    with pytest.raises(ValueError):
        native.kv_append(k[:, :, ::-1], k, 0, cache)
    native.kv_append(k, k, 0, cache)
    q = cp.zeros((1, 24, 256), cp.float32)
    steps = cp.asarray(native.make_steps(0, 1))
    with pytest.raises(TypeError):
        native.prompt_attn_batch(q, cache, cp.zeros((1, 1), cp.int64), steps)
    with pytest.raises(ValueError, match='causal'):
        native.prompt_attn_batch(q, cache, cp.ones((1, 1), cp.int32), steps, check=True)
    steps[0, 3] = 2
    with pytest.raises(ValueError, match='causal'):
        native.prompt_attn_batch(q, cache, cp.zeros((1, 1), cp.int32), steps, check=True)
