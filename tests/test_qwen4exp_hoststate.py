"""Host/device FP32 parity; long histories below are synthetic, not prefilled."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.external.qwen4exp_attention import Qwen4ExpAttention, select_cells
from mini_llm.external import qwen4exp_hoststate as hs
from mini_llm.external import qwen4exp_weights as storage
from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel, Qwen4ExpLayer
from test_0230_qsa_gpu_decode import fixture


@pytest.mark.parametrize('block', [1, 4])
@pytest.mark.parametrize('fast', [False, True])
def test_boundary_parity_and_exact_history(block, fast):
    c, weights, rng = fixture(block=block)
    device = Qwen4ExpAttention(c, **weights, fast_decode=fast)
    host = Qwen4ExpAttention(c, **weights, fast_decode=fast, qsa_state_storage='host')
    a, b = device.create_state(27), host.create_state(27)
    assert b.keys.dtype == np.float32 and isinstance(b.keys, np.ndarray)
    assert b.staged_keys.shape[0] == min(27, c.index_topk+c.block_size-1)
    for t in range(27):
        x = xp.asarray(rng.normal(size=7), dtype=xp.float32)
        diagnostic = t in (0, 3, 15, 16, 17, 23, 26)
        expected = device.step(x, a, position=t, diagnostics=diagnostic)
        actual = host.step(x, b, position=t, diagnostics=diagnostic)
        if diagnostic:
            expected, ed = expected
            actual, ad = actual
            np.testing.assert_array_equal(asnumpy(ad['selected_ids']), asnumpy(ed['selected_ids']))
            # Explicitly detect a changed dot reduction, not only output drift.
            np.testing.assert_allclose(asnumpy(ad['block_scores']), asnumpy(ed['block_scores']),
                                       atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(asnumpy(actual), asnumpy(expected), atol=2e-5, rtol=3e-5)
        for name in ('keys', 'values', 'raw_index', 'pooled'):
            np.testing.assert_array_equal(asnumpy(getattr(a, name)), asnumpy(getattr(b, name)))
    assert a.length == b.length == 27
    host.reset(b)
    assert b.length == 0
    for name in ('keys', 'values', 'raw_index', 'pooled'):
        assert not asnumpy(getattr(b, name)).any()


@pytest.mark.parametrize('bad', ['hidden', 'key', 'index_key', 'index_query', 'output'])
def test_failure_is_transactional(bad):
    c, weights, rng = fixture()
    model = Qwen4ExpAttention(c, **weights, fast_decode=True, qsa_state_storage='host')
    state = model.create_state(8)
    x = xp.asarray(rng.normal(size=7), dtype=xp.float32)
    for t in range(3):
        model.step(x, state, position=t)
    before = {n: asnumpy(getattr(state, n)).copy() for n in ('keys', 'values', 'raw_index', 'pooled')}
    if bad == 'hidden':
        x[:] = xp.nan
    else:
        _, size = model.projections[bad]
        model.projections[bad] = (lambda _: xp.full(size, xp.nan, xp.float32), size)
    # Diagnostics force index query validation even in the early allcells path.
    with pytest.raises(ValueError, match='finite|NaN'):
        model.step(x, state, position=3, diagnostics=True)
    assert state.length == 3
    for name, value in before.items():
        np.testing.assert_array_equal(asnumpy(getattr(state, name)), value)


@pytest.mark.parametrize('n', [262141, 262144])
@pytest.mark.parametrize('production_geometry', [False, True])
def test_synthetic_near_262k_retrieval(n, production_geometry):
    c, weights, rng = fixture(heads=24 if production_geometry else 4,
                              dim=256 if production_geometry else 8)
    if production_geometry:
        c = replace(c, index_heads=4, index_dim=128, index_topk=2048)
        weights.update(index_query=xp.asarray(rng.normal(0, .2, (512, 7)), dtype=xp.float32),
                       index_key=xp.asarray(rng.normal(0, .2, (128, 7)), dtype=xp.float32),
                       gamma_index_q=xp.ones(128, xp.float32),
                       gamma_index_k=xp.ones(128, xp.float32))
    model = Qwen4ExpAttention(c, **weights, qsa_state_storage='host')
    state = model.create_state(262144)
    # Deliberately synthetic: a sparse old marker plus a large tied zero history.
    # Exercise actual step selection, staging, attention, and pool-boundary commit.
    t = n-1
    state.length = t
    marker = 123
    state.pooled[marker//c.block_size] = 100
    state.keys[marker] = .25
    state.values[marker] = 7
    state.raw_index[max(0, t-3):t] = .1
    x = xp.asarray(rng.normal(size=7), dtype=xp.float32)
    # Ensure nonnegative index heads so the old marker wins unambiguously.
    iq_size = c.index_heads*c.index_dim
    model.projections['index_query'] = (lambda _: xp.ones(iq_size, xp.float32), iq_size)
    # Position-independent index query for an independently known membership.
    rotate = model._rotate
    model._rotate = lambda a, p: a if a.shape[-1] == c.index_dim else rotate(a, p)
    result, diagnostic = model.step(x, state, position=t, diagnostics=True)
    scores = asnumpy(diagnostic['cell_scores'])
    assert scores.shape == (n,)
    assert scores[marker] > scores[0]
    expected = select_cells(xp.asarray(scores), c.index_topk+c.block_size-1, 'ggml_cpu')
    np.testing.assert_array_equal(asnumpy(diagnostic['selected_ids']), asnumpy(expected))
    assert marker in asnumpy(expected)
    assert np.isfinite(asnumpy(result)).all()
    assert state.length == n
    gathered, values = hs.gather(state, xp.asarray([marker], xp.int64))
    np.testing.assert_array_equal(asnumpy(gathered[0]), state.keys[marker])
    np.testing.assert_array_equal(asnumpy(values[0]), state.values[marker])


def test_scores_random_and_ties_match_reference():
    rng = np.random.default_rng(818)
    for dim in (6, 128):
        iq = xp.asarray(rng.normal(size=(4, dim)), dtype=xp.float32)
        pooled = xp.asarray(rng.normal(size=(513, dim)), dtype=xp.float32)
        pooled[1:10] = pooled[0]  # Exact repeated-block ties.
        expected = xp.empty(513, xp.float32)
        for b in range(513):
            dots = xp.maximum(iq @ pooled[b], xp.float32(0))
            value = dots[0].copy()
            for h in range(1, 4):
                value += dots[h]
            expected[b] = value
        actual = hs.block_scores(iq, pooled, 513, 0, None, None)
        error = np.abs(asnumpy(actual)-asnumpy(expected))
        print(f'index_dim={dim} FP32 score max_abs={error.max()}')
        np.testing.assert_allclose(asnumpy(actual), asnumpy(expected), atol=2e-5, rtol=3e-6)
        for width in (3, 17, 211):
            np.testing.assert_array_equal(asnumpy(select_cells(actual, width)),
                                          asnumpy(select_cells(expected, width)))


@pytest.mark.parametrize('tail', [0, 1, 3])
def test_cpu_sequential_score_ties_and_head_reduction(tail):
    # Exactly representable dot products isolate index-head accumulation from
    # BLAS backend roundoff. Tiny terms disappear in left-to-right FP32 order;
    # a reassociated head reduction would change the scores and tie membership.
    dim, count = 128, 521
    query = np.zeros((4, dim), np.float32)
    query[:, :4] = np.eye(4, dtype=np.float32)
    blocks = np.zeros((count, dim), np.float32)
    blocks[::3, :4] = [1e8, 1, 2, 3]
    blocks[1::3, :4] = [1e8, 3, 2, 1]
    blocks[2::3, :4] = [np.nextafter(np.float32(1e8), np.float32(np.inf)), 0, 0, 0]
    dead = np.zeros(dim, np.float32)
    dead[:4] = [1, 2, 3, 4]
    reference = []
    for row in list(blocks) + ([dead] if tail else []):
        dots = np.maximum(query @ row, np.float32(0))
        score = dots[0].copy()
        for h in range(1, 4):
            score += dots[h]
        reference.append(score)
    actual = hs.block_scores(xp.asarray(query), xp.asarray(blocks), count, tail,
                             xp.asarray(dead), None)
    np.testing.assert_array_equal(asnumpy(actual), np.asarray(reference))
    scores = xp.repeat(actual, 4)[:count*4+tail]
    expected = np.repeat(np.asarray(reference), 4)[:count*4+tail]
    scores[count*4:] += xp.float32(1e9)
    expected[count*4:] += np.float32(1e9)
    np.testing.assert_array_equal(asnumpy(scores), expected)
    for width in (3, 17, 2048):
        np.testing.assert_array_equal(asnumpy(select_cells(scores, width, 'ggml_cpu')),
                                      asnumpy(select_cells(xp.asarray(expected), width, 'ggml_cpu')))


def test_model_host_state_rejects_poison_and_skips_graphs(monkeypatch):
    c, weights, rng = fixture()
    attention = Qwen4ExpAttention(c, **weights, fast_decode=True, qsa_state_storage='host')
    hc = SimpleNamespace(initialize=lambda x: xp.tile(x, (4, 1)),
        read=lambda x: (x[0], None), write=lambda x, y, g: x+y[None], collapse=lambda x: x[0])
    model = Qwen4ExpGGUFModel.from_components(width=7, vocab=7,
        blocks=[Qwen4ExpLayer(hc, attention, hc, lambda x: xp.zeros_like(x), False)],
        final_hc=hc, embedding=lambda t: xp.ones(7, xp.float32), output=lambda x: x,
        max_length=8)
    monkeypatch.setenv('MINI_LLM_TOKEN_GRAPH', '1')
    monkeypatch.setenv('MINI_LLM_QSA_BLOCK_GRAPH', '1')
    if BACKEND_NAME == 'cupy':
        model._qsa_validation_status = xp.zeros(1, xp.uint32)
        model._qsa_validation_views = {0: model._qsa_validation_status.reshape(())}
    state = model.create_generation_state(1, 8)
    assert np.isfinite(asnumpy(model.step(1, state))).all()
    assert model._orphaned_bound_state_bytes() == 0
    size = attention.projections['value'][1]
    attention.projections['value'] = (lambda _: xp.full(size, xp.nan, xp.float32), size)
    with pytest.raises(ValueError, match='finite'):
        model.step(1, state)
    assert state.failed and state.length == state.qsa[0].length == 1
    with pytest.raises(RuntimeError, match='failed'):
        model.step(1, state)
    model.close()


def test_storage_validation_and_default():
    c, weights, _ = fixture()
    assert Qwen4ExpAttention(c, **weights).qsa_state_storage == 'device'
    with pytest.raises(ValueError, match='qsa_state_storage'):
        Qwen4ExpAttention(c, **weights, qsa_state_storage='disk')


@pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='Mock GPU admission without allocations')
def test_262k_memory_admission_and_split_reservation(monkeypatch):
    GiB = 1024**3
    monkeypatch.setattr(storage, 'BACKEND_NAME', 'cupy')
    monkeypatch.setattr(storage, 'xp', SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(memGetInfo=lambda: (16*GiB, 16*GiB)))))
    w = storage.Qwen4ExpWeights.__new__(storage.Qwen4ExpWeights)
    w._worker_capacity = 2
    w.headers = SimpleNamespace(context=262144, width=3072, vocab=151936,
        dense_bytes=2*GiB, constructor_bytes=32*1024**2,
        recurrent=(True, True, True, False)*12, ple_layers=(),
        mixer=SimpleNamespace(conv_kernel_size=4, qkv_channels=8192, n_value_heads=32, head_dim=128),
        attention=SimpleNamespace(n_kv_heads=2, head_dim=256, index_dim=128,
                                 block_size=4, n_q_heads=24, index_topk=2048))
    w.store = SimpleNamespace(tensors={})
    w.index = SimpleNamespace(d_ff=1024)
    w.plan, w.device_names = None, frozenset()
    budgets = dict(cpu_workers=2, host_budget_bytes=14*GiB,
        static_device_budget_bytes=3*GiB, device_budget_bytes=14*GiB,
        cache_bytes=6*GiB, headroom_bytes=256*1024**2)
    with pytest.raises(MemoryError, match='device budget'):
        w.memory_plan(262144, **budgets)
    plan = w.memory_plan(262144, qsa_state_storage='host', **budgets)
    assert plan['qsa_host_bytes'] == int(13.5*GiB)
    assert plan['qsa_device_bytes'] < .48*GiB
    assert plan['device_reserved_bytes'] < 14*GiB
    assert plan['host_reserved_bytes'] < 14*GiB
    assert plan['state_bytes'] == plan['host_state_bytes']+plan['device_state_bytes']
    print({k: plan[k] for k in ('qsa_host_bytes', 'qsa_device_bytes',
          'host_reserved_bytes', 'device_reserved_bytes')})
    with pytest.raises(MemoryError, match='host budget'):
        w.memory_plan(262144, qsa_state_storage='host', **dict(budgets, host_budget_bytes=13*GiB))


@pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='Synthetic CPU weight-loader fixture')
def test_factory_forwards_host_storage_and_memory_plan():
    from test_0209_qwen4exp_model import TinyStore, factory
    model = factory(TinyStore(), qsa_state_storage='host')
    try:
        plan = model.memory_plan(1, 16)
        assert plan['qsa_state_storage'] == model.plan['qsa_state_storage'] == 'host'
        state = model.create_generation_state(1, 16)
        assert state.host_reserved_bytes == plan['host_state_bytes']
        assert state.device_reserved_bytes == plan['device_state_bytes'] == 0
        assert state.reserved_bytes == state.host_reserved_bytes
        assert all(q.qsa_state_storage == 'host' for q in state.qsa.values())
        assert plan == model.memory_plan(1, 16)
        with pytest.raises(MemoryError, match='aggregate state reservation'):
            model.create_generation_state(1, 1)
    finally:
        model.close()


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Actual nondefault CUDA stream')
def test_nondefault_stream_commit_and_gather():
    c, weights, rng = fixture()
    model = Qwen4ExpAttention(c, **weights, fast_decode=True, qsa_state_storage='host')
    state = model.create_state(24)
    control = Qwen4ExpAttention(c, **weights, fast_decode=True)
    device = control.create_state(24)
    hidden = rng.normal(size=(24, 7)).astype(np.float32)
    expected = xp.stack([control.step(xp.asarray(x), device, position=t)
                         for t, x in enumerate(hidden)])
    xp.cuda.get_current_stream().synchronize()
    streams = [xp.cuda.Stream(non_blocking=True), xp.cuda.Stream(non_blocking=True)]
    outputs = []
    for t, x in enumerate(hidden):
        # Alternate nonblocking streams without a caller-side state fence.
        # Host commit publishes both host history and device pools synchronously.
        with streams[t % 2]:
            out = model.step(xp.asarray(x), state, position=t)
            outputs.append(out.copy())
    for stream in streams:
        stream.synchronize()
    np.testing.assert_allclose(asnumpy(xp.stack(outputs)), asnumpy(expected), atol=2e-5, rtol=3e-5)
    assert state.length == 24
    for name in ('keys', 'values', 'raw_index', 'pooled'):
        np.testing.assert_array_equal(asnumpy(getattr(state, name)), asnumpy(getattr(device, name)))
