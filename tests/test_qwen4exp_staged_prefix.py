"""Incremental allcells staging and exact chunk-filtered GGML heap selection."""
import ctypes
from dataclasses import replace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.external.qwen4exp_attention import Qwen4ExpAttention, select_cells
from mini_llm.external.qwen4exp_model import Qwen4ExpGenerationState
from mini_llm.external.qwen4exp_prompt_checkpoint import PromptCheckpoint
from mini_llm.external import qwen4exp_hoststate as hs
from test_0208_qwen4exp_attention import compiled_source  # Reuse pinned native oracle.
from test_0230_qsa_gpu_decode import fixture


def track_copies(monkeypatch):
    copies = []
    original = hs._copy_rows

    def tracked(source, target):
        copies.append(source.shape[0])
        return original(source, target)

    monkeypatch.setattr(hs, '_copy_rows', tracked)
    return copies


def assert_prefix(state, length):
    assert state.staged_prefix_length == length
    np.testing.assert_array_equal(asnumpy(state.staged_keys[:length]), state.keys[:length])
    np.testing.assert_array_equal(asnumpy(state.staged_values[:length]), state.values[:length])


def test_prefix_backfills_only_missing_rows_and_gather_invalidates(monkeypatch):
    c, weights, rng = fixture()
    model = Qwen4ExpAttention(c, **weights, qsa_state_storage='host')
    state = model.create_state(23)
    state.length = 9  # Synthetic prepopulated committed history.
    state.keys[:9] = rng.normal(size=state.keys[:9].shape).astype(np.float32)
    state.values[:9] = rng.normal(size=state.values[:9].shape).astype(np.float32)
    copies = track_copies(monkeypatch)
    hs.stage_prefix(state, 5)
    assert copies == [5, 5]
    assert_prefix(state, 5)
    hs.stage_prefix(state, 9)
    assert copies == [5, 5, 4, 4]
    assert_prefix(state, 9)
    hs.stage_prefix(state, 9)
    assert copies == [5, 5, 4, 4]
    hs.gather(state, xp.asarray([2, 7], dtype=xp.int64))
    assert state.staged_prefix_length == 0
    hs.stage_prefix(state, 9)
    assert copies[-2:] == [9, 9]
    assert_prefix(state, 9)
    model.reset(state)
    assert state.length == state.staged_prefix_length == 0


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='GPU allcells dispatch')
def test_normal_allcells_decode_never_uploads_old_kv(monkeypatch):
    c, weights, rng = fixture()
    c = replace(c, index_topk=2048)
    host = Qwen4ExpAttention(c, **weights, fast_decode=True, qsa_state_storage='host')
    control = Qwen4ExpAttention(c, **weights, fast_decode=True)
    a, b = host.create_state(1025), control.create_state(1025)
    copies = track_copies(monkeypatch)
    for t in range(1025):
        x = xp.asarray(rng.normal(size=7), dtype=xp.float32)
        expected = control.step(x, b, position=t)
        actual = host.step(x, a, position=t)
        np.testing.assert_allclose(asnumpy(actual), asnumpy(expected), atol=2e-5, rtol=3e-5)
        assert a.staged_prefix_length == t+1
    assert copies == []  # Raw-index block pooling uploads are intentionally separate.
    assert_prefix(a, 1025)
    for name in ('keys', 'values', 'raw_index', 'pooled'):
        np.testing.assert_array_equal(asnumpy(getattr(a, name)), asnumpy(getattr(b, name)))


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='GPU allcells dispatch')
def test_synthetic_prepopulated_prefix_uploads_once_and_then_appends(monkeypatch):
    c, weights, rng = fixture()
    host = Qwen4ExpAttention(c, **weights, fast_decode=True, qsa_state_storage='host')
    control = Qwen4ExpAttention(c, **weights, fast_decode=True)
    a, b = host.create_state(23), control.create_state(23)
    for name in ('keys', 'values', 'raw_index', 'pooled'):
        rows = getattr(a, name)
        value = rng.normal(size=rows.shape).astype(np.float32)
        rows[...] = value if isinstance(rows, np.ndarray) else xp.asarray(value)
        getattr(b, name)[...] = xp.asarray(value)
    a.length = b.length = 5
    copies = track_copies(monkeypatch)
    for t in range(5, 8):
        x = xp.asarray(rng.normal(size=7), dtype=xp.float32)
        expected = control.step(x, b, position=t)
        actual = host.step(x, a, position=t)
        np.testing.assert_allclose(asnumpy(actual), asnumpy(expected), atol=2e-5, rtol=3e-5)
        assert_prefix(a, t+1)
    assert copies == [5, 5]


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='GPU allcells dispatch')
def test_diagnostic_gather_invalidates_then_allcells_backfills(monkeypatch):
    c, weights, rng = fixture()
    host = Qwen4ExpAttention(c, **weights, fast_decode=True, qsa_state_storage='host')
    state = host.create_state(23)
    copies = track_copies(monkeypatch)
    x = xp.asarray(rng.normal(size=7), dtype=xp.float32)
    for t in range(3):
        host.step(x, state, position=t)
    assert copies == []
    host.step(x, state, position=3, diagnostics=True)
    assert state.staged_prefix_length == 0
    copies.clear()
    host.step(x, state, position=4)
    assert copies == [4, 4]
    assert_prefix(state, 5)
    for t in range(5, 17):
        host.step(x, state, position=t)
    # n=17 exceeds this fixture's 16-cell staging/selection width.
    assert state.staged_prefix_length == 0


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='GPU allcells dispatch')
@pytest.mark.parametrize('bad', ['hidden', 'value', 'output'])
def test_failed_validation_never_appends_current_staged_row(bad):
    c, weights, rng = fixture()
    host = Qwen4ExpAttention(c, **weights, fast_decode=True, qsa_state_storage='host')
    state = host.create_state(23)
    x = xp.asarray(rng.normal(size=7), dtype=xp.float32)
    for t in range(3):
        host.step(x, state, position=t)
    before = [asnumpy(a).copy() for a in (state.staged_keys, state.staged_values)]
    if bad == 'hidden':
        x[:] = xp.nan
    else:
        _, size = host.projections[bad]
        host.projections[bad] = (lambda _: xp.full(size, xp.nan, xp.float32), size)
    status = xp.zeros((), dtype=xp.uint32)
    with pytest.raises(ValueError, match='finite'):
        host.step(x, state, position=3, validation_status=status)
    assert int(status.item()) == 1
    assert state.length == state.staged_prefix_length == 3
    for actual, expected in zip((state.staged_keys, state.staged_values), before):
        np.testing.assert_array_equal(asnumpy(actual), expected)
    assert not state.keys[3].any() and not state.values[3].any()


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='GPU allcells dispatch')
def test_checkpoint_restore_invalidates_cached_suffix_and_replays_exactly(monkeypatch):
    c, weights, rng = fixture()
    host = Qwen4ExpAttention(c, **weights, fast_decode=True, qsa_state_storage='host')
    control = Qwen4ExpAttention(c, **weights, fast_decode=True)
    qsa, baseline = host.create_state(23), control.create_state(23)
    state = Qwen4ExpGenerationState(object(), 23, {}, {0: qsa}, {})
    hidden = xp.asarray(rng.normal(size=(7, 7)), dtype=xp.float32)
    for t in range(3):
        logits = host.step(hidden[t], qsa, position=t)
        control.step(hidden[t], baseline, position=t)
        state.length += 1
    checkpoint = PromptCheckpoint.capture(state, [0, 1, 2], logits)
    assert checkpoint is not None
    for t in range(3, 6):
        host.step(hidden[t], qsa, position=t)
        state.length += 1
    assert qsa.staged_prefix_length == 6
    assert checkpoint.restore(state, [0, 1, 2], list(range(6))) is not None
    assert qsa.length == state.length == 3 and qsa.staged_prefix_length == 0
    copies = track_copies(monkeypatch)
    actual = host.step(hidden[6], qsa, position=3)
    expected = control.step(hidden[6], baseline, position=3)
    np.testing.assert_allclose(asnumpy(actual), asnumpy(expected), atol=2e-5, rtol=3e-5)
    assert copies == [3, 3]
    assert_prefix(qsa, 4)


def sequential_heap(values, width):
    """Pre-optimization Python transcription, independent of chunk filtering."""
    ids = list(range(width))

    def adjust(hole, value):
        top, child = hole, hole
        while child < (width-1)//2:
            child = 2*(child+1)
            if values[ids[child]] > values[ids[child-1]]:
                child -= 1
            ids[hole] = ids[child]
            hole = child
        if width % 2 == 0 and child == (width-2)//2:
            child = 2*(child+1)
            ids[hole] = ids[child-1]
            hole = child-1
        parent = (hole-1)//2
        while hole > top and values[ids[parent]] > values[value]:
            ids[hole] = ids[parent]
            hole, parent = parent, (parent-1)//2
        ids[hole] = value

    for parent in range((width-2)//2, -1, -1):
        adjust(parent, ids[parent])
    for candidate in range(width, values.size):
        if values[candidate] > values[ids[0]]:
            adjust(0, candidate)
    return np.sort(ids)


def selection_cases(n):
    rng = np.random.default_rng(819)
    yield rng.normal(size=n).astype(np.float32)
    yield rng.integers(-3, 4, size=n).astype(np.float32)
    yield np.arange(n, dtype=np.float32)
    yield -np.arange(n, dtype=np.float32)
    tied = np.zeros(n, np.float32)
    tied[-3:] = np.float32(1e9)
    yield tied
    extremes = rng.choice(np.array([-np.inf, -0., 0., 1., np.inf], np.float32), n)
    yield extremes


@pytest.mark.parametrize('n,width', [(1, 1), (8195, 1), (8195, 17), (8195, 8195),
                                    (262144, 2051)])
def test_chunk_filtered_selection_exact_sequential_membership(n, width):
    for values in selection_cases(n):
        expected = sequential_heap(values, width)
        actual = select_cells(xp.asarray(values), width, 'ggml_cpu')
        np.testing.assert_array_equal(asnumpy(actual), expected)


def test_long_selection_matches_pinned_native_partial_sort(compiled_source):
    lib = compiled_source
    lib.selection.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
        ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int]
    for values in selection_cases(262144):
        expected = np.empty(2051, np.int32)
        lib.selection(values, values.size, expected.size, expected, 0)
        actual = select_cells(xp.asarray(values), expected.size, 'ggml_cpu')
        np.testing.assert_array_equal(asnumpy(actual), np.sort(expected))
