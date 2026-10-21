"""CPU correctness contract for the 0083B experimental historical reader."""
from dataclasses import replace

import numpy as np
import pytest

from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.ops.historical_schedule import historical_slots, source_indices, sweep_coverage
from mini_llm.ops.historical_read import HistoricalReadMemory
from mini_llm.backend import RandomStream
from mini_llm.parameter import Parameter


@pytest.mark.parametrize('layers,heads,per_slot,period',[(13,12,5120,384),(32,8,3840,512)])
def test_complete_sweep(layers, heads, per_slot, period):
    slots = historical_slots(layers, heads)
    selected = sweep_coverage(layers, heads)
    assert len(selected) == 61440
    assert sorted(selected) == list(range(61440))
    blocks = [s for row in slots for s in row if s.kind == 'block']
    sparse = [s for row in slots for s in row if s.kind == 'sparse']
    learned = [s for row in slots for s in row if s.kind == 'learned']
    assert len(blocks) * 32 == period
    assert len(sparse) == 12
    assert len(learned) == 12
    assert all(len(source_indices(s, 61440)) == per_slot for s in blocks)
    assert len({s.phase for s in sparse}) == 12
    for start in (0, 1, 30, 31, 32, 129, 61439, 61440):
        short = sweep_coverage(layers, heads, history_length=61440, valid_start=start)
        assert sorted(short) == list(range(start, 61440))


@pytest.mark.parametrize('deep',(False, True))
def test_preset_schedules_and_counts(deep):
    c = (ModelConfig.ensemble_deep_512_dense_historical_64k() if deep
         else ModelConfig.ensemble_wide_768_dense_historical_64k())
    assert c.memory_context.active_length == 4096
    assert c.memory_context.distant_memory_length == 61440
    assert c.memory_context.searchable_blocks == 1920
    assert c.memory_context.block_size == 32
    assert c.memory_context.top_k_blocks == 8
    assert c.estimated_parameter_count() > 106_000_000
    slots = historical_slots(c.n_layers, c.n_q_heads)
    assert len(slots) == c.n_layers
    for row in slots:
        assert len({x.head for x in row}) == len(row)
        assert all(0 <= x.head < c.n_q_heads for x in row)
    if deep:
        assert [r[0].layer for r in slots if any(s.kind == 'learned' for s in r)] == [6,9,12,15,18,21]
    else:
        assert [r[0].layer for r in slots if any(s.kind == 'learned' for s in r)] == [5,6,7,8,9,10]


def small_config(deep=False):
    c = (ModelConfig.ensemble_deep_512_dense_historical_64k() if deep
         else ModelConfig.ensemble_wide_768_dense_historical_64k())
    mc = replace(c.memory_context, memory_length=264, target_length=8,
                 recent_length=8, router_query_length=8)
    return replace(c, tokenizer_vocab_size=128, d_ff=64, dtype='float32', memory_context=mc)


@pytest.mark.parametrize('deep',(False, True))
@pytest.mark.parametrize('recompute',(False, True))
def test_trainable_forward_backward_and_history_grad(monkeypatch, deep, recompute):
    monkeypatch.setenv('MINI_LLM_HISTORY_RECOMPUTE', str(int(recompute)))
    c = small_config(deep)
    m = DecoderLanguageModel(c, rng_seed=77)
    assert sum(p.size for p in m.parameters()) == c.estimated_parameter_count()
    ids = np.random.default_rng(78).integers(0, 128, (1, c.memory_context.memory_length))
    y, cache = m.forward_body(ids, memory_metadata={'history_valid_starts': np.array([128])})
    assert y.shape == (1, 8, c.d_model)
    state = cache['memory_context_cache']['state']
    assert (state.encoder_cache is None) == recompute
    m.backward_body(np.ones_like(y) * .001, cache)
    assert np.linalg.norm(m.historical_memory.key.grad) > 0
    assert np.linalg.norm(m.historical_memory.in_proj.grad) > 0
    assert np.linalg.norm(m.historical_memory.router_key.grad) > 0
    assert np.isfinite(m.historical_memory.key.grad).all()


def test_future_current_token_changes_cannot_change_earlier_logits_or_route():
    c = small_config()
    m = DecoderLanguageModel(c, rng_seed=9)
    original = np.random.default_rng(10).integers(0, 128, (1, 264))
    altered = original.copy()
    altered[0, 256+5:] = (altered[0, 256+5:] + 13) % 128
    metadata = {'history_valid_starts': np.array([0])}
    ya, ca = m.forward_body(original, memory_metadata=metadata)
    yb, cb = m.forward_body(altered, memory_metadata=metadata)
    assert np.allclose(ya[:, :5], yb[:, :5], atol=1e-6, rtol=1e-6)
    for i in (5, 6, 7, 8, 9, 10):
        a = ca['block_caches'][i]['attn_cache']['historical_read_cache']
        b = cb['block_caches'][i]['attn_cache']['historical_read_cache']
        assert np.array_equal(a[1][0], b[1][0])


def test_invalid_history_cannot_influence_logits_but_valid_history_can():
    c = small_config()
    m = DecoderLanguageModel(c, rng_seed=5)
    a = np.random.default_rng(6).integers(0, 128, (1, 264))
    b = a.copy(); b[0, 0] = (b[0, 0] + 7) % 128
    meta = {'history_valid_starts': np.array([128])}
    y, _ = m.forward_body(a, memory_metadata=meta)
    yinval, _ = m.forward_body(b, memory_metadata=meta)
    assert np.array_equal(y, yinval)
    c = a.copy(); c[0, 255] = (c[0, 255] + 9) % 128
    yvalid, _ = m.forward_body(c, memory_metadata=meta)
    assert not np.array_equal(y, yvalid)


def test_history_encoder_recompute_gradient_parity(monkeypatch):
    c = small_config()
    ids = np.random.default_rng(100).integers(0, 128, (1, 264))
    metadata = {'history_valid_starts': np.array([64])}
    grads = []
    outputs = []
    for recompute in (False, True):
        monkeypatch.setenv('MINI_LLM_HISTORY_RECOMPUTE', str(int(recompute)))
        m = DecoderLanguageModel(c, rng_seed=23)
        y, cache = m.forward_body(ids, memory_metadata=metadata)
        outputs.append(y.copy())
        m.backward_body(np.ones_like(y)*.001, cache)
        grads.append({p.name: p.grad.copy() for p in m.historical_memory.parameters()})
    assert np.array_equal(outputs[0], outputs[1])
    for name in grads[0]:
        assert np.allclose(grads[0][name], grads[1][name], atol=1e-6, rtol=1e-6), name


def test_checkpoint_parameter_roundtrip(tmp_path):
    from mini_llm.checkpoint import save_checkpoint, load_checkpoint
    c = small_config()
    m = DecoderLanguageModel(c, rng_seed=17)
    params = {p.name: p.data for p in m.parameters()}
    assert len(params) == len(m.parameters())
    save_checkpoint(tmp_path / 'hist', params)
    loaded, _, _ = load_checkpoint(tmp_path / 'hist', skip_optimizer=True)
    assert set(params) == set(loaded)
    for name in params:
        assert np.array_equal(params[name], loaded[name]), name


def test_history_value_gradient_matches_finite_difference():
    c = small_config()
    m = DecoderLanguageModel(c, rng_seed=103)
    mem = m.historical_memory
    ids = np.random.default_rng(104).integers(0, 128, (1, 256))
    q = np.random.default_rng(105).normal(size=(1, 4, 12, 64)).astype('float32')
    # Layer 2 has deterministic heads; no discrete learned route decisions.
    state = mem.open(ids, np.array([0]))
    baseline, cache = mem.apply(np.zeros_like(q), q, 2, state)
    dcontext = np.zeros_like(q)
    dcontext[:, :, 6:8] = 1
    _, dq = mem.backward_reads(dcontext, q, 2, state, cache)
    mem.close(state)
    analytic = mem.value.grad[0, 0]
    assert np.linalg.norm(dq) > 0
    initial = float(mem.value.data[0, 0])
    numerical = []
    eps = .003
    for sign in (+1, -1):
        mem.value.data[0, 0] = initial + sign*eps
        st = mem.open(ids, np.array([0]))
        output, _ = mem.apply(np.zeros_like(q), q, 2, st, return_cache=False)
        numerical.append(float(np.sum(output[:, :, 6:8])))
    mem.value.data[0, 0] = initial
    fd = (numerical[0] - numerical[1]) / (2*eps)
    assert np.isclose(analytic, fd, rtol=.03, atol=1e-4), (analytic, fd)



def test_batched_learned_routes_match_period_loop(monkeypatch):
    c = small_config()
    rng = np.random.default_rng(18)
    ids = rng.integers(0, 128, (2, 256))
    q = rng.normal(size=(2, 256, 12, 64)).astype('float32')
    context = rng.normal(size=q.shape).astype('float32')
    upstream = rng.normal(size=q.shape).astype('float32')
    results = []
    for enabled in (0, 1):
        monkeypatch.setenv('MINI_LLM_HISTORY_BATCH_ROUTES', str(enabled))
        model = DecoderLanguageModel(c, rng_seed=19)
        mem = model.historical_memory
        state = mem.open(ids, np.array([0, 32]))
        output, cache = mem.apply(context, q, 5, state)
        dcontext, dq = mem.backward_reads(upstream, q, 5, state, cache)
        mem.close(state)
        results.append((
            output.copy(), dcontext.copy(), dq.copy(),
            {p.name: p.grad.copy() for p in mem.parameters()},
        ))
    for old, batched in zip(results[0][:3], results[1][:3]):
        np.testing.assert_allclose(old, batched, atol=2e-6, rtol=2e-6)
    for name, old in results[0][3].items():
        np.testing.assert_allclose(
            old, results[1][3][name], atol=3e-6, rtol=3e-6, err_msg=name
        )


def test_route_boundaries_read_only_previous_context():
    c = small_config()
    m = DecoderLanguageModel(c, rng_seed=21)
    mem = m.historical_memory
    state = mem.open(np.random.default_rng(22).integers(0, 128, (1, 256)), np.array([0]))
    q = np.random.default_rng(23).normal(size=(1, 256, 12, 64)).astype('float32')
    slots = mem.slots[5]
    one = mem._route(q, slots, state)[0]
    assert one.shape == (1, 2, 8)
    new = q.copy()
    new[:, 1:128] += 100
    # All changes strictly after r=0, strictly before r=128. The first
    # boundary is unchanged; the second boundary is also unaffected.
    two = mem._route(new, slots, state)[0]
    assert np.array_equal(one, two)
    new = q.copy()
    new[:, 128, 10, :] += 100
    three = mem._route(new, slots, state)[0]
    assert np.array_equal(one[:, 0], three[:, 0])


def test_history_valid_mask_blocks_cross_document_read():
    c = small_config()
    m = DecoderLanguageModel(c, rng_seed=11)
    original = np.random.default_rng(12).integers(0, 128, (1, 264))
    altered = original.copy()
    altered[0, 40] = (altered[0, 40] + 4) % 128
    mask = np.ones((1, 256), dtype=bool)
    mask[:, 0:128] = False
    metadata = {'history_valid_starts': np.array([0]), 'history_valid_mask': mask}
    a = m.forward_body(original, memory_metadata=metadata, return_cache=False)
    b = m.forward_body(altered, memory_metadata=metadata, return_cache=False)
    assert np.array_equal(a, b)


def test_new_memory_parameters_are_optimizer_registered():
    c = small_config()
    m = DecoderLanguageModel(c, rng_seed=24)
    model_names = {p.name for p in m.parameters()}
    optimizer_names = {p.name for p in m.optimization_parameters()}
    memory_names = {p.name for p in m.historical_memory.parameters()}
    assert memory_names <= model_names
    assert memory_names <= optimizer_names
    assert len(memory_names) == len(m.historical_memory.parameters())
    from mini_llm.optim.hybrid_muon import muon_eligible
    assert all(not muon_eligible(p) for p in m.historical_memory.parameters())
