"""Opt-in QSA decode parity, exact state, dispatch, and transactional validation."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.external import qwen4exp_attention as attention
from mini_llm.ops.qsa_decode_gpu import qsa_decode_gpu


pytestmark = pytest.mark.skipif(BACKEND_NAME != "cupy", reason="actual CuPy execution required")


def fixture(*, heads=4, dim=8, block=4, policy="ggml_cpu"):
    c = attention.QSAConfig(
        7, n_q_heads=heads, n_kv_heads=2, head_dim=dim, rotary_dim=4,
        index_heads=2, index_dim=6, index_topk=17-block, block_size=block,
        sections=(1, 1, 0, 0), tie_policy=policy, attention_scale=.37)
    rng = np.random.default_rng(230)
    shapes = dict(query_gate=(2*heads*dim, 7), key=(2*dim, 7),
                  value=(2*dim, 7), output=(7, heads*dim),
                  index_query=(12, 7), index_key=(6, 7))
    weights = {name: xp.asarray(rng.normal(0, .2, shape), dtype=xp.float32)
               for name, shape in shapes.items()}
    for name, width in (("gamma_q", dim), ("gamma_k", dim),
                        ("gamma_index_q", 6), ("gamma_index_k", 6)):
        weights[name] = xp.asarray(rng.uniform(.5, 1.5, width), dtype=xp.float32)
    return c, weights, rng


def parity(actual, expected, label, *, atol=2e-5, rtol=3e-5):
    actual, expected = asnumpy(actual), asnumpy(expected)
    error = np.abs(actual - expected)
    # Report both worst absolute and elementwise relative error (floor 1e-7).
    relative = error / np.maximum(np.abs(expected), 1e-7)
    print(f"{label}: max_abs={error.max():.9g}, max_rel(floor=1e-7)={relative.max():.9g}")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("heads,dim,block,policy", [
    (4, 8, 4, "ggml_cpu"), (24, 256, 4, "strata"), (4, 10, 1, "ggml_cpu")])
def test_sequential_parity_exact_state_and_width_fallback(monkeypatch, heads, dim, block, policy):
    c, weights, rng = fixture(heads=heads, dim=dim, block=block, policy=policy)
    reference = attention.Qwen4ExpAttention(c, **weights)
    fast = attention.Qwen4ExpAttention(c, **weights, fast_decode=True)
    assert not reference.fast_decode
    a, b = reference.create_state(23), fast.create_state(23)
    pointers = [id(getattr(b, name)) for name in ("keys", "values", "raw_index", "pooled")]
    calls = []
    query_calls = []
    query, query_size = fast.projections['index_query']

    def tracked_query(hidden):
        query_calls.append(1)
        return query @ hidden

    fast.projections['index_query'] = (tracked_query, query_size)
    kernel = attention.qsa_decode_gpu
    select = attention.select_cells
    selections = []

    def tracked_kernel(*args):
        calls.append(args[-2])
        return kernel(*args)

    def tracked_select(*args):
        selections.append(args[0].size)
        return select(*args)

    monkeypatch.setattr(attention, "qsa_decode_gpu", tracked_kernel)
    monkeypatch.setattr(attention, "select_cells", tracked_select)
    outputs, expected = [], []
    for t in range(23):
        hidden = xp.asarray(rng.normal(size=7), dtype=xp.float32)
        expected.append(reference.step(hidden, a, position=t))
        before = len(selections)
        outputs.append(fast.step(hidden, b, position=t))
        assert len(selections) - before == (t >= 16)
        assert len(query_calls) == max(0, t - 15)
        assert a.length == b.length == t+1
        for name in ("keys", "values", "raw_index", "pooled"):
            np.testing.assert_array_equal(asnumpy(getattr(a, name)), asnumpy(getattr(b, name)))
    assert calls == list(range(1, 17))
    assert pointers == [id(getattr(b, name)) for name in ("keys", "values", "raw_index", "pooled")]
    parity(xp.stack(outputs[:16]), xp.stack(expected[:16]), f"16 tokens h={heads} d={dim} b={block}")
    np.testing.assert_array_equal(asnumpy(xp.stack(outputs[16:])), asnumpy(xp.stack(expected[16:])))
    # Reusing the exact populated state with diagnostics must remain reference.
    reference.reset(a)
    fast.reset(b)
    for t in range(5):
        hidden = xp.asarray(rng.normal(size=7), dtype=xp.float32)
        x, dx = reference.step(hidden, a, position=t, diagnostics=True)
        y, dy = fast.step(hidden, b, position=t, diagnostics=True)
        np.testing.assert_array_equal(asnumpy(x), asnumpy(y))
        for name in ("selected_ids", "cell_scores", "block_scores"):
            np.testing.assert_array_equal(asnumpy(dx[name]), asnumpy(dy[name]))
    assert calls == list(range(1, 17))
    assert len(query_calls) == 12  # Seven sparse steps plus five diagnostic steps.


@pytest.mark.parametrize("n,dim", [(1, 8), (16, 256), (129, 10), (2051, 32)])
def test_kernel_stable_fp64_oracle_strides_current_and_stream(n, dim):
    rng = np.random.default_rng(230+n)
    heads, kv = 4, 2
    # Views deliberately have nontrivial head/lane/token strides.
    def array(shape, scale=1):
        return xp.asarray(rng.normal(0, scale, shape), dtype=xp.float32)

    stream = xp.cuda.Stream(non_blocking=True)
    with stream:
        q = array((heads, 2*dim), 8)[:, ::2]
        qg = array((heads, 4*dim), 40)[:, ::2]
        k = array((kv, 2*dim))[:, ::2]
        v = array((kv, 2*dim))[:, ::2]
        keys = array((2*n+2, kv, 2*dim))[::2, :, ::2]
        values = array((2*n+2, kv, 2*dim))[::2, :, ::2]
        keys[n-1:] = xp.nan
        values[n-1:] = xp.nan
        result = qsa_decode_gpu(q, qg, k, v, keys, values, n, .37)
        copied = result.copy()  # consumer on the same nondefault stream
    stream.synchronize()
    qh, gh, kh, vh, ks, vs = [asnumpy(a).astype(np.float64) for a in (q, qg, k, v, keys, values)]
    expected = []
    for h in range(heads):
        group = h // (heads//kv)
        all_k = np.concatenate((ks[:n-1, group], kh[group:group+1]))
        all_v = np.concatenate((vs[:n-1, group], vh[group:group+1]))
        logits = (all_k @ qh[h]) * float(np.float32(.37))
        probability = np.exp(logits-logits.max())
        probability /= probability.sum()
        gate = gh[h, dim:]
        z = np.exp(-np.abs(gate))
        expected.append((probability @ all_v) * np.where(gate >= 0, 1/(1+z), z/(1+z)))
    parity(copied, np.asarray(expected).ravel(), f"FP64 n={n} d={dim}", atol=8e-5, rtol=8e-5)
    assert copied.dtype == xp.float32 and copied.shape == (heads*dim,)
    assert bool(xp.all(xp.isnan(keys[n-1:])))


def test_fast_validation_and_output_failure_do_not_commit(monkeypatch):
    c, weights, rng = fixture()
    model = attention.Qwen4ExpAttention(c, **weights, fast_decode=True)
    state = model.create_state(4)
    hidden = xp.asarray(rng.normal(size=7), dtype=xp.float32)
    model.step(hidden, state, position=0)
    snapshot = {name: getattr(state, name).copy() for name in ("keys", "values", "raw_index", "pooled")}

    def unchanged():
        assert state.length == 1
        for name, expected in snapshot.items():
            np.testing.assert_array_equal(asnumpy(getattr(state, name)), asnumpy(expected))

    with pytest.raises(ValueError, match="position"):
        model.step(hidden, state, position=0)
    with pytest.raises(ValueError, match="finite"):
        model.step(xp.full(7, xp.nan, dtype=xp.float32), state, position=1)
    unchanged()
    # Complete a pool block, then fail output validation: neither KV nor the
    # newly computed pool may commit. Diagnostics validate the index query too.
    model.step(hidden, state, position=1)
    model.step(hidden, state, position=2)
    snapshot = {name: getattr(state, name).copy() for name in snapshot}
    for name in ("index_query", "output"):
        original = model.projections[name]
        model.projections[name] = (lambda x: xp.zeros(1, dtype=xp.float32), original[1])
        with pytest.raises(ValueError, match="shape"):
            model.step(hidden, state, position=3, diagnostics=(name == 'index_query'))
        assert state.length == 3
        for field, expected in snapshot.items():
            np.testing.assert_array_equal(asnumpy(getattr(state, field)), asnumpy(expected))
        model.projections[name] = original
