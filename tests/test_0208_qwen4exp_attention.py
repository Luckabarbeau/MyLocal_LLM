"""Independent FP64 graph oracle plus compiled pinned CPU selection/RoPE code."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from mini_llm.backend import asnumpy, xp
from mini_llm.external.qwen4exp_attention import QSAConfig, Qwen4ExpAttention, select_cells


LLAMA = Path('/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/vendor/llama.cpp')
SHA = '171e8846b4af9766c354064cb776cb34a50f053f'


def array(x):
    return xp.asarray(x, dtype=xp.float32)


def fixture(*, topk=5, policy='strata', actual=False):
    c = QSAConfig(7, n_q_heads=4, n_kv_heads=2, head_dim=80 if actual else 8,
                  rotary_dim=64 if actual else 4, index_dim=128 if actual else 6,
                  sections=(11, 11, 10, 0) if actual else (1, 1, 0, 0),
                  index_topk=topk, tie_policy=policy)
    rng = np.random.default_rng(208)
    shapes = dict(query_gate=(2*c.n_q_heads*c.head_dim, c.d_model),
                  key=(c.n_kv_heads*c.head_dim, c.d_model),
                  value=(c.n_kv_heads*c.head_dim, c.d_model),
                  output=(c.d_model, c.n_q_heads*c.head_dim),
                  index_query=(c.index_heads*c.index_dim, c.d_model),
                  index_key=(c.index_dim, c.d_model))
    weights = {name: rng.normal(0, .3, shape).astype(np.float32) for name, shape in shapes.items()}
    for name, dim in [('gamma_q', c.head_dim), ('gamma_k', c.head_dim),
                      ('gamma_index_q', c.index_dim), ('gamma_index_k', c.index_dim)]:
        weights[name] = rng.uniform(.5, 1.5, dim).astype(np.float32)
    model = Qwen4ExpAttention(c, **{k: array(v) for k, v in weights.items()})
    return c, weights, model, rng


def norm(x, gamma, eps):
    return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps) * gamma


def rope(x, p, c):
    # FP64 mathematical oracle: explicitly select IMRoPE's axis per frequency.
    out = x.copy()
    half = c.rotary_dim // 2
    for j in range(half):
        axis = j % 3
        assert j < 3*c.sections[axis]
        angle = (p, p, p)[axis] * c.rope_base**(-2*j/c.rotary_dim)
        out[..., j] = x[..., j]*np.cos(angle) - x[..., j+half]*np.sin(angle)
        out[..., j+half] = x[..., j]*np.sin(angle) + x[..., j+half]*np.cos(angle)
    return out


def oracle(hidden, c, weights):
    w = {k: v.astype(np.float64) for k, v in weights.items()}
    keys, values, raw = [], [], []
    for t, x in enumerate(hidden.astype(np.float64)):
        qg = (w['query_gate'] @ x).reshape(c.n_q_heads, 2*c.head_dim)
        q = rope(norm(qg[:, :c.head_dim], w['gamma_q'], c.epsilon), t, c)
        keys.append(rope(norm((w['key'] @ x).reshape(c.n_kv_heads, c.head_dim),
                              w['gamma_k'], c.epsilon), t, c))
        values.append((w['value'] @ x).reshape(c.n_kv_heads, c.head_dim))
        raw.append(w['index_key'] @ x)
        iq = rope(norm((w['index_query'] @ x).reshape(c.index_heads, c.index_dim),
                       w['gamma_index_q'], c.epsilon), t, c)
        complete = (t+1)//c.block_size
        blocks = []
        for b in range(complete):
            pooled = np.mean(raw[b*c.block_size:(b+1)*c.block_size], axis=0)
            blocks.append(rope(norm(pooled, w['gamma_index_k'], c.epsilon), b*c.block_size, c))
        if (t+1) % c.block_size:
            blocks.append(norm(raw[0], w['gamma_index_k'], c.epsilon))
        block_scores = np.maximum(iq @ np.array(blocks).T, 0).sum(axis=0)
        scores = np.repeat(block_scores, c.block_size)[:t+1]
        # Selection is FP32: finite tail addition can erase small differences.
        selection_scores = scores.astype(np.float32)
        selection_scores[complete*c.block_size:] += np.float32(1e9)
        width = min(t+1, c.index_topk+c.block_size-1)
        ids = np.sort(np.lexsort((np.arange(t+1), -selection_scores))[:width])
        contexts = []
        for h in range(c.n_q_heads):
            kh = h//(c.n_q_heads//c.n_kv_heads)
            logits = np.array(keys)[ids, kh] @ q[h] * (c.attention_scale or c.head_dim**-.5)
            prob = np.exp(logits-logits.max())
            prob /= prob.sum()
            contexts.append(prob @ np.array(values)[ids, kh])
        gated = np.array(contexts) / (1+np.exp(-qg[:, c.head_dim:]))
        yield w['output'] @ gated.ravel(), block_scores, selection_scores, ids


@pytest.mark.parametrize('actual', [False, True])
def test_graph_fp64_scores_selection_and_boundaries(actual):
    c, w, model, rng = fixture(actual=actual)
    hidden = rng.normal(size=(41, c.d_model)).astype(np.float32)
    state = model.create_state(len(hidden))
    pointers = [id(a) for a in (state.keys, state.values, state.raw_index, state.pooled)]
    saw_sparse = False
    for t, reference in enumerate(oracle(hidden, c, w)):
        expected, blocks, scores, ids = reference
        output, diag = model.step(array(hidden[t]), state, position=t, diagnostics=True)
        np.testing.assert_allclose(asnumpy(output), expected, rtol=3e-5, atol=1e-5)
        np.testing.assert_allclose(asnumpy(diag['block_scores']), blocks, rtol=8e-6, atol=2e-5)
        np.testing.assert_allclose(asnumpy(diag['cell_scores']), scores, rtol=8e-6, atol=2e-5)
        np.testing.assert_array_equal(asnumpy(diag['selected_ids']), ids)
        assert diag['tail_start'] == (t+1)//4*4
        assert np.all(ids <= t)
        saw_sparse |= len(ids) < t+1
    assert saw_sparse
    assert pointers == [id(a) for a in (state.keys, state.values, state.raw_index, state.pooled)]
    assert all(a.dtype == xp.float32 for a in (state.keys, state.values, state.raw_index, state.pooled))


@pytest.mark.parametrize('n', [1, 3, 4, 7, 8, 9, 16, 17, 39])
def test_ties_finite_tail_and_partial_token_budget(n):
    scores = np.zeros(n, np.float32)
    tail = n//4*4
    scores[tail:] += np.float32(1e9)
    width = min(n, 5+4-1)
    ids = asnumpy(select_cells(array(scores), width, 'strata'))
    expected = np.sort(np.lexsort((np.arange(n), -scores))[:width])
    np.testing.assert_array_equal(ids, expected)
    assert len(ids) == width
    assert set(range(tail, n)) <= set(ids)
    if n == 16:
        np.testing.assert_array_equal(ids, np.arange(8))


def test_default_budget_long_context_is_not_dense():
    c, _, model, rng = fixture(topk=2048)
    n = 2061
    state = model.create_state(n)
    # Populate synthetic prior state, independently of the step path. Equal
    # block scores isolate exact token-budget/tail semantics at actual width.
    state.length = n-1
    state.keys[:] = array(rng.normal(size=state.keys.shape))
    state.values[:] = array(rng.normal(size=state.values.shape))
    state.raw_index[:] = 0
    state.pooled[:] = 0
    _, diag = model.step(array(rng.normal(size=c.d_model)), state, position=n-1, diagnostics=True)
    ids = asnumpy(diag['selected_ids'])
    assert len(ids) == 2051 < n
    assert ids[-1] == n-1
    np.testing.assert_array_equal(ids[:-1], np.arange(2050))
    assert diag['tail_start'] == 2060


def test_prefill_reset_owner_positions_and_capacity():
    c, _, model, rng = fixture()
    hidden = array(rng.normal(size=(23, c.d_model)))
    a, b = model.create_state(23), model.create_state(23)
    full = model.forward(hidden, a, start_position=0)
    chunk1 = model.forward(hidden[:7], b, start_position=0)
    chunk2 = model.forward(hidden[7:], b, start_position=7)
    np.testing.assert_array_equal(asnumpy(full), asnumpy(xp.concatenate((chunk1, chunk2))))
    for name in ('keys', 'values', 'raw_index', 'pooled'):
        np.testing.assert_array_equal(asnumpy(getattr(a, name)), asnumpy(getattr(b, name)))
    with pytest.raises(ValueError, match='capacity'):
        model.step(hidden[0], a, position=23)
    with pytest.raises(ValueError, match='position'):
        model.step(hidden[0], a, position=0)
    with pytest.raises(ValueError, match='scalar'):
        model.step(hidden[0], a, position=(23, 23, 23))
    other = fixture()[2]
    with pytest.raises(ValueError, match='owner'):
        other.step(hidden[0], a, position=23)
    model.reset(a)
    assert a.length == 0 and not bool(xp.any(a.raw_index))
    np.testing.assert_array_equal(asnumpy(model.forward(hidden, a, start_position=0)), asnumpy(full))


def test_callback_shapes_fail_without_state_mutation():
    c, w, _, rng = fixture()
    kwargs = {k: array(v) for k, v in w.items()}
    for name in ('query_gate', 'key', 'value', 'index_query', 'index_key', 'output'):
        bad = dict(kwargs)
        bad[name] = lambda x: xp.zeros(1, dtype=xp.float32)
        model = Qwen4ExpAttention(c, **bad)
        state = model.create_state(5)
        with pytest.raises(ValueError, match='shape'):
            model.step(array(rng.normal(size=c.d_model)), state, position=0)
        assert state.length == 0
        assert all(not bool(xp.any(a)) for a in (state.keys, state.values, state.raw_index, state.pooled))
    dense = Qwen4ExpAttention(c, **kwargs)
    callbacks = dict(kwargs)
    for name in ('query_gate', 'key', 'value', 'index_query', 'index_key', 'output'):
        callbacks[name] = lambda x, w=kwargs[name]: w @ x
    callback = Qwen4ExpAttention(c, **callbacks)
    x = array(rng.normal(size=(13, c.d_model)))
    np.testing.assert_array_equal(asnumpy(dense.forward(x, dense.create_state(13), start_position=0)),
                                  asnumpy(callback.forward(x, callback.create_state(13), start_position=0)))


@pytest.fixture(scope='module')
def compiled_source(tmp_path_factory):
    if not LLAMA.exists() or not shutil.which('c++'):
        pytest.skip('pinned local llama.cpp and C++ compiler required')
    revision = subprocess.check_output(['git', '--no-pager', '-C', str(LLAMA), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == SHA
    source = (LLAMA/'ggml/src/ggml-cpu/ops.cpp').read_text()
    argsort = source[source.index('template<enum ggml_sort_order order>'):source.index('\nvoid ggml_compute_forward_argsort(')]
    topk = source[source.index('struct cmp_top_k {'):source.index('\nvoid ggml_compute_forward_top_k(')]
    cache = source[source.index('static void ggml_mrope_cache_init('):source.index('\n\ntemplate<typename T>')]
    rotate = source[source.index('template<typename T>\nstatic void rotate_pairs'):source.index('\ntemplate<typename T> //float or ggml_fp16_t')]
    # Compile actual extracted CPU functions, not a second hand-written sorter.
    # Tiny ABI shims replace unrelated graph/thread machinery. rope_yarn shim
    # restricts this harness to unscaled base RoPE, exactly this patch's scope.
    prelude = r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#define GGML_ASSERT(x) do { if (!(x)) abort(); } while(0)
#define GGML_ABORT(x) abort()
#define CACHE_LINE_SIZE_F32 16
#define GGML_TENSOR_UNARY_OP_LOCALS const int64_t ne0=dst->n, ne00=src0->n; const size_t nb0=4, nb01=src0->n*4, nb1=dst->n*4;
enum ggml_sort_order {GGML_SORT_ORDER_ASC, GGML_SORT_ORDER_DESC};
struct ggml_tensor {void *data; int64_t n; ggml_tensor *src[1]; int order;};
struct ggml_compute_params {int ith, nth; void *wdata;};
int64_t ggml_nrows(const ggml_tensor*) {return 1;}
int32_t ggml_get_op_params_i32(const ggml_tensor *t,int) {return t->order;}
void rope_yarn(float theta,float freq,float*,int64_t,float ext,float scale,float*c,float*s) {
    GGML_ASSERT(ext==0); *c=cosf(theta*freq)*scale; *s=sinf(theta*freq)*scale;
}
template<typename T> struct type_conversion_table {
    static float to_f32(T x) {return x;} static T from_f32(float x) {return x;}
};
'''
    wrapper = r'''
extern "C" void selection(float *x,int n,int k,int *out,int sort) {
    ggml_tensor src{x,n,{nullptr},0}, dst{out,sort?n:k,{&src},GGML_SORT_ORDER_DESC};
    auto *scratch = new int32_t[n+16]; ggml_compute_params params{0,1,scratch};
    if(sort) ggml_compute_forward_argsort_f32(&params,&dst);
    else ggml_compute_forward_top_k_f32(&params,&dst);
    delete[] scratch;
}
extern "C" void rotation(float*x,float*out,int dim,int rot,int pos) {
    float cache[512], corr[2]={0,0}; int sections[4]={11,11,10,0};
    ggml_mrope_cache_init(pos,pos,pos,pos,sections,true,false,1,nullptr,corr,dim,0,1,
                         cache,1,powf(1e7f,-2.f/rot));
    memcpy(out,x,dim*4); rotate_pairs<float>(rot,rot/2,cache,x,out);
}
'''
    folder = tmp_path_factory.mktemp('qsa_cpu')
    cpp, so = folder/'source.cpp', folder/'source.so'
    cpp.write_text(prelude+argsort+topk+cache+rotate+wrapper)
    subprocess.run(['c++', '-std=c++17', '-O2', '-shared', '-fPIC', str(cpp), '-o', str(so)], check=True, timeout=60)
    return ctypes.CDLL(str(so))


def test_compiled_actual_cpu_topk_and_argsort_ties(compiled_source):
    lib = compiled_source
    fp = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    ip = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    lib.selection.argtypes = [fp, ctypes.c_int, ctypes.c_int, ip, ctypes.c_int]
    rng = np.random.default_rng(208)
    differs = False
    for n in (1, 4, 17, 33, 129, 2061):
        for k in sorted({1, min(n, 8), n}):
            for scores in (np.zeros(n, np.float32), rng.integers(0, 4, n).astype(np.float32)):
                out = np.empty(k, np.int32)
                lib.selection(scores, n, k, out, 0)
                expected = np.sort(out)
                np.testing.assert_array_equal(asnumpy(select_cells(array(scores), k)), expected)
                differs |= not np.array_equal(expected, asnumpy(select_cells(array(scores), k, 'strata')))
        scores = np.zeros(n, np.float32)
        out = np.empty(n, np.int32)
        lib.selection(scores, n, n, out, 1)
        if n == 33:
            assert not np.array_equal(out, np.arange(n)), 'local std::sort tie behavior changed'
    # A strict improvement evicts one member of a tied heap boundary.
    scores = np.array([0, 0, 0, 0, 1], np.float32)
    out = np.empty(4, np.int32)
    lib.selection(scores, 5, 4, out, 0)
    expected = np.sort(out)
    np.testing.assert_array_equal(expected, [0, 1, 3, 4])
    np.testing.assert_array_equal(asnumpy(select_cells(array(scores), 4)), expected)
    np.testing.assert_array_equal(asnumpy(select_cells(array(scores), 4, 'strata')), [0, 1, 2, 4])
    differs |= not np.array_equal(expected, asnumpy(select_cells(array(scores), 4, 'strata')))
    assert differs, 'must exercise a tie set where CPU and Strata disagree'


def test_compiled_actual_imrope_text_cache_and_pairs(compiled_source):
    fp = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    compiled_source.rotation.argtypes = [fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    c, _, model, rng = fixture(actual=True)
    for dim in (80, 128, 256):
        x = rng.normal(size=dim).astype(np.float32)
        for position in (0, 3, 4, 127, 2057):
            out = np.empty_like(x)
            compiled_source.rotation(x, out, dim, 64, position)
            actual = asnumpy(model._rotate(array(x[None]), position))[0]
            np.testing.assert_allclose(actual, out, rtol=2e-4, atol=3e-4)
            np.testing.assert_allclose(out, rope(x.astype(np.float64), position, c), rtol=2e-4, atol=3e-4)
            np.testing.assert_array_equal(actual[64:], x[64:])


@pytest.mark.parametrize('changes', [dict(n_q_heads=3), dict(rotary_dim=3),
                                      dict(sections=(0, 1, 1, 0)), dict(tie_policy='lower'),
                                      dict(epsilon=0), dict(index_topk=0)])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        QSAConfig(7, **changes)
