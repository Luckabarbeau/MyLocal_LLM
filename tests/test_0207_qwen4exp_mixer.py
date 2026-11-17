"""Synthetic operator validation, not real-weight/full-model reference parity.

Optional pinned CPU primitive: set QWEN4EXP_LLAMA_SOURCE to the llama.cpp
checkout. Builds only GGML in pytest's temporary directory (no vendor edits).
"""

import ctypes as ct
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

if os.environ.get("MINI_LLM_BACKEND", "numpy").lower() != "numpy":
    pytest.skip("synthetic NumPy CPU tests", allow_module_level=True)

from mini_llm.external.qwen4exp_mixer import (
    Qwen4ExpMixerSpec, Qwen4ExpRecurrentMixerInference,
)
from mini_llm.ops.gated_delta_inference import GatedDeltaMixerInference


PIN = "171e8846b4af9766c354064cb776cb34a50f053f"


def fixture(hk=2, hv=6, dim=4, time=9):
    rng = np.random.default_rng(207)
    spec = Qwen4ExpMixerSpec(7, hk, hv, dim, 4, 1e-5)
    def rand(shape, scale=.4):
        return (rng.normal(size=shape) * scale).astype(np.float32)
    weights = dict(qkv=rand((spec.qkv_channels, 7)), z=rand((spec.inner_size, 7)),
                   beta=rand((hv, 7)), alpha=rand((hv, 7)),
                   out=rand((7, spec.inner_size)),
                   conv_weight=rand((spec.qkv_channels, 4)),
                   norm_weight=rand((dim,)) + np.float32(1),
                   ssm_a=-np.exp(rand((hv,))), dt_bias=rand((hv,)))
    hidden = rand((2, time, 7), .7)
    mixer = Qwen4ExpRecurrentMixerInference(spec, **weights)
    state = mixer.create_state(2)
    state.conv.history[:] = rand(state.conv.history.shape, .2)
    state.delta.memory[:] = rand(state.delta.memory.shape, .2)
    state.conv.length = state.delta.length = 3
    return spec, weights, hidden, mixer, state


def oracle(spec, weights, hidden, history, memory, *, grouped=False, silu_z=False):
    """Independent float64 source graph; memory is VALUE-by-KEY, unlike ops.

    Explicit token/batch/head loops, no production normalization or recurrence.
    Optionally operates on pre-converter HF grouped weights/state.
    """
    w = {k: v.astype(np.float64) for k, v in weights.items()}
    x = hidden.astype(np.float64)
    history = history.astype(np.float64).copy()
    memory = memory.astype(np.float64).copy()
    raw = x @ w["qkv"].T
    z = x @ w["z"].T
    beta = np.exp(-np.logaddexp(0., -(x @ w["beta"].T)))
    g = w["ssm_a"] * np.logaddexp(0., x @ w["alpha"].T + w["dt_bias"])
    output = np.empty_like(x)
    qwidth = spec.n_key_heads * spec.head_dim
    qs, ks, vs, reads = [], [], [], []
    for t in range(x.shape[1]):
        window = np.concatenate((history, raw[:, t:t+1]), axis=1)
        conv = np.einsum("bkc,ck->bc", window, w["conv_weight"])
        history = window[:, 1:].copy()
        conv = conv * np.exp(-np.logaddexp(0., -conv))
        q = conv[:, :qwidth].reshape(-1, spec.n_key_heads, spec.head_dim)
        k = conv[:, qwidth:2*qwidth].reshape(q.shape)
        v = conv[:, 2*qwidth:].reshape(-1, spec.n_value_heads, spec.head_dim)
        q = q / np.sqrt(np.sum(q*q, axis=-1, keepdims=True) + spec.rms_eps)
        k = k / np.sqrt(np.sum(k*k, axis=-1, keepdims=True) + spec.rms_eps)
        read = np.empty_like(v)
        for b in range(x.shape[0]):
            for h in range(spec.n_value_heads):
                kh = h // (spec.n_value_heads // spec.n_key_heads) if grouped else h % spec.n_key_heads
                decayed = np.exp(g[b, t, h]) * memory[b, h]
                residual = beta[b, t, h] * (v[b, h] - decayed @ k[b, kh])
                memory[b, h] = decayed + np.outer(residual, k[b, kh])
                read[b, h] = memory[b, h] @ (q[b, kh] / np.sqrt(spec.head_dim))
        normed = read / np.sqrt(np.mean(read*read, axis=-1, keepdims=True) + spec.rms_eps)
        gate = z[:, t].reshape(v.shape)
        activated = np.exp(-np.logaddexp(0., -gate))
        if silu_z:
            activated *= gate
        output[:, t] = (normed * w["norm_weight"] * activated).reshape(x.shape[0], -1) @ w["out"].T
        qs.append(q); ks.append(k); vs.append(v); reads.append(read)
    traces = tuple(np.stack(items, axis=1) for items in (qs, ks, vs, reads)) if qs else ()
    return output, history, memory, (traces, g, beta)


def difference(label, actual, expected):
    delta = actual.astype(np.float64) - expected
    absolute = float(np.max(np.abs(delta), initial=0))
    relative = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(expected.ravel()), 1e-30))
    print(f"{label}: max_abs={absolute:.9g}, relative_l2={relative:.9g}")
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("hk,hv,dim", [(2, 2, 4), (2, 6, 4), (16, 48, 128)])
def test_float64_graph_and_all_streaming_states(hk, hv, dim):
    spec, weights, x, mixer, initial = fixture(hk, hv, dim)
    expected, hist, mem, _ = oracle(spec, weights, x, initial.conv.history,
                                   initial.delta.memory.swapaxes(-1, -2))
    full, step, chunk = initial.copy(), initial.copy(), initial.copy()
    y = mixer.forward(x, full)
    label = f"Hk={hk},Hv={hv},D={dim}"
    difference(label + " output", y, expected)
    difference(label + " memory", full.delta.memory, mem.swapaxes(-1, -2))
    difference(label + " history", full.conv.history, hist)
    stepped = []
    for t in range(x.shape[1]):
        stepped.append(mixer.step(x[:, t], step))
        prefix = initial.copy()
        prefix_y = mixer.forward(x[:, :t+1], prefix)
        np.testing.assert_allclose(stepped[-1], prefix_y[:, -1], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(step.delta.memory, prefix.delta.memory, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(step.conv.history, prefix.conv.history, rtol=2e-5, atol=2e-6)
        assert step.length == prefix.length == initial.length + t + 1
    difference(label + " sequence/step", np.stack(stepped, axis=1), y)
    pieces = [mixer.forward(x[:, a:b], chunk) for a, b in ((0, 2), (2, 2), (2, 5), (5, 9))]
    np.testing.assert_allclose(np.concatenate(pieces, axis=1), y, rtol=2e-5, atol=2e-6)
    for state in (step, chunk):
        np.testing.assert_allclose(state.delta.memory, full.delta.memory, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(state.conv.history, full.conv.history, rtol=2e-5, atol=2e-6)
        assert state.length == full.length == 12
    assert initial.length == 3
    full.reset()
    assert full.length == 0
    assert not np.any(full.delta.memory) and not np.any(full.conv.history)
    reset_y = mixer.forward(x, full)
    fresh = mixer.create_state(2)
    np.testing.assert_array_equal(reset_y, mixer.forward(x, fresh))
    np.testing.assert_array_equal(full.delta.memory, fresh.delta.memory)
    np.testing.assert_array_equal(full.conv.history, fresh.conv.history)


def test_converter_grouped_to_tiled_order():
    spec, grouped, x, _, initial = fixture()
    hk, hv, d = spec.n_key_heads, spec.n_value_heads, spec.head_dim
    # Each tiled index selects this grouped head: [0,3,1,4,2,5].
    heads = np.arange(hv).reshape(hk, hv // hk).T.reshape(-1)
    channels = (heads[:, None] * d + np.arange(d)).reshape(-1)
    qw = 2 * hk * d
    qkv_channels = np.r_[np.arange(qw), qw + channels]
    tiled = {name: value.copy() for name, value in grouped.items()}
    for name in ("qkv", "conv_weight"):
        tiled[name] = grouped[name][qkv_channels]
    tiled["z"] = grouped["z"][channels]
    tiled["out"] = grouped["out"][:, channels]
    for name in ("beta", "alpha", "ssm_a", "dt_bias"):
        tiled[name] = grouped[name][heads]
    expected, hist, mem, _ = oracle(spec, grouped, x, initial.conv.history,
        initial.delta.memory.swapaxes(-1, -2), grouped=True)
    mixer = Qwen4ExpRecurrentMixerInference(spec, **tiled)
    state = initial.copy()
    state.conv.history[:] = initial.conv.history[..., qkv_channels]
    state.delta.memory[:] = initial.delta.memory[:, heads]
    difference("grouped/tiled output", mixer.forward(x, state), expected)
    difference("grouped/tiled history", state.conv.history, hist[..., qkv_channels])
    difference("grouped/tiled memory", state.delta.memory, mem[:, heads].swapaxes(-1, -2))
    assert mixer.mixer.rule.head_map == (0, 1, 0, 1, 0, 1)


def test_sigmoid_not_silu_and_generic_default_unchanged():
    spec, weights, x, mixer, state = fixture()
    expected, _, _, _ = oracle(spec, weights, x, state.conv.history,
                               state.delta.memory.swapaxes(-1, -2))
    wrong, _, _, _ = oracle(spec, weights, x, state.conv.history,
                            state.delta.memory.swapaxes(-1, -2), silu_z=True)
    difference("sigmoid Z", mixer.forward(x, state), expected)
    gap = np.max(np.abs(expected - wrong))
    print(f"sigmoid-vs-SiLU Z max_abs={gap:.9g}")
    assert gap > .1
    generic = GatedDeltaMixerInference(2, 6, 4, 4, 4)
    assert generic.norm.gate_activation == "silu"
    assert mixer.mixer.norm.gate_activation == "sigmoid"
    # Qwen4Exp must not enter the SiLU-only fused decode implementation.
    from mini_llm.ops.gated_delta_decode import _eligible
    assert not _eligible(mixer.mixer, np.zeros((2, 1, spec.qkv_channels), np.float32),
                         None, None, None, state)


def test_projection_callbacks_and_invalid_result_before_state_mutation():
    spec, weights, x, dense, initial = fixture()
    calls = []
    def callback(name):
        def project(hidden):
            calls.append((name, hidden.shape))
            return hidden @ weights[name].T
        return project
    callbacks = dict(weights)
    for name in ("qkv", "z", "beta", "alpha", "out"):
        callbacks[name] = callback(name)
    mixer = Qwen4ExpRecurrentMixerInference(spec, **callbacks)
    a, b = initial.copy(), initial.copy()
    np.testing.assert_array_equal(mixer.forward(x, a), dense.forward(x, b))
    assert [name for name, _ in calls] == ["qkv", "z", "beta", "alpha", "out"]
    np.testing.assert_array_equal(a.delta.memory, b.delta.memory)
    np.testing.assert_array_equal(a.conv.history, b.conv.history)
    mixer.projections["beta"] = lambda hidden: np.zeros((*hidden.shape[:2], spec.n_value_heads), np.float64)
    before = a.copy()
    with pytest.raises(TypeError, match="beta output"):
        mixer.forward(x, a)
    np.testing.assert_array_equal(a.delta.memory, before.delta.memory)
    np.testing.assert_array_equal(a.conv.history, before.conv.history)
    assert a.length == before.length
    with pytest.raises(TypeError, match="hidden"):
        dense.forward(x.astype(np.float16), initial)
    with pytest.raises(ValueError, match="already-negative"):
        Qwen4ExpRecurrentMixerInference(spec, **dict(weights, ssm_a=-weights["ssm_a"]))


def test_metadata_and_extreme_coefficients():
    metadata = {"general.architecture": "qwen4exp", "qwen4exp.embedding_length": 2560,
                "qwen4exp.ssm.group_count": 16, "qwen4exp.ssm.time_step_rank": 48,
                "qwen4exp.ssm.state_size": 128, "qwen4exp.ssm.conv_kernel": 4,
                "qwen4exp.ssm.inner_size": 6144,
                "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6}
    real = Qwen4ExpMixerSpec.from_metadata(metadata)
    assert (real.n_key_heads, real.n_value_heads, real.head_dim, real.conv_kernel_size,
            real.qkv_channels) == (16, 48, 128, 4, 10240)
    with pytest.raises(KeyError):
        Qwen4ExpMixerSpec.from_metadata({})
    with pytest.raises(ValueError, match="inner_size"):
        Qwen4ExpMixerSpec.from_metadata(dict(metadata, **{"qwen4exp.ssm.inner_size": 6145}))
    spec, weights, x, _, state = fixture()
    weights["beta"].fill(1000)
    weights["alpha"].fill(-1000)
    weights["z"].fill(1000)
    # Existing generic sigmoid saturates via exp overflow -> inf -> zero.
    # Check the numerical result, not an unsupported warning-free contract.
    with np.errstate(over="ignore", invalid="raise", divide="raise"):
        expected, _, mem, _ = oracle(spec, weights, x, state.conv.history,
                                     state.delta.memory.swapaxes(-1, -2))
        mixer = Qwen4ExpRecurrentMixerInference(spec, **weights)
        difference("extreme coefficients", mixer.forward(x, state), expected)
        difference("extreme memory", state.delta.memory, mem.swapaxes(-1, -2))


@pytest.fixture(scope="module")
def pinned_ggml(tmp_path_factory):
    source = os.environ.get("QWEN4EXP_LLAMA_SOURCE")
    if not source:
        pytest.skip("set QWEN4EXP_LLAMA_SOURCE to build/run pinned GGML CPU primitive")
    source = Path(source)
    sha = subprocess.check_output(["git", "--no-pager", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert sha == PIN, f"reference revision {sha} != {PIN}"
    for tool in ("cmake", "c++"):
        if not shutil.which(tool):
            pytest.skip(f"pinned reference build needs {tool}")
    parent = tmp_path_factory.mktemp("0207_ggml")
    # This pinned vendor tree omits standalone ggml.pc.in; embed GGML instead.
    (parent / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.14)\nproject(patch0207 C CXX)\n'
        f'add_subdirectory("{source / "ggml"}" ggml)\n')
    build = parent / "build"
    subprocess.run(["cmake", "-S", str(parent), "-B", str(build),
                    "-DBUILD_SHARED_LIBS=ON", "-DGGML_NATIVE=OFF", "-DGGML_OPENMP=OFF",
                    "-DGGML_CCACHE=OFF", "-DGGML_BUILD_TESTS=OFF", "-DGGML_BUILD_EXAMPLES=OFF"],
                   check=True, text=True, timeout=60)
    subprocess.run(["cmake", "--build", str(build), "-j", "2"], check=True,
                   text=True, timeout=180)
    base = ct.CDLL(str(next(build.rglob("libggml-base.so"))), mode=ct.RTLD_GLOBAL)
    cpu = ct.CDLL(str(next(build.rglob("libggml-cpu.so"))), mode=ct.RTLD_GLOBAL)
    ptr = ct.c_void_p
    class Init(ct.Structure):
        _fields_ = [("mem_size", ct.c_size_t), ("mem_buffer", ptr), ("no_alloc", ct.c_bool)]
    for lib, name, args, result in (
        (base, "ggml_init", [Init], ptr),
        (base, "ggml_free", [ptr], None),
        (base, "ggml_new_tensor_4d", [ptr, ct.c_int] + [ct.c_int64]*4, ptr),
        (base, "ggml_get_data", [ptr], ptr),
        (base, "ggml_gated_delta_net", [ptr]*7 + [ct.c_int64], ptr),
        (base, "ggml_new_graph", [ptr], ptr),
        (base, "ggml_build_forward_expand", [ptr, ptr], None),
        (cpu, "ggml_graph_compute_with_ctx", [ptr, ptr, ct.c_int], ct.c_int),
    ):
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    return base, cpu, Init


@pytest.mark.parametrize("time", [1, 7])
def test_actual_pinned_ggml_recurrence(pinned_ggml, time):
    base, cpu, Init = pinned_ggml
    spec, weights, x, mixer, initial = fixture(2, 6, 128, time)
    expected, _, memory, ((q, k, v, reads), g, beta) = oracle(
        spec, weights, x, initial.conv.history, initial.delta.memory.swapaxes(-1, -2))
    context = base.ggml_init(Init(64*1024*1024, None, False))
    assert context
    def tensor(array):
        array = np.ascontiguousarray(array, dtype=np.float32)
        result = base.ggml_new_tensor_4d(context, 0, *reversed(array.shape))
        ct.memmove(base.ggml_get_data(result), array.ctypes.data, array.nbytes)
        return result
    try:
        # GGML fastest-first [D,H,T,B]; native memory is value-by-key in Python.
        args = [tensor(a) for a in (q, k, v, g[..., None], beta[..., None],
                                    initial.delta.memory.swapaxes(-1, -2))]
        result = base.ggml_gated_delta_net(context, *args, 1)
        graph = base.ggml_new_graph(context)
        base.ggml_build_forward_expand(graph, result)
        assert cpu.ggml_graph_compute_with_ctx(context, graph, 2) == 0
        count = reads.size + memory.size
        data = np.ctypeslib.as_array(ct.cast(base.ggml_get_data(result), ct.POINTER(ct.c_float)),
                                    shape=(count,)).copy()
        native_read = data[:reads.size].reshape(reads.shape)
        native_memory = data[reads.size:].reshape(memory.shape)
        difference(f"GGML T={time} read vs float64", native_read, reads)
        difference(f"GGML T={time} memory vs float64", native_memory, memory)
        state = initial.copy()
        difference(f"adapter T={time} output", mixer.forward(x, state), expected)
        difference(f"adapter T={time} memory vs GGML", state.delta.memory,
                   native_memory.swapaxes(-1, -2))
    finally:
        base.ggml_free(context)
