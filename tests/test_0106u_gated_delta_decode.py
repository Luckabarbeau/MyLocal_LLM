"""Generic T==1 decode contracts; run separately with numpy and cupy backends.

Optional bounded synthetic timing (no checkpoint):
MINI_LLM_BACKEND=cupy MINI_LLM_BENCH_GATED_DELTA_DECODE=1 \
    .venv/bin/python -m pytest -s -q tests/test_0106u_gated_delta_decode.py
"""

import os
import time
import weakref

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops import gated_delta_decode as decode
from mini_llm.ops.gated_delta_inference import GatedDeltaMixerInference


GPU = pytest.mark.skipif(BACKEND_NAME != "cupy", reason="real CuPy backend required")
RTOL, ATOL = 3e-4, 5e-6  # Existing 0106N FP32 mixer / independent FP64 tolerances.


def fixture(dk=3, dv=5, taps=4, dtype="float32", batch=2, tokens=9,
            activation="silu", default_map=False):
    rng = np.random.default_rng(10621)
    hk, hv = (2, 4) if default_map else (3, 5)
    mapping = None if default_map else (2, 0, 2, 1, 0)
    mixer = GatedDeltaMixerInference(hk, hv, dk, dv, taps, head_map=mapping,
        l2_eps=2e-5, rms_eps=3e-6, state_dtype=dtype, gate_activation=activation)
    mixer.rule.query_scale = -.37  # Not the default; only query readout scales.
    def array(value):
        return xp.asarray(value, dtype=dtype)
    mixer.set_weights(array(rng.normal(0, .3, (mixer.channels, taps))),
                      array(rng.uniform(.3, 1.4, dv)),
                      array(rng.normal(0, .05, mixer.channels)))
    state = mixer.create_state(batch)
    state.conv.history[...] = array(rng.normal(0, .2, state.conv.history.shape))
    state.delta.memory[...] = array(rng.normal(0, .1, state.delta.memory.shape))
    # Explicit nonzero initial history participates even when length is zero.
    qkv = array(rng.normal(0, .4, (batch, tokens, mixer.channels)))
    gate = array(rng.normal(0, .8, (batch, tokens, hv, dv)))
    log_decay = array(rng.uniform(-.7, -.01, (batch, tokens, hv)))
    beta = array(rng.uniform(.1, .9, (batch, tokens, hv)))
    if tokens:
        log_decay[:, 0] = 0
        beta[:, 0, 0] = 0
        beta[:, 0, -1] = 1
    return mixer, (qkv, gate, log_decay, beta), state


def reference(mixer, arrays, initial):
    """Independent FP64 convolution + affine recurrence (not delta primitives)."""
    raw, gate, log_decay, beta = (asnumpy(a).astype(np.float64) for a in arrays)
    history = asnumpy(initial.conv.history).astype(np.float64)
    memory = asnumpy(initial.delta.memory).astype(np.float64)
    weight = asnumpy(mixer.conv.weight).astype(np.float64)
    bias = (0 if mixer.conv.bias is None else
            asnumpy(mixer.conv.bias).astype(np.float64))
    gamma = asnumpy(mixer.norm.norm.gamma).astype(np.float64)
    rule = mixer.rule
    output = np.empty(gate.shape, np.float64)
    for t in range(raw.shape[1]):
        window = np.concatenate((history, raw[:, t:t+1]), axis=1)
        z = np.einsum("btc,ct->bc", window, weight) + bias
        z = z / (1 + np.exp(-z))
        q = z[:, :mixer.q_width].reshape(-1, rule.n_key_heads, rule.key_dim)
        k = z[:, mixer.q_width:2*mixer.q_width].reshape(q.shape)
        v = z[:, 2*mixer.q_width:].reshape(-1, rule.n_value_heads, rule.value_dim)
        q = q / np.sqrt(np.sum(q*q, axis=-1, keepdims=True) + mixer.l2_eps)
        k = k / np.sqrt(np.sum(k*k, axis=-1, keepdims=True) + mixer.l2_eps)
        for b in range(raw.shape[0]):
            for h, kh in enumerate(rule.head_map):
                transform = np.exp(log_decay[b, t, h]) * (
                    np.eye(rule.key_dim) - beta[b, t, h] * np.outer(k[b, kh], k[b, kh]))
                memory[b, h] = transform @ memory[b, h] + beta[b, t, h] * np.outer(k[b, kh], v[b, h])
                y = (q[b, kh] * rule.query_scale) @ memory[b, h]
                normalized = (y / np.sqrt(np.mean(y*y) + mixer.norm.norm.eps)) * gamma
                g = gate[b, t, h]
                activated = 1 / (1 + np.exp(-g))
                if mixer.norm.gate_activation == "silu":
                    activated *= g
                output[b, t, h] = normalized * activated
        history = window[:, 1:].copy()
    return output, history, memory


def token_arrays(arrays, t):
    # A batch>1 time slice is not contiguous. Real decode projections are.
    return tuple(xp.ascontiguousarray(a[:, t:t+1]) for a in arrays)


def check(actual, expected, label):
    actual, expected = asnumpy(actual), asnumpy(expected)
    error = float(np.max(np.abs(actual.astype(np.float64) - expected), initial=0))
    print(f"{label}: max_abs_error={error:.9g}")
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, err_msg=label)
    return error


@pytest.mark.parametrize("dk,dv,taps,default_map", [
    (3, 5, 4, False), (128, 128, 4, False), (7, 11, 1, True), (256, 33, 2, False)])
def test_multistep_portable_and_independent_fp64(dk, dv, taps, default_map, monkeypatch):
    mixer, arrays, initial = fixture(dk, dv, taps, default_map=default_map)
    expected, history64, memory64 = reference(mixer, arrays, initial)
    fast, portable = initial.copy(), initial.copy()
    memory_id, history_id = id(fast.delta.memory), id(fast.conv.history)
    mixer._decode_fastpath_enabled = False
    expected32 = mixer.forward(*arrays, portable)
    if BACKEND_NAME == "cupy":
        mixer._decode_fastpath_enabled = True
        def forbidden(*args, **kwargs):
            pytest.fail("eligible decode unexpectedly fell back to portable primitives")
        monkeypatch.setattr(mixer.conv, "forward", forbidden)
        monkeypatch.setattr(mixer.rule, "forward", forbidden)
        monkeypatch.setattr(mixer.norm, "forward", forbidden)
    actual = xp.concatenate([mixer.forward(*token_arrays(arrays, t), fast)
                             for t in range(arrays[0].shape[1])], axis=1)
    assert actual.dtype == xp.float32 and fast.length == arrays[0].shape[1]
    assert id(fast.delta.memory) == memory_id and id(fast.conv.history) == history_id
    label = f"dk={dk},dv={dv},taps={taps}"
    check(actual, expected32, label + "/portable32/output")
    check(fast.delta.memory, portable.delta.memory, label + "/portable32/memory")
    check(actual, expected, label + "/fp64/output")
    check(fast.delta.memory, memory64, label + "/fp64/memory")
    np.testing.assert_array_equal(asnumpy(fast.conv.history), history64)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_fork_reset_empty_causal_prefix_and_raw_history(dtype):
    mixer, arrays, initial = fixture(dtype=dtype)
    prefix = initial.copy()
    first = mixer.forward(*token_arrays(arrays, 0), prefix)
    fork = prefix.copy()
    before = (asnumpy(prefix.conv.history).copy(), asnumpy(prefix.delta.memory).copy())
    changed = tuple(a.copy() for a in arrays)
    changed[0][:, 1:] += 2
    other = initial.copy()
    check(mixer.forward(*changed, other)[:, :1], first, "causal-prefix")
    empty = mixer.forward(*(a[:, :0] for a in arrays), prefix)
    assert empty.shape == (2, 0, 5, 5) and prefix.length == 1
    np.testing.assert_array_equal(asnumpy(prefix.conv.history), before[0])
    np.testing.assert_array_equal(asnumpy(prefix.delta.memory), before[1])
    next_inputs = token_arrays(arrays, 1)
    a = mixer.forward(*next_inputs, prefix)
    b = mixer.forward(*next_inputs, fork)
    np.testing.assert_array_equal(asnumpy(a), asnumpy(b))
    assert fork.delta.memory is not prefix.delta.memory
    assert fork.conv.history is not prefix.conv.history
    np.testing.assert_array_equal(asnumpy(prefix.conv.history[:, -1]), asnumpy(arrays[0][:, 1]))
    ids = id(fork.conv.history), id(fork.delta.memory)
    fork.reset()
    assert fork.length == 0 and ids == (id(fork.conv.history), id(fork.delta.memory))
    assert not np.any(asnumpy(fork.conv.history)) and not np.any(asnumpy(fork.delta.memory))
    fresh = mixer.create_state(2)
    check(mixer.forward(*next_inputs, fork), mixer.forward(*next_inputs, fresh), "reset-replay")


@pytest.mark.parametrize("bad", ["qkv_shape", "gate_shape", "coefficient_shape", "coefficient_dtype",
    "history_shape", "history_dtype", "memory_shape", "memory_dtype", "conv_length",
    "delta_length", "position_mismatch", "missing_conv_weight", "missing_gamma"])
def test_validation_precedes_any_mutation(bad, monkeypatch):
    mixer, arrays, state = fixture(tokens=1)
    arrays = list(arrays)
    if bad == "qkv_shape": arrays[0] = arrays[0][..., :-1]
    elif bad == "gate_shape": arrays[1] = arrays[1][..., :-1]
    elif bad == "coefficient_shape": arrays[3] = arrays[3][..., :1]
    elif bad == "coefficient_dtype": arrays[2] = arrays[2].astype(xp.int32)
    elif bad == "history_shape": state.conv.history = state.conv.history[:, :1].copy()
    elif bad == "history_dtype": state.conv.history = state.conv.history.astype(xp.float64)
    elif bad == "memory_shape": state.delta.memory = state.delta.memory[..., :1].copy()
    elif bad == "memory_dtype": state.delta.memory = state.delta.memory.astype(xp.float64)
    elif bad == "conv_length": state.conv.length = True
    elif bad == "delta_length": state.delta.length = -1
    elif bad == "position_mismatch": state.delta.length = 1
    elif bad == "missing_conv_weight": mixer.conv.weight = None
    else: mixer.norm.norm.gamma = None
    before = asnumpy(state.conv.history).copy(), asnumpy(state.delta.memory).copy()
    lengths = state.conv.length, state.delta.length
    def forbidden(*args):
        pytest.fail("dispatch must follow all existing validators")
    monkeypatch.setattr(decode, "_get_kernels", forbidden)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        mixer.forward(*arrays, state)
    assert lengths == (state.conv.length, state.delta.length)
    np.testing.assert_array_equal(asnumpy(state.conv.history), before[0])
    np.testing.assert_array_equal(asnumpy(state.delta.memory), before[1])


def strided(a):
    storage = xp.empty((*a.shape[:-1], 2 * a.shape[-1]), dtype=a.dtype)
    view = storage[..., ::2]
    view[...] = a
    return view


@pytest.mark.parametrize("unsupported", ["qkv_stride", "gate_stride", "decay_stride", "beta_stride",
    "history_stride", "memory_stride", "weight_stride", "gamma_stride", "bias_stride",
    "qkv_half", "gate_half", "coefficient_double", "gamma_double", "float64_state",
    "sigmoid", "dk257", "multitoken", "empty", "overlap"])
def test_unsupported_metadata_uses_unchanged_portable_path(unsupported, monkeypatch):
    mixer, arrays, state = fixture(dk=257 if unsupported == "dk257" else 3,
        dtype="float64" if unsupported == "float64_state" else "float32",
        activation="sigmoid" if unsupported == "sigmoid" else "silu",
        tokens={"multitoken": 2, "empty": 0}.get(unsupported, 1))
    arrays = list(arrays)
    if unsupported.endswith("_stride"):
        name = unsupported.removesuffix("_stride")
        if name in ("qkv", "gate", "decay", "beta"):
            i = ("qkv", "gate", "decay", "beta").index(name)
            arrays[i] = strided(arrays[i])
        elif name == "history": state.conv.history = strided(state.conv.history)
        elif name == "memory": state.delta.memory = strided(state.delta.memory)
        elif name == "weight": mixer.conv.weight = strided(mixer.conv.weight)
        elif name == "bias": mixer.conv.bias = strided(mixer.conv.bias)
        else: mixer.norm.norm.gamma = strided(mixer.norm.norm.gamma)
    elif unsupported == "qkv_half": arrays[0] = arrays[0].astype(xp.float16)
    elif unsupported == "gate_half": arrays[1] = arrays[1].astype(xp.float16)
    elif unsupported == "coefficient_double": arrays[2] = arrays[2].astype(xp.float64)
    elif unsupported == "gamma_double": mixer.norm.norm.gamma = mixer.norm.norm.gamma.astype(xp.float64)
    elif unsupported == "overlap":
        # Raw input aliases the final history slot: portable snapshots current.
        arrays[0] = state.conv.history[:, -1:]
    expected_state = state.copy()
    mixer._decode_fastpath_enabled = False
    expected = mixer.forward(*arrays, expected_state)
    mixer._decode_fastpath_enabled = True
    def forbidden(*args):
        pytest.fail("unsupported metadata must not compile/launch decode kernels")
    monkeypatch.setattr(decode, "_get_kernels", forbidden)
    actual = mixer.forward(*arrays, state)
    np.testing.assert_array_equal(asnumpy(actual), asnumpy(expected))
    np.testing.assert_array_equal(asnumpy(state.conv.history), asnumpy(expected_state.conv.history))
    np.testing.assert_array_equal(asnumpy(state.delta.memory), asnumpy(expected_state.delta.memory))
    assert state.length == expected_state.length


@GPU
@pytest.mark.gpu
def test_three_launches_nondefault_stream_and_full_array_lifetimes(monkeypatch):
    stream = xp.cuda.Stream(non_blocking=True)
    with stream:
        mixer, arrays, state = fixture(tokens=1)
        expected_state = state.copy()
        mixer._decode_fastpath_enabled = False
        expected = mixer.forward(*arrays, expected_state)
        mixer._decode_fastpath_enabled = True
        get_kernels = decode._get_kernels
        launched = []
        def counted(device):
            kernels = get_kernels(device)
            assert kernels is not None, "NVRTC compilation unexpectedly unavailable"
            def wrap(kernel, name):
                def launch(*args, **kwargs):
                    assert kwargs["stream"].ptr == stream.ptr
                    launched.append(name)
                    return kernel(*args, **kwargs)
                return launch
            return tuple(wrap(k, n) for k, n in zip(kernels, decode._NAMES))
        monkeypatch.setattr(decode, "_get_kernels", counted)
        output = mixer.forward(*arrays, state)
        assert launched == list(decode._NAMES)
        _, refs = decode._PENDING[(arrays[0].device.id, stream.ptr)][-1]
        for array in (*arrays, mixer.conv.weight, mixer.norm.norm.gamma,
                      state.conv.history, state.delta.memory, output):
            assert any(r is array for r in refs)
        ref = weakref.ref(arrays[0])
        del arrays, refs, array
        assert ref() is not None
    stream.synchronize()
    check(output, expected, "nondefault-stream")
    check(state.delta.memory, expected_state.delta.memory, "nondefault-stream/memory")


@GPU
@pytest.mark.gpu
def test_small_l2_vectors_biasless_and_scaled_updated_memory():
    with xp.cuda.Stream(non_blocking=True) as stream:
        mixer, arrays, state = fixture(tokens=1)
        mixer.conv.bias = None
        state.conv.history *= 1e-4
        arrays = list(arrays)
        arrays[0] *= 1e-4
        expected, history, memory = reference(mixer, arrays, state)
        output = mixer.forward(*arrays, state)
    stream.synchronize()
    check(output, expected, "sum-l2-epsilon")
    check(state.delta.memory, memory, "sum-l2-epsilon/memory")
    np.testing.assert_array_equal(asnumpy(state.conv.history), history)


@GPU
@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("MINI_LLM_BENCH_GATED_DELTA_DECODE") != "1",
                    reason="opt-in bounded synthetic timing")
def test_bounded_isolated_latency():
    stream = xp.cuda.Stream(non_blocking=True)
    with stream:
        # Synthetic generic geometry only, 1 MiB recurrent memory, no model.
        rng = np.random.default_rng(10622)
        mixer = GatedDeltaMixerInference(8, 16, 128, 128, 4)
        mixer.set_weights(xp.asarray(rng.normal(0, .1, (4096, 4)), dtype=xp.float32),
                          xp.ones(128, dtype=xp.float32))
        arrays = (xp.asarray(rng.normal(0, .3, (1, 1, 4096)), dtype=xp.float32),
                  xp.ones((1, 1, 16, 128), dtype=xp.float32),
                  xp.full((1, 1, 16), -.1, dtype=xp.float32),
                  xp.full((1, 1, 16), .4, dtype=xp.float32))
        results = {}
        for enabled in (False, True):
            mixer._decode_fastpath_enabled = enabled
            state = mixer.create_state(1)
            for _ in range(5):
                mixer.forward(*arrays, state)
            stream.synchronize()
            start, end = xp.cuda.Event(), xp.cuda.Event()
            start.record(stream)
            wall_start = time.perf_counter()
            for _ in range(40):
                mixer.forward(*arrays, state)
            host_ms = (time.perf_counter() - wall_start) * 1000 / 40
            end.record(stream)
            end.synchronize()
            span_ms = xp.cuda.get_elapsed_time(start, end) / 40
            results[enabled] = host_ms, span_ms
        print("synthetic B1 Hk8 Hv16 Dk128 Dv128 K4; 5 warmups, 40 calls/path; "
              f"portable host/span ms={results[False]}; decode host/span ms={results[True]}; "
              f"host speedup={results[False][0] / results[True][0]:.3f}x")
        assert decode._MODULES[arrays[0].device.id] is not None
