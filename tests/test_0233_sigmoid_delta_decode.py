"""Bounded real-geometry mixer tests; no model execution or latency claims.

Optional independent 0226 replay: set MINI_LLM_0233_CHECKPOINT to the first
GGUF shard and optionally MINI_LLM_0233_CAPTURE to a retained capture directory.
Run with MINI_LLM_BACKEND=cupy and pytest -s to record FP32 error metrics.
"""

import gc
import json
import os
from pathlib import Path
import weakref

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops import gated_delta_decode as decode
from mini_llm.ops.gated_delta_inference import GatedDeltaMixerInference


GPU = pytest.mark.skipif(BACKEND_NAME != "cupy", reason="actual CuPy required")
RTOL, ATOL = 3e-4, 5e-6  # Existing 0106u FP32 mixer tolerances.


def fixture(activation="sigmoid", tokens=16):
    rng = np.random.default_rng(233)
    mapping = tuple(h % 16 for h in range(48)) if activation == "sigmoid" else None
    mixer = GatedDeltaMixerInference(16, 48, 128, 128, 4,
        head_map=mapping, gate_activation=activation)
    a = lambda value: xp.asarray(value, dtype=xp.float32)
    mixer.set_weights(a(rng.normal(0, .3, (10240, 4))),
                      a(rng.uniform(.3, 1.4, 128)))
    state = mixer.create_state(1)
    state.conv.history[...] = a(rng.normal(0, .2, state.conv.history.shape))
    state.delta.memory[...] = a(rng.normal(0, .1, state.delta.memory.shape))
    arrays = (a(rng.normal(0, .4, (1, tokens, 10240))),
              a(rng.normal(0, .8, (1, tokens, 48, 128))),
              a(rng.uniform(-.7, -.01, (1, tokens, 48))),
              a(rng.uniform(.1, .9, (1, tokens, 48))))
    arrays[2][:, 0] = 0
    arrays[3][:, 0, 0] = 0
    arrays[3][:, 0, -1] = 1
    return mixer, arrays, state


def check(actual, expected, label, *, rtol=RTOL, atol=ATOL):
    actual, expected = asnumpy(actual), asnumpy(expected)
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    error = float(np.max(np.abs(actual.astype(np.float64) - expected)))
    scale = float(np.max(np.abs(expected)))
    relative = error / max(scale, np.finfo(np.float32).tiny)
    print(f"{label}: max_abs={error:.9g}, normwise_rel={relative:.9g}, "
          f"rtol={rtol:g}, atol={atol:g}, dtype=float32")
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=label)
    return error, relative


def count_launches(monkeypatch, stream=None):
    original = decode._get_kernels
    launched = []
    def counted(device):
        kernels = original(device)
        assert kernels is not None, "decode compilation unavailable, not a portable pass"
        def wrap(kernel, name):
            def launch(*args, **kwargs):
                if stream is not None:
                    assert kwargs["stream"].ptr == stream.ptr
                launched.append(name)
                return kernel(*args, **kwargs)
            return launch
        return tuple(wrap(k, n) for k, n in zip(kernels, decode._NAMES))
    monkeypatch.setattr(decode, "_get_kernels", counted)
    return launched


@GPU
@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
def test_real_geometry_16_continuous_tokens(activation, monkeypatch):
    mixer, arrays, initial = fixture(activation)
    fast, portable = initial.copy(), initial.copy()
    history = asnumpy(initial.conv.history).copy()
    snapshots = [asnumpy(a).copy() for a in arrays]
    ids = id(fast.conv.history), id(fast.delta.memory)
    launched = count_launches(monkeypatch)
    outputs, expected_outputs, memories, expected_memories = [], [], [], []
    assert mixer.rule.head_map == tuple(
        h % 16 if activation == "sigmoid" else h // 3 for h in range(48))
    for t in range(16):
        inputs = tuple(xp.ascontiguousarray(a[:, t:t+1]) for a in arrays)
        assert decode._eligible(mixer, *inputs, fast)
        mixer._decode_fastpath_enabled = False
        expected_outputs.append(mixer.forward(*inputs, portable))
        mixer._decode_fastpath_enabled = True
        outputs.append(mixer.forward(*inputs, fast))
        memories.append(fast.delta.memory.copy())
        expected_memories.append(portable.delta.memory.copy())
        assert launched == list(decode._NAMES) * (t + 1)
        assert fast.length == portable.length == t + 1
        assert ids == (id(fast.conv.history), id(fast.delta.memory))
        history = np.concatenate((history[:, 1:], snapshots[0][:, t:t+1]), axis=1)
        np.testing.assert_array_equal(asnumpy(fast.conv.history), history)
        np.testing.assert_array_equal(asnumpy(portable.conv.history), history)
    check(xp.concatenate(outputs, axis=1), xp.concatenate(expected_outputs, axis=1),
          activation + "/16-token/output")
    check(xp.stack(memories), xp.stack(expected_memories), activation + "/16-token/memory")
    for a, saved in zip(arrays, snapshots):
        np.testing.assert_array_equal(asnumpy(a), saved)


@GPU
def test_sigmoid_saturation_and_gate_independent_state(monkeypatch):
    mixer, arrays, initial = fixture(tokens=1)
    gate_values = np.array([-np.inf, -1000, -100, -20, -1, 0, 1, 20, 100, 1000, np.inf],
                           dtype=np.float32)
    gates = np.resize(gate_values, arrays[1].shape)
    arrays[1][...] = xp.asarray(gates)
    portable, saturated, zero_gate = initial.copy(), initial.copy(), initial.copy()
    mixer._decode_fastpath_enabled = False
    expected = mixer.forward(*arrays, portable)
    launched = count_launches(monkeypatch)
    mixer._decode_fastpath_enabled = True
    actual = mixer.forward(*arrays, saturated)
    half = mixer.forward(arrays[0], xp.zeros_like(arrays[1]), *arrays[2:], zero_gate)
    assert launched == list(decode._NAMES) * 2
    check(actual, expected, "saturated sigmoid/portable")
    z = np.exp(-np.abs(gates.astype(np.float64)))
    sigmoid64 = np.where(gates >= 0, 1 / (1 + z), z / (1 + z))
    check(actual, (asnumpy(half).astype(np.float64) * 2 * sigmoid64), "saturated sigmoid/FP64 gate")
    np.testing.assert_array_equal(asnumpy(actual)[gates == -np.inf], 0)
    np.testing.assert_array_equal(asnumpy(saturated.delta.memory), asnumpy(zero_gate.delta.memory))
    np.testing.assert_array_equal(asnumpy(saturated.conv.history), asnumpy(zero_gate.conv.history))
    check(saturated.delta.memory, portable.delta.memory, "saturated sigmoid/memory")


@GPU
def test_sigmoid_nondefault_stream_and_full_array_lifetimes(monkeypatch):
    stream = xp.cuda.Stream(non_blocking=True)
    launched = count_launches(monkeypatch, stream)
    with stream:
        mixer, arrays, state = fixture(tokens=1)
        portable = state.copy()
        mixer._decode_fastpath_enabled = False
        expected = mixer.forward(*arrays, portable)
        mixer._decode_fastpath_enabled = True
        output = mixer.forward(*arrays, state)
        consumed = output.copy()  # Same-stream consumer, no default-stream barrier.
        assert launched == list(decode._NAMES)
        pending = decode._PENDING[(arrays[0].device.id, stream.ptr)]
        event, refs = pending[-1]
        owned = (*arrays, mixer.conv.weight, mixer.norm.norm.gamma,
                 mixer.rule._head_indices, state.conv.history, state.delta.memory, output)
        for array in owned:
            assert any(r is array for r in refs)
        weak = [weakref.ref(a) for a in owned]
        del arrays, mixer, state, output, owned, refs, array
        gc.collect()
        assert all(ref() is not None for ref in weak)
        # Pressure the allocator while the pending entry owns all launch operands.
        churn = [xp.empty((10240,), dtype=xp.float32) for _ in range(8)]
        for a in churn:
            a.fill(-233)
    stream.synchronize()
    assert event.done
    check(consumed, expected, "sigmoid/nondefault stream/lifetimes")
    check(weak[-2](), portable.delta.memory, "sigmoid/nondefault stream/memory")
    np.testing.assert_array_equal(asnumpy(weak[-3]()), asnumpy(portable.conv.history))


@pytest.mark.parametrize("value", ["0", "false", "off", "no"])
def test_sigmoid_environment_escape_hatch(value, monkeypatch):
    monkeypatch.setenv("MINI_LLM_GATED_DELTA_DECODE", value)
    mixer, arrays, state = fixture(tokens=1)
    assert mixer._decode_fastpath_enabled is False
    def forbidden(*args):
        pytest.fail("disabled decode tried to compile kernels")
    monkeypatch.setattr(decode, "_get_kernels", forbidden)
    expected_state = state.copy()
    expected = mixer.forward(*arrays, expected_state)
    actual = mixer.forward(*arrays, state)
    np.testing.assert_array_equal(asnumpy(actual), asnumpy(expected))
    np.testing.assert_array_equal(asnumpy(state.delta.memory), asnumpy(expected_state.delta.memory))
    np.testing.assert_array_equal(asnumpy(state.conv.history), asnumpy(expected_state.conv.history))
    assert state.length == expected_state.length == 1


@GPU
def test_optional_independent_0226_captured_operands(monkeypatch):
    checkpoint = os.environ.get("MINI_LLM_0233_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set MINI_LLM_0233_CHECKPOINT for retained 0226 independent replay")
    from mini_llm.external.gguf_store import MultiShardTensorStore
    from mini_llm.external.qwen4exp_mixer import Qwen4ExpMixerSpec

    path = Path(os.environ.get("MINI_LLM_0233_CAPTURE", str(
        Path(__file__).resolve().parents[1] / "docs/qwen4exp/diagnostics_0226_v1")))
    events = {}
    for line in (path / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        key = event["position"], event["key"]
        assert key not in events
        events[key] = event
    def captured(p, key, shape):
        e = events[p, key]
        assert e["tensor"]["type"] == 0
        raw = np.fromfile(path / e["file"], dtype=np.float32)
        assert raw.size == int(np.prod(e["tensor"]["ne"])) == int(np.prod(shape))
        assert np.isfinite(raw).all()
        return raw.reshape(shape)
    with MultiShardTensorStore.open(checkpoint) as store:
        spec = Qwen4ExpMixerSpec.from_metadata(store.metadata)
        assert (spec.n_key_heads, spec.n_value_heads, spec.head_dim, spec.conv_kernel_size) == (16, 48, 128, 4)
        def dense(name):
            t = store.tensors[name]
            raw = store.tensor_bytes(name)
            if t.quant_type == "F32":
                return raw.view("<f4").reshape(t.shape).copy()
            assert t.quant_type == "BF16"
            return (raw.view("<u2").astype(np.uint32) << 16).view(np.float32).reshape(t.shape)
        mixer = GatedDeltaMixerInference(16, 48, 128, 128, 4,
            head_map=tuple(h % 16 for h in range(48)), gate_activation="sigmoid",
            l2_eps=spec.rms_eps, rms_eps=spec.rms_eps)
        mixer.set_weights(xp.asarray(dense("blk.0.ssm_conv1d.weight")),
                          xp.asarray(dense("blk.0.ssm_norm.weight")))
        ssm_a = xp.asarray(dense("blk.0.ssm_a"))
        dt_bias = xp.asarray(dense("blk.0.ssm_dt.bias"))
    launched = count_launches(monkeypatch)
    state = mixer.create_state(1)
    for p in range(16):
        # Same independently captured pre-state per token, not a full-model replay.
        history = captured(p, "conv_state_at-0", (10240, 3)).T[None].copy()
        memory = captured(p, "state_predelta-0", (1, 48, 128, 128)).swapaxes(-1, -2).copy()
        state.conv.history[...] = xp.asarray(history)
        state.delta.memory[...] = xp.asarray(memory)
        state.conv.length = state.delta.length = p
        portable = state.copy()
        qkv = xp.asarray(captured(p, "linear_attn_qkv_mixed-0", (1, 1, 10240)))
        gate = xp.asarray(captured(p, "z-0", (1, 1, 48, 128)))
        alpha = xp.asarray(captured(p, "alpha-0", (1, 1, 48)))
        logits = xp.asarray(captured(p, "beta-0", (1, 1, 48)))
        beta = xp.exp(-xp.logaddexp(xp.float32(0), -logits))
        decay = ssm_a * xp.logaddexp(xp.float32(0), alpha + dt_bias)
        mixer._decode_fastpath_enabled = False
        reference = mixer.forward(qkv, gate, decay, beta, portable)
        mixer._decode_fastpath_enabled = True
        actual = mixer.forward(qkv, gate, decay, beta, state)
        assert launched == list(decode._NAMES) * (p + 1)
        check(actual, reference, f"0226/p{p}/portable")
        check(actual, captured(p, "final_output-0", actual.shape),
              f"0226/p{p}/independent output", rtol=3e-5, atol=2e-5)
        check(state.delta.memory, captured(p, "new_state-0", (1, 48, 128, 128)).swapaxes(-1, -2),
              f"0226/p{p}/independent memory", rtol=3e-5, atol=2e-5)
        expected_history = np.concatenate((history[:, 1:], asnumpy(qkv)), axis=1)
        np.testing.assert_array_equal(asnumpy(state.conv.history), expected_history)
        if p < 15:
            np.testing.assert_array_equal(expected_history,
                captured(p + 1, "conv_state_at-0", (10240, 3)).T[None])
        assert state.length == portable.length == p + 1
