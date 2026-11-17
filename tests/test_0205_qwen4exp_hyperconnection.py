"""Synthetic mathematical tests, NOT llama.cpp/Radiance execution parity."""

import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mini_llm.backend import asnumpy, xp
from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection


def array(value):
    return xp.asarray(value, dtype=xp.float32)


def connection(d=3, rank=2, seed=205, inject=True):
    rng = np.random.default_rng(seed)
    weights = [rng.uniform(0.5, 1.5, (4, d)),
               rng.normal(0, 0.4, (rank, 4 * d)),
               rng.normal(0, 0.4, (4 * d, rank))]
    if inject:
        weights.append(rng.normal(0, 0.4, (4, 4 * d)))
    return Qwen4ExpHyperConnection(*map(array, weights), epsilon=1e-5)


def oracle(streams, hc):
    """Independent Float64 scalar-loop mathematical oracle, canonical weights."""
    h = np.asarray(streams, dtype=np.float64)
    gamma, down, up = (asnumpy(w).astype(np.float64) for w in
                       (hc.gamma, hc.down, hc.up))
    inj = None if hc.inject is None else asnumpy(hc.inject).astype(np.float64)
    d, rank = hc.width, down.shape[0]
    rows = h.reshape(-1, 4, d)
    mixes, gains = [], []
    for row in rows:
        n = []
        for c in range(4):
            scale = math.sqrt(sum(float(v) ** 2 for v in row[c]) / d + float(hc.epsilon))
            n.extend(row[c, i] / scale * gamma[c, i] for i in range(d))
        low = []
        for j in range(rank):
            a = sum(down[j, k] * n[k] for k in range(4 * d)) / 4
            low.append(a / (1 + math.exp(-a)))
        mixed = []
        for i in range(d):
            total = 0.0
            for c in range(4):
                j = c * d + i
                a = sum(up[j, k] * low[k] for k in range(rank))
                total += n[j] / (1 + math.exp(-a))
            mixed.append(total / 4)
        mixes.append(mixed)
        if inj is not None:
            gains.append([2 / (1 + math.exp(-sum(inj[c, k] * n[k]
                                                for k in range(4 * d)) / 4))
                          for c in range(4)])
    leading = h.shape[:-2]
    return (np.array(mixes).reshape(leading + (d,)),
            None if inj is None else np.array(gains).reshape(leading + (4,)))


def check(actual, expected, label):
    actual = asnumpy(actual)
    expected = asnumpy(expected)
    assert actual.dtype == np.float32
    diff = float(np.max(np.abs(actual - expected)))
    print(f"{label} maximum_absdiff={diff:.9g}")
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_exact_zero_gates_and_original_residual():
    hc = Qwen4ExpHyperConnection(array([[2], [4], [6], [8]]),
                                array(np.zeros((1, 4))), array(np.zeros((4, 1))),
                                array(np.zeros((4, 4))), epsilon=3)
    h = array([[1], [1], [1], [1]])
    mixed, gains = hc.read(h)
    np.testing.assert_array_equal(asnumpy(mixed), [1.25])  # half mean([1,2,3,4])
    np.testing.assert_array_equal(asnumpy(gains), np.ones(4))
    np.testing.assert_array_equal(asnumpy(hc.write(h, array([2]), gains)), np.full((4, 1), 3))


def test_hand_nonzero_down_scaling_and_injection_scaling():
    hc = Qwen4ExpHyperConnection(array(np.full((4, 1), 2)), array([[1, 1, 1, 1]]),
                                array([[1], [2], [-1], [0]]),
                                array([[0] * 4, [1] * 4, [-1] * 4, [2] * 4]), epsilon=3)
    mixed, gains = hc.read(array(np.ones((4, 1))))  # normalized features all exactly 1
    silu_one = 1 / (1 + math.exp(-1))
    sigmoid = lambda v: 1 / (1 + math.exp(-v))
    check(mixed, [(sigmoid(silu_one) + sigmoid(2 * silu_one)
                   + sigmoid(-silu_one) + 0.5) / 4], "hand read")
    check(gains, [1, 2 * sigmoid(1), 2 * sigmoid(-1), 2 * sigmoid(2)], "hand gains")


@pytest.mark.parametrize("leading,d,rank", [((), 1, 1), ((5,), 3, 2), ((2, 3), 7, 5)])
def test_full_paired_blocks_and_final_collapse(leading, d, rank):
    rng = np.random.default_rng(205 + d)
    first, second = connection(d, rank), connection(d, rank, seed=206)
    final = connection(d, rank, seed=207, inject=False)
    embedding = array(rng.normal(size=leading + (d,)))
    h = first.initialize(embedding)
    reference = np.repeat(asnumpy(embedding)[..., None, :], 4, axis=-2).astype(np.float64)
    snapshots = [(w, asnumpy(w).copy()) for hc in (first, second, final)
                 for w in (hc.gamma, hc.down, hc.up) + (() if hc.inject is None else (hc.inject,))]
    for index, hc in enumerate((first, second)):
        saved = asnumpy(h).copy()
        expected_x, expected_gains = oracle(reference, hc)
        x, gains = hc.read(h)
        check(x, expected_x, f"block{index} read D={d}")
        check(gains, expected_gains, f"block{index} gains D={d}")
        # Deliberately nonidentity block: two successive, independently paired connections.
        output = xp.tanh(x) * xp.float32(index + 0.75)
        expected_output = np.tanh(expected_x) * (index + 0.75)
        output_saved, gains_saved = asnumpy(output).copy(), asnumpy(gains).copy()
        updated = hc.write(h, output, gains)
        reference = reference + expected_gains[..., :, None] * expected_output[..., None, :]
        check(updated, reference, f"block{index} write D={d}")
        np.testing.assert_array_equal(asnumpy(h), saved)
        np.testing.assert_array_equal(asnumpy(output), output_saved)
        np.testing.assert_array_equal(asnumpy(gains), gains_saved)
        h = updated
    check(final.collapse(h), oracle(reference, final)[0], f"collapse D={d}")
    x, gains = final.read(h)
    assert gains is None
    np.testing.assert_array_equal(asnumpy(x), asnumpy(final.collapse(h)))
    for w, saved in snapshots:
        np.testing.assert_array_equal(asnumpy(w), saved)
    np.testing.assert_array_equal(asnumpy(embedding), asnumpy(first.initialize(embedding))[..., 0, :])


def test_initialize_per_token_independence_and_flat_gamma_copy():
    gamma = array(np.arange(1, 13).reshape(4, 3))
    down, up = array(np.zeros((2, 12))), array(np.zeros((12, 2)))
    hc = Qwen4ExpHyperConnection(gamma.reshape(-1), down, up, epsilon=1e-5)
    canonical = Qwen4ExpHyperConnection(gamma, down, up, epsilon=1e-5)
    gamma[...] = 99
    embedding = array([[1, 2, 3], [-3, -2, -1]])
    h = hc.initialize(embedding)
    for c in range(4):
        np.testing.assert_array_equal(asnumpy(h[:, c]), asnumpy(embedding))
    np.testing.assert_array_equal(asnumpy(hc.collapse(h)), asnumpy(canonical.collapse(h)))
    check(hc.collapse(h)[0], hc.collapse(h[0]), "token independent")
    h[0, 0, 0] = 100
    assert float(embedding[0, 0]) == 1 and float(h[0, 1, 0]) == 1


@pytest.mark.parametrize("kind", ["gamma_broadcast", "gamma_transposed", "gamma_flat", "down", "up", "inject", "rank", "fp16", "fp64"])
def test_constructor_validation(kind):
    gamma, down, up, inj = array(np.ones((4, 3))), array(np.zeros((2, 12))), array(np.zeros((12, 2))), array(np.zeros((4, 12)))
    if kind == "gamma_broadcast": gamma = array(np.ones((1, 3)))
    if kind == "gamma_transposed": gamma = array(np.ones((3, 4)))
    if kind == "gamma_flat": gamma = array(np.ones(11))
    if kind == "down": down = down.T
    if kind == "up": up = up.T
    if kind == "inject": inj = inj.T
    if kind == "rank": down = array(np.zeros((0, 12)))
    if kind == "fp16": gamma = gamma.astype(xp.float16)
    if kind == "fp64": down = down.astype(xp.float64)
    with pytest.raises((ValueError, TypeError)):
        Qwen4ExpHyperConnection(gamma, down, up, inj, epsilon=1e-5)


@pytest.mark.parametrize("epsilon", [0, -1, float("nan"), float("inf"), 1e-100, 1e100])
def test_epsilon_validation(epsilon):
    with pytest.raises(ValueError):
        Qwen4ExpHyperConnection(array(np.ones((4, 2))), array(np.zeros((1, 8))),
                                array(np.zeros((8, 1))), epsilon=epsilon)


def test_activation_shape_dtype_validation():
    hc = connection()
    h = array(np.ones((2, 4, 3)))
    for bad in (array(np.ones((2, 12))), array(np.ones((2, 1, 3))), h.astype(xp.float16)):
        with pytest.raises((ValueError, TypeError)):
            hc.read(bad)
    for out, gains in ((array(np.ones(3)), array(np.ones((2, 4)))),
                       (array(np.ones((2, 3))), array(np.ones(4))),
                       (array(np.ones((2, 3))), array(np.ones((2, 4, 1)))),
                       (array(np.ones((2, 3))).astype(xp.float64), array(np.ones((2, 4))))):
        with pytest.raises((ValueError, TypeError)):
            hc.write(h, out, gains)
    with pytest.raises(ValueError):
        hc.initialize(array(np.ones((2, 4))))
    with pytest.raises(TypeError):
        hc.initialize(array(np.ones(3)).astype(xp.float16))


def test_stable_saturated_sigmoid_and_zero_norm():
    hc = Qwen4ExpHyperConnection(array(np.ones((4, 1))), array([[4000] * 4]),
                                array([[4000], [-4000], [4000], [-4000]]),
                                array([[4000] * 4, [-4000] * 4] * 2), epsilon=1e-5)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mixed, gains = hc.read(array(np.ones((4, 1))))
        assert bool(xp.isfinite(mixed).all())
        np.testing.assert_array_equal(asnumpy(gains), [2, 0, 2, 0])
        zero, gains = hc.read(array(np.zeros((4, 1))))
        np.testing.assert_array_equal(asnumpy(zero), [0])
        np.testing.assert_array_equal(asnumpy(gains), [1] * 4)


def test_optional_cupy_actual_backend():
    if xp.__name__ == "cupy":
        assert isinstance(connection().initialize(array([1, 2, 3])), xp.ndarray)
        return
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy not installed in this interpreter")
    env = dict(os.environ, MINI_LLM_BACKEND="cupy")
    probe = subprocess.run([sys.executable, "-c", "import cupy as cp; cp.ones(1); cp.cuda.Stream.null.synchronize()"],
                           env=env, capture_output=True, text=True, timeout=30)
    if probe.returncode:
        pytest.skip("CuPy device unavailable: " + probe.stderr[-500:])
    result = subprocess.run([sys.executable, "-m", "pytest", str(Path(__file__).resolve()),
                             "-q", "-s", "-k", "not optional_cupy"], env=env,
                            capture_output=True, text=True, timeout=120)
    print(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
