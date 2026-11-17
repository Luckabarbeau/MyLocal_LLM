"""Bounded synthetic HC decode tests, not full-model or execution parity."""

import numpy as np
import pytest

from mini_llm.backend import asnumpy, xp
from mini_llm.external import qwen4exp_hyperconnection as hc_module
from mini_llm.ops import hyperconnection_decode_gpu as gpu


GPU = pytest.mark.skipif(xp.__name__ != "cupy", reason="actual CuPy required")
HC = hc_module.Qwen4ExpHyperConnection


def pair(d=7, rank=3, inject=True, seed=232):
    rng = np.random.default_rng(seed)
    weights = [rng.uniform(-1.5, 1.5, (4, d)),
               rng.normal(0, .4 / np.sqrt(d), (rank, 4*d)),
               rng.normal(0, .4, (4*d, rank))]
    if inject:
        weights.append(rng.normal(0, .4 / np.sqrt(d), (4, 4*d)))
    arrays = [xp.asarray(w, dtype=xp.float32) for w in weights]
    return (HC(*arrays, epsilon=1e-5),
            HC(*arrays, epsilon=1e-5, fast_decode=True), rng)


def sigmoid64(x):
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + z), z / (1 + z))


def oracle(h, hc):
    h = np.asarray(h, dtype=np.float64)
    gamma, down, up = [asnumpy(w).astype(np.float64)
                       for w in (hc.gamma, hc.down, hc.up)]
    n = (h / np.sqrt(np.mean(h*h, axis=-1, keepdims=True)
                     + float(hc.epsilon))) * gamma
    flat = n.reshape(h.shape[:-2] + (-1,))
    low = (flat @ down.T) / 4
    low = low * sigmoid64(low)
    mixed = np.mean(n * sigmoid64(low @ up.T).reshape(h.shape), axis=-2)
    gains = None if hc.inject is None else 2 * sigmoid64(
        (flat @ asnumpy(hc.inject).astype(np.float64).T) / 4)
    return mixed, gains


def check(actual, expected, label):
    actual, expected = asnumpy(actual), np.asarray(expected, dtype=np.float64)
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    error = float(np.max(np.abs(actual - expected)))
    scale = float(np.max(np.abs(expected)))
    relative = error / max(scale, np.finfo(np.float32).tiny)
    print(f"{label}: max_abs={error:.9g}, normwise_rel={relative:.9g}")
    # 2e-6 is the existing HC relative tolerance; use an infinity-norm
    # comparison to avoid unjustified elementwise demands near cancellation.
    assert error <= 2e-6 * scale, (label, error, scale, relative)


@pytest.mark.parametrize("leading", [(), (1,), (1, 1)])
@GPU
@pytest.mark.parametrize("d,rank,scale", [(1, 1, 1), (7, 3, .001),
                                          (257, 16, 1), (1024, 32, 1000)])
def test_read_write_fp64_reference_and_fresh_inputs(leading, d, rank, scale):
    ref, fast, rng = pair(d, rank)
    assert ref.fast_decode is False and fast.fast_decode is True
    h = xp.asarray(rng.normal(0, scale, leading + (4, d)), dtype=xp.float32)
    output = xp.asarray(rng.normal(0, scale, leading + (d,)), dtype=xp.float32)
    snapshots = [asnumpy(a).copy() for a in (h, output, fast.gamma, fast.down,
                                            fast.up, fast.inject)]
    x64, g64 = oracle(asnumpy(h), fast)
    x, g = fast.read(h)
    xr, gr = ref.read(h)
    check(x, x64, f"FP64 read D={d} leading={leading}")
    check(g, g64, f"FP64 gain D={d}")
    check(x, asnumpy(xr), f"reference read D={d}")
    check(g, asnumpy(gr), f"reference gain D={d}")
    updated = fast.write(h, output, g)
    check(updated, snapshots[0].astype(np.float64)
          + g64[..., :, None] * snapshots[1][..., None, :], f"FP64 write D={d}")
    check(updated, asnumpy(ref.write(h, output, gr)), f"reference write D={d}")
    # Exactly two roundings, even where a fused multiply-add would differ.
    np.testing.assert_array_equal(asnumpy(updated), asnumpy(
        h + g[..., :, None] * output[..., None, :]))
    for a, saved in zip((h, output, fast.gamma, fast.down, fast.up, fast.inject), snapshots):
        np.testing.assert_array_equal(asnumpy(a), saved)
    saved_g = asnumpy(g).copy()
    fast.read(h * xp.float32(-.7))
    np.testing.assert_array_equal(asnumpy(g), saved_g)


@GPU
def test_paired_tokens_four_streams_and_final_excludes_injection(monkeypatch):
    ref, fast, rng = pair(19, 5)
    h = fast.initialize(xp.asarray(rng.normal(size=19), dtype=xp.float32))
    for c in range(4):
        np.testing.assert_array_equal(asnumpy(h[c]), asnumpy(h[0]))
    reference = asnumpy(h).astype(np.float64)
    for token in range(4):
        for block in range(2):
            x64, g64 = oracle(reference, fast)
            x, g = fast.read(h)
            check(x, x64, f"token={token} block={block} read")
            check(g, g64, f"token={token} block={block} gain")
            output = xp.tanh(x) * xp.float32(.75)
            h = fast.write(h, output, g)
            reference += g64[:, None] * (.75 * np.tanh(x64))[None, :]
            check(h, reference, f"token={token} block={block} four streams")
        if token != 3:
            # Independent next-token embedding, not a time-axis interpretation.
            embedding = xp.asarray(rng.normal(size=19), dtype=xp.float32)
            h = fast.initialize(embedding)
            reference = asnumpy(h).astype(np.float64)
    expected = fast.collapse(h)
    fast.inject.fill(xp.nan)
    monkeypatch.setattr(hc_module, "gains_gpu", lambda *a: pytest.fail("collapse injected"))
    np.testing.assert_array_equal(asnumpy(fast.collapse(h)), asnumpy(expected))
    check(expected, oracle(reference, ref)[0], "final FP64 collapse")
    _, final, _ = pair(19, 5, inject=False)
    assert final.read(h)[1] is None
    np.testing.assert_array_equal(asnumpy(final.collapse(h)), asnumpy(final.read(h)[0]))


@GPU
def test_negatives_saturation_and_zero():
    a = lambda x: xp.asarray(x, dtype=xp.float32)
    ref = HC(a(np.ones((4, 1))), a([[4000]*4]),
             a([[4000], [-4000], [4000], [-4000]]),
             a([[4000]*4, [-4000]*4]*2), epsilon=1e-5)
    fast = HC(ref.gamma, ref.down, ref.up, ref.inject,
              epsilon=1e-5, fast_decode=True)
    for value in (0, 1, -1):
        h = xp.full((4, 1), value, dtype=xp.float32)
        x, g = fast.read(h)
        x64, g64 = oracle(asnumpy(h), fast)
        check(x, x64, f"saturation read {value}")
        check(g, g64, f"saturation gain {value}")
        np.testing.assert_array_equal(asnumpy(x), asnumpy(ref.read(h)[0]))
        np.testing.assert_array_equal(asnumpy(g), asnumpy(ref.read(h)[1]))


@GPU
def test_current_nondefault_stream():
    stream = xp.cuda.Stream(non_blocking=True)
    with stream:
        ref, fast, rng = pair(513, 9)
        h = xp.asarray(rng.normal(size=(1, 4, 513)), dtype=xp.float32)
        h *= xp.float32(-3)
        x, g = fast.read(h)
        updated = fast.write(h, x, g)
        # A same-stream consumer must see completed outputs.
        consumed = updated.copy()
        xr, gr = ref.read(h)
        expected = ref.write(h, xr, gr)
    stream.synchronize()
    check(consumed, asnumpy(expected), "nondefault stream")


@pytest.mark.parametrize("kind", ["tokens", "strided", "output_strided", "gains_strided"])
def test_exact_fallback_and_numpy_unchanged(monkeypatch, kind):
    ref, fast, rng = pair()
    leading = (2, 3) if kind == "tokens" else (1,)
    base = xp.asarray(rng.normal(size=leading + (4, 14)), dtype=xp.float32)
    h = base[..., ::2] if kind == "strided" else base[..., :7].copy()
    output = xp.ones(leading + (14,), dtype=xp.float32)[..., ::2]
    gains = xp.ones(leading + (8,), dtype=xp.float32)[..., ::2]
    if kind != "output_strided":
        output = output.copy()
    if kind != "gains_strided":
        gains = gains.copy()
    def forbidden(*args):
        pytest.fail("ineligible path launched a fused kernel")
    if kind in ("tokens", "strided") or xp.__name__ != "cupy":
        monkeypatch.setattr(hc_module, "mix_gpu", forbidden)
        monkeypatch.setattr(hc_module, "gains_gpu", forbidden)
        x, g = fast.read(h)
        xr, gr = ref.read(h)
        np.testing.assert_array_equal(asnumpy(x), asnumpy(xr))
        np.testing.assert_array_equal(asnumpy(g), asnumpy(gr))
        np.testing.assert_array_equal(asnumpy(fast.collapse(h)), asnumpy(ref.collapse(h)))
    monkeypatch.setattr(hc_module, "write_gpu", forbidden)
    np.testing.assert_array_equal(asnumpy(fast.write(h, output, gains)),
                                  asnumpy(ref.write(h, output, gains)))
    if xp.__name__ != "cupy":
        assert not gpu.eligible(h)


@pytest.mark.parametrize("enabled", [False, True])
def test_existing_validation_before_dispatch(enabled):
    _, fast, _ = pair()
    fast.fast_decode = enabled
    for bad in (xp.ones((4, 8), dtype=xp.float32),
                xp.ones((4, 7), dtype=xp.float64)):
        with pytest.raises((ValueError, TypeError)):
            fast.read(bad)
        with pytest.raises((ValueError, TypeError)):
            fast.collapse(bad)
    h = xp.ones((4, 7), dtype=xp.float32)
    for output, gains in ((xp.ones((1, 7), dtype=xp.float32), xp.ones(4, dtype=xp.float32)),
                          (xp.ones(7, dtype=xp.float32), xp.ones(4, dtype=xp.float64))):
        with pytest.raises((ValueError, TypeError)):
            fast.write(h, output, gains)
