"""0203 direct packed kernels: bounded 0202 oracle and pinned GGML C fixtures."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from mini_llm.ops import native_packed_cpu as native
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight
from test_0202_iq_formats import FORMATS, c_decode, c_oracles  # noqa: F401


def packed(fmt, n, k, seed):
    qk, qb = BLOCK_FORMATS[fmt]
    assert k % qk == 0
    raw = np.random.default_rng(seed).integers(0, 256, (n * k // qk, qb), np.uint8)
    raw[:, :2] = np.frombuffer(np.float16(1 / 1024).tobytes(), np.uint8)
    raw.flags.writeable = False
    return PackedGGUFWeight(raw.reshape(-1), (n, k), fmt)


def compare(got, expected, label, *, atol=2e-5, rtol=5e-5):
    error = float(np.max(np.abs(got.astype(np.float64) - expected.astype(np.float64))))
    print(f"{label}: dtype={got.dtype} max_abs_diff={error:.17g} atol={atol} rtol={rtol}")
    np.testing.assert_allclose(got, expected, atol=atol, rtol=rtol)


def reject_reference(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Native execution must not call a reference decoder")
    for method in ("dequantize_reference", "dequantize_rows_reference", "matvec_reference"):
        monkeypatch.setattr(PackedGGUFWeight, method, reject)


def projection(x, w):
    # Bounded oracle even for representative model dimensions, not full FP32 weights.
    return np.stack([w.matvec_reference(row, tile_rows=8) for row in x])


def expert_oracle(x, weights):
    gate, up, down = weights
    g, u = projection(x, gate), projection(x, up)
    return projection((g / (1 + np.exp(-g))) * u, down)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_direct_gemv_oracle_immutable_noncontiguous(fmt, dtype, monkeypatch, tmp_path):
    w = packed(fmt, 7, 512, 203)
    path = tmp_path / "packed.bin"
    path.write_bytes(w.data.tobytes())
    mapped = np.memmap(path, mode="r", dtype=np.uint8)
    w = PackedGGUFWeight(mapped, w.shape, fmt)
    before = w.data.tobytes()
    x = np.random.default_rng(204).normal(0, .02, (2, 3, 1024)).astype(dtype)[..., ::2]
    x_before = x.copy()
    expected = projection(x.astype(np.float32).reshape(-1, 512), w).reshape(2, 3, 7).astype(dtype)
    reject_reference(monkeypatch)
    got = native.packed_cpu_linear(x, w)
    assert got.shape == expected.shape and got.dtype == dtype
    compare(got, expected, f"{fmt} GEMV", atol=2e-5 if dtype != np.float16 else 1e-4)
    assert w.data.tobytes() == before and not w.data.flags.writeable
    np.testing.assert_array_equal(x, x_before)


@pytest.mark.parametrize("fmt", FORMATS)
def test_independent_ggml_decoded_fixture(fmt, c_oracles, monkeypatch):
    w = packed(fmt, 5, 512, 205)
    raw = w.data.reshape(-1, FORMATS[fmt][1])
    decoded = c_decode(c_oracles, fmt, raw).reshape(w.shape)
    # Identity probes verify every decoded lane, not just a random reduction.
    basis = np.eye(512, dtype=np.float32)
    x = np.random.default_rng(206).normal(0, .02, (3, 512)).astype(np.float32)
    reject_reference(monkeypatch)
    got = native.packed_cpu_linear(basis, w)
    compare(got, decoded.T, f"{fmt} GGML lanes", atol=0, rtol=0)
    compare(native.packed_cpu_linear(x, w), x @ decoded.T, f"{fmt} GGML GEMV")


@pytest.mark.parametrize("fmt", FORMATS)
def test_complete_expert_all_six_formats(fmt, monkeypatch):
    d = ff = 256
    weights = tuple(packed(fmt, ff, d, 210 + i) for i in range(3))
    x = np.random.default_rng(211).normal(0, .02, (3, d)).astype(np.float32)
    expected = expert_oracle(x, weights)
    before = [w.data.tobytes() for w in weights]
    reject_reference(monkeypatch)
    got = native.packed_cpu_expert(x, weights)
    compare(got, expected, f"{fmt} unweighted expert")
    assert got.dtype == x.dtype
    assert [w.data.tobytes() for w in weights] == before


@pytest.mark.parametrize("down_fmt", ["IQ4_NL", "Q2_0"])
def test_representative_mixed_expert_d2560_f640(down_fmt, monkeypatch):
    d, ff = 2560, 640  # FFN width is deliberately not divisible by 256.
    weights = (packed("IQ2_S", ff, d, 220), packed("IQ3_S", ff, d, 221),
               packed(down_fmt, d, ff, 222))
    x = np.random.default_rng(223).normal(0, .01, (1, d)).astype(np.float32)
    expected = expert_oracle(x, weights)
    reject_reference(monkeypatch)
    compare(native.packed_cpu_expert(x, weights), expected,
            f"D2560 F640 IQ2_S/IQ3_S/{down_fmt}")


def test_top10_parallel_unweighted_experts(monkeypatch):
    formats = list(FORMATS)
    d = ff = 256
    experts = [tuple(packed(formats[(e + i) % len(formats)], ff, d, 230 + 3 * e + i)
                     for i in range(3)) for e in range(10)]
    x = np.random.default_rng(231).normal(0, .02, (2, d)).astype(np.float32)
    expected = [expert_oracle(x, ws) for ws in experts]
    before = [[w.data.tobytes() for w in ws] for ws in experts]
    reject_reference(monkeypatch)
    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(lambda ws: native.packed_cpu_expert(x, ws), experts))
    for e, (output, reference) in enumerate(zip(got, expected)):
        compare(output, reference, f"top10 expert {e} unweighted")
        np.testing.assert_array_equal(native.packed_cpu_expert(x, experts[e]), output)
    # Router weighting belongs to the caller; no weighting is applied by the kernel.
    router = np.arange(1, 11, dtype=np.float32) / 55
    compare(sum(p * y for p, y in zip(router, got)),
            sum(p * y for p, y in zip(router, expected)), "top10 caller-weighted sum")
    assert [[w.data.tobytes() for w in ws] for ws in experts] == before


def test_native_c_argument_alignment_guards():
    lib = native._load_native()
    w = packed("IQ4_NL", 2, 640, 240)
    x = np.ones((1, 640), np.float32)
    out = np.full((1, 2), np.nan, np.float32)
    for fmt, k in ((-1, 640), (12, 640), (native._FORMAT["IQ3_S"], 640),
                   (native._FORMAT["IQ4_NL"], 639), (native._FORMAT["Q2_0"], 608)):
        assert lib.mylocal_packed_gemv(w.data.ctypes.data, fmt, 2, k,
                                      x.ctypes.data, 1, out.ctypes.data) == 1
        assert np.isnan(out).all()


def test_header_bytes_invalidate_native_cache(tmp_path, monkeypatch):
    source = Path(native.__file__).with_name("native_packed_cpu.cpp")
    header = source.with_name("gguf_codebooks.h")
    (tmp_path / source.name).write_bytes(source.read_bytes())
    header_copy = tmp_path / header.name
    header_copy.write_bytes(header.read_bytes())
    monkeypatch.setattr(native, "__file__", str(tmp_path / "native_packed_cpu.py"))
    monkeypatch.setenv("MINI_LLM_NATIVE_BUILD_DIR", str(tmp_path / "build"))
    monkeypatch.setattr(native, "_LIB", None)
    monkeypatch.setattr(native, "_FAILURE", None)
    first = native._load_native()
    header_copy.write_bytes(header_copy.read_bytes() + b"\n// cache invalidation probe\n")
    monkeypatch.setattr(native, "_LIB", None)
    second = native._load_native()
    assert first._name != second._name
    assert len(list((tmp_path / "build").glob("packed_cpu_*.so"))) == 2
