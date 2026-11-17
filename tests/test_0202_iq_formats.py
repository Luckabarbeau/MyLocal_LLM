"""0202 CPU oracles: compiled pinned GGML, gguf 0.19, no real weights.

Optional external references skip explicitly when unavailable. Override roots
with PATCH0202_LLAMA / PATCH0202_STRATA; their commits must match the pins.
"""
import ctypes
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from mini_llm.ops import gguf_codebooks as books
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight

FORMATS = {"IQ2_S": (256, 82), "IQ3_S": (256, 110),
           "IQ3_XXS": (256, 98), "IQ4_NL": (32, 18),
           "IQ4_XS": (256, 136), "Q2_0": (64, 18)}
PINS = {"llama": "171e8846b4af9766c354064cb776cb34a50f053f",
        "strata": "c77572806996349b49995e4f98d41b5294d36a21"}
ROOTS = {"llama": "/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/vendor/llama.cpp",
         "strata": "/home/lucka/Documents/my_local_LLM/qwen4exp-references/Strata"}
TABLES = {"ksigns_iq2xs": (128,), "iq2s_grid": (1024, 8),
          "iq3xxs_grid": (256, 4), "iq3s_grid": (512, 4),
          "kvalues_iq4nl": (16,)}


def reference_root(kind):
    root = Path(os.environ.get("PATCH0202_" + kind.upper(), ROOTS[kind]))
    if not root.is_dir():
        pytest.skip(f"pinned {kind} reference unavailable: {root}")
    sha = subprocess.check_output(["git", "--no-pager", "-C", str(root),
                                   "rev-parse", "HEAD"], text=True).strip()
    assert sha == PINS[kind]
    return root


def extract_function(source, name):
    # Copy the actual upstream function, not a second handwritten decoder.
    begin = source.index("void " + name + "(")
    opening = source.index("{", begin)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def compile_shared(path, source):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler unavailable for independent GGML oracle")
    path.with_suffix(".c").write_text(source)
    subprocess.run([cc, "-std=c11", "-O2", "-ffp-contract=off", "-shared", "-fPIC",
                    str(path.with_suffix(".c")), "-o", str(path)], check=True,
                   capture_output=True, text=True, timeout=60)
    return ctypes.CDLL(str(path))


@pytest.fixture(scope="module")
def c_oracles(tmp_path_factory):
    directory = tmp_path_factory.mktemp("0202_ggml")
    libs = {}
    for kind in ("llama", "strata"):
        root = reference_root(kind)
        src = root / ("ggml/src" if kind == "llama" else "third_party/strata-ggml/ggml/src")
        header = src / "ggml-common.h"
        source = (src / "ggml-quants.c").read_text()
        preamble = ('#include <assert.h>\n#include <string.h>\n'
                    '#define GGML_COMMON_DECL_C\n#define GGML_COMMON_IMPL_C\n'
                    f'#include "{header}"\n#define GGML_RESTRICT restrict\n'
                    'static float half_to_float(uint16_t bits) {\n'
                    '  _Float16 h; memcpy(&h, &bits, 2); return (float) h;\n}\n'
                    '#define GGML_FP16_TO_FP32(x) half_to_float(x)\n')
        formats = [f for f in FORMATS if (f == "Q2_0") == (kind == "strata")]
        text = preamble + "\n".join(extract_function(source, "dequantize_row_" + f.lower())
                                     for f in formats)
        if kind == "llama":
            text += "\n" + "\n".join(
                f'const void *ref_{name}(void) {{ return {name}; }}' for name in TABLES)
        lib = compile_shared(directory / (kind + ".so"), text)
        for fmt in formats:
            fn = getattr(lib, "dequantize_row_" + fmt.lower())
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            fn.restype = None
            libs[fmt] = fn
        libs[kind] = lib
    return libs


def c_decode(oracles, fmt, raw):
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    result = np.empty((len(raw), FORMATS[fmt][0]), np.float32)
    oracles[fmt](raw.ctypes.data, result.ctypes.data, result.size)
    return result


def random_blocks(fmt, count=17, seed=202):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (count, FORMATS[fmt][1]), dtype=np.uint8)
    # Finite half values include subnormal, +/-0, extrema and signed scales.
    bits = rng.integers(0, 65536, count, dtype=np.uint16)
    bits[(bits & 0x7c00) == 0x7c00] &= np.uint16(0xfbff)
    edge = np.array([0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff], dtype="<u2")
    bits[:min(count, len(edge))] = edge[:min(count, len(edge))]
    raw[:, :2] = bits.astype("<u2").view(np.uint8).reshape(-1, 2)
    return raw


def compare_c(c_oracles, fmt, raw, label):
    expected = c_decode(c_oracles, fmt, raw)
    got = PackedGGUFWeight(raw, expected.shape, fmt).dequantize_reference()
    error = float(np.max(np.abs(got - expected)))
    print(f"{fmt} {label}: compiled GGML max_abs_error={error:g}")
    np.testing.assert_array_equal(got, expected)
    return expected


@pytest.mark.parametrize("fmt", FORMATS)
def test_layout_and_finite_half_random_c(c_oracles, fmt):
    assert BLOCK_FORMATS[fmt] == FORMATS[fmt]
    compare_c(c_oracles, fmt, random_blocks(fmt, 257), "finite-half/random")


@pytest.mark.parametrize("fmt", [f for f in FORMATS if f != "Q2_0"])
def test_installed_gguf_oracle(fmt):
    pytest.importorskip("gguf")
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize
    raw = random_blocks(fmt, 73)
    expected = dequantize(raw, GGMLQuantizationType[fmt])
    got = PackedGGUFWeight(raw, expected.shape, fmt).dequantize_reference()
    print(f"{fmt}: gguf max_abs_error={np.max(np.abs(got - expected)):g}")
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("fmt", ["IQ2_S", "IQ3_S", "IQ3_XXS"])
def test_exhaustive_grid_indices_highbits_and_signs(c_oracles, fmt):
    count = {"IQ2_S": 1024, "IQ3_S": 512, "IQ3_XXS": 256}[fmt]
    raw = np.zeros((count, FORMATS[fmt][1]), np.uint8)
    raw[:, :2] = np.array([0, 60], np.uint8)  # d = 1
    for i in range(count):
        if fmt == "IQ2_S":
            raw[i, 2:34] = i & 255
            raw[i, 34:66] = np.arange(32, dtype=np.uint8) + np.uint8(i & 255)
            raw[i, 66:74] = (i >> 8) * 0x55
            raw[i, 74:] = np.arange(8, dtype=np.uint8) * 17 + (i & 15)
        elif fmt == "IQ3_S":
            raw[i, 2:66] = i & 255
            raw[i, 66:74] = 255 if i >= 256 else 0
            raw[i, 74:106] = np.arange(32, dtype=np.uint8) + np.uint8(i & 255)
            raw[i, 106:] = np.arange(4, dtype=np.uint8) * 17 + (i & 15)
        else:
            raw[i, 2:66] = i
            aux = np.array([sum(((i + g + lane) & 127) << (7 * lane) for lane in range(4))
                            | (((i + g) & 15) << 28) for g in range(8)], dtype="<u4")
            raw[i, 66:] = aux.view(np.uint8)
    compare_c(c_oracles, fmt, raw, "exhaustive grids/highbits/signs")


@pytest.mark.parametrize("fmt", ["IQ2_S", "IQ3_S", "IQ3_XXS", "IQ4_NL", "IQ4_XS", "Q2_0"])
def test_scale_and_lane_fixtures(c_oracles, fmt):
    raw = random_blocks(fmt, 256, 902)
    raw[:, :2] = [0, 60]
    for i in range(256):
        if fmt == "IQ2_S":
            raw[i, 34:66] = i  # every independent sign mask
            raw[i, 74:] = i    # every pair of scale nibbles
        elif fmt == "IQ3_S":
            raw[i, 74:106] = i
            raw[i, 106:] = i
        elif fmt == "IQ3_XXS":
            raw[i, 66:] = i  # full scale nibble and sign-index range
        elif fmt == "IQ4_XS":
            scale = i % 64
            raw[i, 2:4] = (scale >> 4) * 85
            raw[i, 4:8] = (scale & 15) * 17
            raw[i, 8:] = np.arange(128, dtype=np.uint8) + np.uint8(i)
        else:
            raw[i, 2:] = i  # all nibbles / two-bit lanes
    compare_c(c_oracles, fmt, raw, "all signs/scales/lanes")


@pytest.mark.parametrize("name", TABLES)
def test_vendored_python_and_c_tables_against_upstream(c_oracles, tmp_path, name):
    header = Path(books.__file__).with_suffix(".h")
    lib = compile_shared(tmp_path / "tables.so", f'#include "{header}"\n'
                         + f'const void *table(void) {{ return {name}; }}\n')
    lib.table.restype = ctypes.c_void_p
    ref = getattr(c_oracles["llama"], "ref_" + name)
    ref.restype = ctypes.c_void_p
    table = getattr(books, name)
    assert table.shape == TABLES[name]
    assert table.dtype == (np.int8 if name == "kvalues_iq4nl" else np.uint8)
    assert not table.flags.writeable
    assert ctypes.string_at(lib.table(), table.nbytes) == table.tobytes()
    assert ctypes.string_at(ref(), table.nbytes) == table.tobytes()


@pytest.mark.parametrize("fmt", FORMATS)
def test_selected_rows_and_bounded_matvec(c_oracles, fmt, monkeypatch, tmp_path):
    block, size = FORMATS[fmt]
    raw = random_blocks(fmt, 22)
    raw[:, :2] = [0, 44]  # modest scales for reduction-error checks
    path = tmp_path / "small.blocks"
    path.write_bytes(raw.tobytes())
    mapped = np.memmap(path, mode="r", dtype=np.uint8)
    experts = PackedGGUFWeight(mapped, (2, 11, block), fmt)
    weight = experts.select_outer(1)
    assert np.shares_memory(weight.data, mapped)
    expected = c_decode(c_oracles, fmt, raw[11:])
    original = PackedGGUFWeight.dequantize_reference
    decoded_rows = []
    def bounded(self):
        decoded_rows.append(self.shape[0])
        assert self.shape[0] <= 3
        return original(self)
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", bounded)
    indices = np.array([10, 0, 10, 3, 2])
    np.testing.assert_array_equal(weight.dequantize_rows_reference(indices, tile_rows=3), expected[indices])
    vector = np.random.default_rng(903).normal(size=block).astype(np.float32)
    got = weight.matvec_reference(vector, tile_rows=3)
    reference = expected @ vector
    print(f"{fmt} tiled GEMV max_abs_error={np.max(np.abs(got - reference)):g}")
    np.testing.assert_allclose(got, reference, rtol=3e-6, atol=2e-5)
    assert max(decoded_rows) <= 3
    assert weight.dequantize_rows_reference(np.array([], dtype=int)).shape == (0, block)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="tile_rows"):
            weight.matvec_reference(vector, tile_rows=bad)
    for bad in ([-1], [11]):
        with pytest.raises(IndexError):
            weight.dequantize_rows_reference(bad)
    with pytest.raises(ValueError, match="integer"):
        weight.dequantize_rows_reference([1.5])
    with pytest.raises(ValueError, match="vector shape"):
        weight.matvec_reference(np.zeros((1, block)))
    with pytest.raises(ValueError, match="matrix"):
        experts.matvec_reference(vector)
    with pytest.raises(ValueError, match="matrix"):
        experts.dequantize_rows_reference(indices)
    with pytest.raises(ValueError, match="innermost"):
        PackedGGUFWeight(np.zeros(size, np.uint8), (2, block // 2), fmt)
    with pytest.raises(ValueError, match="expected"):
        PackedGGUFWeight(np.zeros(size - 1, np.uint8), (1, block), fmt)


def test_q2_raw42_store_and_float_types(tmp_path):
    from test_0201_gguf_store import write_gguf
    from mini_llm.external.gguf_store import MultiShardTensorStore
    raw = np.zeros((1, 18), np.uint8)
    raw[:, :2] = [0, 56]  # d = .5
    raw[:, 2:] = 0xe4  # [-1, 0, 1, 2] in adjacent lanes
    path = tmp_path / "raw42.gguf"
    write_gguf(path, [("q2", (64, 1), 42, 0, raw.tobytes())])
    with MultiShardTensorStore.open(path) as store:
        weight = store.packed_weight("q2")
        assert weight.quant_type == "Q2_0"
        np.testing.assert_array_equal(weight.dequantize_reference(),
                                      np.tile([-.5, 0, .5, 1], 16).reshape(1, 64))
    # Non-packed floating point types stay outside BLOCK_FORMATS.
    assert not {"BF16", "F16", "F32"}.intersection(BLOCK_FORMATS)
