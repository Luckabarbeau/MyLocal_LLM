"""Sparse real GGUF rows, independent uint64 C++ hash and float64 PLE graph."""

import importlib.util
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

from mini_llm.backend import asnumpy, xp
from mini_llm.external.qwen4exp_ple import PLEConfig, PLEState, Qwen4ExpPLE
from mini_llm.ops.quantized_weight import PackedGGUFWeight


D = 2560
EOS = 248044
TOKENS = [7, 9, EOS, 3, 248046, 2**31 - 1, 11, 11, 1, 19, 8, 0, 13, EOS, 5]
CODEBOOK = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                     1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


def metadata():
    return {"qwen4exp.ple.ngram_size": 3, "qwen4exp.ple.heads_per_ngram": 8,
            "qwen4exp.ple.conv_kernel": 4, "qwen4exp.embedding_length_per_layer_input": 160,
            "qwen4exp.embedding_length": D, "qwen4exp.ple.eos_token_id": EOS,
            "tokenizer.ggml.eos_token_id": 248046,
            "qwen4exp.ple.layer_multipliers": [2**64 - 1, 20109073645365, 8052911324071],
            "qwen4exp.ple.head_offsets": [h * 1000003 for h in range(16)],
            "qwen4exp.ple.head_vocab_sizes": [1000003] * 16,
            "qwen4exp.attention.layer_norm_rms_epsilon": 1e-5}


def hash_oracle(token, previous, config):
    # Independent byte truncation rather than the implementation's integer mask.
    ctx = [token, *previous[::-1]]
    for j in range(1, 3):
        if ctx[j] < 0 or ctx[j] == config.eos:
            ctx[j:] = [config.eos] * (3 - j)
            break
    terms = [int.from_bytes((t * m).to_bytes(16, "little")[:8], "little")
             for t, m in zip(ctx, config.multipliers)]
    return tuple((terms[0] ^ terms[1] ^ (terms[2] if h >= 8 else 0)) % config.vocabs[h]
                 + config.offsets[h] for h in range(16))


def raw_row(index):
    return b"".join(struct.pack("<e", (b + 1) / 128) +
                    bytes(((index + b + j) % 16) | (((index + 3*b + j + 5) % 16) << 4)
                          for j in range(16)) for b in range(5))


def row_oracle(index):
    # IQ4_NL has low nibble -> first 16, high nibble -> last 16 in each block.
    return np.concatenate([np.concatenate((CODEBOOK[(index + b + np.arange(16)) % 16],
                                          CODEBOOK[(index + 3*b + np.arange(16) + 5) % 16]))
                           * np.float32((b + 1) / 128) for b in range(5)])


def gguf_string(text):
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


@pytest.fixture
def store(tmp_path):
    pytest.importorskip("gguf")
    from mini_llm.external.gguf_store import MultiShardTensorStore

    md = metadata()
    rows = 16 * 1000003
    config = PLEConfig.from_metadata(md, rows)
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, len(md))
    for name, value in md.items():
        header += gguf_string(name)
        if isinstance(value, list):
            header += struct.pack("<IIQ", 9, 10, len(value))
            header += struct.pack("<" + "Q" * len(value), *value)
        elif isinstance(value, float):
            header += struct.pack("<If", 6, value)
        else:
            header += struct.pack("<IQ", 10, value)
    header += gguf_string("per_layer_token_embd.weight")
    header += struct.pack("<IQQIQ", 2, 160, rows, 20, 0)  # GGML IQ4_NL=20
    offset = (len(header) + 31) // 32 * 32
    path = tmp_path / "sparse.gguf"
    selected = set()
    for sequence in (TOKENS, [1, 2, 3], [9, 8, 7]):
        prev = (-1, -1)
        for token in sequence:
            selected.update(hash_oracle(token, prev, config))
            prev = (prev[1], token)
    with path.open("wb") as file:
        file.write(header)
        file.truncate(offset + rows * 90)  # 1.44 GB logical, only selected pages allocated
        for row in selected:
            file.seek(offset + row * 90)
            file.write(raw_row(row))
    with MultiShardTensorStore.open(path) as result:
        yield result


def weights():
    rng = np.random.default_rng(206)
    gammas = [rng.uniform(0.3, 1.4, (4, D)).astype(np.float32) for _ in range(3)]
    conv = rng.normal(0, 0.13, (4 * D, 4)).astype(np.float32)
    # Structured exact linear projections exercise all streams, sign and zero gates.
    def key(e):
        return xp.concatenate((e, -e, xp.zeros_like(e), xp.roll(e, 17)))
    def value(e):
        return xp.roll(e, 3) * xp.float32(0.7) + e * xp.float32(0.2)
    args = dict(key=key, value=value, gamma_key=xp.asarray(gammas[0]),
                gamma_query=xp.asarray(gammas[1]), gamma_conv=xp.asarray(gammas[2]),
                conv=xp.asarray(conv))
    return args, gammas, conv


def make_module(store):
    return Qwen4ExpPLE.from_store(store, **weights()[0])


def test_hash_boundaries_overflow_and_current_eos():
    config = PLEConfig.from_metadata(metadata(), 16 * 1000003)
    for token in TOKENS:
        for prev in [(-1, -1), (-1, 7), (7, 9), (9, EOS), (EOS, 9), (248046, 9)]:
            assert config.rows(token, prev) == hash_oracle(token, prev, config)
    assert config.rows(EOS, (7, 9)) != config.rows(EOS, (-1, -1))
    assert config.rows(2, (7, EOS)) == config.rows(2, (-1, EOS))
    assert config.rows(2, (7, 248046)) != config.rows(2, (-1, EOS))


def test_original_strata_cpp_hash(tmp_path):
    """Compile extracted original functions if the pinned checkout is available.

    Runtime module never imports/opens this checkout. Portable Python oracle
    above remains mandatory even when this optional provenance check skips.
    """
    root = Path(os.environ.get("QWEN4EXP_STRATA_SOURCE",
                "/home/lucka/Documents/my_local_LLM/qwen4exp-references/Strata"))
    source = root / "src/kernels/ngram.cpp"
    compiler = shutil.which("c++")
    if not compiler or not source.is_file():
        pytest.skip("optional original Strata checkout/C++ compiler unavailable")
    revision = subprocess.check_output(["git", "--no-pager", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert revision == "c77572806996349b49995e4f98d41b5294d36a21"
    original = source.read_text()
    extracted = original[original.index("uint64_t ngram_mixed("):original.index("\nnamespace {", original.index("void ngram_rows("))]
    cpp = tmp_path / "hash.cpp"
    cpp.write_text('''#include <cstdint>
#include <iostream>
constexpr int NGRAM_SIZE=3, PLE_N_HEADS=16, HEADS_PER_NGRAM=8;
constexpr int TOKEN_NULL=-1, PLE_EOS_TOKEN_ID=248044;
struct PleConsts { uint64_t mult[3], vocab[16], offset[16]; };
''' + extracted + '''
int main() {
 PleConsts c{}; for(auto &m:c.mult) std::cin>>m;
 for(auto &v:c.vocab) std::cin>>v; for(auto &o:c.offset) std::cin>>o;
 int32_t t, p[2]; uint32_t out[16];
 while(std::cin>>t>>p[0]>>p[1]) { ngram_rows(&t,p,1,c,out);
 for(auto r:out) std::cout<<r<<" "; std::cout<<"\\n"; }
}
''')
    binary = tmp_path / "hash"
    subprocess.run([compiler, "-std=c++17", str(cpp), "-o", str(binary)], check=True, timeout=30)
    config = PLEConfig.from_metadata(metadata(), 16 * 1000003)
    cases = [(t, p) for t in TOKENS for p in [(-1, -1), (EOS, 9), (7, EOS), (7, 9)]]
    text = " ".join(map(str, (*config.multipliers, *config.vocabs, *config.offsets))) + "\n"
    text += "\n".join(f"{t} {p[0]} {p[1]}" for t, p in cases)
    output = subprocess.check_output([str(binary)], input=text, text=True, timeout=10)
    assert [tuple(map(int, line.split())) for line in output.splitlines()] == [config.rows(t, p) for t, p in cases]


def test_bounded_rows_and_embedding_order(store, monkeypatch):
    module = make_module(store)
    calls = []
    original_rows = store.rows
    original_decode = PackedGGUFWeight.dequantize_reference
    def bounded_rows(name, first, last, **kwargs):
        assert last == first + 1
        calls.append(first)
        return original_rows(name, first, last, **kwargs)
    def bounded_decode(weight):
        assert weight.shape == (1, 160), "full-table decode attempted"
        return original_decode(weight)
    def forbidden(*args, **kwargs):
        pytest.fail("whole table accessor invoked")
    monkeypatch.setattr(store, "rows", bounded_rows)
    monkeypatch.setattr(store, "packed_weight", forbidden)
    monkeypatch.setattr(store, "tensor_bytes", forbidden)
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", bounded_decode)
    e = module.embedding(7, (-1, -1))
    expected_indices = hash_oracle(7, (-1, -1), module.config)
    assert calls == list(expected_indices)
    np.testing.assert_array_equal(asnumpy(e), np.concatenate([row_oracle(i) for i in expected_indices]))
    assert e.dtype == xp.float32 and e.shape == (D,)


def test_graph_float64_oracle_and_state(store):
    args, gammas, kernel = weights()
    module = Qwen4ExpPLE.from_store(store, **args)
    state = module.fresh_state()
    rng = np.random.default_rng(81)
    inputs = rng.normal(0, 0.4, (len(TOKENS), 4, D)).astype(np.float32)
    hist = np.zeros((9, 4, D), np.float64)
    prev = (-1, -1)
    norm = lambda a, g: a / np.sqrt(np.mean(a*a, axis=-1, keepdims=True) + float(module.epsilon)) * g
    for t, x in zip(TOKENS, inputs):
        emb = np.concatenate([row_oracle(i) for i in hash_oracle(t, prev, module.config)]).astype(np.float64)
        key = np.stack((emb, -emb, np.zeros(D), np.roll(emb, 17)))
        k, q = norm(key, gammas[0]), norm(x.astype(np.float64), gammas[1])
        v = np.roll(emb, 3) * float(np.float32(.7)) + emb * float(np.float32(.2))
        score = np.sum(k*q, axis=-1) / np.sqrt(D)
        gate = 1 / (1 + np.exp(-np.sign(score)*np.sqrt(np.maximum(np.abs(score), 1e-6))))
        assert gate[2] == 0.5
        u = gate[:, None] * v
        z = norm(u, gammas[2])
        conv = sum(a * kernel.reshape(4, D, 4)[..., j] for j, a in enumerate((hist[0], hist[3], hist[6], z)))
        expected = x + (u + conv / (1 + np.exp(-conv)))
        old_history = asnumpy(state.normalized_history).copy()
        output, new_state = module.step(t, xp.asarray(x), state)
        np.testing.assert_allclose(asnumpy(output), expected, atol=2e-6, rtol=3e-5)
        hist = np.concatenate((hist[1:], z[None]))
        prev = (prev[1], t)
        np.testing.assert_allclose(asnumpy(new_state.normalized_history), hist, atol=1e-6, rtol=2e-5)
        np.testing.assert_array_equal(asnumpy(state.normalized_history), old_history)
        assert new_state.predecessors == prev
        assert output.shape == (4, D) and output.dtype == xp.float32
        assert bool(xp.all(xp.isfinite(output)))
        state = new_state
    # Replay from an explicit reset; an interleaved request must not affect it.
    def sequence(tokens):
        state = module.fresh_state()
        result = []
        for token in tokens:
            out, state = module.step(token, xp.asarray(inputs[0]), state)
            result.append(asnumpy(out))
        return result, state
    first, state1 = sequence([1, 2, 3])
    sequence([9, 8, 7])
    repeated, state2 = sequence([1, 2, 3])
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(asnumpy(state1.normalized_history), asnumpy(state2.normalized_history))


def test_validation_and_projection_contract(store):
    args, _, _ = weights()
    module = Qwen4ExpPLE.from_store(store, **args)
    x = xp.zeros((4, D), dtype=xp.float32)
    state = module.fresh_state()
    for shape in [(1, 4, D), (2, 4, D), (D,)]:
        with pytest.raises(ValueError, match="one token/sequence"):
            module.step(1, xp.zeros(shape, dtype=xp.float32), state)
    with pytest.raises(TypeError, match="float32"):
        module.step(1, x.astype(xp.float64), state)
    with pytest.raises(TypeError, match="explicit"):
        module.step(1, x, None)
    with pytest.raises(ValueError, match="normalized_history"):
        module.step(1, x, PLEState((-1, -1), x))
    with pytest.raises(ValueError, match="two oldest"):
        module.step(1, x, PLEState((-1,), state.normalized_history))
    for name, shape in [("key", (D, 4*D)), ("value", (1, D)),
                        ("gamma_key", (4*D,)), ("conv", (4, 4*D))]:
        # Wrong-shape zero-stride view: no large allocation before validation.
        bad = xp.broadcast_to(xp.asarray(0, dtype=xp.float32), shape)
        with pytest.raises(ValueError, match=name):
            Qwen4ExpPLE.from_store(store, **{**args, name: bad})
    for epsilon in [0, -1, float("nan"), float("inf"), 1e-100]:
        with pytest.raises(ValueError, match="epsilon"):
            Qwen4ExpPLE.from_store(store, **args, epsilon=epsilon)
    bad_module = Qwen4ExpPLE.from_store(store, **{**args, "key": lambda e: e})
    with pytest.raises(ValueError, match="projection output"):
        bad_module.step(1, x, state)
    # Exercise the dense projection operation without allocating real-model weights.
    small = xp.asarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    np.testing.assert_array_equal(asnumpy(module._project(small, xp.ones(4, dtype=xp.float32), 3)), [6, 22, 38])
    for name, value in [("qwen4exp.ple.ngram_size", 4),
                        ("qwen4exp.ple.head_vocab_sizes", [0]*16),
                        ("qwen4exp.ple.layer_multipliers", [1.5, 2, 3]),
                        ("qwen4exp.ple.head_offsets", [16*1000003]*16)]:
        with pytest.raises(ValueError):
            PLEConfig.from_metadata({**metadata(), name: value}, 16*1000003)
    with pytest.raises(ValueError, match="IQ4_NL"):
        Qwen4ExpPLE(store, module.table_name,
                    PLEConfig((1, 2, 3), (0,)*16, (1,)*16, EOS, 2), **args, epsilon=1e-5)


def test_optional_actual_cupy():
    if os.environ.get("MINI_LLM_BACKEND") == "cupy":
        pytest.skip("already executing on actual CuPy")
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy not installed")
    probe = subprocess.run([sys.executable, "-c", "import cupy; print(cupy.cuda.runtime.getDeviceCount())"],
                           capture_output=True, text=True, timeout=30)
    if probe.returncode or int(probe.stdout.strip().splitlines()[-1]) == 0:
        pytest.skip("no usable CuPy device")
    env = {**os.environ, "MINI_LLM_BACKEND": "cupy"}
    subprocess.run([sys.executable, "-m", "pytest", "-q", __file__, "-k",
                    "not optional_actual_cupy and not original_strata_cpp_hash"],
                   env=env, check=True, timeout=120)
