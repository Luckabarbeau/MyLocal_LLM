"""CPU-only Qwen2 GGUF tokenization, independent of model/weight loading.

Source: llama.cpp 171e8846b4af9766c354064cb776cb34a50f053f,
llama-vocab.cpp LLAMA_VOCAB_PRE_TYPE_QWEN2 and unicode.cpp
unicode_regex_split_custom_qwen2. Expected Unicode splits are explicit, not
computed using the production regex. A small Python rank-BPE implementation
checks actual GGUF vocabulary/merges independently of Rust tokenizers.

The discovered local packs/*/tokenizer/tokenizer.json files are qwen35
metadata exports with a different vocabulary, NOT matching HF references.
These tests do not claim full HF/llama.cpp runtime parity or chat parity.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from mini_llm.external.gguf_tokenizer import (
    load_gguf_tokenizer,
    tokenizer_spec_from_gguf,
)


INSTALLED_GGUF = Path(
    "/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/"
    "Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
)

# Each expected chunk follows the ordered Qwen2 alternatives before UTF-8
# byte encoding. In particular, Qwen2 does NOT use Qwen35's letter+mark runs.
SPLITS = [
    ("", []),
    ("I'm I'M WE'RE we'd he'll you've can't",
     ["I", "'m", " I", "'M", " WE", "'RE", " we", "'d", " he", "'ll",
      " you", "'ve", " can", "'t"]),
    ("café 中文 Ελληνικά Привет 🙂e\u0301x",
     ["café", " 中文", " Ελληνικά", " Привет", " 🙂", "e", "\u0301x"]),
    ("e\u0301\u0308!", ["e", "\u0301\u0308!"]),
    ("123 ٤٥６ Ⅳ²", ["1", "2", "3", " ", "٤", "٥", "６", " ", "Ⅳ", "²"]),
    ("def f(x):\r\n\treturn x + 12\n\n",
     ["def", " f", "(x", "):\r\n", "\treturn", " x", " +", " ", "1", "2", "\n\n"]),
    ("a \t\r\n  b\t \n\n   ", ["a", " \t\r\n", " ", " b", "\t \n\n", "   "]),
    ("  \tfoo\rbar\n baz", ["  ", "\tfoo", "\r", "bar", "\n", " baz"]),
    ("foo_bar... ==\r\n", ["foo", "_bar", "...", " ==\r\n"]),
    ("\u00a0x\u2003\u2003y\u0085", ["\u00a0x", "\u2003", "\u2003y", "\u0085"]),
]


def _byte_alphabet():
    # GPT-2's reversible byte alphabet, without using tokenizers internals.
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {byte: chr(byte) for byte in visible}
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(256 + len(mapping) - len(visible))
    return mapping


BYTE_ALPHABET = _byte_alphabet()


def _byte_encode(text):
    return "".join(BYTE_ALPHABET[byte] for byte in text.encode("utf-8"))


def _reader(**overrides):
    tokens = [BYTE_ALPHABET[byte] for byte in range(256)] + [
        "ab", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>", "[PAD]",
    ]
    raw = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.token_type": [1] * 257 + [3, 3, 3, 4, 5],
        "tokenizer.ggml.bos_token_id": 257,
        "tokenizer.ggml.eos_token_id": 259,
        # Even if requested by metadata, raw completion adds nothing implicitly.
        "tokenizer.ggml.add_bos_token": True,
        "tokenizer.chat_template": "{{ 1 / 0 }}",
    }
    raw.update(overrides)
    fields = {}
    for name, value in raw.items():
        if value is not None:
            fields[name] = SimpleNamespace(contents=lambda offset, value=value: (
                list(value) if isinstance(offset, slice) else
                value[0] if isinstance(value, (list, tuple)) else value
            ))
    return SimpleNamespace(fields=fields)


@pytest.fixture(scope="module")
def qwen():
    pytest.importorskip("tokenizers")
    return load_gguf_tokenizer(_reader(), vocab_size=262)


@pytest.mark.parametrize("text,chunks", SPLITS)
def test_qwen2_pretokenization(qwen, text, chunks):
    assert "".join(chunks) == text
    actual = qwen._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    assert [piece for piece, _ in actual] == [_byte_encode(chunk) for chunk in chunks]
    assert qwen.decode(qwen.encode(text).ids) == text


def test_qwen2_special_ids_raw_completion_and_vocab(qwen):
    assert qwen.spec.bos_id == 257
    assert qwen.spec.eos_id == 259
    assert qwen.spec.add_bos is True
    assert qwen.encode("").ids == []
    assert qwen.encode("ab").ids == [256]
    assert qwen.encode("<|endoftext|>ab<|im_end|><tool_call>").ids == [257, 256, 259, 260]
    assert qwen.decode([257, 256, 259, 260]) == "<|endoftext|>ab<|im_end|><tool_call>"
    # USER_DEFINED stays an added token, not a control/stop token.
    assert qwen.decode([257, 256, 259, 260], skip_special_tokens=True) == "ab<tool_call>"
    assert qwen._tokenizer.get_vocab() == dict(zip(qwen.spec.tokens, range(262)))
    assert qwen.token_to_id("[PAD]") == 261
    with pytest.raises(ValueError, match="explicit special-token framing"):
        qwen.encode("ab", add_special_tokens=True)


def test_qwen_chat_is_rejected_even_with_chatml_and_template(qwen):
    assert qwen.chat_template == "{{ 1 / 0 }}"
    with pytest.raises(ValueError, match="chat framing is not validated"):
        qwen.chat_prompt("Hi")


@pytest.mark.parametrize("model,pre", [
    ("gpt2", "qwen35"), ("gpt2", "qwen3"), ("gpt2", "qwen2.5"),
    ("gpt2", "llama3"), ("gpt2", None), ("gpt2", ""),
    ("llama", "qwen2"), (None, "qwen2"),
])
def test_unsupported_variants_fail_before_tokenizer_construction(model, pre):
    reader = _reader(**{"tokenizer.ggml.model": model, "tokenizer.ggml.pre": pre})
    with pytest.raises(ValueError, match="Unsupported GGUF tokenizer"):
        tokenizer_spec_from_gguf(reader)
    with pytest.raises(ValueError, match="Unsupported GGUF tokenizer"):
        load_gguf_tokenizer(reader)


@pytest.mark.parametrize("key,value", [
    ("tokenizer.ggml.bos_token_id", -1),
    ("tokenizer.ggml.eos_token_id", 262),
])
def test_qwen_invalid_special_id_fails_closed(key, value):
    with pytest.raises(ValueError, match="outside tokenizer vocabulary"):
        tokenizer_spec_from_gguf(_reader(**{key: value}))


def test_smollm_variant_selection_keeps_legacy_pipeline():
    pytest.importorskip("tokenizers")
    smollm = load_gguf_tokenizer(_reader(**{"tokenizer.ggml.pre": "smollm"}))
    # Qwen2 consumes tab+word and punctuation+CRLF together; GPT-2 does not.
    pieces = smollm._tokenizer.pre_tokenizer.pre_tokenize_str("\tword!\r\n")
    assert [piece for piece, _ in pieces] == [
        _byte_encode(chunk) for chunk in ["\t", "word", "!", "\r\n"]
    ]
    assert smollm.chat_prompt("Hi") == (
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    )


@pytest.fixture(scope="module")
def installed():
    if not INSTALLED_GGUF.is_file():
        pytest.skip("installed Qwen3-Coder-Next GGUF not available")
    gguf = pytest.importorskip("gguf")
    pytest.importorskip("tokenizers")

    class MetadataOnlyReader(gguf.GGUFReader):
        def _build_tensors(self, *args, **kwargs):
            # Parse metadata/directory only: never even construct payload views.
            pass

    reader = MetadataOnlyReader(INSTALLED_GGUF, mode="r")
    assert reader.tensors == []
    tokenizer = load_gguf_tokenizer(reader, vocab_size=151936)
    vocab = dict(zip(tokenizer.spec.tokens, range(len(tokenizer.spec.tokens))))
    ranks = {tuple(merge.split(" ")): rank
             for rank, merge in enumerate(tokenizer.spec.merges)}
    return tokenizer, vocab, ranks


def _reference_bpe(chunks, vocab, ranks):
    """Slow greedy pair-rank BPE, independent of production tokenizers code."""
    ids = []
    for chunk in chunks:
        symbols = list(_byte_encode(chunk))
        while len(symbols) > 1:
            candidates = [(ranks[pair], index)
                          for index, pair in enumerate(zip(symbols, symbols[1:]))
                          if pair in ranks]
            if not candidates:
                break
            _, index = min(candidates)
            symbols[index:index + 2] = [symbols[index] + symbols[index + 1]]
        ids.extend(vocab[symbol] for symbol in symbols)
    return ids


@pytest.mark.parametrize("text,chunks", SPLITS)
def test_installed_qwen_raw_ids_match_independent_rank_bpe(installed, text, chunks):
    tokenizer, vocab, ranks = installed
    expected = _reference_bpe(chunks, vocab, ranks)
    actual = tokenizer.encode(text).ids
    assert actual == expected
    assert tokenizer.decode(actual) == text


def test_installed_metadata_vocab_and_specials(installed):
    tokenizer, vocab, _ = installed
    assert (tokenizer.spec.model, tokenizer.spec.pre) == ("gpt2", "qwen2")
    assert len(tokenizer.spec.merges) == 151387
    assert tokenizer.spec.bos_id is None
    assert tokenizer.spec.add_bos is False
    assert tokenizer.spec.eos_id == 151645
    assert tokenizer._tokenizer.get_vocab() == vocab
    expected = {
        "<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645,
        "<|vision_pad|>": 151654, "<tool_call>": 151657, "</tool_call>": 151658,
        "<|fim_prefix|>": 151659, "<|fim_middle|>": 151660,
        "<|fim_suffix|>": 151661, "<think>": 151667, "</think>": 151668,
        "[PAD151935]": 151935,
    }
    for token, token_id in expected.items():
        assert tokenizer.token_to_id(token) == token_id
    for token_id, token_type in enumerate(tokenizer.spec.token_types):
        if token_type not in (3, 4):
            continue
        token = tokenizer.spec.tokens[token_id]
        assert tokenizer.encode(token).ids == [token_id]
        assert tokenizer.decode([token_id]) == token
        assert tokenizer.decode([token_id], skip_special_tokens=True) == (
            "" if token_type == 3 else token
        )
    raw = "<|fim_prefix|>def f():\n    pass<|fim_suffix|>\n<|fim_middle|>"
    assert tokenizer.decode(tokenizer.encode(raw).ids) == raw
    with pytest.raises(ValueError, match="chat framing is not validated"):
        tokenizer.chat_prompt("Write code")
