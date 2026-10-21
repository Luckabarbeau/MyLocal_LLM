"""Validate GGUF tokenizer metadata and fail-closed reconstruction behavior.

The pure-metadata tests run without the optional Rust tokenizers package.
Real SmolLM2 / HF equivalence is an opt-in CLI test using the actual GGUF.
"""
from types import SimpleNamespace

import pytest

from mini_llm.external.gguf_tokenizer import (
    GGUFTokenizer, GGUFTokenizerSpec, tokenizer_spec_from_gguf, load_gguf_tokenizer,
)
from benchmarks.try_external_gguf import _validate_embedded_tokenizer


class Field:
    def __init__(self, value):
        self.value = value

    def contents(self, offset):
        if isinstance(offset, slice):
            return list(self.value)
        return self.value[0] if isinstance(self.value, (tuple, list)) else self.value


def fake_reader(**overrides):
    raw = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "smollm",
        "tokenizer.ggml.tokens": ["a", "b", "ab", "<|im_start|>", "<|im_end|>"],
        "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.token_type": [1, 1, 1, 3, 3],
        "tokenizer.ggml.eos_token_id": 4,
        "tokenizer.ggml.add_bos_token": False,
        "tokenizer.chat_template": "{{ messages }}",
    }
    raw.update(overrides)
    return SimpleNamespace(fields={key: Field(value) for key, value in raw.items()
                                   if value is not None})


def test_metadata_reads_specials_and_template():
    spec = tokenizer_spec_from_gguf(fake_reader(), vocab_size=5)
    assert spec.tokens[2] == "ab"
    assert spec.merges == ("a b",)
    assert spec.token_types == (1, 1, 1, 3, 3)
    assert spec.eos_id == 4 and spec.add_bos is False
    assert spec.chat_template == "{{ messages }}"


@pytest.mark.parametrize("changes, message", [
    ({"tokenizer.ggml.pre": "llama3"}, "Unsupported GGUF tokenizer"),
    ({"tokenizer.ggml.model": "llama"}, "Unsupported GGUF tokenizer"),
    ({"tokenizer.ggml.merges": None}, "requires tokenizer.ggml.merges"),
    ({"tokenizer.ggml.tokens": ["a", "a"]}, "duplicates"),
    ({"tokenizer.ggml.merges": ["not-a-merge"]}, "Invalid GGUF BPE merge"),
    ({"tokenizer.ggml.token_type": [1, 3]}, "token types"),
    ({"tokenizer.ggml.token_type": [1, 1, 1, 3, 8]}, "token types"),
    ({"tokenizer.ggml.eos_token_id": 99}, "outside tokenizer vocabulary"),
])
def test_invalid_metadata_fails_closed(changes, message):
    with pytest.raises(ValueError, match=message):
        tokenizer_spec_from_gguf(fake_reader(**changes))


def test_tokenizer_vocab_mismatch_fails_closed():
    with pytest.raises(ValueError, match="model expects 49152"):
        tokenizer_spec_from_gguf(fake_reader(), vocab_size=49152)


class DummyTokenizer:
    def __init__(self, change=False):
        self.change = change

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        ids = list(text.encode("utf-8"))
        if self.change and "convergence" in text:
            ids = [123] + ids
        return SimpleNamespace(ids=ids)

    def token_to_id(self, token):
        return {"<|im_start|>": 3, "<|im_end|>": 4}.get(token)


def test_chatml_wrapper_and_parity_diagnostic(capsys):
    spec = GGUFTokenizerSpec("gpt2", "smollm", (), (), (), None, None, False, None)
    wrapped = GGUFTokenizer(DummyTokenizer(), spec)
    assert wrapped.chat_prompt("Hi") == (
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    )
    _validate_embedded_tokenizer(wrapped, DummyTokenizer())
    assert "6 varied inputs" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="first differing index"):
        _validate_embedded_tokenizer(wrapped, DummyTokenizer(change=True))
    with pytest.raises(ValueError, match="explicit special-token framing"):
        wrapped.encode("Hi", add_special_tokens=True)


def test_optional_tokenizers_dependency_failure(monkeypatch):
    # Detect optional-dependency errors without altering normal import flow.
    import builtins
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "tokenizers":
            raise ImportError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="pip install tokenizers"):
        load_gguf_tokenizer(fake_reader(), vocab_size=5)
