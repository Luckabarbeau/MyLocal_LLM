"""Rebuild a supported byte-level BPE tokenizer from GGUF metadata.

No model weights or network access are required.  Only the explicitly
validated GGUF GPT-2 BPE/SmolLM pre-tokenizer variant is supported; other
algorithms are rejected rather than silently approximated.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GGUFTokenizerSpec:
    model: str
    pre: str
    tokens: tuple[str, ...]
    merges: tuple[str, ...]
    token_types: tuple[int, ...]
    bos_id: Optional[int]
    eos_id: Optional[int]
    add_bos: bool
    chat_template: Optional[str]


def _value(reader, name, default=None):
    field = reader.fields.get(name)
    return default if field is None else field.contents(0)


def _array(reader, name):
    field = reader.fields.get(name)
    if field is None:
        raise ValueError(f"GGUF tokenizer requires {name}")
    values = field.contents(slice(None))
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"GGUF tokenizer field {name} is not an array")
    return tuple(values)


def tokenizer_spec_from_gguf(reader, *, vocab_size=None):
    """Parse and validate tokenizer metadata without requiring tokenizers."""
    model = _value(reader, "tokenizer.ggml.model")
    pre = _value(reader, "tokenizer.ggml.pre")
    if (model, pre) != ("gpt2", "smollm"):
        raise ValueError(
            f"Unsupported GGUF tokenizer (model={model!r}, pre={pre!r}); "
            "only the verified SmolLM GPT-2 BPE variant is supported"
        )
    tokens = _array(reader, "tokenizer.ggml.tokens")
    merges = _array(reader, "tokenizer.ggml.merges")
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError("GGUF tokenizer tokens are empty or contain duplicates")
    if vocab_size is not None and len(tokens) != int(vocab_size):
        raise ValueError(f"GGUF tokenizer has {len(tokens)} entries but model expects {vocab_size}")
    if not merges:
        raise ValueError("GGUF tokenizer has no BPE merges")
    for merge in merges:
        if not isinstance(merge, str) or len(merge.split(" ")) != 2:
            raise ValueError(f"Invalid GGUF BPE merge: {merge!r}")
    types = _value(reader, "tokenizer.ggml.token_type")
    if types is None:
        token_types = tuple(1 for _ in tokens)
    else:
        token_types = tuple(int(t) for t in _array(reader, "tokenizer.ggml.token_type"))
    if len(token_types) != len(tokens) or not set(token_types).issubset({1, 2, 3, 4, 5, 6}):
        raise ValueError("GGUF tokenizer token types are invalid or inconsistent")

    def optional_id(key):
        val = _value(reader, key)
        if val is None:
            return None
        val = int(val)
        if val < 0 or val >= len(tokens):
            raise ValueError(f"GGUF {key} ID {val} is outside tokenizer vocabulary")
        return val

    template = _value(reader, "tokenizer.chat_template")
    return GGUFTokenizerSpec(
        model=model, pre=pre, tokens=tokens, merges=merges,
        token_types=token_types,
        bos_id=optional_id("tokenizer.ggml.bos_token_id"),
        eos_id=optional_id("tokenizer.ggml.eos_token_id"),
        add_bos=bool(_value(reader, "tokenizer.ggml.add_bos_token", False)),
        chat_template=template,
    )


class GGUFTokenizer:
    """Compatibility wrapper with the subset of `tokenizers.Tokenizer` used by CLI."""

    def __init__(self, tokenizer, spec):
        self._tokenizer = tokenizer
        self.spec = spec
        self.chat_template = spec.chat_template

    def encode(self, text, add_special_tokens=False):
        # The CLI supplies its own ChatML framing; no implicit BOS is inserted.
        # add_special_tokens=True is deliberately unsupported until model-specific
        # BOS/EOS postprocessing has been validated against the source tokenizer.
        if add_special_tokens:
            raise ValueError("GGUF tokenizer requires explicit special-token framing")
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids, skip_special_tokens=False):
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token):
        return self._tokenizer.token_to_id(token)

    def chat_prompt(self, text):
        """Format one user message for verified SmolLM2 ChatML-style tokens."""
        # SmolLM2-Instruct uses this plain ChatML form, independently of whether
        # the GGUF has a metadata template. Do not guess for other model families.
        if self.token_to_id("<|im_start|>") is None or self.token_to_id("<|im_end|>") is None:
            raise ValueError("Cannot build SmolLM2 chat prompt: ChatML tokens missing")
        return "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"


def load_gguf_tokenizer(reader, *, vocab_size=None):
    """Construct an offline tokenizer using only GGUF metadata and `tokenizers`."""
    spec = tokenizer_spec_from_gguf(reader, vocab_size=vocab_size)
    try:
        from tokenizers import Tokenizer, AddedToken, Regex, models, pre_tokenizers, decoders
    except ImportError as exc:
        raise RuntimeError("Install offline GGUF tokenization: pip install tokenizers") from exc
    vocab = {value: idx for idx, value in enumerate(spec.tokens)}
    merges = [tuple(merge.split(" ", 1)) for merge in spec.merges]
    unknown_id = next((i for i, t in enumerate(spec.token_types) if t == 2), None)
    bpe = models.BPE(
        vocab=vocab,
        merges=merges,
        unk_token=spec.tokens[unknown_id] if unknown_id is not None else None,
    )
    tokenizer = Tokenizer(bpe)
    # SmolLM pre-tokenization is GPT-2 byte-level BPE with additional single-
    # digit isolation. Ref: llama.cpp's LLAMA_VOCAB_PRE_TYPE_SMOLLM regexes.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"\p{N}"), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    specials = [AddedToken(value, single_word=False, lstrip=False, rstrip=False,
                           normalized=False, special=True)
                for value, t in zip(spec.tokens, spec.token_types) if t == 3]
    if specials:
        tokenizer.add_special_tokens(specials)
    # Preserve explicit USER_DEFINED tokens as no-normalization added tokens,
    # but never turn them into stop tokens automatically.
    custom = [AddedToken(value, single_word=False, lstrip=False, rstrip=False,
                         normalized=False, special=False)
              for value, t in zip(spec.tokens, spec.token_types) if t == 4]
    if custom:
        tokenizer.add_tokens(custom)
    if tokenizer.get_vocab_size(with_added_tokens=True) != len(spec.tokens):
        raise ValueError("GGUF tokenizer construction altered vocabulary size")
    for idx, token in enumerate(spec.tokens):
        if tokenizer.token_to_id(token) != idx:
            raise ValueError(f"GGUF tokenizer changed token ID for {token!r}")
    return GGUFTokenizer(tokenizer, spec)
