"""Tokenizer module.

This module provides multiple tokenizer backends:
- SimpleBPETokenizer: Reference Python implementation (educational)
- FastBPETokenizer: Production BPE using Hugging Face tokenizers (fast)

Special tokens are consistently defined as:
- <|pad|>: Padding token (ID 0)
- <|eos|>: End of sequence token (ID 1)
- <|unk|>: Unknown token (ID 2)
"""

from mini_llm.tokenizer.tokenizer import (
    SimpleBPETokenizer,
    FastBPETokenizer,
    TokenizerProtocol,
    SPECIAL_TOKEN_IDS,
    get_tokenizer,
)
from mini_llm.tokenizer.config import TokenizerConfig, TokenizerManager

__all__ = [
    "SimpleBPETokenizer",
    "FastBPETokenizer",
    "TokenizerProtocol",
    "SPECIAL_TOKEN_IDS",
    "get_tokenizer",
    "TokenizerConfig",
    "TokenizerManager",
]
