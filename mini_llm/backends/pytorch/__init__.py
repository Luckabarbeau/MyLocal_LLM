"""Optional first-class PyTorch execution backend for MyLocal_LLM."""

from .model import DecoderLanguageModel, TorchDecoderLanguageModel
from .optimizer import HybridMuonAdamW

__all__ = ["DecoderLanguageModel", "HybridMuonAdamW", "TorchDecoderLanguageModel"]
