from .config import ModelConfig
from .parameter import Parameter

# Blocks
from .blocks.transformer_block import TransformerBlock
from .blocks.moe_block import MoE

# Ops
from .ops.router import Router
from .ops.experts import Experts, ExpertFFN

# Model
from .model.decoder_lm import DecoderLanguageModel

# Training
from .train import MiniTrainer
from .train_extended import ExtendedTrainer

# Inference (inference module is at package root)
import sys
from pathlib import Path

# Add parent directory to path for inference module
_parent = Path(__file__).parent.parent
_inference_path = str(_parent / "inference.py")
if _inference_path not in sys.path:
    sys.path.insert(0, str(_parent))

from inference import TextGenerator
