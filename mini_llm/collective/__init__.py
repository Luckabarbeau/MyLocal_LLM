"""Learned output-distribution fusion for shared-vocabulary language models."""

from .probability_fusion import ProbabilityResidualCombiner, FusionConfig

__all__ = ["ProbabilityResidualCombiner", "FusionConfig"]
