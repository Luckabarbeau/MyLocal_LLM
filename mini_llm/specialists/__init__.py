"""Experimental shared-foundation / independently trained specialist tools.

The initial module contains only the frozen-foundation interface. Specialist
Transformer blocks and private historical readers are intentionally separate
later phases; none of the existing training objectives are affected.
"""
from .frozen_foundation import FrozenFoundation
from .residual_specialist import ResidualSpecialist, SpecialistConfig

__all__ = ["FrozenFoundation", "ResidualSpecialist", "SpecialistConfig"]
