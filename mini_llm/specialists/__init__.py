"""Experimental shared-foundation / independently trained specialist tools.

The initial module contains only the frozen-foundation interface. Specialist
Transformer blocks and private historical readers are intentionally separate
later phases; none of the existing training objectives are affected.
"""
from .frozen_foundation import FrozenFoundation
from .residual_specialist import ResidualSpecialist, SpecialistConfig

__all__ = ["FrozenFoundation", "ResidualSpecialist", "SpecialistConfig", "FrozenResidualComposition", "fingerprint_foundation",
           "save_specialist_checkpoint", "load_specialist_checkpoint"]

from .residual_composition import FrozenResidualComposition
from .checkpoint import (fingerprint_foundation, save_specialist_checkpoint,
                         load_specialist_checkpoint)
