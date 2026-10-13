"""0080 opt-in two-stage probabilistic block proposal architecture.

This module is deliberately independent from the old 0079 refinement trainer.
The production autoregressive model remains unchanged until integrated 0080D.
"""

from .contracts import BlockSchedule, block_schedule, prefix_lm_allow_mask, shifted_decoder_tokens

__all__ = [
    'BlockSchedule', 'block_schedule', 'prefix_lm_allow_mask',
    'shifted_decoder_tokens',
]
