"""0104E: minimal composition for testing frozen-foundation capacity expansion.

No separate lexical embeddings or specialist history readers are introduced.
This numerical component does not claim production ExtendedTrainer support.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrozenResidualComposition:
    foundation: object
    specialist: object

    def __post_init__(self):
        if self.foundation.config.d_model != self.specialist.config.d_model:
            raise ValueError("foundation and specialist hidden widths must match")
        if self.foundation.config.dtype != self.specialist.config.dtype:
            raise ValueError("foundation and specialist compute dtypes must match")

    def parameters(self):
        """Only specialist parameters are optimizer-eligible."""
        return self.specialist.parameters()

    def zero_grad(self):
        self.specialist.zero_grad()

    def loss(self, inputs, targets, *, loss_mask=None, position_ids=None,
             memory_metadata=None, dual_resolution_metadata=None,
             chunk_tokens=512, activation_checkpoint=False, return_cache=True):
        h, target_slice = self.foundation.hidden(
            inputs, position_ids=position_ids, memory_metadata=memory_metadata,
            dual_resolution_metadata=dual_resolution_metadata,
        )
        delta_result = self.specialist.forward_delta(
            h, position_ids=position_ids,
            dual_resolution_metadata=dual_resolution_metadata,
            return_cache=return_cache,
            activation_checkpoint=activation_checkpoint,
        )
        if return_cache:
            delta, specialist_cache = delta_result
        else:
            delta = delta_result
        corrected = self.specialist.corrected_hidden(h, delta)
        loss, head_cache = self.foundation.head_loss(
            corrected, targets, loss_mask=loss_mask,
            target_slice=target_slice, chunk_tokens=chunk_tokens,
        )
        if not return_cache:
            return loss
        return loss, {"head": head_cache, "specialist": specialist_cache}

    def backward(self, cache, *, grad_scale=1.0):
        """Propagate dCE exclusively through the specialist; never foundation."""
        dh = self.foundation.head_backward(cache.pop("head"), grad_scale=grad_scale)
        return self.specialist.backward_delta(dh, cache.pop("specialist"))
