"""0104E: minimal composition for testing frozen-foundation capacity expansion.

No separate lexical embeddings or specialist history readers are introduced.
This numerical component does not claim production ExtendedTrainer support.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from mini_llm.backend import synchronize


def _stage(name, callback, timings):
    """Synchronize only when an explicit diagnostic profile is requested."""
    if timings is None:
        return callback()
    synchronize()
    start = perf_counter()
    value = callback()
    synchronize()
    timings[name] = timings.get(name, 0.0) + perf_counter() - start
    return value


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
             chunk_tokens=512, activation_checkpoint=False, return_cache=True,
             timings=None, return_foundation_ce=False):
        h, target_slice = _stage("foundation_forward_s", lambda: self.foundation.hidden(
            inputs, position_ids=position_ids, memory_metadata=memory_metadata,
            dual_resolution_metadata=dual_resolution_metadata,
        ), timings)
        # Optional paired, PRE-UPDATE frozen-foundation CE on the SAME
        # supervised targets and hidden states. The head is frozen; discard its
        # forward cache and never call its backward operation for this metric.
        foundation_loss = None
        if return_foundation_ce:
            def _baseline_head():
                loss, baseline_cache = self.foundation.head_loss(
                    h, targets, loss_mask=loss_mask, target_slice=target_slice,
                    chunk_tokens=chunk_tokens)
                del baseline_cache
                return loss
            foundation_loss = _stage("foundation_head_ce_s", _baseline_head, timings)
        delta_result = _stage("specialist_forward_s", lambda: self.specialist.forward_delta(
            h, position_ids=position_ids,
            dual_resolution_metadata=dual_resolution_metadata,
            return_cache=return_cache,
            activation_checkpoint=activation_checkpoint,
        ), timings)
        if return_cache:
            delta, specialist_cache = delta_result
        else:
            delta = delta_result
        corrected = self.specialist.corrected_hidden(h, delta)
        loss, head_cache = _stage("head_forward_s", lambda: self.foundation.head_loss(
            corrected, targets, loss_mask=loss_mask,
            target_slice=target_slice, chunk_tokens=chunk_tokens,
        ), timings)
        if not return_cache:
            if return_foundation_ce:
                return loss, foundation_loss
            return loss
        cache = {"head": head_cache, "specialist": specialist_cache}
        if return_foundation_ce:
            cache["foundation_loss"] = foundation_loss
        return loss, cache

    def backward(self, cache, *, grad_scale=1.0, timings=None):
        """Propagate dCE exclusively through the specialist; never foundation."""
        dh = _stage("head_backward_s", lambda: self.foundation.head_backward(
            cache.pop("head"), grad_scale=grad_scale), timings)
        return _stage("specialist_backward_s", lambda: self.specialist.backward_delta(
            dh, cache.pop("specialist")), timings)
