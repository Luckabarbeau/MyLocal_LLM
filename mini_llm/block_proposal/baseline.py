"""Matched conventional AR next-token baseline for 0080 CPU throughput experiments.

Uses the *same* production routed-memory DecoderLanguageModel, source/target
alignment, document-first sampling, attention heads and optimizer as 0080.
No horizon proposal head, soft plans or extra decoder are constructed.
"""
from mini_llm.backend import xp, asnumpy


class MatchedAutoregressiveTrainer:
    def __init__(self, model):
        self.model = model

    def parameters(self):
        return self.model.parameters()

    def zero_grad(self):
        return self.model.zero_grad()

    def train_window(self, source_inputs, target_labels, *, valid_target_mask,
                     memory_metadata=None, compute_gradients=True,
                     grad_scale=1.0):
        if target_labels.ndim != 2 or source_inputs.ndim != 2:
            raise ValueError('expected batched source and target arrays')
        if valid_target_mask.shape != target_labels.shape:
            raise ValueError('mask shape differs from next-token targets')
        n_valid = int(asnumpy(xp.sum(valid_target_mask)))
        if not n_valid:
            raise ValueError('empty supervised window')
        # The same x[t] -> y[t+1] representation as 0080; never roll labels.
        md = None if memory_metadata is None else dict(memory_metadata)
        if md is not None:
            if not md.get('causal_prefix_query', False):
                raise ValueError('a noncausal history route is forbidden')
            if not md.get('causal_masked_target_refinement', False):
                raise ValueError('padded targets require causal masking')
        src = xp.asarray(source_inputs, dtype=xp.int64)
        y = xp.asarray(target_labels, dtype=xp.int64)
        mask = xp.asarray(valid_target_mask, dtype=xp.bool_)
        if compute_gradients:
            logits, model_cache = self.model.forward(
                src, memory_metadata=md, return_cache=True)
        else:
            logits = self.model.forward(src, memory_metadata=md, return_cache=False)
            model_cache = None
        if logits.shape[:2] != y.shape:
            raise RuntimeError('baseline logits do not align with the 0080 target window')
        ce, ce_cache = self.model.compute_loss(
            logits, xp.where(mask, y, 0), loss_mask=mask)
        if compute_gradients:
            dlogits = self.model.backward_loss(ce_cache)
            self.model.backward(dlogits * float(grad_scale), model_cache)
        return dict(final_ce=float(asnumpy(ce)), unique_supervised_tokens=n_valid,
                    main_forward_passes=1,
                    main_backward_passes=1 if compute_gradients else 0)
