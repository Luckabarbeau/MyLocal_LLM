"""0080M: minimal production-training adapter for the two-stage objective.

The production ExtendedTrainer owns sampling, optimizer, scheduler, clipping,
loss scaling, validation and checkpoint orchestration. This adapter owns only
the joint K-horizon objective and its single-pass backward; the legacy 0080
research trainers and checkpoint loader are unchanged.
"""

from mini_llm.backend import xp
from .checkpoint import architecture_metadata
from .training import SinglePassBlockTrainer


class BlockProposalObjective:
    """Opt-in objective bound to the *same* production DecoderLanguageModel."""

    name = "block-proposal"

    def __init__(self, net, *, block_group_size=8):
        if int(block_group_size) < 1:
            raise ValueError("block_group_size must be positive")
        self.net = net
        self.model = net.proposal_head.model
        if net.decoder.model is not self.model:
            raise ValueError("proposal and decoder must share the original main model")
        if self.model.hierarchical_memory_enabled:
            memory = self.model.config.memory_context
            if memory.integration_mode != "routed_prefix" or memory.memory_training != "joint":
                raise ValueError("0080 objective requires joint routed-prefix memory training")
        if getattr(self.model, "progressive_depth_enabled", False):
            raise ValueError("0080M has not validated progressive-depth parameter growth")
        self.block_group_size = int(block_group_size)
        self.single_pass = SinglePassBlockTrainer(net)
        params = self.parameters()
        if len(params) != len({id(p) for p in params}):
            raise ValueError("duplicate optimizer parameter objects")
        if len(params) != len({p.name for p in params}):
            raise ValueError("duplicate optimizer parameter names")
        if {id(p) for p in self.model.parameters()} - {id(p) for p in params}:
            raise ValueError("original model parameters must remain trainable")

    def parameters(self):
        return self.single_pass.parameters()

    @staticmethod
    def prepare_metadata(metadata):
        if metadata is None:
            return None
        prepared = dict(metadata)
        if prepared.get("causal_prefix_query", True) is not True:
            raise ValueError("noncausal pre-window routing is forbidden for 0080")
        prepared["causal_prefix_query"] = True
        prepared["causal_masked_target_refinement"] = True
        return prepared

    @staticmethod
    def _valid_mask(targets, loss_mask):
        if loss_mask is None:
            return xp.ones(targets.shape, dtype=xp.bool_)
        mask = xp.asarray(loss_mask)
        if mask.shape != targets.shape:
            raise ValueError("0080 validity mask shape differs from target shape")
        # Unlike conventional CE, the block contract is *binary*, not a set
        # of fractional per-token weights. Refuse rather than silently change
        # the normalization or valid target-length semantics.
        if not bool(xp.all((mask == 0) | (mask == 1))):
            raise ValueError("0080 requires a binary target validity mask")
        return mask.astype(xp.bool_, copy=False)

    def _run(self, inputs, targets, loss_mask, metadata, *, compute_gradients, grad_scale):
        source = xp.asarray(inputs, dtype=xp.int64)
        labels = xp.asarray(targets, dtype=xp.int64)
        valid = self._valid_mask(labels, loss_mask)
        return self.single_pass.train_window(
            source, labels, valid_target_mask=valid,
            memory_metadata=self.prepare_metadata(metadata),
            block_group_size=self.block_group_size,
            compute_gradients=compute_gradients, grad_scale=grad_scale,
        )

    def forward_backward(self, inputs, targets, loss_mask, metadata, *, grad_scale=1.0):
        return self._run(inputs, targets, loss_mask, metadata,
                         compute_gradients=True, grad_scale=grad_scale)

    def evaluate_batch(self, inputs, targets, loss_mask, metadata):
        return self._run(inputs, targets, loss_mask, metadata,
                         compute_gradients=False, grad_scale=1.0)

    def checkpoint_metadata(self):
        return architecture_metadata(self.net)


class BlockValidationAccumulator:
    """Unique-token weighted production validation with K-horizon detail."""

    def __init__(self, k):
        import numpy as np
        self.k = int(k)
        self.counts = np.zeros(self.k, dtype=np.float64)
        self.proposal_horizon_nll = np.zeros(self.k, dtype=np.float64)
        self.final_horizon_nll = np.zeros(self.k, dtype=np.float64)
        self.proposal_nll = 0.
        self.final_nll = 0.
        self.valid_tokens = 0

    def add(self, metrics):
        from mini_llm.backend import asnumpy
        import numpy as np
        n = int(metrics['unique_supervised_tokens'])
        if n <= 0:
            raise ValueError('empty 0080 validation batch')
        self.valid_tokens += n
        self.proposal_nll += float(metrics['proposal_ce']) * n
        self.final_nll += float(metrics['final_ce']) * n
        self.counts += np.asarray(asnumpy(metrics['position_counts']), dtype=np.float64)
        self.proposal_horizon_nll += np.asarray(asnumpy(metrics['proposal_position_ce_sums']), dtype=np.float64)
        self.final_horizon_nll += np.asarray(asnumpy(metrics['final_position_ce_sums']), dtype=np.float64)

    def summary(self):
        if not self.valid_tokens:
            raise ValueError('no 0080 validation batches')
        if int(self.counts.sum()) != self.valid_tokens:
            raise ValueError('K-horizon and unique-token validation counts differ')
        record = {
            'proposal_ce_nats_per_target': self.proposal_nll / self.valid_tokens,
            'final_ce_nats_per_target': self.final_nll / self.valid_tokens,
            'decoder_gain_nats_per_target': (self.proposal_nll - self.final_nll) / self.valid_tokens,
            'block_unique_supervised_tokens': int(self.valid_tokens),
        }
        for h in range(self.k):
            count = int(self.counts[h])
            record[f'h{h+1}_count'] = count
            if count:
                record[f'h{h+1}_proposal_ce'] = self.proposal_horizon_nll[h] / count
                record[f'h{h+1}_final_ce'] = self.final_horizon_nll[h] / count
                record[f'h{h+1}_decoder_gain'] = (self.proposal_horizon_nll[h] - self.final_horizon_nll[h]) / count
            else:
                record[f'h{h+1}_proposal_ce'] = None
                record[f'h{h+1}_final_ce'] = None
                record[f'h{h+1}_decoder_gain'] = None
        return record
