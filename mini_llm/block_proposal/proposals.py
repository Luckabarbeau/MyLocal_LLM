"""0080B: nonlinear, full-vocabulary K-horizon proposal head.

This is an explicit, handwritten-backward component like the production model.
All horizons share the *existing* untied model.output_proj.W. Position one
uses the exact same normalized causal hidden vector as ordinary next-token LM.

It intentionally computes only bounded groups of blocks: full-vocab logits
are transient and the 0080D trainer must chunk block groups by peak VRAM.
"""

from dataclasses import dataclass

from mini_llm.backend import xp, RandomStream, is_low_precision_dtype
from mini_llm.init import matrix_parameter
from mini_llm.parameter import Parameter
from mini_llm.performance_profiler import block_proposal_detail_scope
from mini_llm.ops.loss import cross_entropy_forward, cross_entropy_backward


@dataclass(frozen=True)
class HorizonCache:
    context: object
    states: object
    main_hidden: object
    output_cache: dict


class HorizonProposalHead:
    """One shared nonlinear MLP with horizon embeddings, no K vocab heads.

    For j = 0: u_j = h (exact conventional LM baseline).
    For j > 0: u_j = h + residual_scale * tanh(h W_in + pos_j) W_out.

    The nonlinear transform depends on both h and the horizon, unlike simply
    adding a horizon-specific vocabulary bias. A separate W_out is NEVER made
    for the 65K vocabulary: self.model.output_proj is shared in both paths.
    """

    def __init__(self, model, block_size=8, *, hidden_width=None, rng_seed=80,
                 residual_scale=0.1):
        k = int(block_size)
        if k <= 0:
            raise ValueError('block_size must be positive')
        d = int(model.config.d_model)
        width = d if hidden_width is None else int(hidden_width)
        if width <= 0:
            raise ValueError('hidden_width must be positive')
        if not 0 < float(residual_scale) <= 1:
            raise ValueError('residual_scale must lie in (0,1]')
        self.model = model
        self.block_size = k
        self.residual_scale = float(residual_scale)
        rng = RandomStream(rng_seed)
        dtype = model.dtype
        self.W_context = matrix_parameter(
            (d, width), 0.02, rng, 'block_proposal.horizon_context', dtype=dtype)
        self.horizon_embedding = Parameter(
            rng.normal((k, width), std=0.02, dtype=dtype),
            name='block_proposal.horizon_embedding', decay=False)
        self.W_back = matrix_parameter(
            (width, d), 0.01, rng, 'block_proposal.horizon_back', dtype=dtype)

    def parameters(self):
        return [self.W_context, self.horizon_embedding, self.W_back]

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def forward(self, h, *, return_cache=True):
        """h: [number_of_active_blocks,d_model] in production normalized space."""
        if h.ndim != 2 or h.shape[-1] != self.model.config.d_model:
            raise ValueError('expected [Nblocks,d_model] normalized causal h')
        if h.shape[0] == 0:
            raise ValueError('expected at least one boundary state')
        compute = h.astype(self.model.dtype, copy=False)
        k = self.block_size
        if k == 1:
            states = None
            u = compute[:, None, :]
        else:
            # Two-dimensional GEMMs also work with the CuPy BF16 backend.
            context = compute @ self.W_context.data
            pre = context[:, None, :] + self.horizon_embedding.data[None, 1:, :]
            # FP32 tanh for numerical stability; keep activations in compute dtype.
            states = xp.tanh(pre.astype(xp.float32, copy=False)).astype(self.model.dtype)
            n = int(h.shape[0])
            delta = (states.reshape(-1, states.shape[-1]) @ self.W_back.data).reshape(n,k-1,-1)
            later = compute[:, None, :] + self.residual_scale * delta
            u = xp.concatenate((compute[:, None, :], later), axis=1)
        with block_proposal_detail_scope('block_proposal.proposal_vocab.forward'):
            logits, output_cache = self.model.output_proj.forward(u)
        if not return_cache:
            return logits
        return logits, HorizonCache(
            context=None if k==1 else context,
            states=states, main_hidden=compute,
            output_cache=output_cache,
        )

    def backward(self, d_logits, cache: HorizonCache):
        """Accumulate W_vocab and horizon grads; return [N,d] main boundary grad.

        Call model.backward_body() exactly once later with the *scattered*
        boundary gradients. No main Transformer replay per horizon/block.
        """
        if d_logits.ndim != 3 or d_logits.shape[1] != self.block_size:
            raise ValueError('expected [N,K,V] proposal logit gradient')
        # The projection is BF16 for 5080 training; model output projection's
        # existing backward handles its FP32 Parameter.grad accumulator.
        compute_dy = (d_logits.astype(self.model.dtype, copy=False)
                      if is_low_precision_dtype(self.model.dtype) else d_logits)
        with block_proposal_detail_scope('block_proposal.proposal_vocab.backward'):
            d_u = self.model.output_proj.backward(compute_dy, cache.output_cache)
        d_h = xp.sum(d_u.astype(xp.float32, copy=False), axis=1)
        if self.block_size > 1:
            n = int(d_u.shape[0]); width = self.W_back.shape[0]
            d_later = (d_u[:,1:,:] * self.residual_scale).reshape(-1,d_u.shape[-1])
            states = cache.states.reshape(-1,width)
            self.W_back.grad += states.T @ d_later
            d_states = (d_later @ self.W_back.data.T).reshape(n,self.block_size-1,width)
            d_pre = d_states.astype(xp.float32, copy=False) * (
                1.0 - cache.states.astype(xp.float32, copy=False)**2)
            # BF16 GEMM is supported on GPU only for BF16 operands.
            d_pre_gemm = (d_pre.astype(self.model.dtype, copy=False)
                          if is_low_precision_dtype(self.model.dtype) else d_pre)
            self.horizon_embedding.grad[1:,:] += xp.sum(d_pre,axis=0)
            d_context = xp.sum(d_pre_gemm,axis=1)
            self.W_context.grad += cache.main_hidden.T @ d_context
            d_h += (d_context @ self.W_context.data.T).astype(xp.float32, copy=False)
        return d_h.astype(self.model.dtype,copy=False) if is_low_precision_dtype(self.model.dtype) else d_h

    def supervised_loss(self, logits, target_blocks, valid_mask):
        """Per-token CE normalized by unique valid target count."""
        if target_blocks.shape != logits.shape[:2] or valid_mask.shape != target_blocks.shape:
            raise ValueError('target block/valid shapes mismatch')
        targets = xp.where(valid_mask, target_blocks, 0)
        if bool(xp.any((targets < 0) | (targets >= logits.shape[-1]))):
            raise ValueError('valid target ID out of vocabulary')
        return cross_entropy_forward(logits, targets, loss_mask=valid_mask)

    def supervised_loss_backward(self, ce_cache):
        return cross_entropy_backward(ce_cache)
