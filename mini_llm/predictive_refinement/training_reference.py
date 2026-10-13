"""Scientific reference for full predictive-refinement BPTT.

The original 0079C/0079F mode recomputes short committed prefixes for an
independent mathematical reference. Optional 0079G mode encodes the causal
context once and reuses its per-layer K/V across K+1 active-block passes,
including an exact single causal-context backward. Both remain bounded
research paths, NOT the production 4K/64K trainer or learned-memory router.

Pass 0 predicts all unknown target positions. Pass j (1 <= j <= K) replaces
positions < j with teacher-forced tokens and positions >= j with soft token
expectations from pass j-1. The same DecoderLanguageModel instance is called
K+1 times, with separate forward caches, and manually backpropagated in
reverse time through every QE edge and all shared Transformer parameters.
"""

from dataclasses import dataclass

from mini_llm.backend import xp
from mini_llm.predictive_refinement.block_attention import block_causal_allow_mask
from mini_llm.predictive_refinement.soft_embeddings import (
    FuturePositionEmbeddings, SoftTokenExpectation, stable_softmax,
)
from mini_llm.backend import RandomStream
from mini_llm.performance_profiler import performance_scope


@dataclass(frozen=True)
class RefinementReferenceConfig:
    block_size: int = 4
    feedback_temperature: float = 1.0
    feedback_mode: str = "full"  # "unknown" = matched-pass no-feedback ablation
    initial_loss_weight: float = 0.1
    max_reference_tokens: int = 256
    enable_compact_static_attention: bool = False
    enable_shared_context_training: bool = False
    # 0079P1: retain only directly supervised CE rows and a rolling
    # backward gradient. Disabled by default to preserve inspectable tapes.
    compact_recurrent_tape: bool = False

    def __post_init__(self):
        if self.feedback_mode not in {"full", "unknown"}:
            raise ValueError("feedback_mode must be full or unknown")
        if self.block_size < 1 or self.block_size > 16:
            raise ValueError("reference block_size must be in [1,16]")
        if self.feedback_temperature <= 0 or self.feedback_temperature == float("inf"):
            raise ValueError("feedback_temperature must be finite and positive")
        if self.initial_loss_weight < 0 or self.initial_loss_weight == float("inf"):
            raise ValueError("initial_loss_weight must be finite and >= 0")
        if self.max_reference_tokens <= 0:
            raise ValueError("max_reference_tokens must be positive")


@dataclass
class ReferencePass:
    model_cache: object
    logits: object
    loss_gradient: object
    feedback_cache: object = None
    fixed_prefix_length: int = 0
    # Compact mode stores only the direct CE rows; a full [B,T,V] backward
    # workspace is built once per reverse step. None in the reference mode.
    loss_position: int = -1
    logits_shape: tuple = ()
    logits_dtype: object = None


class DenseRefinementTrainingReference:
    """Correctness-only short-context training, with **full** BPTT.

    ``forward`` returns a dictionary of scalar losses plus an opaque tape.
    After ``model.zero_grad()`` and ``position_states.zero_grad()``, invoke
    ``backward(tape)`` to add all gradients. It does not step an optimizer.
    """

    def __init__(self, model, config=None, *, rng_seed=0):
        self.model = model
        self.config = config or RefinementReferenceConfig()
        self.compact_static = bool(self.config.enable_compact_static_attention)
        self.shared_context = bool(self.config.enable_shared_context_training)
        if model.hierarchical_memory_enabled:
            # Q2 uses the original learned routed-prefix encoder, followed by
            # P2's shared K/V views. Per-layer *learned retrieval* for active
            # provisional blocks is Q3 and must never silently become static.
            mc = model.config.memory_context
            if not self.shared_context or mc.integration_mode != "routed_prefix":
                raise NotImplementedError("Q2 requires routed_prefix and shared-context training")
            if any(b.attention.terminal_memory is not None
                   for b in model.blocks[:model.active_layers]):
                raise NotImplementedError("provisional terminal-landmark memory is unsupported")
        mixed = [block.attention.attention_config is not None for block in model.blocks]
        if any(mixed) and (not (self.compact_static or self.shared_context) or not all(mixed)):
            raise NotImplementedError("mixed static blocks require explicit enable_compact_static_attention")
        if self.compact_static and not all(mixed):
            raise ValueError("compact static mode needs configured attention on every layer")
        if self.compact_static and any(
            block.attention._retrieval_groups or block.attention.terminal_memory is not None
            for block in model.blocks
        ):
            raise NotImplementedError("retrieval/terminal memory requires a separate causal router")
        self.states = FuturePositionEmbeddings(
            model.config.d_model, RandomStream(rng_seed),
            std=model.config.init_std, dtype=model.dtype,
        )
        self.expectation = SoftTokenExpectation(model.embedding.W)

    def parameters(self):
        return list(self.model.parameters()) + self.states.parameters()

    def zero_grad(self):
        self.model.zero_grad()
        for param in self.states.parameters():
            param.zero_grad()

    def forward(self, context_ids, target_ids, *, valid_target_mask=None,
                shared_context_override=None, loss_normalization_tokens=None,
                record_tape=True, record_row_ce=False):
        """Compute initial + progressive teacher-forced commitment CE.

        ``target_ids`` have shape [B,K] including trailing padded slots. Valid
        target positions must form a prefix (EOS, if present, is valid).
        Losses are normalized by the number of *unique valid target tokens*,
        including those in shorter final blocks.
        """
        if context_ids.ndim != 2 or target_ids.ndim != 2:
            raise ValueError("context_ids/target_ids must have shape [B,T]")
        batch, context_len = context_ids.shape
        batch_targets, k = target_ids.shape
        if batch != batch_targets or k != self.config.block_size or batch == 0:
            raise ValueError("target_ids must be [B,config.block_size]")
        valid = (
            xp.ones((batch, k), dtype=xp.bool_)
            if valid_target_mask is None else xp.asarray(valid_target_mask, dtype=xp.bool_)
        )
        if valid.shape != (batch, k):
            raise ValueError("valid_target_mask must have shape [B,K]")
        # Valid slots must form a prefix. False->True would imply that a
        # teacher-forced prefix included a missing/padded target.
        if bool(xp.any(valid[:, 1:] & ~valid[:, :-1])):
            raise ValueError("valid target masks must be prefix-contiguous")
        local_nvalid = int(xp.sum(valid))
        nvalid = local_nvalid if loss_normalization_tokens is None else int(loss_normalization_tokens)
        if local_nvalid == 0 or nvalid <= 0:
            raise ValueError("target block has no supervised positions")
        # Padding may contain an out-of-vocabulary sentinel such as -100;
        # replace it without affecting valid loss or attention semantics.
        safe_targets = xp.where(valid, target_ids, 0)
        if bool(xp.any((safe_targets < 0) | (safe_targets >= self.model.config.vocab_size))):
            raise ValueError("valid target IDs must be inside the vocabulary")
        allow = None if (self.compact_static or self.shared_context) else block_causal_allow_mask(
            context_len, k, valid_block_mask=valid,
            max_reference_tokens=self.config.max_reference_tokens,
        )
        embedding = self.model.embedding
        context_cache = None
        context_shape = None
        if shared_context_override is not None and not self.shared_context:
            raise ValueError("grouped causal context requires shared-context training")
        if self.shared_context and shared_context_override is not None:
            context_kv = shared_context_override["context_kv"]
            context_cache = None
            context_shape = None
            embedding_cache = None
            context_input = None
        elif self.shared_context:
            # One fully causal context encoding per training example. The K/V
            # tensors in its per-layer caches are read-only during refinement.
            if context_len:
                context_head, context_cache = self.model.forward_body(
                    context_ids, return_cache=True, activation_checkpoint=False,
                )
                context_shape = context_head.shape
                del context_head
                context_kv = tuple(
                    (layer_cache["attn_cache"]["k"], layer_cache["attn_cache"]["v"])
                    for layer_cache in context_cache["block_caches"]
                )
            else:
                context_kv = tuple(
                    (xp.zeros((batch, 0, block.attention.n_kv_heads, block.attention.d_head),
                              dtype=self.model.dtype),
                     xp.zeros((batch, 0, block.attention.n_kv_heads, block.attention.d_head),
                              dtype=self.model.dtype))
                    for block in self.model.blocks[:self.model.active_layers]
                )
            embedding_cache = None
            context_input = None
        else:
            context_input, embedding_cache = embedding.forward(context_ids)
        logit_offset = 0 if self.shared_context else context_len
        active = self.states.unknown_block(batch, k)
        passes = []
        position_ces = []
        initial_pos_sums = xp.zeros((k,), dtype=xp.float64)
        commit_pos_sums = xp.zeros((k,), dtype=xp.float64)
        position_counts = xp.sum(valid, axis=0, dtype=xp.int64)
        row_commitment_ce_sums = (xp.zeros((batch,), dtype=xp.float64)
                                  if record_row_ce else None)
        previous_logits = None
        compact_tape = bool(record_tape and self.config.compact_recurrent_tape)
        initial_ce = None
        loss_total_sum = xp.asarray(0.0, dtype=xp.float64)
        aux_loss_sum = xp.asarray(0.0, dtype=xp.float64)
        for step in range(k + 1):
            feedback_cache = None
            fixed_prefix_length = max(0, step - 1)
            if step:
                # A single bounded operation for the *still-uncommitted* suffix.
                # Do not materialize qE for positions already committed.
                previous = (previous_logits if compact_tape or not record_tape
                            else passes[-1].logits)
                if self.config.feedback_mode == "full":
                    with performance_scope("refinement.soft_expectation_forward"):
                        soft, feedback_cache = self.expectation.forward(
                            previous[:, logit_offset + fixed_prefix_length:, :],
                            temperature=self.config.feedback_temperature,
                        )
                    remaining = self.states.soft_block(soft)
                else:
                    # Identical number of complete Transformer passes and
                    # legitimate fixed-token prefixes, but no probabilistic
                    # feedback. This is NOT autoregressive training.
                    remaining = self.states.unknown_block(batch, k - fixed_prefix_length)
                if compact_tape:
                    # No future pass needs the previous logits after their
                    # probability-weighted embedding has been constructed.
                    previous_logits = None
                    previous = None
                if fixed_prefix_length:
                    fixed = self.states.fixed_block(
                        safe_targets[:, :fixed_prefix_length], embedding.W
                    )
                    active = xp.concatenate((fixed, remaining), axis=1)
                else:
                    active = remaining
            with performance_scope("refinement.full_model_forward"):
                if self.shared_context:
                    block_output = self.model.forward_shared_context_block(
                        active, context_kv, context_length=context_len,
                        valid_block_mask=valid, return_cache=record_tape,
                        block_position_starts=(
                            None if shared_context_override is None
                            else shared_context_override["block_position_starts"]
                        ),
                        layer_group_plans=(
                            None if shared_context_override is None
                            else shared_context_override["layer_group_plans"]
                        ),
                        layer_retrieval_tasks=(
                            None if shared_context_override is None else
                            shared_context_override.get("layer_retrieval_tasks")
                        ),
                    )
                else:
                    full_input = xp.concatenate((context_input, active), axis=1)
                    block_output = self.model.forward(
                        None, input_embeddings=full_input,
                        attention_allow_mask=allow,
                        block_context_length=context_len if self.compact_static else None,
                        block_valid_mask=valid if self.compact_static else None,
                        return_cache=record_tape,
                    )
            logits, model_cache = (block_output if record_tape else (block_output, None))
            with performance_scope("refinement.vocab_loss"):
                block_logits = logits[:, logit_offset:, :]
                dloss = (xp.zeros_like(logits) if record_tape and not compact_tape
                         else None)
                if step == 0:
                    p = stable_softmax(block_logits) if record_tape else None
                    # Numerically stable CE from logits using logsumexp.
                    row_max = xp.max(block_logits, axis=-1, keepdims=True)
                    shifted = block_logits - row_max
                    logsum = xp.log(xp.sum(xp.exp(shifted), axis=-1))
                    target_shifted = xp.take_along_axis(
                        shifted, safe_targets[..., None], axis=-1
                    )[..., 0]
                    row_ce = logsum - target_shifted
                    initial_pos_sums = xp.sum(row_ce * valid, axis=0)
                    initial_ce = xp.sum(initial_pos_sums) / nvalid
                    aux_loss_sum = initial_ce
                    # Auxiliary loss scales with lambda; still no early target
                    # access in the input path.
                    if record_tape:
                        if compact_tape:
                            # Same BF16 rounding order as the legacy dense
                            # gradient (store p -> subtract one -> scale).
                            dloss = p.astype(logits.dtype, copy=True)
                            dloss[xp.arange(batch)[:, None], xp.arange(k)[None, :],
                                  safe_targets] -= 1.0
                            dloss *= valid[..., None] * (self.config.initial_loss_weight / nvalid)
                        else:
                            dloss[:, logit_offset:, :] = p
                            dloss[
                                xp.arange(batch)[:, None],
                                logit_offset + xp.arange(k)[None, :],
                                safe_targets,
                            ] -= 1.0
                            dloss[:, logit_offset:, :] *= (
                                valid[..., None] * (self.config.initial_loss_weight / nvalid)
                            )
                else:
                    j = step - 1
                    row = block_logits[:, j, :]
                    p = stable_softmax(row) if record_tape else None
                    row_max = xp.max(row, axis=-1, keepdims=True)
                    shifted = row - row_max
                    row_ce = xp.log(xp.sum(xp.exp(shifted), axis=-1)) - (
                        xp.take_along_axis(shifted, safe_targets[:, j, None], axis=-1)[:, 0]
                    )
                    if record_row_ce:
                        row_commitment_ce_sums += row_ce * valid[:, j]
                    per_pos_sum = xp.sum(row_ce * valid[:, j])
                    commit_pos_sums[j] = per_pos_sum
                    position_ces.append(per_pos_sum / xp.maximum(xp.sum(valid[:, j]), 1))
                    loss_total_sum += per_pos_sum
                    if record_tape:
                        if compact_tape:
                            dloss = p.astype(logits.dtype, copy=True)
                            dloss[xp.arange(batch), safe_targets[:, j]] -= 1.0
                            dloss *= valid[:, j, None] / nvalid
                        else:
                            dloss[:, logit_offset + j, :] = p
                            dloss[xp.arange(batch), logit_offset + j, safe_targets[:, j]] -= 1.0
                            dloss[:, logit_offset + j, :] *= valid[:, j, None] / nvalid
            if record_tape:
                passes.append(ReferencePass(
                    model_cache=model_cache,
                    logits=None if compact_tape else logits,
                    loss_gradient=dloss,
                    feedback_cache=feedback_cache,
                    fixed_prefix_length=fixed_prefix_length,
                    loss_position=(step - 1 if step else -1),
                    logits_shape=tuple(logits.shape),
                    logits_dtype=logits.dtype,
                ))
            if compact_tape or not record_tape:
                previous_logits = logits
        # Do not keep the last vocabulary projection alive beyond forward.
        previous_logits = None
        commit_ce = loss_total_sum / nvalid
        total_loss = commit_ce + self.config.initial_loss_weight * initial_ce
        # Count only the vocabulary-related buffers owned by the recurrent
        # tape (not shared model caches/activations, which have separate lives).
        tape_vocab_bytes = 0
        if record_tape:
            for entry in passes:
                if entry.logits is not None:
                    tape_vocab_bytes += entry.logits.nbytes
                if entry.loss_gradient is not None:
                    tape_vocab_bytes += entry.loss_gradient.nbytes
                if entry.feedback_cache is not None:
                    tape_vocab_bytes += entry.feedback_cache.probs.nbytes
        return {
            "commitment_ce": commit_ce,
            "initial_ce": initial_ce,
            "total_loss": total_loss,
            "position_ce": tuple(position_ces),
            # Unnormalized sums/counts: callers combine groups and documents
            # before forming per-position averages. Zero-count positions stay NaN.
            "initial_position_ce_sums": initial_pos_sums,
            "commitment_position_ce_sums": commit_pos_sums,
            "position_counts": position_counts,
            **({"row_commitment_ce_sums": row_commitment_ce_sums}
               if record_row_ce else {}),
            "supervised_tokens": local_nvalid,
            "recurrent_vocab_tape_bytes": tape_vocab_bytes,
            "compact_recurrent_tape": compact_tape,
            "full_model_passes": k + 1,
            "causal_context_encodes": (0 if shared_context_override is not None else 1) if self.shared_context and context_len else (
                k + 1 if context_len else 0
            ),
        }, (None if not record_tape else {
            "passes": passes,
            "context_ids": context_ids,
            "target_ids": safe_targets,
            "embedding_cache": embedding_cache,
            "context_length": context_len,
            "shared_context": self.shared_context,
            "context_cache": context_cache,
            "context_shape": context_shape,
            "defer_context_backward": shared_context_override is not None,
            "compact_recurrent_tape": compact_tape,
        })

    def backward(self, tape):
        """Reverse K+1 shared Transformer passes and all soft feedback edges."""
        passes = tape["passes"]
        c = tape["context_length"]
        ids = tape["target_ids"]
        compact = bool(tape.get("compact_recurrent_tape", False))
        # The legacy path is preserved for loss/gradient comparisons and for
        # callers that inspect the complete forward tape. Compact mode has
        # only a SINGLE inter-pass dependency: (step -> step-1) via Q @ E.
        downstream_logits = (
            None if compact else [xp.zeros_like(step.logits) for step in passes]
        )
        next_gradient = None
        shared = bool(tape.get("shared_context", False))
        offset = 0 if shared else c
        accumulated_kv = None
        for step in range(len(passes) - 1, -1, -1):
            record = passes[step]
            if compact:
                # Reuse the incoming inter-pass gradient as the current full
                # gradient buffer. There is no list of K+1 [B,T,V] buffers.
                dlogits = (next_gradient if next_gradient is not None
                           else xp.zeros(record.logits_shape, dtype=record.logits_dtype))
                next_gradient = None
                if record.loss_position == -1:
                    dlogits[:, offset:offset + ids.shape[1], :] += record.loss_gradient
                else:
                    dlogits[:, offset + record.loss_position, :] += record.loss_gradient
                # CE gradient can be released before the expensive model BWD.
                record.loss_gradient = None
            else:
                dlogits = record.loss_gradient + downstream_logits[step]
            if shared:
                with performance_scope("refinement.full_model_backward"):
                    dx, kv_grads = self.model.backward_shared_context_block(
                        dlogits, record.model_cache
                    )
                # P2's virtual K/V source has already scattered historical
                # gradients into the shared causal tensors. None indicates
                # that no per-refinement history gradient array was allocated.
                if accumulated_kv is None:
                    accumulated_kv = [[dk, dv] for dk, dv in kv_grads]
                else:
                    for pair, (dk, dv) in zip(accumulated_kv, kv_grads):
                        if dk is not None:
                            pair[0] += dk
                            pair[1] += dv
            else:
                with performance_scope("refinement.full_model_backward"):
                    dx = self.model.backward(dlogits, record.model_cache)
                if c:
                    self.model.embedding.backward(dx[:, :c, :], tape["embedding_cache"])
            if compact:
                # This tape is intentionally single-use when compact. The
                # model's backward has consumed this pass's saved activations.
                record.model_cache = None
                del dlogits
            if step == 0:
                self.states.backward_unknown(dx[:, offset:, :])
                continue
            prefix = record.fixed_prefix_length
            if prefix:
                self.states.backward_fixed(
                    dx[:, offset:offset + prefix, :], ids[:, :prefix], self.model.embedding.W
                )
            if self.config.feedback_mode == "full":
                remaining_grad = self.states.backward_soft(dx[:, offset + prefix:, :])
                with performance_scope("refinement.soft_expectation_backward"):
                    dprev = self.expectation.backward(remaining_grad, record.feedback_cache)
                if compact:
                    # Preserve the original accumulator dtype/casting order,
                    # including BF16 rounding when dprev is FP32.
                    next_gradient = xp.zeros(passes[step - 1].logits_shape,
                                             dtype=passes[step - 1].logits_dtype)
                    next_gradient[:, offset + prefix:, :] += dprev
                    record.feedback_cache = None
                    del dprev
                else:
                    downstream_logits[step - 1][:, offset + prefix:, :] += dprev
            else:
                self.states.backward_unknown(dx[:, offset + prefix:, :])
            if compact:
                del dx
        if tape.get("defer_context_backward", False):
            return accumulated_kv
        if shared and c:
            # Crucial: execute ONE backward through the context encoder.
            # Every refinement's dK/dV affects the shared weights, norm1,
            # lower-layer context states and actual committed token embeddings.
            dhead = xp.zeros(tape["context_shape"], dtype=self.model.dtype)
            self.model.backward_body(
                dhead, tape["context_cache"],
                shared_context_kv_grads=accumulated_kv,
            )
