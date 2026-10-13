"""0079H: bounded parallel teacher-forced target blocks with one causal trunk.

A single causal pass over each independent document row supplies the immutable
per-layer K/V needed at *all* target block boundaries. Each bounded block group
uses its own prefix cutoff and absolute RoPE positions; only K active queries
run per block, and groups are backpropagated immediately to keep refinement
activation storage bounded. The context tape is backpropagated exactly once.

This is the short-window static-attention research path, not the 64K routed
production trainer. Grouped reference indexed attention still assembles bounded
[B*G,Cmax+K] K/V tensors; use small groups and monitor peak GPU memory.
"""

from mini_llm.backend import xp, is_low_precision_dtype
from mini_llm.ops.attention_selection import KeySelectionPlan
from mini_llm.performance_profiler import performance_scope
from mini_llm.predictive_refinement.compact_sparse import grouped_provisional_plan
from mini_llm.predictive_refinement.compact_history import pack_group_history
from mini_llm.predictive_refinement.training_reference import (
    DenseRefinementTrainingReference, RefinementReferenceConfig,
)


class WindowRefinementTrainer(DenseRefinementTrainingReference):
    """Train non-overlapping K-token blocks of a document-first token window.

    `train_window` does both forward and backward, *without* stepping an
    optimizer. Call `zero_grad` first and step the optimizer afterwards.
    Source rows must be separate documents, with prefix-contiguous validity
    (including EOS as a valid final token). No valid target is duplicated.
    """

    def __init__(self, model, config=None, *, rng_seed=0):
        cfg = config or RefinementReferenceConfig(enable_shared_context_training=True)
        if not cfg.enable_shared_context_training:
            raise ValueError("0079H requires enable_shared_context_training=True")
        super().__init__(model, cfg, rng_seed=rng_seed)

    def train_routed_window(self, source_ids, *, memory_metadata=None, **kwargs):
        """Q2: train K-token blocks with ONE original routed-prefix causal trunk.

        Source contains [same-document historical store | dense target window].
        The learned top-level route is chosen only from PRE-WINDOW history;
        its 2K selected tokens and the entire working window share one
        causal Transformer cache. Q3 is required for learned retrieval heads.
        No [B,65536,...] deep cache or dense historical attention mask exists.
        """
        cfg = self.model.config.memory_context
        if not self.model.hierarchical_memory_enabled or cfg.integration_mode != 'routed_prefix':
            raise ValueError('train_routed_window requires a routed-prefix model')
        if source_ids.ndim != 2 or int(source_ids.shape[1]) != cfg.source_input_length:
            raise ValueError('source IDs must contain exactly the configured 64K horizon')
        if any(layer.attention.terminal_memory is not None
               for layer in self.model.blocks[:self.model.active_layers]):
            raise NotImplementedError('Q3 does not support terminal-landmark memory')
        if any(layer.attention._retrieval_groups
               for layer in self.model.blocks[:self.model.active_layers]):
            if not kwargs.get('resident_history', False):
                raise ValueError('Q3 learned retrieval requires resident_history=True')
        if 'prefix_length' in kwargs:
            raise ValueError('routed windows supervise the entire working region')
        if not kwargs.get('compact_history', False) or not kwargs.get('compact_active_blocks', False):
            raise ValueError('Q2 requires compact_history and compact_active_blocks')
        # A physical gathered-K/V reference is available with resident_history=False;
        # the optimized implementation should set resident_history=True.
        return self.train_window(
            source_ids[:, cfg.distant_memory_length:], prefix_length=0,
            routed_source_ids=source_ids, routed_memory_metadata=memory_metadata,
            **kwargs,
        )

    def _precompute_group_plans(self, starts, batch, valid, stored_length):
        plans = []
        # Most transformer layers share the same static topology. A selection
        # plan is immutable throughout all refinement passes; build each
        # distinct topology once per window group, not once per layer/head.
        plan_cache = {}
        for layer in self.model.blocks[:self.model.active_layers]:
            attention = layer.attention
            groups = attention._static_head_groups if attention.attention_config is not None else ({
                "kind": "dense", "head_indices": tuple(range(attention.n_q_heads)),
                "window": None, "dilation": None, "offset": 0, "stride": None,
            },)
            per_layer = []
            for group in groups:
                signature = (group["kind"], group["window"], group["dilation"],
                             group["offset"] or 0, group["stride"])
                if signature not in plan_cache:
                    plan_cache[signature] = grouped_provisional_plan(
                        stored_length, starts, batch, self.config.block_size,
                        valid_block_mask=valid, kind=signature[0], window=signature[1],
                        dilation=signature[2], offset=signature[3], stride=signature[4],
                    )
                per_layer.append(plan_cache[signature])
            per_layer = tuple(per_layer)
            plans.append(per_layer)
        return tuple(plans)

    def train_window(self, token_ids, *, prefix_length=0, valid_token_mask=None,
                     group_size=2, max_window_tokens=256, loss_normalization_tokens=None,
                     compute_gradients=True, record_row_ce=False,
                     compact_active_blocks=False, max_active_queries=0,
                     compact_history=False, max_history_keys=4096,
                     resident_history=False, routed_source_ids=None,
                     routed_memory_metadata=None, max_supervised_blocks=None):
        """Return globally token-normalized losses and accumulate exact grads.

        `token_ids`: [B,T] one document per row, padding only at the tail.
        `prefix_length`: committed warm-up prefix (not scored). All subsequent
        valid positions are supervised exactly once, in aligned K-sized blocks.
        `group_size`: maximum independent target blocks in a batched pass.
        `max_window_tokens`: explicit safety bound for the reference causal
        encoding (does not allocate a 64K square mask).
        `compact_active_blocks`: pack only document/block pairs with at least
        one valid target; mathematically equivalent to padded execution.
        `max_active_queries`: optional hard per-pass guard (0 disables).
        This limits memory exposure, not the number of supervised targets.
        `compact_history`: use only statically selected committed K/V keys per
        prediction block.  This is recommended for long sparse windows; it
        avoids replicating a full prefix for every provisional row.
        Neither setting changes the model weights or attention semantics.
        """
        if routed_source_ids is not None:
            mc = self.model.config.memory_context
            if not self.model.hierarchical_memory_enabled or mc.integration_mode != 'routed_prefix':
                raise ValueError('routed source requires a routed-prefix model')
            if not compact_history or not compact_active_blocks:
                raise ValueError('routed Q2 requires compact historical selection and active blocks')
            if int(prefix_length) != 0:
                raise ValueError('routed Q2 supervises the entire dense working window')
            if routed_source_ids.ndim != 2 or int(routed_source_ids.shape[1]) != mc.source_input_length:
                raise ValueError('invalid routed source shape')
            if token_ids.shape != (routed_source_ids.shape[0], mc.target_length):
                raise ValueError('working token window has the wrong shape')
            if bool(xp.any(token_ids != routed_source_ids[:, mc.distant_memory_length:])):
                raise ValueError('working tokens must match the routed source suffix')
            if any(layer.attention.terminal_memory is not None
                   for layer in self.model.blocks[:self.model.active_layers]):
                raise NotImplementedError('Q3 does not support terminal-landmark memory')
            if any(layer.attention._retrieval_groups
                   for layer in self.model.blocks[:self.model.active_layers]) and not resident_history:
                raise ValueError('Q3 learned retrieval requires shared-resident history')
        elif self.model.hierarchical_memory_enabled:
            raise ValueError('hierarchical model requires routed_source_ids')
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [B,T]")
        batch, length = map(int, token_ids.shape)
        prefix = int(prefix_length)
        groups_limit = int(group_size)
        if not batch or not length or not 0 <= prefix < length:
            raise ValueError("non-empty documents and 0 <= prefix_length < T required")
        if groups_limit < 1 or groups_limit > 64:
            raise ValueError("reference group_size must lie in [1,64]")
        if resident_history and not compact_history:
            raise ValueError('resident_history requires compact_history=True')
        if compact_history and not compact_active_blocks:
            raise ValueError("compact_history requires compact_active_blocks=True")
        if int(max_history_keys) < 1:
            raise ValueError("max_history_keys must be positive")
        if int(max_active_queries) < 0:
            raise ValueError("max_active_queries must be >= 0")
        if length > int(max_window_tokens):
            raise ValueError("window exceeds bounded reference maximum")
        if length > int(self.model.config.context_length):
            raise ValueError("window length exceeds model context_length")
        valid = (xp.ones((batch, length), dtype=xp.bool_)
                 if valid_token_mask is None else xp.asarray(valid_token_mask, dtype=xp.bool_))
        if valid.shape != (batch, length):
            raise ValueError("valid_token_mask must have shape [B,T]")
        if bool(xp.any(valid[:, 1:] & ~valid[:, :-1])):
            raise ValueError("document valid masks must be prefix-contiguous")
        if routed_source_ids is not None and not bool(xp.all(valid)):
            # Q4: tail padding is present in the 4K causal encoder, but never
            # occurs before a genuine token. Every scored block has a valid
            # prefix-only history selection, so no pad can enter its context.
            # Opt in explicitly; the Q2 strict reference remains the default.
            if not (routed_memory_metadata or {}).get('causal_masked_target_refinement', False):
                raise ValueError('routed Q2 requires a fully valid working window')
        # Debug-only: encode the FULL causal window, but supervise only
        # the first N blocks so a large model can run a bounded GPU smoke.
        if max_supervised_blocks is not None:
            if int(max_supervised_blocks) < 1:
                raise ValueError('max_supervised_blocks must be >= 1')
            score_limit = min(length, prefix + int(max_supervised_blocks) * self.config.block_size)
            supervised_valid = valid.copy()
            supervised_valid[:, score_limit:] = False
        else:
            supervised_valid = valid
        valid_targets = supervised_valid[:, prefix:]
        actual_nvalid = int(xp.sum(valid_targets))
        nvalid = actual_nvalid if loss_normalization_tokens is None else int(loss_normalization_tokens)
        if nvalid < actual_nvalid:
            raise ValueError("loss_normalization_tokens must be >= local valid tokens")
        if actual_nvalid == 0:
            raise ValueError("no valid supervised tokens after prefix")
        safe_ids = xp.where(valid, token_ids, 0)
        if bool(xp.any((safe_ids < 0) | (safe_ids >= self.model.config.vocab_size))):
            raise ValueError("valid token IDs must be inside the vocabulary")

        # Full teacher-forced causal encoding is safe: the block-level key
        # plans can only read positions strictly *before* each block start.
        # Later actual tokens are NEVER supplied as active block embeddings.
        with performance_scope("refinement.context_forward"):
            if routed_source_ids is None:
                context_head, context_cache = self.model.forward_body(
                    safe_ids, return_cache=True, activation_checkpoint=False,
                )
                routed_prefix_slots = 0
                routed_prefix_mode = 'none'
            else:
                # Q1's original routing/embedding logic, executed ONCE. Do
                # not let caller metadata override the causality constraint.
                metadata = dict(routed_memory_metadata or {})
                # The packed document-first sampler must give the exact start
                # of genuine same-document history in every external store.
                # Assuming a padded store is all real would invalidate routing.
                if 'history_valid_starts' not in metadata:
                    raise ValueError('routed Q2 needs history_valid_starts from the document sampler')
                starts_valid = xp.asarray(metadata['history_valid_starts'], dtype=xp.int64)
                if starts_valid.shape != (batch,) or bool(xp.any(
                    (starts_valid < 0) | (starts_valid > int(self.model.config.memory_context.distant_memory_length))
                )):
                    raise ValueError('invalid routed history_valid_starts')
                actual_lengths = xp.sum(valid.astype(xp.int64), axis=1)
                if 'target_valid_lengths' in metadata:
                    lengths_valid = xp.asarray(metadata['target_valid_lengths'], dtype=xp.int64)
                    if lengths_valid.shape != (batch,):
                        raise ValueError('invalid routed target_valid_lengths shape')
                    if not bool(xp.all(lengths_valid == actual_lengths)):
                        raise ValueError('target_valid_lengths disagree with valid_token_mask')
                elif not bool(xp.all(valid)):
                    raise ValueError('routed masked targets require explicit target_valid_lengths')
                if not bool(xp.all(valid)):
                    if metadata.get('causal_masked_target_refinement') is not True:
                        raise ValueError('masked routed refinement must explicitly enable causal masking')
                if metadata.get('causal_prefix_query', True) is not True:
                    raise ValueError('Q2 requires causal_prefix_query=True')
                metadata['causal_prefix_query'] = True
                context_head, context_cache = self.model.forward_body(
                    routed_source_ids, memory_metadata=metadata,
                    return_cache=True, activation_checkpoint=False,
                )
                routed_prefix_slots = int(context_cache['memory_context_cache']['prefix_length'])
                routed_prefix_mode = context_cache['memory_context_cache']['prefix_mode']
                if int(context_head.shape[1]) != routed_prefix_slots + length:
                    raise RuntimeError('routed causal trunk and working window disagree')
        context_shape = context_head.shape
        del context_head
        full_kv = tuple(
            (cache["attn_cache"]["k"], cache["attn_cache"]["v"])
            for cache in context_cache["block_caches"]
        )
        grad_dtype = (xp.float32 if is_low_precision_dtype(self.model.dtype)
                      else self.model.dtype)
        context_grad = ([
            [xp.zeros(kv[0].shape, dtype=grad_dtype),
             xp.zeros(kv[1].shape, dtype=grad_dtype)]
            for kv in full_kv
        ] if compute_gradients else None)
        block_size = self.config.block_size
        starts = tuple(range(prefix, length, block_size))
        if max_supervised_blocks is not None:
            starts = starts[:int(max_supervised_blocks)]
        # Add only up to K-1 trailing dummy tokens, never an unrelated document.
        extra = block_size - 1
        padded_ids = xp.pad(safe_ids, ((0, 0), (0, extra)))
        padded_valid = xp.pad(supervised_valid, ((0, 0), (0, extra)))
        loss_total = xp.asarray(0.0, dtype=xp.float64)
        initial_total = xp.asarray(0.0, dtype=xp.float64)
        commitment_total = xp.asarray(0.0, dtype=xp.float64)
        initial_position_ce_sums = xp.zeros((block_size,), dtype=xp.float64)
        commitment_position_ce_sums = xp.zeros((block_size,), dtype=xp.float64)
        position_counts = xp.zeros((block_size,), dtype=xp.int64)
        row_commitment_ce_sums = (xp.zeros((batch,), dtype=xp.float64)
                                  if record_row_ce else None)
        active_groups = 0
        active_blocks = 0
        executed_rows = []
        historical_slots = []
        removed_empty_rows = 0
        recurrent_vocab_tape_peak_bytes = 0
        avoided_history_copy_bytes = 0
        for first in range(0, len(starts), groups_limit):
            selected = starts[first:first + groups_limit]
            g = len(selected)
            target_group = xp.stack([padded_ids[:, s:s + block_size] for s in selected], axis=1)
            mask_group = xp.stack([padded_valid[:, s:s + block_size] for s in selected], axis=1)
            if not bool(xp.any(mask_group)):
                continue
            cmax = max(selected)
            # All contexts were causally encoded only once above. Here each
            # [document, target-block] is an *independent* refinement example.
            # Avoid computing full-model passes on rows that contain no target
            # (common for short, unbiased document-first samples).
            full_rows = batch * g
            flat_valid = mask_group.reshape(full_rows, block_size)
            if compact_active_blocks:
                active_idx = xp.nonzero(xp.any(flat_valid, axis=1))[0]
                active_rows = int(active_idx.shape[0])
            else:
                active_idx = None
                active_rows = full_rows
            if active_rows == 0:
                continue
            if max_active_queries and active_rows * block_size > int(max_active_queries):
                raise ValueError(
                    f"active query guard exceeded: {active_rows * block_size} > "
                    f"{max_active_queries}; lower batch/group or raise budget"
                )
            removed_empty_rows += full_rows - active_rows
            executed_rows.append(active_rows * block_size)
            with performance_scope("refinement.group_kv_materialize"):
                all_doc_rows = xp.repeat(xp.arange(batch, dtype=xp.int64), g)
                doc_rows = (all_doc_rows if active_idx is None
                            else all_doc_rows[active_idx])
                if compact_history:
                    # Build the compact plan *before* materializing K/V.
                    # It is shared by every refinement pass and Transformer
                    # layer; only selected causal keys are gathered.
                    attention_layouts = []
                    for layer in self.model.blocks[:self.model.active_layers]:
                        attn = layer.attention
                        attention_layouts.append(
                            attn._static_head_groups if attn.attention_config is not None
                            else ({"kind": "dense", "window": None, "dilation": None,
                                   "offset": 0, "stride": None},)
                        )
                    hist_index, hist_valid, compact_plans = pack_group_history(
                        (tuple(v + routed_prefix_slots for v in selected)
                         if routed_source_ids is not None else selected),
                        batch, block_size, mask_group, attention_layouts,
                        max_history_keys=max_history_keys,
                    )
                    if active_idx is not None:
                        hist_index = hist_index[active_idx]
                        hist_valid = hist_valid[active_idx]
                        compact_plans = tuple(tuple(KeySelectionPlan(
                            p.key_indices[active_idx], p.valid_mask[active_idx],
                            None if p.logit_bias is None else p.logit_bias[active_idx],
                        ) for p in layer) for layer in compact_plans)
                    if resident_history:
                        avoided_history_copy_bytes += sum(
                            (k.dtype.itemsize + v.dtype.itemsize) * active_rows *
                            int(hist_index.shape[1]) * int(k.shape[2]) * int(k.shape[3])
                            for k, v in full_kv
                        )
                        # Keep one per-layer causal K/V cache resident. Every
                        # pass sees virtual historical slots through the same
                        # doc/key map, and scatters gradients into context_grad.
                        # No [active_blocks, history_slots, heads, dim] copies.
                        from mini_llm.ops.shared_indexed_kv import SharedIndexedKV
                        grouped_kv = tuple((
                            SharedIndexedKV(k, None if context_grad is None else context_grad[i][0],
                                            doc_rows, hist_index),
                            SharedIndexedKV(v, None if context_grad is None else context_grad[i][1],
                                            doc_rows, hist_index),
                        ) for i, (k, v) in enumerate(full_kv))
                    else:
                        grouped_kv = tuple((
                            k[doc_rows[:, None], hist_index],
                            v[doc_rows[:, None], hist_index],
                        ) for k, v in full_kv)
                    context_slots = int(hist_index.shape[1])
                elif compact_active_blocks:
                    grouped_kv = tuple((k[doc_rows, :cmax], v[doc_rows, :cmax])
                                       for k, v in full_kv)
                    context_slots = cmax
                else:
                    # Preserve the original 0079H implementation as a
                    # throughput and gradient-equivalence reference.
                    grouped_kv = tuple((
                        xp.broadcast_to(k[:, None, :cmax],
                                        (batch, g, cmax) + k.shape[2:]).reshape(
                                            full_rows, cmax, *k.shape[2:]),
                        xp.broadcast_to(v[:, None, :cmax],
                                        (batch, g, cmax) + v.shape[2:]).reshape(
                                            full_rows, cmax, *v.shape[2:]),
                    ) for k, v in full_kv)
                    context_slots = cmax
            group_positions = xp.broadcast_to(
                (xp.asarray(selected, dtype=xp.int64) +
                 (int(self.model.config.memory_context.distant_memory_length)
                  if routed_source_ids is not None else 0))[None, :], (batch, g)
            ).reshape(full_rows)
            plans = (compact_plans if compact_history else
                     self._precompute_group_plans(selected, batch, mask_group, cmax))
            target_rows = target_group.reshape(full_rows, block_size)
            historical_slots.append(context_slots)
            if active_idx is not None:
                group_positions = group_positions[active_idx]
                target_rows = target_rows[active_idx]
                flat_valid = flat_valid[active_idx]
                if not compact_history:
                    plans = tuple(tuple(KeySelectionPlan(
                        plan.key_indices[active_idx], plan.valid_mask[active_idx],
                        (None if plan.logit_bias is None
                         else plan.logit_bias[active_idx]),
                    ) for plan in layer) for layer in plans)
            # Q3 reuses causal-trunk router selections/gradients. Retrieval
            # keys address the original shared trunk directly rather than the
            # statically compacted history view above.
            retrieval_tasks = None
            if routed_source_ids is not None and any(
                layer.attention._retrieval_groups
                for layer in self.model.blocks[:self.model.active_layers]
            ):
                from mini_llm.predictive_refinement.routed_retrieval import build_group_retrieval_tasks
                retrieval_tasks = build_group_retrieval_tasks(
                    self.model, context_cache, full_kv, context_grad,
                    doc_rows, group_positions, block_size, flat_valid,
                    compute_gradients=compute_gradients,
                )
            dummy_ctx = xp.zeros((active_rows, context_slots), dtype=xp.int64)
            with performance_scope("refinement.group_forward"):
                metrics, tape = self.forward(
                    dummy_ctx, target_rows,
                    valid_target_mask=flat_valid,
                    shared_context_override={
                        "context_kv": grouped_kv,
                        "block_position_starts": group_positions,
                        "layer_group_plans": plans,
                        "layer_retrieval_tasks": retrieval_tasks,
                    },
                    loss_normalization_tokens=nvalid, record_tape=compute_gradients,
                    record_row_ce=record_row_ce,
                )
            loss_total = loss_total + metrics["total_loss"]
            initial_total = initial_total + metrics["initial_ce"]
            commitment_total = commitment_total + metrics["commitment_ce"]
            initial_position_ce_sums += metrics["initial_position_ce_sums"]
            commitment_position_ce_sums += metrics["commitment_position_ce_sums"]
            position_counts += metrics["position_counts"]
            if record_row_ce:
                if compact_active_blocks:
                    xp.add.at(row_commitment_ce_sums, doc_rows,
                              metrics["row_commitment_ce_sums"].reshape(active_rows))
                else:
                    row_commitment_ce_sums += metrics["row_commitment_ce_sums"].reshape(batch, g).sum(axis=1)
            if compute_gradients:
                recurrent_vocab_tape_peak_bytes = max(
                    recurrent_vocab_tape_peak_bytes,
                    metrics["recurrent_vocab_tape_bytes"],
                )
                with performance_scope("refinement.group_backward"):
                    kv_grads = self.backward(tape)
                    if context_slots and not resident_history:
                        for (acc_k, acc_v), (dk, dv) in zip(context_grad, kv_grads):
                            if compact_history:
                                valid_w = hist_valid[..., None, None]
                                xp.add.at(acc_k, (doc_rows[:, None], hist_index),
                                          dk * valid_w)
                                xp.add.at(acc_v, (doc_rows[:, None], hist_index),
                                          dv * valid_w)
                                continue
                            # Different blocks of the same document share KV;
                            # scatter-add, never overwrite their gradients.
                            if compact_active_blocks:
                                xp.add.at(acc_k[:, :cmax], doc_rows, dk)
                                xp.add.at(acc_v[:, :cmax], doc_rows, dv)
                            else:
                                acc_k[:, :cmax] += dk.reshape(batch, g, cmax, *dk.shape[2:]).sum(axis=1)
                                acc_v[:, :cmax] += dv.reshape(batch, g, cmax, *dv.shape[2:]).sum(axis=1)
                del kv_grads
            # Release recurrent state before the next block group.
            del tape, grouped_kv, plans
            if compact_history:
                del hist_index, hist_valid, compact_plans
            active_groups += 1
            active_blocks += active_rows
        if compute_gradients:
            with performance_scope("refinement.context_backward"):
                self.model.backward_body(
                    xp.zeros(context_shape, dtype=self.model.dtype), context_cache,
                    shared_context_kv_grads=context_grad,
                )
        return {
            "total_loss": loss_total,
            "initial_ce": initial_total,
            "commitment_ce": commitment_total,
            "supervised_tokens": actual_nvalid,
            "recurrent_vocab_tape_peak_bytes": recurrent_vocab_tape_peak_bytes,
            "initial_position_ce_sums": initial_position_ce_sums,
            "commitment_position_ce_sums": commitment_position_ce_sums,
            "position_counts": position_counts,
            **({"row_commitment_ce_sums": row_commitment_ce_sums}
               if record_row_ce else {}),
            "target_blocks": active_blocks,
            "refinement_batches": active_groups,
            "causal_context_encodes": 1,
            "refinement_passes": active_groups * (block_size + 1),
            "executed_active_rows_per_pass": tuple(executed_rows),
            "active_block_queries": sum(executed_rows) * (block_size + 1),
            "context_keys_per_group_max": max(historical_slots, default=0),
            "compact_history": bool(compact_history),
            "resident_history": bool(resident_history),
            "routed_prefix_history": routed_source_ids is not None,
            "routed_prefix_slots": int(routed_prefix_slots),
            "routed_prefix_mode": routed_prefix_mode,
            "avoided_group_history_copy_bytes": avoided_history_copy_bytes,

            "skipped_empty_block_rows": removed_empty_rows,
            "active_block_row_fill": (
                active_blocks / (active_blocks + removed_empty_rows)
                if active_blocks + removed_empty_rows else 0.0
            ),
        }
