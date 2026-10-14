"""0080D reference integration: one causal main trunk and K-block decoder groups.

Original routed-prefix memory, learned per-layer retrieval heads, and existing
production Transformer are kept intact. Only boundary *hidden states* are
read by the K-horizon proposal head; the entire target window passes once
through the causal main body and once through its backward.

This trainer does not perform optimizer updates, checkpointing, or GPU peak
profiling; those belong to 0080F. Calls are synchronous and train one window.
"""

import os

from mini_llm.backend import xp, asnumpy, is_low_precision_dtype, BACKEND_NAME
from mini_llm.performance_profiler import performance_scope, block_proposal_scope, block_proposal_detail_scope
from .contracts import block_schedule


def compact_working_length(full_length, maximum_valid, block_size):
    """Choose a compact causal working-window bucket without changing samples.

    Default 4096-token windows use 512/1024/2048/4096. Smaller test
    configurations use proportionally smaller buckets. Intermediate buckets
    remain multiples of K; the configured full length may be nondivisible and
    its final partial K block is preserved.
    """
    full_length, maximum_valid, block_size = (
        int(full_length), int(maximum_valid), int(block_size)
    )
    if full_length <= 0 or block_size <= 0:
        raise ValueError("working-window length and K must be positive")
    if maximum_valid < 1 or maximum_valid > full_length:
        raise ValueError("valid target extent outside configured working window")
    bucket = max(block_size, full_length // 8)
    bucket = min(full_length, ((bucket + block_size - 1) // block_size) * block_size)
    while bucket < maximum_valid:
        bucket = min(full_length, bucket * 2)
    return bucket


class SinglePassBlockTrainer:
    def __init__(self,two_stage_model):
        self.net=two_stage_model
        self.model=two_stage_model.proposal_head.model

    def parameters(self):
        return self.net.parameters()

    def zero_grad(self):
        return self.net.zero_grad()

    def train_window(self,source_inputs,target_labels,*,valid_target_mask,
                     memory_metadata=None,block_group_size=8,
                     compute_gradients=True,grad_scale=1.0,
                     plan_condition='normal',prefix_condition='normal',
                     defer_metrics=None):
        """Perform one shared causal forward/backward for an original LM batch.

        source_inputs: production [D history | T *current* inputs], not Q4
        shifted labels; target_labels: next-token labels corresponding to T
        current inputs. Boundary h at active prefix + b*K has seen x_(b*K).

        All groups have exact shared-denominator loss normalization. When
        accumulating gradient across microbatches, caller should pass a
        per-microbatch ``grad_scale`` proportional to its unique target count.
        """
        # GPU production training retains loss scalars on device until a
        # bounded logging/optimizer boundary; standalone CPU references use
        # the historical Python-float reporting contract by default.
        if defer_metrics is None:
            defer_metrics = BACKEND_NAME == 'cupy'
        if target_labels.ndim!=2 or source_inputs.ndim!=2:
            raise ValueError('expected source [B,D+T] and targets [B,T]')
        b,t=map(int,target_labels.shape)
        if source_inputs.shape[0]!=b or b==0:
            raise ValueError('batch dimensions mismatch')
        if plan_condition not in ('normal','neutral','shuffled'):
            raise ValueError('plan_condition must be normal, neutral or shuffled')
        if prefix_condition not in ('normal','masked'):
            raise ValueError('prefix_condition must be normal or masked')
        if compute_gradients and (plan_condition!='normal' or prefix_condition!='normal'):
            raise ValueError('ablation modes are evaluation-only; do not backpropagate')
        group_size=int(block_group_size)
        if group_size<=0:
            raise ValueError('block_group_size must be positive')
        memory_enabled=self.model.hierarchical_memory_enabled
        if memory_enabled:
            cfg=self.model.config.memory_context
            if cfg.integration_mode!='routed_prefix':
                raise ValueError('0080D requires existing production routed-prefix memory')
            if cfg.memory_training!='joint':
                raise ValueError('0080D needs joint router+main model training')
            if source_inputs.shape[1]!=cfg.source_input_length or t!=cfg.target_length:
                raise ValueError('production routed source/target length mismatch')
            if memory_metadata is None or 'history_valid_starts' not in memory_metadata:
                raise ValueError('document-first routed samples need genuine history_valid_starts')
            metadata=dict(memory_metadata)
            if metadata.get('causal_prefix_query',True) is not True:
                raise ValueError('noncausal routing is forbidden for 0080D')
            metadata['causal_prefix_query']=True
            metadata['causal_masked_target_refinement']=True
            if 'target_valid_lengths' not in metadata:
                raise ValueError('routed training requires target_valid_lengths')
            valid_lengths=xp.sum(xp.asarray(valid_target_mask,dtype=xp.bool_),axis=1)
            if not bool(xp.all(xp.asarray(metadata['target_valid_lengths'])==valid_lengths)):
                raise ValueError('target lengths and mask disagree')
        else:
            if source_inputs.shape[1]!=t:
                raise ValueError('direct context inputs and labels must have same length')
            if memory_metadata is not None:
                raise ValueError('memory_metadata requires routed main model')
            metadata=None
        # 0080O11: keep document-first sampling and the entire [D|4096]
        # source unchanged. Only the trailing *working* padding is absent from
        # the shared Transformer. The compact length is common to this B-row
        # microbatch; no per-row attention mask or large 64K activation exists.
        full_working_length = t
        if (compute_gradients and
                os.environ.get('MINI_LLM_BLOCK_PROPOSAL_COMPACT_TARGET', '0') == '1'):
            # Validate the *original* full-length mask before slicing. A
            # non-prefix mask must never hide valid later targets by reducing
            # the apparent extent to its population count.
            full_valid = xp.asarray(valid_target_mask, dtype=xp.bool_)
            if full_valid.shape != target_labels.shape or bool(xp.any(
                    full_valid[:, 1:] & ~full_valid[:, :-1])):
                raise ValueError('compact targets require a full-length prefix-valid mask')
            if memory_enabled:
                # Already checked against valid_target_mask, and retained on
                # CPU by the original document-first sampler.
                maximum_valid = max(int(x) for x in metadata['target_valid_lengths'])
            else:
                maximum_valid = int(asnumpy(xp.max(xp.sum(valid_target_mask, axis=1))))
            compact_length = compact_working_length(
                full_working_length, maximum_valid, self.net.proposal_head.block_size
            )
            if compact_length < t:
                target_labels = target_labels[:, :compact_length]
                valid_target_mask = valid_target_mask[:, :compact_length]
                if memory_enabled:
                    metadata['compact_working_length'] = compact_length
                else:
                    source_inputs = source_inputs[:, :compact_length]
                t = compact_length
        schedule=block_schedule(target_labels,valid_target_mask,
                                self.net.proposal_head.block_size,
                                vocab_size=self.model.config.vocab_size)
        nvalid=schedule.unique_supervised_tokens
        with performance_scope('block_proposal.main_trunk.forward'):
            if compute_gradients:
                main_hidden,trunk_cache=self.model.forward_body(
                    source_inputs,memory_metadata=metadata,return_cache=True,
                    activation_checkpoint=False)
            else:
                main_hidden=self.model.forward_body(source_inputs,memory_metadata=metadata,
                                                     return_cache=False)
                trunk_cache=None
        if memory_enabled:
            prefix=int(trunk_cache['memory_context_cache']['prefix_length']) if trunk_cache is not None else int(main_hidden.shape[1]-t)
        else:
            prefix=0
        # Reuse production routing-mode accounting without retaining the full
        # backward cache / large intermediate tensors across microbatches.
        used_mode = ((trunk_cache.get('memory_context_cache') or {}).get('prefix_mode', 'none')
                     if trunk_cache is not None else ('routed_blocks' if prefix else 'none'))
        routing_cache = {'prefix_mode': used_mode, 'prefix_length': int(prefix)}
        if int(main_hidden.shape[1])!=prefix+t:
            raise RuntimeError('unexpected main hidden working-window length')
        # Reusable gradient buffer is [B,selected_2K+T,d], never 64K.
        dh_buffer=xp.zeros(main_hidden.shape,dtype=xp.float32 if is_low_precision_dtype(self.model.dtype) else main_hidden.dtype) if compute_gradients else None
        sums={'proposal_ce':0.,'final_ce':0.,'total_loss':0.}
        horizon_count=xp.zeros((self.net.proposal_head.block_size,),dtype=xp.int64)
        proposal_horizon_sum=xp.zeros((self.net.proposal_head.block_size,),dtype=xp.float64)
        final_horizon_sum=xp.zeros((self.net.proposal_head.block_size,),dtype=xp.float64)
        row_final_sum=xp.zeros((b,),dtype=xp.float64)
        selected_rows=schedule.source_row
        boundaries=schedule.boundary_working_index+prefix
        nblocks=int(schedule.targets.shape[0])
        # One compact device->host transfer for all block counts, rather
        # than one implicit GPU synchronization per group and per CE loss.
        # The same schedule also establishes the global unique denominator.
        block_counts=asnumpy(xp.sum(schedule.valid,axis=1)).astype('int64')
        if int(block_counts.sum()) != nvalid:
            raise RuntimeError('block schedule supervised-token count mismatch')
        shuffled_plan_bank=None
        shuffled_indices=None
        if plan_condition=='shuffled':
            # Compute *only compact P@E plans* for all boundaries in bounded
            # vocabulary chunks. Shuffle across independently sampled rows,
            # not nearby windows in the same row. The sampler can occasionally
            # repeat a document. Never swap the direct residual prior logits.
            original_rows=[int(v) for v in asnumpy(selected_rows).tolist()]
            unique_rows=sorted(set(original_rows))
            if len(unique_rows)<2:
                raise ValueError('shuffled-plan control requires >=2 independent source rows')
            row_blocks={row:[] for row in unique_rows}
            for i,row in enumerate(original_rows):
                row_blocks[row].append(i)
            offset_in_row={row:0 for row in unique_rows}
            rotated={row:unique_rows[(j+1)%len(unique_rows)]
                     for j,row in enumerate(unique_rows)}
            shuffled_indices=[]
            for row in original_rows:
                target_row=rotated[row]
                candidates=row_blocks[target_row]
                shuffled_indices.append(candidates[offset_in_row[row]%len(candidates)])
                offset_in_row[row]+=1
            bank=[]
            for pre_start in range(0,nblocks,group_size):
                pre_end=min(pre_start+group_size,nblocks)
                hpre=main_hidden[selected_rows[pre_start:pre_end],
                                 boundaries[pre_start:pre_end],:]
                lp,_=self.net.proposal_head.forward(hpre)
                z,_=self.net.decoder.expectation.forward(lp)
                bank.append(z)
            shuffled_plan_bank=xp.concatenate(bank,axis=0)
        for start in range(0,nblocks,group_size):
            end=min(start+group_size,nblocks)
            rows=selected_rows[start:end]
            positions=boundaries[start:end]
            h=main_hidden[rows,positions,:].copy()
            y=schedule.targets[start:end]
            valid=schedule.valid[start:end]
            with block_proposal_scope('block_proposal.decoder_group.forward'):
                override=(shuffled_plan_bank[shuffled_indices[start:end]]
                          if shuffled_plan_bank is not None else None)
                metrics,cache=self.net.forward(h,y,valid,
                    plan_condition=plan_condition,prefix_condition=prefix_condition,
                    plan_override=override,defer_metrics=defer_metrics,
                    known_valid_count=int(block_counts[start:end].sum()),
                    trusted_block_schedule=True)
            ng=int(metrics['unique_supervised_tokens'])
            group_weight=ng/nvalid
            for key in sums:
                sums[key]+=metrics[key]*group_weight
            horizon_count+=metrics['position_counts']
            proposal_horizon_sum+=metrics['proposal_position_ce_sums']
            final_horizon_sum+=metrics['final_position_ce_sums']
            xp.add.at(row_final_sum,rows,metrics['block_row_final_ce_sums'])
            if compute_gradients:
                with block_proposal_scope('block_proposal.decoder_group.backward'):
                    boundary_grad=self.net.backward(cache,scale=float(grad_scale)*group_weight)
                with block_proposal_detail_scope('block_proposal.boundary_gradient.scatter'):
                    xp.add.at(dh_buffer,(rows,positions),boundary_grad.astype(dh_buffer.dtype,copy=False))
            del h,cache
        del main_hidden
        if compute_gradients:
            with performance_scope('block_proposal.main_trunk.backward'):
                # All horizon and decoder chunks accumulated into this SINGLE backward.
                self.model.backward_body(dh_buffer,trunk_cache)
        return dict(sums,unique_supervised_tokens=nvalid,active_blocks=nblocks,
                    position_counts=horizon_count,
                    proposal_position_ce_sums=proposal_horizon_sum,
                    final_position_ce_sums=final_horizon_sum,
                    decoder_position_gain=(proposal_horizon_sum-final_horizon_sum)/xp.maximum(horizon_count,1),
                    row_final_ce_sums=row_final_sum,
                    main_forward_passes=1,
                    main_backward_passes=1 if compute_gradients else 0,
                    routed_prefix_slots=prefix, routing_cache=routing_cache,
                    trunk_working_tokens_per_row=t,
                    trunk_padding_tokens_saved_per_row=full_working_length-t)
