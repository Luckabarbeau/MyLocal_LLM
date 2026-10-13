"""0079Q3: causally reuse production learned routes in provisional-block attention.

The production causal trunk has already computed each layer's learned routes.
A block starting at compressed-trunk position s may use ONLY the most recent
route whose causal boundary is <= s. Its selected exact keys stay in the one
resident trunk K/V allocation; the route softmax gradients are accumulated in
that original router tape for ONE backward after all independent blocks.

No routing computation, 6K K/V copy, or 64K dense attention is performed per
refinement pass. Learned route identities remain locally fixed as in the
original model's top-k backward convention.
"""

from mini_llm.backend import xp, is_low_precision_dtype
from mini_llm.ops.attention_selection import KeySelectionPlan
from mini_llm.ops.shared_indexed_kv import SharedIndexedKV


def build_group_retrieval_tasks(model, context_cache, full_kv, context_grad,
                                doc_rows, absolute_starts, block_size, valid,
                                *, compute_gradients=True):
    """Return per-layer learned-head tasks; use only preceding causal routes."""
    mc = model.config.memory_context
    prefix = int(context_cache['memory_context_cache']['prefix_length'])
    starts = xp.asarray(absolute_starts, dtype=xp.int64) - int(mc.distant_memory_length) + prefix
    rows = int(starts.shape[0])
    if prefix % int(block_size) and not context_cache['memory_context_cache'].get(
        'allow_unaligned_refinement_blocks', False
    ):
        raise NotImplementedError('Q3 requires routed-prefix slots aligned to K')
    # Q4 short documents can expose an exact 1..K-1-token genuine prefix.
    # For those cases each provisional block freezes the MOST RECENT route
    # strictly available by its start. No later real-token route is reused.
    if xp.any(starts < prefix):
        raise ValueError('route starts before the actual working window')
    tasks_per_layer = []
    for layer_idx, layer in enumerate(model.blocks[:model.active_layers]):
        router_groups = layer.attention._retrieval_groups
        if not router_groups:
            tasks_per_layer.append(())
            continue
        ctx_cache = context_cache['block_caches'][layer_idx]['attn_cache']
        ctx_groups = {entry['name']: entry for entry in ctx_cache['pattern_caches']
                      if entry['kind'] == 'retrieval'}
        k, v = full_kv[layer_idx]
        full_len = int(k.shape[1])
        if int(xp.max(starts)) > full_len:
            raise ValueError('block start is outside the shared causal cache')
        # Broadcasted indices are metadata only: no per-block historical K/V.
        # Only selected keys are gathered by the existing P2 split kernel.
        full_history = xp.broadcast_to(xp.arange(full_len, dtype=xp.int64)[None, :],
                                       (rows, full_len))
        gradients = None if context_grad is None else context_grad[layer_idx]
        resident = (
            SharedIndexedKV(k, None if gradients is None else gradients[0],
                            doc_rows, full_history),
            SharedIndexedKV(v, None if gradients is None else gradients[1],
                            doc_rows, full_history),
        )
        layer_tasks = []
        for group in router_groups:
            name = group['name']
            if name not in ctx_groups:
                raise RuntimeError(f'missing original learned route cache: {name}')
            route_cache = ctx_groups[name]['cache']
            attn_cache = route_cache['attention_cache']
            selected = attn_cache['selected_blocks']
            weights = attn_cache['selected_weights']
            route_starts = attn_cache['route_starts']
            config = group['module'].config
            r = int(route_starts.shape[0])
            h = len(group['head_indices'])
            nh = int(selected.shape[2])
            kb = int(selected.shape[3])
            bs = int(config.history_block_size)
            # In the production architecture routing_stride=128 and K=8.
            # A route then remains fixed throughout each eight-token block.
            # Reject other geometries rather than silently omitting a route
            # refresh that autoregressive inference would perform inside K.
            if int(config.routing_stride) % int(block_size):
                raise NotImplementedError(
                    'Q3 requires K to divide retrieval routing_stride'
                )
            slots = kb * bs
            if slots < 1 or nh not in (1, h):
                raise ValueError('invalid production router head geometry')
            if r:
                # Last route established no later than the provisional block.
                selected_route = xp.searchsorted(route_starts, starts, side='right') - 1
                route_valid = selected_route >= 0
                safe_route = xp.maximum(selected_route, 0)
                blocks = selected[doc_rows, safe_route]
                route_weights = weights[doc_rows, safe_route]
                boundary = route_starts[safe_route]
                # We require complete historical blocks strictly before the
                # route boundary and therefore before the target block.
                ends = (blocks + 1) * bs
                if bool(xp.any(route_valid[:, None, None] & (
                    ends > (boundary[:, None, None] - int(config.exclude_recent_tokens))
                ))):
                    raise ValueError('learned route selects future/ineligible history')
                token_keys = (blocks[..., None] * bs +
                              xp.arange(bs, dtype=xp.int64)).reshape(rows, nh, slots)
                if bool(xp.any(token_keys >= full_len)):
                    raise ValueError('selected retrieval key outside compressed trunk')
                route_mask = xp.broadcast_to(route_valid[:, None, None, None],
                                              (rows, nh, block_size, slots))
                token_keys = xp.broadcast_to(token_keys[:, :, None, :],
                                             (rows, nh, block_size, slots))
                if config.router_weight_mode == 'logit_bias':
                    w = route_weights.astype('float32', copy=False)
                    biases = float(config.router_weight_scale) * xp.log(
                        w + float(config.router_weight_eps)
                    )
                    biases = xp.repeat(biases, bs, axis=-1)
                    biases = xp.broadcast_to(biases[:, :, None, :],
                                              (rows, nh, block_size, slots))
                elif config.router_weight_mode == 'none':
                    biases = None
                else:
                    raise ValueError('unsupported learned routing weight mode')
            else:
                selected_route = xp.zeros((rows,), dtype=xp.int64)
                route_valid = xp.zeros((rows,), dtype=xp.bool_)
                route_weights = xp.zeros((rows, nh, kb), dtype=weights.dtype)
                token_keys = xp.zeros((rows, nh, block_size, slots), dtype=xp.int64)
                route_mask = xp.zeros(token_keys.shape, dtype=xp.bool_)
                biases = xp.zeros(token_keys.shape, dtype=xp.float32) if config.router_weight_mode == 'logit_bias' else None
            if valid is not None:
                route_mask = route_mask & valid[:, None, :, None]
            plan = KeySelectionPlan(token_keys, route_mask, biases)
            # The original causal cache will run its router.backward ONCE.
            if compute_gradients and config.router_weight_mode == 'logit_bias' and r:
                extra = route_cache.get('refinement_extra_dweights')
                if extra is None:
                    extra = xp.zeros(weights.shape, dtype=xp.float32)
                    route_cache['refinement_extra_dweights'] = extra
            else:
                extra = None
            layer_tasks.append({
                'heads': group['head_indices'], 'kv': resident, 'plan': plan,
                'doc_rows': doc_rows, 'route_slots': selected_route,
                'route_valid': route_valid, 'route_weights': route_weights,
                'router_heads': nh, 'block_size': bs,
                'weight_scale': float(config.router_weight_scale),
                'weight_eps': float(config.router_weight_eps),
                'extra_dweights': extra,
            })
        tasks_per_layer.append(tuple(layer_tasks))
    return tuple(tasks_per_layer)


def accumulate_retrieval_bias_gradient(task, dbias):
    """Reduce provisional attention prior gradient into the ORIGINAL route tape."""
    extra = task['extra_dweights']
    if extra is None:
        return
    rows, n_heads, n_queries, slots = map(int, dbias.shape)
    bs = task['block_size']
    if slots % bs:
        raise ValueError('learned retrieval slot count is not block aligned')
    nw = slots // bs
    # Same derivative as the production block-retrieval kernel:
    # d/dw [scale * log(w + eps)] = scale / (w + eps).
    bgrad = dbias.reshape(rows, n_heads, n_queries, nw, bs).sum(axis=(2, 4))
    if task['router_heads'] == 1 and n_heads != 1:
        bgrad = bgrad.sum(axis=1, keepdims=True)
    weights = task['route_weights'].astype(bgrad.dtype, copy=False)
    dweights = bgrad * (task['weight_scale'] / (weights + task['weight_eps']))
    dweights *= task['route_valid'][:, None, None]
    xp.add.at(extra, (task['doc_rows'], task['route_slots']),
              dweights.astype(extra.dtype, copy=False))
