"""0079F: compact block-causal key plans for static heterogeneous heads.

Only K active/provisional queries use these plans. Committed-context rows keep
exactly their original causal attention kernels. No [C+K,C+K] masks or
[B,H,C,K,D] selected-value intermediates are constructed here.

This first integration intentionally excludes learned retrieval/terminal memory:
those require a separate router evaluated at the *committed* block boundary.
"""

from mini_llm.backend import xp, BACKEND_NAME
from mini_llm.ops.attention_selection import KeySelectionPlan


def compact_provisional_plan(
    context_length, block_size, batch_size, *, kind, window=None,
    dilation=None, offset=0, stride=None, valid_block_mask=None,
    max_selected_keys=8192, absolute_block_start=None,
):
    """Create only [B,1,K,S] indexed keys for a provisional block.

    Static context selections match the specified original causal topology at
    each *absolute provisional query position*, while all K provisional keys
    are visible bidirectionally. Padded future keys are never visible to valid
    queries. Duplicate provisional/context keys are deliberately avoided.
    """
    c, k, b = int(context_length), int(block_size), int(batch_size)
    start = c if absolute_block_start is None else int(absolute_block_start)
    if not 0 <= start <= c:
        raise ValueError("absolute block start must lie inside stored causal context")
    if c < 0 or k < 1 or b < 1:
        raise ValueError("context_length >= 0, block_size >= 1 and batch_size >= 1 required")
    if kind not in {"dense", "local", "dilated", "global_sparse"}:
        raise NotImplementedError("compact 0079F supports only static attention groups")
    if int(max_selected_keys) < k:
        raise ValueError("max_selected_keys must fit the provisional block")
    if kind in {"local", "dilated"} and (window is None or int(window) < 1):
        raise ValueError("local/dilated require a positive window")
    if kind == "dilated" and (dilation is None or int(dilation) < 1 or not 0 <= offset < dilation):
        raise ValueError("dilation must be positive with 0 <= offset < dilation")
    if kind == "global_sparse" and (stride is None or int(stride) < 1 or not 0 <= offset < stride):
        raise ValueError("global_sparse requires stride >= 1 and 0 <= offset < stride")
    if valid_block_mask is None:
        valid = xp.ones((b, k), dtype=xp.bool_)
    else:
        valid = xp.asarray(valid_block_mask, dtype=xp.bool_)
        if valid.shape != (b, k):
            raise ValueError("valid_block_mask must be [B,K]")
        # The parent refinement trainer validates this once. Do not introduce
        # one synchronizing device->host reduction *per head per layer per pass*.
        if BACKEND_NAME == "numpy" and bool(xp.any(valid[:, 1:] & ~valid[:, :-1])):
            raise ValueError("valid targets must form a prefix")

    # Construct *small* query-dependent historical selections. The maximum
    # number of keys is bounded by max_selected_keys before allocating plans.
    histories = []
    for j in range(k):
        pos = start + j
        if kind == "dense":
            keys = range(start)
        elif kind == "local":
            keys = range(max(0, pos - int(window) + 1), start)
        elif kind == "dilated":
            slots = (int(window) - 1 - int(offset)) // int(dilation) + 1
            keys = (pos - int(offset) - m * int(dilation) for m in range(slots))
            keys = sorted(key for key in keys if 0 <= key < start)
        else:
            keys = range(int(offset), start, int(stride))
        if len(keys) + k > int(max_selected_keys):
            raise ValueError("compact block exceeds max_selected_keys; reduce context/selection")
        histories.append(tuple(keys))

    slots = max(len(keys) for keys in histories) + k
    positions = xp.zeros((b, 1, k, slots), dtype=xp.int64)
    mask = xp.zeros((b, 1, k, slots), dtype=xp.bool_)
    for j, keys in enumerate(histories):
        count = len(keys)
        if count:
            positions[:, 0, j, :count] = xp.asarray(keys, dtype=xp.int64)
            mask[:, 0, j, :count] = True
        positions[:, 0, j, count:count + k] = xp.arange(c, c + k)
        mask[:, 0, j, count:count + k] = valid
        # Invalid rows do not contribute to the loss, but self attention
        # remains valid to avoid an empty row for c == 0.
        mask[:, 0, j, count + j] |= ~valid[:, j]
    return KeySelectionPlan(positions, mask)


def grouped_provisional_plan(context_storage_length, block_starts, batch_size, block_size,
                             *, valid_block_mask=None, **group):
    """0079H: B independent document rows x G block starts, packed as [B*G,K].

    Historical keys use absolute indices below *each block's own* start;
    provisional keys occupy the same physical K slots after the maximum
    stored prefix. The plan is built once per group and reused in all K+1
    refinement passes. No square context mask or cross-block active keys.
    """
    starts = tuple(int(s) for s in block_starts)
    b, g, k, c = int(batch_size), len(starts), int(block_size), int(context_storage_length)
    if not starts or any(s < 0 or s > c for s in starts):
        raise ValueError("each block start must fit the stored causal prefix")
    if valid_block_mask is None:
        valid = xp.ones((b, g, k), dtype=xp.bool_)
    else:
        valid = xp.asarray(valid_block_mask, dtype=xp.bool_)
        if valid.shape != (b, g, k):
            raise ValueError("group validity must have shape [B,G,K]")
    individual = [
        compact_provisional_plan(
            c, k, b, absolute_block_start=start,
            valid_block_mask=valid[:, gi], **group,
        ) for gi, start in enumerate(starts)
    ]
    width = max(int(plan.key_indices.shape[-1]) for plan in individual)
    keys = xp.zeros((b, g, 1, k, width), dtype=xp.int64)
    mask = xp.zeros((b, g, 1, k, width), dtype=xp.bool_)
    for gi, plan in enumerate(individual):
        width_i = int(plan.key_indices.shape[-1])
        keys[:, gi, :, :, :width_i] = plan.key_indices
        mask[:, gi, :, :, :width_i] = plan.valid_mask
    return KeySelectionPlan(
        keys.reshape(b * g, 1, k, width),
        mask.reshape(b * g, 1, k, width),
    )
