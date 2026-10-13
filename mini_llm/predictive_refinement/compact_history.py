"""0079N: compact, block-boundary-causal K/V views for grouped refinement.

The full teacher-forced context is encoded once, but a refinement group reads
only the union of historical keys selected by its static attention heads.  A
provisional row never accesses teacher-forced tokens at/after its block start.
This opt-in reference deliberately does *not* implement learned retrieval.
"""

import numpy as np

from mini_llm.backend import xp, asnumpy
from mini_llm.ops.attention_selection import KeySelectionPlan


def _history(start, j, group):
    """Match compact_provisional_plan's exact historical selection semantics."""
    kind = group['kind']
    pos = start + j
    if kind == 'dense':
        return tuple(range(start))
    if kind == 'local':
        return tuple(range(max(0, pos - int(group['window']) + 1), start))
    if kind == 'dilated':
        dilation = int(group['dilation'])
        offset = int(group['offset'] or 0)
        slots = (int(group['window']) - 1 - offset) // dilation + 1
        return tuple(sorted(p for m in range(slots)
                            if 0 <= (p := pos - offset - m * dilation) < start))
    if kind == 'global_sparse':
        return tuple(range(int(group['offset'] or 0), start, int(group['stride'])))
    raise NotImplementedError(f'Unsupported group in compact history: {kind}')


def pack_group_history(block_starts, batch_size, block_size, valid_block_mask,
                       attention_groups_per_layer, *, max_history_keys=4096):
    """Build per-row compact historical key indices and remapped attention plans.

    Returns (history_index, history_valid, layer_plans), with history_index
    [B*G,H], histories padded with index 0 and masked.  The plans address
    [0,H) for historical keys and [H,H+K) for provisional keys.  Context
    gathering/scattering is performed separately so cache ownership is clear.

    Metadata generation is on CPU; only bounded index/plan arrays are copied
    to the backend.  This avoids per-head device-to-host synchronization.
    """
    starts = tuple(int(s) for s in block_starts)
    b, g, k = int(batch_size), len(starts), int(block_size)
    if b < 1 or g < 1 or k < 1 or any(s < 0 for s in starts):
        raise ValueError('invalid grouped compact-history dimensions')
    mask = np.asarray(asnumpy(valid_block_mask), dtype=bool)
    if mask.shape != (b, g, k):
        raise ValueError('valid masks must be [B,G,K]')
    if not attention_groups_per_layer:
        raise ValueError('attention group layouts are required')
    layouts = tuple(tuple(layer) for layer in attention_groups_per_layer)
    unique_specs = {}
    for layer in layouts:
        for group in layer:
            signature = (group['kind'], group.get('window'), group.get('dilation'),
                         group.get('offset') or 0, group.get('stride'))
            unique_specs[signature] = group
    selection = {}
    for start in starts:
        for signature, spec in unique_specs.items():
            for j in range(k):
                selection[(start, signature, j)] = _history(start, j, spec)
    histories = []
    for doc in range(b):
        for gi, start in enumerate(starts):
            union = set()
            for signature in unique_specs:
                for j in range(k):
                    union.update(selection[(start, signature, j)])
            keys = sorted(union)
            if len(keys) > int(max_history_keys):
                raise ValueError('compact history exceeds max_history_keys budget')
            histories.append(keys)
    rows = b * g
    width = max(map(len, histories), default=0)
    hist_index = np.zeros((rows, width), dtype=np.int64)
    hist_valid = np.zeros((rows, width), dtype=bool)
    per_row_maps = []
    for row, keys in enumerate(histories):
        hist_index[row, :len(keys)] = keys
        hist_valid[row, :len(keys)] = True
        per_row_maps.append({key: idx for idx, key in enumerate(keys)})
    layer_plans = []
    plan_cache = {}
    for layer in layouts:
        group_plans = []
        for spec in layer:
            signature = (spec['kind'], spec.get('window'), spec.get('dilation'),
                         spec.get('offset') or 0, spec.get('stride'))
            if signature not in plan_cache:
                key_lists = [[selection[(start, signature, j)] for j in range(k)]
                             for start in starts]
                slots = max((len(keys) for per_start in key_lists for keys in per_start),
                            default=0) + k
                indices = np.zeros((rows, 1, k, slots), dtype=np.int64)
                allowed = np.zeros((rows, 1, k, slots), dtype=bool)
                for doc in range(b):
                    for gi, start in enumerate(starts):
                        row = doc * g + gi
                        remap = per_row_maps[row]
                        for j, keys in enumerate(key_lists[gi]):
                            n = len(keys)
                            if n:
                                indices[row, 0, j, :n] = [remap[key] for key in keys]
                                allowed[row, 0, j, :n] = True
                            indices[row, 0, j, n:n+k] = np.arange(width, width+k)
                            allowed[row, 0, j, n:n+k] = mask[doc, gi]
                            # Padded query: make self-key visible to avoid empty rows.
                            if not mask[doc, gi, j]:
                                allowed[row, 0, j, n+j] = True
                plan_cache[signature] = KeySelectionPlan(
                    xp.asarray(indices), xp.asarray(allowed))
            group_plans.append(plan_cache[signature])
        layer_plans.append(tuple(group_plans))
    return (xp.asarray(hist_index), xp.asarray(hist_valid), tuple(layer_plans))
