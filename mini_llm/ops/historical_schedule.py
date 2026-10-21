"""0083B: pure source-coordinate schedules for historical *read* heads.

Heads are numbered in the original GQA query layout. A historical role replaces
(not augments) that head's current-window attention in its designated layers.
Layer numbers are zero-based. No backend operations are required to audit the
full 61,440-position deterministic sweep.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class HistoricalSlot:
    layer: int
    head: int
    kind: str
    phase: int = 0
    # Explicit period prevents ambiguous inference for new head layouts.
    period: int = 0


def historical_slots(n_layers, n_heads):
    if (n_layers, n_heads) == (13, 12):
        sweep = range(2, 8)
        sweep_heads = (6, 7)
        sparse = range(0, 6)
        sparse_heads = (8, 9)
        learned = (5, 6, 7, 8, 9, 10)
        learned_heads = (10, 11)
    elif (n_layers, n_heads) == (32, 8):
        sweep = range(2, 10)
        sweep_heads = (3, 4)
        sparse = range(0, 12)
        sparse_heads = (5,)
        learned = (6, 9, 12, 15, 18, 21)
        learned_heads = (6, 7)
    elif (n_layers, n_heads) == (20, 12):
        # 0083E intermediate dense ensemble: preserve the 0083C Wide
        # deterministic 12-phase exact-token sweep, sparse global overview,
        # and six learned reread layers spread across its additional depth.
        # Head zero stays full-current causal throughout the model.
        sweep = range(2, 8)
        sweep_heads = (6, 7)
        sparse = range(0, 6)
        sparse_heads = (8, 9)
        learned = (4, 6, 9, 12, 14, 16)
        learned_heads = (10, 11)
    elif n_heads == 16 and n_layers in (16, 20, 22, 24):
        # 0083F: 1024-wide, 16/20/22/24-layer dense family. Head 0 is
        # always full-current causal; the remaining non-historical heads
        # are sliding-window 512. Reuse 0083E's 12-phase 384-period sweep,
        # 12 sparse source-coordinate offsets, and six learned reread layers.
        # Spread learned reads across depth; leave >= 3 integration layers.
        sweep = range(2, 8)
        sweep_heads = (10, 11)
        sparse = range(0, 6)
        sparse_heads = (12, 13)
        learned_by_depth = {
            16: (3, 5, 7, 9, 11, 12),
            20: (4, 6, 9, 12, 14, 16),
            22: (4, 7, 10, 13, 16, 18),
            24: (4, 8, 12, 16, 18, 20),
        }
        learned = learned_by_depth[n_layers]
        learned_heads = (14, 15)
    elif (n_layers, n_heads) in {(36, 24), (32, 28)}:
        # 0083D: large dense AR family. Retain 12 sparse views,
        # 16 sweep phases (complete coverage), and six learned reread layers.
        # Head zero is reserved for full 4K current-context attention.
        sweep = range(2, 10)
        sweep_heads = (n_heads - 8, n_heads - 7)
        sparse = range(0, 6)
        sparse_heads = (n_heads - 6, n_heads - 5)
        learned = ((7, 12, 17, 22, 27, 31) if n_layers == 36
                   else (6, 11, 16, 21, 25, 28))
        learned_heads = (n_heads - 2, n_heads - 1)
    else:
        raise ValueError(
            "historical_read requires 13x12, 20x12, 32x8, 16/20/22/24x16, 36x24, or 32x28 layout"
        )
    period = 32 * len(sweep) * len(sweep_heads)
    result = [[] for _ in range(n_layers)]
    for s, (layer, head) in enumerate((l, h) for l in sweep for h in sweep_heads):
        result[layer].append(HistoricalSlot(layer, head, 'block', s * 32, period))
    for s, (layer, head) in enumerate((l, h) for l in sparse for h in sparse_heads):
        result[layer].append(HistoricalSlot(layer, head, 'sparse', (s * 128) // 12))
    for layer in learned:
        for head in learned_heads:
            result[layer].append(HistoricalSlot(layer, head, 'learned'))
    assert period in (384, 512)
    for slots in result:
        heads = [slot.head for slot in slots]
        if len(heads) != len(set(heads)):
            raise AssertionError("historical head assigned twice on one layer")
    return tuple(tuple(slots) for slots in result)


def source_indices(slot, history_length, *, valid_start=0, block_size=32):
    """Exact source positions; no compact-index coordinate reinterpretation.

    For the 32-token deterministic sweep, the phase period is 384 on Wide and
    512 on Deep. The period is inferred from the layer schedule's width/depth.
    A negative valid_start or an incomplete document-prefix block is rejected.
    """
    if history_length < 0 or not 0 <= valid_start <= history_length:
        raise ValueError("invalid source history length or valid start")
    if slot.kind == 'block':
        # Legacy slots without an explicit period still use the original
        # 0083B mapping; new layouts carry the period directly.
        period = slot.period or (384 if slot.layer <= 7 and slot.head in (6, 7) else 512)
        if period < block_size or period % block_size:
            raise ValueError("invalid deterministic historical sweep period")
        return tuple(i for start in range(slot.phase, history_length, period)
                     for i in range(max(start, valid_start), min(start + block_size, history_length)))
    if slot.kind == 'sparse':
        return tuple(range(slot.phase + ((max(0, valid_start - slot.phase) + 127) // 128) * 128,
                           history_length, 128))
    raise ValueError("learned retrieval depends on a causal route query")


def sweep_coverage(n_layers, n_heads, history_length=61440, valid_start=0):
    slots = historical_slots(n_layers, n_heads)
    seen = []
    for layer in slots:
        for slot in layer:
            if slot.kind == 'block':
                seen.extend(source_indices(slot, history_length, valid_start=valid_start))
    return seen
