"""0080A: exact source/target, block-boundary and information-flow contracts.

Use the production causal LM alignment: inputs x[:,i] predict labels y[:,i]
(i.e., the next real token). The main hidden row at working index b*K
already saw x[b*K], equivalent to seeing exactly the committed prefix for
labels[b*K]. Do NOT use Q4's shifted-label training window unchanged: its
first target would lack an immediately preceding working-window hidden row.

Document tails are prefix-contiguous. Invalid labels can contain -100 and are
never consumed as previous-token decoder inputs.
"""

from dataclasses import dataclass
from mini_llm.backend import xp


@dataclass(frozen=True)
class BlockSchedule:
    """Packed independent blocks in document-major, then time order.

    ``source_row`` and ``boundary_working_index`` select causal main hidden
    vectors. Add the actual active routed-prefix length to the latter when
    gathering ``forward_body`` outputs. A block's proposal must see ONLY its
    boundary hidden state, never later states from the teacher-forced trunk.
    """
    source_row: object           # [blocks]
    boundary_working_index: object  # [blocks], 0,K,2K,...
    targets: object              # [blocks,K], invalid slots replaced with 0
    valid: object                # [blocks,K]
    unique_supervised_tokens: int


def block_schedule(labels, valid_mask, block_size, *, vocab_size=None):
    """Pack nonempty K-blocks, preserving short documents and masked tails.

    ``labels`` are original next-token *targets*, not the Q4 shifted source.
    The associated unshifted inputs are the ones supplied to the main trunk.
    """
    if labels.ndim != 2:
        raise ValueError('labels must be [B,T]')
    batch, time = map(int, labels.shape)
    k = int(block_size)
    if batch <= 0 or time <= 0 or k <= 0:
        raise ValueError('nonempty batch/time and positive K required')
    valid = xp.ones((batch, time), dtype=xp.bool_) if valid_mask is None else xp.asarray(valid_mask, dtype=xp.bool_)
    if valid.shape != labels.shape:
        raise ValueError('valid mask shape must match labels')
    if bool(xp.any(valid[:, 1:] & ~valid[:, :-1])):
        raise ValueError('valid targets must be prefix-contiguous within each document')
    if vocab_size is not None and bool(xp.any(valid & ((labels < 0) | (labels >= int(vocab_size))))):
        raise ValueError('valid target IDs outside vocabulary')
    ids = xp.where(valid, labels, 0)
    rows, starts, blocks, masks = [], [], [], []
    for b in range(batch):
        for start in range(0, time, k):
            if not bool(xp.any(valid[b, start:start+k])):
                continue
            end = min(time, start + k)
            length = end - start
            block = xp.zeros((k,), dtype=labels.dtype)
            mask = xp.zeros((k,), dtype=xp.bool_)
            block[:length] = ids[b, start:end]
            mask[:length] = valid[b, start:end]
            rows.append(b)
            starts.append(start)
            blocks.append(block)
            masks.append(mask)
    if not blocks:
        raise ValueError('no supervised targets in batch')
    return BlockSchedule(
        source_row=xp.asarray(rows, dtype=xp.int64),
        boundary_working_index=xp.asarray(starts, dtype=xp.int64),
        targets=xp.stack(blocks), valid=xp.stack(masks),
        unique_supervised_tokens=int(xp.sum(valid)),
    )


def shifted_decoder_tokens(target_blocks, valid_mask, *, bos_token=-1):
    """Teacher-forcing inputs: [BOS,y1,...,y(K-1)], never y_j itself.

    ``bos_token`` is a *symbol*, not an input vocabulary ID. The decoder maps
    it to a separate learned block-start embedding. Trailing invalid inputs
    remain BOS rather than reading arbitrary padding.
    """
    if target_blocks.ndim != 2:
        raise ValueError('target blocks must be [N,K]')
    valid = xp.asarray(valid_mask, dtype=xp.bool_)
    if valid.shape != target_blocks.shape:
        raise ValueError('target validity shape mismatch')
    if bool(xp.any(valid[:, 1:] & ~valid[:, :-1])):
        raise ValueError('valid block slots must be prefix-contiguous')
    shifted = xp.full(target_blocks.shape, int(bos_token), dtype=xp.int64)
    shifted[:, 1:] = xp.where(valid[:, :-1] & valid[:, 1:], target_blocks[:, :-1], int(bos_token))
    return shifted


def prefix_lm_allow_mask(block_size):
    """[2K,2K] Boolean attention mask. True means a query can read a key.

    PLAN rows: all PLAN rows only. DEC_j rows: all PLAN rows and DEC_<=j.
    Independent blocks must be batched, not concatenated into one sequence.
    """
    k = int(block_size)
    if k <= 0:
        raise ValueError('K must be positive')
    mask = xp.zeros((2*k, 2*k), dtype=xp.bool_)
    mask[:k, :k] = True
    mask[k:, :k] = True
    mask[k:, k:] = xp.tri(k, k, dtype=xp.bool_)
    return mask
