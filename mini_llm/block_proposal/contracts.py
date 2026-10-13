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
import numpy as np
from mini_llm.backend import xp, asnumpy


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
    # The mask is prefix-contiguous (verified above). Copy only B valid
    # lengths to the host, then construct compact block indices in one batch.
    # The historical per-block xp.any() caused one GPU synchronization for
    # *every* K-block (e.g. 1,024 syncs for B=2, T=4096, K=8).
    lengths = asnumpy(xp.sum(valid, axis=1)).astype(np.int64)
    counts = (lengths + k - 1) // k
    nblocks = int(counts.sum())
    if nblocks == 0:
        raise ValueError('no supervised targets in batch')
    rows_host = np.repeat(np.arange(batch, dtype=np.int64), counts)
    starts_host = np.concatenate([
        np.arange(int(count) * k, dtype=np.int64)[::k]
        for count in counts if count
    ])
    rows = xp.asarray(rows_host, dtype=xp.int64)
    starts = xp.asarray(starts_host, dtype=xp.int64)
    columns = starts[:, None] + xp.arange(k, dtype=xp.int64)[None, :]
    masks = columns < xp.asarray(lengths, dtype=xp.int64)[rows, None]
    # Clamping handles a short final block before masking padded values.
    gather_columns = xp.minimum(columns, time - 1)
    blocks = xp.where(masks, labels[rows[:, None], gather_columns], 0)
    return BlockSchedule(
        source_row=rows, boundary_working_index=starts,
        targets=blocks, valid=masks,
        unique_supervised_tokens=int(lengths.sum()),
    )


def shifted_decoder_tokens(target_blocks, valid_mask, *, bos_token=-1,
                           trusted_prefix_contiguous=False):
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
    if not trusted_prefix_contiguous and bool(xp.any(valid[:, 1:] & ~valid[:, :-1])):
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
