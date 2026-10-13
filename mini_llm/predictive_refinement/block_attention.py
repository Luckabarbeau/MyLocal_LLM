"""0079B: compact *reference* block-causal attention masks.

Only use on deliberately small direct dense context windows. The training
attention implementation materializes O((C+K)^2) scores, so this is *not* the
production long-history or cached-refinement attention implementation.
"""

from mini_llm.backend import xp


def block_causal_allow_mask(
    context_length, block_size, *, valid_block_mask=None, max_reference_tokens=256,
):
    """Return allow mask [T,T] or [B,T,T] (True means key is visible).

    Context queries are strictly causal and never see a provisional key.
    Valid provisional queries see the context and all *valid* provisional keys
    (including those later in the block). Invalid query rows can see context
    but are never visible as keys to valid queries. The target values are never
    involved in mask construction.
    """
    c, k = int(context_length), int(block_size)
    if c < 0 or k <= 0:
        raise ValueError("context_length must be nonnegative and block_size positive")
    n = c + k
    if n > int(max_reference_tokens):
        raise ValueError("dense block-causal reference is limited to short sequences")
    positions = xp.arange(c)
    allow = xp.zeros((n, n), dtype=xp.bool_)
    allow[:c, :c] = positions[:, None] >= positions[None, :]
    allow[c:, :c] = True
    if valid_block_mask is None:
        allow[c:, c:] = True
        return allow
    valid = xp.asarray(valid_block_mask, dtype=xp.bool_)
    if valid.ndim == 1 and valid.shape == (k,):
        allow[c:, c:] = xp.broadcast_to(valid[None, :], (k, k))
        # Invalid query positions are never supervised, but keep self visible.
        invalid = xp.where(~valid)[0]
        allow[c + invalid, c + invalid] = True
        return allow
    if valid.ndim == 2 and valid.shape[1] == k:
        batched = xp.broadcast_to(allow, (valid.shape[0], n, n)).copy()
        batched[:, c:, c:] = valid[:, None, :]
        # No invalid query row is all masked, including for C=0.
        indices = xp.arange(k)
        batched[:, c + indices, c + indices] |= ~valid
        return batched
    raise ValueError("valid_block_mask must be [K] or [B,K]")
