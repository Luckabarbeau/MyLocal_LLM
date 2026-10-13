"""0080K: read-only horizon-specific future-plan controls.

This module deliberately does not enter the training path.  A single production
causal main forward gives every block-boundary hidden state. Proposal logits
are computed once per bounded group. The K distinct decoder input patterns
needed to score the K horizons are batched, so they do not repeat the main
trunk. Labels/prefixes remain exactly those used in 0080I validation.
"""

import numpy as np

from mini_llm.backend import xp, asnumpy
from .contracts import block_schedule

SCOPES = (
    'all',                 # all K position plans, reference
    'own_only',            # only proposal P_j
    'past_and_current',    # proposals P_1..P_j (no future)
    'all_except_current',  # all except P_j
    'neutral',             # no informative P@E plan
)


def plan_mask(scope, k):
    """Boolean [query horizon, PLAN position] selection matrix."""
    if scope not in SCOPES:
        raise ValueError(f'unknown horizon-plan control: {scope}')
    j = np.arange(int(k))[:, None]
    i = np.arange(int(k))[None, :]
    if scope == 'all':
        return np.ones((k, k), dtype=bool)
    if scope == 'neutral':
        return np.zeros((k, k), dtype=bool)
    if scope == 'own_only':
        return i == j
    if scope == 'past_and_current':
        return i <= j
    return i != j


def _token_nll(logits, target, valid):
    """Numerically stable individual CE values for one selected decoder row."""
    safe = xp.where(valid, target, 0).astype(xp.int64)
    x = logits.astype(xp.float64 if logits.dtype == xp.float64 else xp.float32,
                      copy=False)
    m = xp.max(x, axis=-1)
    correct = xp.take_along_axis(x, safe[..., None], axis=-1)[..., 0]
    nll = m + xp.log(xp.sum(xp.exp(x-m[..., None]), axis=-1)) - correct
    return xp.where(valid, nll, 0.)


def score_window(trainer, source, targets, valid, metadata, *, group_size=2):
    """Score all controls on one original mixed-source validation window.

    No weights or gradients are mutated. Crucially the last query dimension of
    each replicated decoder row is scored *only* at its own horizon j, while
    all genuine earlier teacher-forced tokens remain visible to that decoder.
    """
    net = trainer.net
    model = trainer.model
    k = int(net.proposal_head.block_size)
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError('group_size must be positive')
    if source.ndim != 2 or targets.ndim != 2 or valid.shape != targets.shape:
        raise ValueError('expected aligned source/target/mask batches')
    b, t = targets.shape
    if source.shape[0] != b:
        raise ValueError('source batch differs from target batch')
    if model.hierarchical_memory_enabled:
        cfg = model.config.memory_context
        if cfg.integration_mode != 'routed_prefix' or cfg.memory_training != 'joint':
            raise ValueError('requires existing causal routed-prefix model')
        if source.shape[1] != cfg.source_input_length or t != cfg.target_length:
            raise ValueError('source/target length mismatch for routed prefix')
        if metadata is None or 'history_valid_starts' not in metadata:
            raise ValueError('source-specific valid history starts required')
        if metadata.get('causal_prefix_query', True) is not True:
            raise ValueError('noncausal top-level routing not allowed')
        m = dict(metadata)
        m['causal_prefix_query'] = True
        m['causal_masked_target_refinement'] = True
        if 'target_valid_lengths' not in m:
            raise ValueError('missing genuine target_valid_lengths')
        if not bool(xp.all(xp.asarray(m['target_valid_lengths']) == xp.sum(valid, axis=1))):
            raise ValueError('target length/valid mask mismatch')
    else:
        if source.shape[1] != t or metadata is not None:
            raise ValueError('direct-context source/target length mismatch')
        m = None
    schedule = block_schedule(targets, valid, k, vocab_size=model.config.vocab_size)
    main = model.forward_body(source, memory_metadata=m, return_cache=False)
    prefix = int(main.shape[1]) - t
    if prefix < 0 or (not model.hierarchical_memory_enabled and prefix != 0):
        raise RuntimeError('unexpected main causal window length')
    positions = schedule.boundary_working_index + prefix
    rows = schedule.source_row
    counts = np.zeros(k, dtype=np.int64)
    prop = np.zeros(k, dtype=np.float64)
    final = {scope: np.zeros(k, dtype=np.float64) for scope in SCOPES}
    neutral = xp.mean(model.embedding.W.data.astype(xp.float32), axis=0)
    masks = {scope:xp.asarray(plan_mask(scope,k))[None,:,:]
             for scope in SCOPES}
    replicas = xp.arange(k, dtype=xp.int64)
    for start in range(0, schedule.targets.shape[0], group_size):
        end = min(start + group_size, schedule.targets.shape[0])
        h = main[rows[start:end], positions[start:end], :].copy()
        y = schedule.targets[start:end]
        v = schedule.valid[start:end]
        n = int(y.shape[0])
        logits = net.proposal_head.forward(h, return_cache=False)
        p_nll = _token_nll(logits, y, v)
        prop += np.asarray(asnumpy(xp.sum(p_nll, axis=0)), dtype=np.float64)
        counts += np.asarray(asnumpy(xp.sum(v,axis=0)), dtype=np.int64)
        original, _ = net.decoder.expectation.forward(logits)
        # K copies per block, one for each query horizon. Bound this to one
        # group at a time: the expensive main causal forward is never repeated.
        original = xp.repeat(original, k, axis=0)
        neutral_plan = xp.broadcast_to(neutral.astype(original.dtype), original.shape)
        rep_logits = xp.repeat(logits, k, axis=0)
        rep_y = xp.repeat(y, k, axis=0)
        rep_v = xp.repeat(v, k, axis=0)
        selected = xp.tile(replicas, n)
        linear_row = xp.arange(n*k, dtype=xp.int64)
        selected_valid = rep_v[linear_row, selected]
        selected_labels = rep_y[linear_row, selected]
        for scope in SCOPES:
            allow = xp.broadcast_to(masks[scope], (n,k,k)).reshape(n*k,k)
            plan = xp.where(allow[:,:,None], original, neutral_plan)
            # Only the plan embeddings are changed. The residual prior logits
            # and every real committed predecessor are unchanged.
            corrected, _ = net.decoder.forward(
                rep_logits, rep_y, rep_v,
                plan_condition='external', prefix_condition='normal',
                plan_override=plan)
            j_logits = corrected[linear_row, selected, :]
            loss = _token_nll(j_logits, selected_labels, selected_valid)
            horizon_sums = xp.sum(loss.reshape(n,k),axis=0)
            final[scope] += np.asarray(asnumpy(horizon_sums), dtype=np.float64)
    if not bool(np.all(counts > 0)):
        raise RuntimeError('missing one or more valid horizons')
    if int(sum(counts)) != schedule.unique_supervised_tokens:
        raise RuntimeError('token counts changed in ablation')
    return dict(counts=counts, proposal_sums=prop, final_sums=final,
                main_forward_passes=1)
