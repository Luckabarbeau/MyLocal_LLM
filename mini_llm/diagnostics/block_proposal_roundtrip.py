"""0080AD: bounded, non-mutating checkpoint output fingerprints for K-block models.

A report contains full K-horizon CE and a single block's actual intermediate
values, not just optimizer or parameter names.  The logits capture is bounded
by K * vocabulary size, independent of the 4K/64K sequence lengths.
"""

import numpy as np

from mini_llm.backend import asnumpy, synchronize
from mini_llm.block_proposal.objective import BlockProposalObjective


CAPTURE_KEYS = (
    'main_boundary_hidden', 'proposal_logits', 'soft_plan_embeddings',
    'decoder_correction_logits', 'final_logits',
)


def decoder_pack_integrity(net):
    """Refuse a supposedly checkpoint-ready decoder with stale QKV buffers."""
    results = []
    for index, layer in enumerate(net.decoder.layers):
        attn = layer.attn
        if not getattr(attn, '_packed_qkv_valid', False):
            raise AssertionError(f'decoder layer {index}: packed QKV buffer invalid')
        actual = asnumpy(attn._packed_qkv_weight)
        canonical = np.concatenate([
            asnumpy(attn.Wq.data), asnumpy(attn.Wk.data),
            asnumpy(attn.Wv.data),
        ], axis=1)
        if actual.shape != canonical.shape or not np.array_equal(actual, canonical):
            difference = (np.max(np.abs(actual.astype(np.float32) - canonical.astype(np.float32)))
                          if actual.shape == canonical.shape else float('inf'))
            raise AssertionError(f'decoder layer {index}: stale packed QKV; max abs={difference}')
        results.append({'layer': index, 'shape': list(actual.shape),
                        'matches_canonical': True})
    return results


def fixed_batch_fingerprint(net, source, targets, valid, metadata, *, group_size):
    """Score a fixed batch and capture the *actual* first decoder block's tensors.

    The original methods are restored in all cases; this does not change any
    trainable weights, optimizer state, or model RNG. Caller must choose a
    deterministic routing configuration to make before/after comparable.
    """
    head = net.proposal_head
    decoder = net.decoder
    expectation = decoder.expectation
    originals = (head.forward, expectation.forward, decoder.forward)
    captured = {}

    def on_head(*args, **kwargs):
        result = originals[0](*args, **kwargs)
        if 'proposal_logits' not in captured:
            logits = result[0] if isinstance(result, tuple) else result
            captured['proposal_logits'] = np.asarray(asnumpy(logits[:1])).copy()
            captured['main_boundary_hidden'] = np.asarray(asnumpy(args[0][:1])).copy()
        return result

    def on_expectation(*args, **kwargs):
        result = originals[1](*args, **kwargs)
        if 'soft_plan_embeddings' not in captured:
            captured['soft_plan_embeddings'] = np.asarray(asnumpy(result[0][:1])).copy()
        return result

    def on_decoder(*args, **kwargs):
        result = originals[2](*args, **kwargs)
        if 'final_logits' not in captured:
            logits = np.asarray(asnumpy(result[0][:1])).copy()
            proposal = np.asarray(asnumpy(args[0][:1])).copy()
            captured['final_logits'] = logits
            captured['decoder_correction_logits'] = (
                logits.astype(np.float32) - proposal.astype(np.float32))
        return result

    try:
        head.forward = on_head
        expectation.forward = on_expectation
        decoder.forward = on_decoder
        metrics = BlockProposalObjective(net, block_group_size=group_size).evaluate_batch(
            source, targets, valid, metadata)
        synchronize()
    finally:
        head.forward, expectation.forward, decoder.forward = originals

    if set(captured) != set(CAPTURE_KEYS):
        raise AssertionError('missing captured decoder stages: ' + str(set(CAPTURE_KEYS)-set(captured)))
    counts = np.asarray(asnumpy(metrics['position_counts']), dtype=np.int64)
    prop_sums = np.asarray(asnumpy(metrics['proposal_position_ce_sums']), dtype=np.float64)
    final_sums = np.asarray(asnumpy(metrics['final_position_ce_sums']), dtype=np.float64)
    if np.any(counts == 0):
        raise AssertionError('fixed batch must have nonempty supervision at each horizon')
    return {
        'counts': counts, 'proposal_ce': prop_sums / counts,
        'final_ce': final_sums / counts,
        'decoder_gain': (prop_sums - final_sums) / counts,
        'stages': captured,
    }


def compare_fingerprints(before, after, *, ce_atol, logits_atol):
    """Raise when a checkpoint roundtrip changed any measured output."""
    if not np.array_equal(before['counts'], after['counts']):
        raise AssertionError('supervision counts changed across checkpoint reload')
    report = {'max_ce_error': 0., 'stage_max_abs_errors': {}}
    for key in ('proposal_ce', 'final_ce', 'decoder_gain'):
        error = float(np.max(np.abs(before[key] - after[key])))
        report['max_ce_error'] = max(report['max_ce_error'], error)
        if not np.isfinite(error) or error > ce_atol:
            raise AssertionError(f'checkpoint roundtrip {key} differs: {error:.8g} > {ce_atol}')
    for key in CAPTURE_KEYS:
        a, b = before['stages'][key], after['stages'][key]
        if a.shape != b.shape:
            raise AssertionError(f'checkpoint roundtrip {key} shape changed')
        error = float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
        report['stage_max_abs_errors'][key] = error
        if not np.isfinite(error) or error > logits_atol:
            raise AssertionError(f'checkpoint roundtrip {key} differs: {error:.8g} > {logits_atol}')
    return report
