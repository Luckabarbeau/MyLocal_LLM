"""0080O11: strictly trailing padded working rows may be skipped, not samples."""

import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.block_proposal.training import compact_working_length
from tests.test_block_proposal_0080d import _build, _sample


@pytest.mark.parametrize("valid,bucket", [(1, 512), (511, 512), (512, 512),
                                         (513, 1024), (1500, 2048), (3000, 4096),
                                         (4096, 4096)])
def test_default_4096_buckets(valid, bucket):
    assert compact_working_length(4096, valid, 4) == bucket


def test_small_model_buckets_and_invalid_length():
    assert compact_working_length(16, 3, 4) == 4
    assert compact_working_length(16, 5, 4) == 8
    assert compact_working_length(16, 9, 4) == 16
    with pytest.raises(ValueError, match="outside"):
        compact_working_length(16, 0, 4)
    assert compact_working_length(15, 5, 4) == 8
    assert compact_working_length(15, 14, 4) == 15
    with pytest.raises(ValueError, match="positive"):
        compact_working_length(0, 1, 4)


def sample_short(lengths=(7, 3), *, use_retrieval=True):
    _, _, _, md = _sample()
    source = xp.asarray(np.mod(np.arange(2 * 48).reshape(2, 48) * 7 + 3, 67))
    labels = (source[:, 32:] + 1) % 67
    valid = xp.ones((2, 16), dtype=xp.bool_)
    for row, length in enumerate(lengths):
        valid[row, length:] = False
        labels[row, length:] = -100
        source[row, 32 + length:] = 0
    md = dict(md, target_valid_lengths=np.asarray(lengths, dtype=np.int64),
              use_retrieval=use_retrieval, causal_masked_target_refinement=True)
    return source, labels, valid, md


def run(model, sample, *, monkeypatch, enabled, record_length=False):
    source, labels, valid, md = sample
    monkeypatch.setenv('MINI_LLM_BLOCK_PROPOSAL_COMPACT_TARGET', str(int(enabled)))
    observed = []
    original = model.model.forward_body
    if record_length:
        def forward(*args, **kwargs):
            output = original(*args, **kwargs)
            head = output[0] if isinstance(output, tuple) else output
            observed.append(int(head.shape[1]))
            return output
        model.model.forward_body = forward
    try:
        model.zero_grad()
        metrics = model.train_window(source, labels, valid_target_mask=valid,
                                     memory_metadata=md, block_group_size=2)
        grads = {p.name: asnumpy(p.grad).copy() for p in model.parameters()}
    finally:
        model.model.forward_body = original
    return metrics, grads, observed


@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.parametrize("lengths", [(7, 3), (4, 3), (12, 9), (16, 5)])
def test_same_logits_loss_and_full_gradients(monkeypatch, learned, lengths):
    model = _build(learned=learned)
    sample = sample_short(lengths)
    first, ref_grad, _ = run(model, sample, monkeypatch=monkeypatch, enabled=False)
    second, new_grad, observed = run(model, sample, monkeypatch=monkeypatch,
                                     enabled=True, record_length=True)
    expected = compact_working_length(16, max(lengths), 4)
    assert observed == [int(first['routed_prefix_slots']) + expected]
    assert second['trunk_working_tokens_per_row'] == expected
    assert second['trunk_padding_tokens_saved_per_row'] == 16 - expected
    assert first['unique_supervised_tokens'] == second['unique_supervised_tokens'] == sum(lengths)
    assert first['active_blocks'] == second['active_blocks']
    for key in ['proposal_ce', 'final_ce', 'total_loss']:
        np.testing.assert_allclose(asnumpy(first[key]), asnumpy(second[key]),
                                   rtol=1e-8, atol=1e-9, err_msg=key)
    for key in ['position_counts', 'proposal_position_ce_sums',
                'final_position_ce_sums', 'decoder_position_gain', 'row_final_ce_sums']:
        np.testing.assert_allclose(asnumpy(first[key]), asnumpy(second[key]),
                                   rtol=1e-8, atol=1e-9, err_msg=key)
    assert ref_grad.keys() == new_grad.keys()
    for name in ref_grad:
        np.testing.assert_allclose(ref_grad[name], new_grad[name],
                                   rtol=1e-5, atol=1e-7, err_msg=name)


def test_padding_values_and_original_source_unchanged(monkeypatch):
    model = _build(learned=True)
    src, labels, mask, md = sample_short((7, 2))
    original = asnumpy(src).copy()
    a, ga, _ = run(model, (src, labels, mask, md), monkeypatch=monkeypatch, enabled=True)
    changed = src.copy()
    changed[:, 32+7:] = 47
    changed[1, 32+2:] = 45
    different_labels = labels.copy()
    different_labels[~mask] = 51
    b, gb, _ = run(model, (changed, different_labels, mask, md),
                   monkeypatch=monkeypatch, enabled=True)
    np.testing.assert_array_equal(asnumpy(src), original)
    for key in ('proposal_ce', 'final_ce', 'total_loss'):
        np.testing.assert_allclose(asnumpy(a[key]), asnumpy(b[key]), atol=1e-10)
    for name in ga:
        np.testing.assert_allclose(ga[name], gb[name], rtol=1e-6, atol=1e-8, err_msg=name)


def test_reject_unsanctioned_compact_override():
    model = _build()
    src, _, _, md = sample_short((7, 3))
    md['compact_working_length'] = 8
    md['causal_masked_target_refinement'] = False
    with pytest.raises(ValueError, match='requires causal block refinement'):
        model.model.forward_body(src, memory_metadata=md, return_cache=False)


def test_noncontiguous_original_validity_is_not_hidden_by_compaction(monkeypatch):
    model = _build()
    src, labels, mask, md = sample_short((3, 2))
    # A later True after a hole must not disappear merely because the
    # population count fits a small compact bucket.
    mask[0, 10] = True
    labels[0, 10] = 3
    md['target_valid_lengths'] = np.array([4, 2])
    monkeypatch.setenv('MINI_LLM_BLOCK_PROPOSAL_COMPACT_TARGET', '1')
    with pytest.raises(ValueError, match='prefix-valid'):
        model.train_window(src, labels, valid_target_mask=mask,
                           memory_metadata=md)


def test_numpy_target_length_metadata_is_not_sent_to_gpu_converter(monkeypatch):
    """GPU backend's asnumpy expects .get(), but sampler metadata is NumPy."""
    import mini_llm.block_proposal.training as training_module

    model = _build()
    src, labels, valid, md = sample_short((7, 3))
    cpu_lengths = md['target_valid_lengths']
    assert isinstance(cpu_lengths, np.ndarray)
    original_convert = training_module.asnumpy

    def reject_numpy_metadata(value):
        if value is cpu_lengths:
            raise AssertionError('CPU target_valid_lengths passed to GPU-only asnumpy')
        return original_convert(value)

    monkeypatch.setattr(training_module, 'asnumpy', reject_numpy_metadata)
    monkeypatch.setenv('MINI_LLM_BLOCK_PROPOSAL_COMPACT_TARGET', '1')
    metrics = model.train_window(src, labels, valid_target_mask=valid,
                                 memory_metadata=md, block_group_size=2)
    assert metrics['trunk_working_tokens_per_row'] == 8
    assert metrics['unique_supervised_tokens'] == 10
