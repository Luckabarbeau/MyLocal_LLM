"""0080O1: numerical CPU guards for delayed device losses / compact scheduling.

Real CuPy/BF16 and true GPU memory measurement remain separate hardware tests.
"""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp, asnumpy
from mini_llm.block_proposal.contracts import block_schedule
from mini_llm.ops.loss import cross_entropy_forward, cross_entropy_backward
from test_block_proposal_0080l_baselines import _make, _window

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='NumPy reference regression')


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('masked', [False, True])
def test_preverified_ce_normalizer_and_delayed_loss_match_default(dtype, masked):
    rng = np.random.default_rng(8401)
    logits = rng.normal(size=(2, 4, 17)).astype(dtype)
    targets = rng.integers(0, 17, size=(2, 4))
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=np.bool_) if masked else None
    reference, cache = cross_entropy_forward(logits.copy(), targets, loss_mask=mask)
    grad_reference = cross_entropy_backward(cache)
    n = int(mask.sum()) if mask is not None else targets.size
    delayed, cache = cross_entropy_forward(logits.copy(), targets, loss_mask=mask,
                                           return_device_loss=True,
                                           normalizer_count=n)
    grad_delayed = cross_entropy_backward(cache)
    assert isinstance(delayed, (np.ndarray, np.generic))
    np.testing.assert_allclose(delayed, reference, atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(grad_delayed, grad_reference, atol=1e-8, rtol=1e-7)


def test_known_ce_denominator_rejects_nonpositive():
    with pytest.raises(ValueError, match='positive'):
        cross_entropy_forward(np.ones((2, 4)), np.zeros(2, dtype=np.int64),
                              normalizer_count=0)


@pytest.mark.parametrize('k', [1, 4, 8])
def test_vectorized_schedule_preserves_document_major_boundaries(k):
    labels = np.arange(3 * 11, dtype=np.int64).reshape(3, 11)
    lengths = [0, 6, 11]
    mask = np.zeros_like(labels, dtype=np.bool_)
    for row, length in enumerate(lengths):
        mask[row, :length] = True
    labels[~mask] = -100
    schedule = block_schedule(labels, mask, k, vocab_size=100)
    expected = [(row, start) for row, length in enumerate(lengths)
                for start in range(0, length, k)]
    assert [(int(r), int(t)) for r, t in zip(schedule.source_row, schedule.boundary_working_index)] == expected
    assert schedule.unique_supervised_tokens == sum(lengths)
    assert int(schedule.valid.sum()) == sum(lengths)
    for (row, start), target_block, valid_block in zip(expected, schedule.targets, schedule.valid):
        for offset in range(k):
            valid = start + offset < lengths[row]
            assert bool(valid_block[offset]) == valid
            assert int(target_block[offset]) == (int(labels[row, start + offset]) if valid else 0)


@pytest.mark.parametrize('k,group_size', [(1, 1), (8, 1), (8, 2), (8, 4)])
def test_deferred_group_losses_gradients_and_horizon_counts_equal_legacy(k, group_size):
    source, labels, mask, meta = _window()
    legacy, deferred = _make(k), _make(k)
    old = legacy.train_window(source, labels, valid_target_mask=mask,
                              memory_metadata=meta, block_group_size=group_size,
                              defer_metrics=False)
    new = deferred.train_window(source, labels, valid_target_mask=mask,
                                memory_metadata=meta, block_group_size=group_size,
                                defer_metrics=True)
    assert old['unique_supervised_tokens'] == new['unique_supervised_tokens'] == 13
    assert old['main_forward_passes'] == new['main_forward_passes'] == 1
    assert old['main_backward_passes'] == new['main_backward_passes'] == 1
    for key in ['proposal_ce', 'final_ce', 'total_loss', 'position_counts',
                'proposal_position_ce_sums', 'final_position_ce_sums', 'row_final_ce_sums']:
        np.testing.assert_allclose(asnumpy(new[key]), asnumpy(old[key]), rtol=1e-12, atol=1e-12)
    for a, b in zip(legacy.parameters(), deferred.parameters()):
        assert a.name == b.name
        np.testing.assert_allclose(asnumpy(a.grad), asnumpy(b.grad), rtol=1e-12,
                                   atol=1e-12, err_msg=a.name)


def test_legacy_public_joint_forward_still_returns_python_float():
    net = _make(8).net
    x = np.zeros((1, 16), dtype=np.float64)
    labels = np.arange(8, dtype=np.int64)[None, :]
    valid = np.ones_like(labels, dtype=np.bool_)
    metrics, _ = net.forward(x, labels, valid)
    assert type(metrics['proposal_ce']) is float
    assert type(metrics['final_ce']) is float
    assert type(metrics['total_loss']) is float
