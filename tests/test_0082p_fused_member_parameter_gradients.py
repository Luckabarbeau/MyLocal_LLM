"""0082P: fused GPU gradient reduction, with independent NumPy reference."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner


def _reference(dpre, ratio, identity, context, entropy, wid, wctx):
    return (
        np.sum(dpre * ratio[..., None], axis=(0, 1, 2)),
        np.sum(dpre * identity[None, ..., None], axis=(0, 1, 2)),
        np.sum(dpre * context[..., None], axis=(0, 1, 2)),
        np.sum(dpre * entropy[:, :, None, None], axis=(0, 1, 2)),
        np.sum(dpre, axis=(0, 1, 2)),
        np.sum(dpre * wid, axis=-1),
        np.sum(dpre * wctx, axis=-1),
    )


def test_numpy_reference_gradient_shapes():
    rng = np.random.default_rng(92)
    dpre = rng.normal(size=(3, 2, 17, 8)).astype(np.float32)
    ratio = rng.normal(size=(3, 2, 17)).astype(np.float32)
    ident = rng.normal(size=(2, 17)).astype(np.float32)
    context = rng.normal(size=(3, 2, 17)).astype(np.float32)
    entropy = rng.normal(size=(3, 2)).astype(np.float32)
    w = rng.normal(size=(8,)).astype(np.float32)
    got = _reference(dpre, ratio, ident, context, entropy, w, w)
    assert all(x.shape == (8,) for x in got[:5])
    assert got[5].shape == (3, 2, 17)
    assert got[6].shape == (3, 2, 17)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CUDA CuPy')
def test_fused_member_gradients_match_numpy_reference():
    from mini_llm.collective.fusion_cuda_kernels import member_parameter_grads

    rng = np.random.default_rng(93)
    # Test incomplete tiles, multiple tiles, variable model counts, and widths.
    for rows, members, vocab, hidden in (
        (1, 1, 11, 3), (3, 2, 17, 8), (4, 3, 129, 12),
        (2, 4, 67, 16), (2, 2, 131, 64), (2, 5, 599, 8),
    ):
        dpre = rng.normal(0., 0.1, (rows, members, vocab, hidden)).astype('float32')
        ratio = rng.normal(size=(rows, members, vocab)).astype('float32')
        identity = rng.normal(size=(members, vocab)).astype('float32')
        context = rng.normal(size=(rows, members, vocab)).astype('float32')
        entropy = rng.normal(size=(rows, members)).astype('float32')
        wid = rng.normal(size=hidden).astype('float32')
        wctx = rng.normal(size=hidden).astype('float32')
        args = (dpre, ratio, identity, context, entropy, wid, wctx)
        expected = _reference(*args)
        actual = member_parameter_grads(*[xp.asarray(x) for x in args])
        for n, (got, want) in enumerate(zip(actual, expected)):
            np.testing.assert_allclose(xp.asnumpy(got), want, rtol=5e-5,
                                       atol=3e-5, err_msg=f'gradient[{n}] shape={dpre.shape}')


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CUDA CuPy')
def test_full_cuda_combiner_gradients_match_unfused(monkeypatch):
    import mini_llm.collective.probability_fusion as fusion_mod

    rng = np.random.default_rng(117)
    for members, depth in ((1, 1), (2, 1), (3, 2)):
        cfg = FusionConfig(vocab_size=41, max_models=4, decoder_layers=depth)
        faster = ProbabilityResidualCombiner(cfg)
        base = ProbabilityResidualCombiner(cfg)
        # Nonzero output head exercises all paths before and after reductions.
        faster.output_weight.data[...] = xp.asarray(
            rng.normal(0, 0.12, faster.output_weight.data.shape).astype('float32'))
        base.load_state_dict(faster.state_dict())
        logits = xp.asarray(rng.normal(size=(3, members, 41)).astype('float32'))
        y = rng.integers(0, 41, size=3)
        ids = np.arange(members)
        faster.zero_grad()
        base.zero_grad()
        fast_loss = faster.loss_and_backward(logits, ids, y, return_weighted=True)
        with monkeypatch.context() as m:
            m.setattr(fusion_mod, 'BACKEND_NAME', 'numpy')
            slow_loss = base.loss_and_backward(logits, ids, y, return_weighted=True)
        np.testing.assert_allclose(fast_loss, slow_loss, rtol=2e-5, atol=2e-5)
        for name in faster.parameters_by_name:
            np.testing.assert_allclose(xp.asnumpy(faster.parameters_by_name[name].grad),
                                       xp.asnumpy(base.parameters_by_name[name].grad),
                                       rtol=6e-4, atol=7e-5, err_msg=name)
