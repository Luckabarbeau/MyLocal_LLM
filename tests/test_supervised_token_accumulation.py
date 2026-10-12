"""0077K: verify valid-token weighting through complete training backward."""

import numpy as np
import pytest

from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train_extended import ExtendedTrainer


def _run_gradients(weighting, batches):
    model = DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=101)
    trainer = ExtendedTrainer(
        model=model, train_shard_paths=['unused.bin'], val_shard_paths=['unused.bin'],
        batch_size=1, seq_length=8, grad_accum_steps=len(batches),
        gradient_accumulation_weighting=weighting,
        total_steps=100, warmup_steps=5, grad_clip=1e10,
    )
    seq = iter(batches)

    def fake_batch(return_metadata=False):
        x, y, mask = next(seq)
        trainer._current_mask = mask.copy()
        data = (x.copy(), y.copy(), None)
        return data if return_metadata else data[:2]

    trainer.get_train_batch = fake_batch
    trainer._effective_loss_mask = lambda metadata, unused: trainer._current_mask
    collected = {}

    def capture(*, lr=None):
        for p in trainer.optimization_parameters:
            if p.grad is not None:
                collected[p.name] = np.array(p.grad, copy=True)

    trainer.optimizer.step = capture
    loss, _ = trainer.train_step()
    return collected, loss, trainer


def _inputs_and_targets():
    x1 = np.array([[3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int32)
    y1 = np.array([[4, 5, 6, 7, 8, 9, 10, 11]], dtype=np.int32)
    x2 = np.array([[12, 13, 14, 15, 16, 17, 18, 19]], dtype=np.int32)
    y2 = np.array([[13, 14, 15, 16, 17, 18, 19, 20]], dtype=np.int32)
    return [
        (x1, y1, np.array([[1, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)),
        (x2, y2, np.ones((1, 8), dtype=np.float32)),
    ]


@pytest.mark.parametrize("chunk_tokens", [0, 4])
def test_microbatch_and_token_weighting_match_respective_reference_objectives(chunk_tokens, monkeypatch):
    monkeypatch.setenv("MINI_LLM_LM_HEAD_CHUNK_TOKENS", str(chunk_tokens))
    batches = _inputs_and_targets()
    g1, l1, _ = _run_gradients('supervised_tokens', batches[:1])
    g2, l2, _ = _run_gradients('supervised_tokens', batches[1:])
    weighted, loss_weighted, trainer = _run_gradients('supervised_tokens', batches)
    equal, loss_equal, _ = _run_gradients('microbatch', batches)

    assert trainer._last_step_supervised_tokens == 10
    assert trainer._last_step_supervised_utilization == pytest.approx(10 / 16)
    assert loss_weighted == pytest.approx((2 * l1 + 8 * l2) / 10, abs=1e-5)
    assert loss_equal == pytest.approx((l1 + l2) / 2, abs=1e-5)

    for name in g1:
        np.testing.assert_allclose(
            weighted[name], (2 * g1[name] + 8 * g2[name]) / 10,
            rtol=2e-5, atol=2e-6,
        )
        np.testing.assert_allclose(
            equal[name], (g1[name] + g2[name]) / 2,
            rtol=2e-5, atol=2e-6,
        )


def test_invalid_token_weighting_rejected():
    with pytest.raises(ValueError, match='gradient_accumulation_weighting'):
        ExtendedTrainer(
            model=DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=101),
            train_shard_paths=['unused.bin'], val_shard_paths=['unused.bin'],
            batch_size=1, seq_length=8, gradient_accumulation_weighting='other',
        )
