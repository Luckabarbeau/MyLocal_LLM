"""0079E model/Adam checkpoint exact resume with strict architecture metadata."""

import tempfile
import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.predictive_refinement.training_reference import (
    DenseRefinementTrainingReference, RefinementReferenceConfig,
)
from mini_llm.predictive_refinement.checkpoint import (
    save_refinement_checkpoint, load_refinement_checkpoint,
)


def make_ref(k=2):
    model = DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=5, dtype="float64")
    return DenseRefinementTrainingReference(model, RefinementReferenceConfig(block_size=k), rng_seed=6)


def train_step(ref, opt):
    context = xp.asarray([[1, 2, 3]], dtype="int64")
    targets = xp.asarray([[4, 5]], dtype="int64")
    ref.zero_grad()
    metrics, tape = ref.forward(context, targets)
    ref.backward(tape)
    opt.step()
    return float(metrics["total_loss"])


def test_checkpoint_round_trip_restores_model_and_adam_trajectory():
    reference = make_ref()
    optimizer = AdamW(reference.parameters(), lr=0.005, weight_decay=0.0)
    for _ in range(3):
        train_step(reference, optimizer)
    with tempfile.TemporaryDirectory() as tmp:
        save_refinement_checkpoint(
            tmp, reference, optimizer=optimizer, training_step=3,
            supervised_tokens=6, data_provenance={"source": "synthetic"},
        )
        restored = make_ref()
        restored_opt = AdamW(restored.parameters(), lr=0.005, weight_decay=0.0)
        state = load_refinement_checkpoint(tmp, restored, optimizer=restored_opt)
        assert state["step"] == 3
        assert state["supervised_tokens"] == 6
        assert state["architecture"]["type"] == "block_causal_predictive_refinement_v1"
        for a, b in zip(reference.parameters(), restored.parameters()):
            np.testing.assert_array_equal(asnumpy(a.data), asnumpy(b.data))
        np.testing.assert_allclose(train_step(reference, optimizer), train_step(restored, restored_opt), atol=1e-10)
        for a, b in zip(reference.parameters(), restored.parameters()):
            np.testing.assert_allclose(asnumpy(a.data), asnumpy(b.data), atol=1e-10)


def test_rejects_wrong_block_size_and_missing_or_ar_metadata():
    ref = make_ref()
    with tempfile.TemporaryDirectory() as tmp:
        save_refinement_checkpoint(tmp, ref)
        with pytest.raises(ValueError, match="block_size"):
            load_refinement_checkpoint(tmp, make_ref(k=3))
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="missing"):
            load_refinement_checkpoint(tmp, ref)
