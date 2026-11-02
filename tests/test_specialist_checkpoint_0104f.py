"""Specialist-only roundtrip, foundation fingerprint, stale checkpoint checks."""
from dataclasses import asdict
import json
import numpy as np
import pytest

from mini_llm.checkpoint import save_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.specialists import (
    FrozenFoundation, ResidualSpecialist, SpecialistConfig,
    FrozenResidualComposition, fingerprint_foundation,
    save_specialist_checkpoint, load_specialist_checkpoint,
)


def prepared(tmp_path):
    model_config = ModelConfig(tokenizer_vocab_size=64, context_length=8,
        n_layers=1, d_model=16, n_q_heads=2, n_kv_heads=1, d_head=8,
        d_ff=32, n_experts=1, top_k=1, ffn_mode='dense', dtype='float32')
    model = DecoderLanguageModel(model_config, rng_seed=2)
    base_path = tmp_path / 'foundation'
    base_path.mkdir()
    (base_path / 'config.json').write_text(json.dumps(asdict(model_config)))
    save_checkpoint(base_path, {p.name: p.data for p in model.parameters()})
    foundation = FrozenFoundation.from_checkpoint(base_path)
    spec = SpecialistConfig(specialist_id='general', n_layers=2, d_model=16,
        d_ff=32, n_q_heads=2, n_kv_heads=1, d_head=8,
        full_window=8, local_window=4, dtype='float32')
    specialist = ResidualSpecialist(spec, rng_seed=5)
    return foundation, FrozenResidualComposition(foundation, specialist)


def test_residual_checkpoint_only_saves_specialist_and_verifies_foundation(tmp_path):
    foundation, composed = prepared(tmp_path)
    initial = fingerprint_foundation(foundation.checkpoint_path)
    path = tmp_path / 'specialist'
    first = composed.parameters()[0]
    first.data[...] = 0.4
    save_specialist_checkpoint(path, composed,
        optimizer_state={'step': 18, 'lr': .0002},
        training_state={'supervised_tokens': 12345})
    loaded, optimizer, state = load_specialist_checkpoint(path, foundation)
    assert optimizer['step'] == 18 and optimizer['lr'] == pytest.approx(.0002)
    assert state['supervised_tokens'] == 12345
    assert state['specialist']['foundation_sha256'] == initial
    np.testing.assert_array_equal(loaded.parameters()[0].data, first.data)
    _, _, _ = load_specialist_checkpoint(path, foundation, load_optimizer=False)
    assert {p.name for p in loaded.parameters()} == {
        p.name for p in composed.parameters()}
    assert all(p.grad is None for p in foundation.model.parameters())
    (foundation.checkpoint_path / 'config.json').write_text(
        (foundation.checkpoint_path / 'config.json').read_text() + ' ')
    with pytest.raises(ValueError, match='digest mismatch'):
        load_specialist_checkpoint(path, foundation)


def test_specialist_checkpoint_requires_checkpoint_backed_foundation(tmp_path):
    foundation, composed = prepared(tmp_path)
    bare = FrozenResidualComposition(FrozenFoundation(foundation.model), composed.specialist)
    with pytest.raises(ValueError, match='checkpoint-backed'):
        save_specialist_checkpoint(tmp_path / 'invalid', bare)
