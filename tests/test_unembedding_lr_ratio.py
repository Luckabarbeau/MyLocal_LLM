"""Targeted tests for output-specific AdamW learning-rate scaling."""

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from mini_llm.backend import xp
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.optim.hybrid_muon import HybridMuonAdamW
from mini_llm.parameter import Parameter
from mini_llm.train_extended import ExtendedTrainer
import train_model


def _param(name, shape=(3, 4)):
    return Parameter(xp.ones(shape, dtype="float32"), name=name)


def _value(param):
    return np.asarray(param.data.get() if hasattr(param.data, "get") else param.data)


def _adam_params():
    return [_param("embedding.W"), _param("output_proj.W"), _param("blocks.0.norm1.gamma")]


def test_adam_multiplier_scales_only_output_update_including_decay():
    base, slow = _adam_params(), _adam_params()
    a = AdamW(base, lr=.02, weight_decay=.1)
    b = AdamW(slow, lr=.02, weight_decay=.1, unembedding_lr_ratio=.1)
    for _ in range(3):
        for p in base + slow:
            p.grad[...] = .25
        a.step(lr=.02)
        b.step(lr=.02)
    np.testing.assert_array_equal(_value(base[0]), _value(slow[0]))
    np.testing.assert_array_equal(_value(base[2]), _value(slow[2]))
    assert np.abs(_value(slow[1]) - 1).max() < np.abs(_value(base[1]) - 1).max()
    # Separately verify the 10x multiplier also affects decoupled decay.
    x = _param("output_proj.W")
    x.grad[...] = 0
    wd = AdamW([x], lr=.02, weight_decay=.1, unembedding_lr_ratio=.1)
    wd.step()
    np.testing.assert_allclose(_value(x), 1 - .02 * .1 * .1, rtol=1e-6)


def test_ratio_matches_reference_optimizer_with_lower_lr_and_same_gradients():
    # Compare after several varying scheduled learning rates: the head's
    # parameter update must be identical to an ordinary AdamW head whose
    # scheduled LR is 10x lower, without modifying Adam moments.
    scaled_p = _param("output_proj.W")
    reference_p = _param("output_proj.W")
    scaled = AdamW([scaled_p], lr=.01, weight_decay=.1, unembedding_lr_ratio=.1)
    reference = AdamW([reference_p], lr=.001, weight_decay=.1)
    for step, lr in enumerate([.01, .002, .007], start=1):
        grad = .3 * step
        scaled_p.grad[...] = grad
        reference_p.grad[...] = grad
        scaled.step(lr=lr)
        reference.step(lr=lr * .1)
        np.testing.assert_allclose(_value(scaled_p), _value(reference_p), rtol=1e-6)
        np.testing.assert_array_equal(scaled.m[0], reference.m[0])
        np.testing.assert_array_equal(scaled.v[0], reference.v[0])


def test_ratio_zero_freezes_weights_moments_and_adam_age():
    embedding, head = _param("embedding.W"), _param("output_proj.W")
    opt = AdamW([embedding, head], lr=.05, unembedding_lr_ratio=0.0)
    before = _value(head).copy()
    for _ in range(4):
        embedding.grad[...] = .3
        head.grad[...] = .3
        opt.step()
    np.testing.assert_array_equal(before, _value(head))
    np.testing.assert_array_equal(opt.m[1], np.zeros_like(opt.m[1]))
    np.testing.assert_array_equal(opt.v[1], np.zeros_like(opt.v[1]))
    assert opt.parameter_birth_steps[1] == 4
    assert np.abs(_value(embedding) - 1).max() > .01
    # Changing the setting on resume must resume the head at local Adam step 1.
    opt.unembedding_lr_ratio = .1
    opt.step()
    assert opt.step_index - opt.parameter_birth_steps[1] == 1
    assert np.abs(_value(head) - 1).max() > 0


def test_hybrid_routes_only_head_to_reduced_adam_lr():
    def build():
        return [_param("embedding.W"), _param("output_proj.W"),
                _param("blocks.0.attention.Wq", (3, 3))]
    p1, p2 = build(), build()
    kwargs = dict(lr=.01, beta1=.9, beta2=.95, eps=1e-8,
                  weight_decay=.1, muon_matmul_precision="fp32")
    a = HybridMuonAdamW(p1, **kwargs)
    b = HybridMuonAdamW(p2, unembedding_lr_ratio=.1, **kwargs)
    for p in p1 + p2:
        p.grad[...] = .2
    a.step(lr=.01)
    b.step(lr=.01)
    np.testing.assert_array_equal(_value(p1[0]), _value(p2[0]))
    np.testing.assert_array_equal(_value(p1[2]), _value(p2[2]))
    assert np.abs(_value(p2[1]) - 1).max() < np.abs(_value(p1[1]) - 1).max()
    assert b.adam.unembedding_lr_ratio == .1


@pytest.mark.parametrize("ratio", [-.1, float("nan"), float("inf")])
def test_invalid_ratios_rejected(ratio):
    with pytest.raises(ValueError, match="ratio"):
        AdamW([_param("output_proj.W")], unembedding_lr_ratio=ratio)


def test_trainer_tied_embedding_warns_and_uses_shared_weight(tmp_path, capsys):
    config = dataclasses.replace(ModelConfig.tiny_inspection(), tie_embeddings=True)
    model = DecoderLanguageModel(config, rng_seed=42, dtype="float32")
    shard = tmp_path / "train.bin"
    np.zeros(512, dtype=np.uint16).tofile(shard)
    trainer = ExtendedTrainer(model, [str(shard)], [str(shard)],
                              batch_size=1, seq_length=8,
                              unembedding_lr_ratio=.1)
    assert trainer.unembedding_parameter_name == "embedding.W"
    assert trainer.optimizer.unembedding_parameter_name == "embedding.W"
    assert trainer.optimizer.unembedding_lr_ratio == .1
    assert "also scales input" in capsys.readouterr().out
    assert "output_proj.W" not in {p.name for p in trainer.optimizer.parameters}


def test_trainer_untied_output_and_checkpoint_metadata(tmp_path):
    model = DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=42, dtype="float32")
    shard = tmp_path / "train.bin"
    np.zeros(512, dtype=np.uint16).tofile(shard)
    dest = tmp_path / "ckpt"
    trainer = ExtendedTrainer(model, [str(shard)], [str(shard)],
                              batch_size=1, seq_length=8, checkpoint_dir=str(dest),
                              unembedding_lr_ratio=.2)
    assert trainer.unembedding_parameter_name == "output_proj.W"
    assert trainer.optimizer.unembedding_lr_ratio == .2
    trainer.save()
    with (dest / "training_state.json").open() as fp:
        saved = json.load(fp)
    assert saved["unembedding_lr_ratio"] == .2


def test_cli_accepts_ratio(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_model.py", "--unembedding-lr-ratio", "0.1"])
    assert train_model.parse_args().unembedding_lr_ratio == .1
    monkeypatch.setattr(sys, "argv", ["train_model.py"])
    assert train_model.parse_args().unembedding_lr_ratio is None
