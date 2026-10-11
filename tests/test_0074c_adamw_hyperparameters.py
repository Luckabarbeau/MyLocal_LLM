"""0074C: CLI-configurable AdamW moments, defaults, and resume metadata."""

import sys
from argparse import Namespace

import numpy as np
import pytest

import train_model
from mini_llm.backend import xp
from mini_llm.checkpoint import load_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.parameter import Parameter
from mini_llm.train_extended import ExtendedTrainer


def _args(beta1=None, beta2=None, eps=None):
    return Namespace(adam_beta1=beta1, adam_beta2=beta2, adam_eps=eps)


def test_cli_defaults_and_overrides(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_model.py"])
    defaults = train_model.parse_args()
    assert defaults.adam_beta1 is None
    assert defaults.adam_beta2 is None
    assert defaults.adam_eps is None
    assert train_model.resolve_adamw_hyperparameters(defaults) == {
        "beta1": 0.9, "beta2": 0.95, "eps": 1e-8,
    }
    monkeypatch.setattr(sys, "argv", [
        "train_model.py", "--adam-beta1", "0.974", "--adam-beta2", "0.9873",
        "--adam-eps", "1e-7",
    ])
    args = train_model.parse_args()
    assert train_model.resolve_adamw_hyperparameters(args) == {
        "beta1": 0.974, "beta2": 0.9873, "eps": 1e-7,
    }


@pytest.mark.parametrize("key,bad", [
    ("beta1", -0.01), ("beta1", 1.0), ("beta1", float("nan")),
    ("beta2", 1.0), ("beta2", float("inf")),
    ("eps", 0.0), ("eps", -1e-8), ("eps", float("nan")),
    ("eps", float("inf")),
])
def test_invalid_hparams_rejected_by_cli_and_optimizer(key, bad):
    names = {"beta1": "adam_beta1", "beta2": "adam_beta2", "eps": "adam_eps"}
    args = _args()
    setattr(args, names[key], bad)
    with pytest.raises(ValueError, match="--adam-"):
        train_model.resolve_adamw_hyperparameters(args)
    with pytest.raises(ValueError, match="AdamW"):
        AdamW([], **{key: bad})


def test_explicit_values_reach_actual_optimizer_moments():
    p = Parameter(xp.asarray([1.0, -1.0], dtype="float32"), name="p")
    opt = AdamW([p], lr=1e-3, beta1=0.8, beta2=0.7, eps=1e-7,
                weight_decay=0.0)
    p.grad[...] = xp.asarray([2.0, -4.0], dtype="float32")
    opt.step()
    np.testing.assert_allclose(opt.m[0], [0.4, -0.8], rtol=1e-6)
    np.testing.assert_allclose(opt.v[0], [1.2, 4.8], rtol=1e-6)
    assert opt.eps == 1e-7


def test_default_optimizer_unchanged_with_explicit_defaults():
    params = [Parameter(xp.asarray([1.2, -1.3], dtype="float32"), name="p")
              for _ in range(2)]
    optimizers = [AdamW([params[0]], lr=1e-3),
                  AdamW([params[1]], lr=1e-3, beta1=.9, beta2=.95, eps=1e-8)]
    for value in ([1.0, -2.0], [-.5, .75], [.2, .1]):
        for p, opt in zip(params, optimizers):
            p.grad[...] = xp.asarray(value, dtype="float32")
            opt.step()
    np.testing.assert_array_equal(params[0].data, params[1].data)
    np.testing.assert_array_equal(optimizers[0].m[0], optimizers[1].m[0])
    np.testing.assert_array_equal(optimizers[0].v[0], optimizers[1].v[0])


def test_saved_hyperparameters_inherited_and_legacy_fallback(tmp_path):
    model = DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=42,
                                 dtype="float32")
    shard = tmp_path / "train_shard_00000.bin"
    np.arange(4096, dtype="uint16").tofile(shard)
    trainer = ExtendedTrainer(
        model, [str(shard)], [str(shard)], batch_size=1, seq_length=16,
        total_steps=2, warmup_steps=0, checkpoint_dir=str(tmp_path / "ckpt"),
        adam_beta1=0.94, adam_beta2=0.975, adam_eps=1e-7,
    )
    assert trainer.optimizer.beta1 == 0.94
    assert trainer.optimizer.beta2 == 0.975
    assert trainer.optimizer.eps == 1e-7
    trainer.save()
    _, _, state = load_checkpoint(tmp_path / "ckpt", skip_optimizer=True)
    saved = state["adamw_hyperparameters"]
    assert saved == {"beta1": 0.94, "beta2": 0.975, "eps": 1e-7}
    assert train_model.resolve_adamw_hyperparameters(_args(), state) == saved
    assert train_model.resolve_adamw_hyperparameters(_args(), {"step": 12}) == {
        "beta1": 0.9, "beta2": 0.95, "eps": 1e-8,
    }


def test_explicit_override_warns_on_resume_and_preserves_unset_values():
    state = {"adamw_hyperparameters": {"beta1": .94, "beta2": .975, "eps": 1e-7}}
    with pytest.warns(UserWarning, match="optimizer transition"):
        result = train_model.resolve_adamw_hyperparameters(_args(beta1=.97), state)
    assert result == {"beta1": .97, "beta2": .975, "eps": 1e-7}
    with pytest.warns(UserWarning, match="Legacy checkpoint"):
        result = train_model.resolve_adamw_hyperparameters(_args(beta2=.9873), {"step": 1})
    assert result == {"beta1": .9, "beta2": .9873, "eps": 1e-8}
    # No warning if the explicit value matches the checkpoint.
    assert train_model.resolve_adamw_hyperparameters(_args(beta1=.94), state) == state["adamw_hyperparameters"]
