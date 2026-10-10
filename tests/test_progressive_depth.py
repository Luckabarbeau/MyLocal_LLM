"""0064B progressive-depth model/optimizer state tests."""

import dataclasses
import tempfile
from pathlib import Path

import numpy as np

from mini_llm.backend import xp
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.parameter import Parameter
from mini_llm.train_extended import ExtendedTrainer
from mini_llm.checkpoint import load_checkpoint


def progressive_config(n_layers=3, initial=1):
    return dataclasses.replace(
        ModelConfig.tiny_inspection(),
        n_layers=n_layers,
        progressive_depth=True,
        progressive_initial_layers=initial,
    )


def create_dummy_shard(path: Path, num_tokens: int = 4096):
    rng = np.random.default_rng(6402)
    rng.integers(0, 256, size=num_tokens, dtype=np.uint16).tofile(path)


def make_trainer(model, tmpdir, growth_steps=None):
    train = Path(tmpdir) / "train_shard_00000.bin"
    val = Path(tmpdir) / "val_shard_00000.bin"
    create_dummy_shard(train)
    create_dummy_shard(val, 1024)
    return ExtendedTrainer(
        model=model,
        train_shard_paths=[str(train)],
        val_shard_paths=[str(val)],
        batch_size=1,
        seq_length=model.config.context_length,
        grad_accum_steps=1,
        total_steps=20,
        warmup_steps=2,
        val_interval=100,
        save_interval=100,
        checkpoint_dir=tmpdir,
        progressive_growth_steps=growth_steps,
    )


def test_progressive_model_starts_with_only_requested_optimizer_blocks():
    model = DecoderLanguageModel(progressive_config(3, 1), rng_seed=9, dtype="float32")

    assert model.active_layers == 1
    assert model.max_layers == 3
    assert float(model.blocks[0].alpha_attn.data) == 1.0
    assert float(model.blocks[0].alpha_mlp.data) == 1.0
    assert float(model.blocks[1].alpha_attn.data) == 0.0
    assert float(model.blocks[2].alpha_mlp.data) == 0.0

    optimized_names = {p.name for p in model.optimization_parameters()}
    assert any(name.startswith("blocks.0.") for name in optimized_names)
    assert not any(name.startswith("blocks.1.") for name in optimized_names)
    assert not any(name.startswith("blocks.2.") for name in optimized_names)
    assert "blocks.0.alpha_attn" in optimized_names


def test_inactive_blocks_are_not_executed():
    model = DecoderLanguageModel(progressive_config(3, 1), rng_seed=10, dtype="float32")
    token_ids = xp.asarray([[1, 2, 3, 4]], dtype=xp.int64)

    def forbidden(*args, **kwargs):
        raise AssertionError("inactive Transformer block executed")

    model.blocks[1].forward = forbidden
    model.blocks[2].forward = forbidden
    hidden = model.forward_body(token_ids, return_cache=False)
    assert hidden.shape == (1, 4, model.config.d_model)


def test_activating_zero_gated_random_block_preserves_function_exactly():
    model = DecoderLanguageModel(progressive_config(2, 1), rng_seed=11, dtype="float32")
    token_ids = xp.asarray([[1, 5, 7, 3, 2]], dtype=xp.int64)

    before = model.forward_body(token_ids, return_cache=False).copy()
    layer_idx = model.activate_next_layer()
    after = model.forward_body(token_ids, return_cache=False)

    assert layer_idx == 1
    assert model.active_layers == 2
    assert float(model.blocks[1].alpha_attn.data) == 0.0
    assert float(model.blocks[1].alpha_mlp.data) == 0.0
    np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


def test_adam_new_parameter_uses_fresh_local_bias_correction():
    p_old = Parameter(xp.asarray([0.75], dtype="float32"), name="old", decay=False)
    opt = AdamW([p_old], lr=1e-3, weight_decay=0.0)

    for _ in range(7):
        p_old.grad[...] = 0.25
        opt.step()
        opt.zero_grad()

    old_m = np.asarray(opt.m[0]).copy()
    old_v = np.asarray(opt.v[0]).copy()
    old_master = np.asarray(opt.master_weights[0]).copy()

    p_new = Parameter(xp.asarray([0.5], dtype="float32"), name="new", decay=False)
    assert opt.add_parameters([p_new]) == 1
    assert opt.parameter_birth_steps[-1] == 7
    np.testing.assert_array_equal(np.asarray(opt.m[0]), old_m)
    np.testing.assert_array_equal(np.asarray(opt.v[0]), old_v)
    np.testing.assert_array_equal(np.asarray(opt.master_weights[0]), old_master)
    np.testing.assert_array_equal(np.asarray(opt.m[-1]), np.zeros(1, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(opt.v[-1]), np.zeros(1, dtype=np.float32))

    # Compare the newly added parameter's first update against a truly fresh Adam.
    p_fresh = Parameter(xp.asarray([0.5], dtype="float32"), name="fresh", decay=False)
    fresh = AdamW([p_fresh], lr=1e-3, weight_decay=0.0)
    p_new.grad[...] = 0.4
    p_old.grad[...] = 0.0
    p_fresh.grad[...] = 0.4
    opt.step()
    fresh.step()
    np.testing.assert_allclose(
        np.asarray(p_new.data), np.asarray(p_fresh.data), rtol=0.0, atol=1e-7
    )


def test_trainer_growth_preserves_old_optimizer_state_and_adds_new_block():
    model = DecoderLanguageModel(progressive_config(3, 1), rng_seed=12, dtype="float32")
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = make_trainer(model, tmpdir)
        trainer.optimizer.step_index = 13
        trainer.step = 13
        trainer.optimizer.m[0][...] = 0.125
        trainer.optimizer.v[0][...] = 0.25
        old_m = np.asarray(trainer.optimizer.m[0]).copy()
        old_v = np.asarray(trainer.optimizer.v[0]).copy()
        old_count = len(trainer.optimizer.parameters)

        layer_idx = trainer.activate_next_progressive_layer()

        assert layer_idx == 1
        assert model.active_layers == 2
        assert len(trainer.optimizer.parameters) > old_count
        np.testing.assert_array_equal(np.asarray(trainer.optimizer.m[0]), old_m)
        np.testing.assert_array_equal(np.asarray(trainer.optimizer.v[0]), old_v)
        new_indices = [
            i for i, p in enumerate(trainer.optimizer.parameters)
            if p.name.startswith("blocks.1.")
        ]
        assert new_indices
        assert all(trainer.optimizer.parameter_birth_steps[i] == 13 for i in new_indices)
        assert all(np.all(np.asarray(trainer.optimizer.m[i]) == 0) for i in new_indices)
        assert all(np.all(np.asarray(trainer.optimizer.v[i]) == 0) for i in new_indices)


def test_scheduled_growth_is_idempotent_and_checkpointed():
    model = DecoderLanguageModel(progressive_config(3, 1), rng_seed=13, dtype="float32")
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = make_trainer(model, tmpdir, growth_steps=[2, 5])
        trainer.step = 2
        trainer.optimizer.step_index = 2

        trainer._apply_scheduled_progressive_growth()
        assert model.active_layers == 2
        count = len(trainer.optimizer.parameters)
        trainer._apply_scheduled_progressive_growth()
        assert model.active_layers == 2
        assert len(trainer.optimizer.parameters) == count

        trainer.save()
        _, _, state = load_checkpoint(
            tmpdir,
            param_names=[p.name for p in model.parameters()],
            skip_optimizer=True,
        )
        assert state["progressive_active_layers"] == 2
        assert state["progressive_growth_steps"] == [2, 5]
        births = state["optimizer_parameter_birth_steps"]
        assert births["blocks.1.alpha_attn"] == 2
        assert births["blocks.1.alpha_mlp"] == 2


def test_progressive_parameter_estimate_counts_only_active_optimizer_subset():
    config = progressive_config(3, 1)
    model = DecoderLanguageModel(config, rng_seed=14, dtype="float32")
    assert config.estimated_parameter_count() == sum(p.size for p in model.parameters())
    assert config.estimated_parameter_count(active_layers=1) == sum(
        p.size for p in model.optimization_parameters()
    )


def test_progressive_training_runs_across_growth_boundary():
    """End-to-end smoke: scheduled growth must train scalar gates and new blocks."""
    model = DecoderLanguageModel(progressive_config(2, 1), rng_seed=15, dtype="float32")
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = make_trainer(model, tmpdir, growth_steps=[1])
        trainer.total_steps = 2
        trainer.scheduler.total_steps = 2
        losses = trainer.train(num_steps=2, log_interval=100)

        assert len(losses) == 2
        assert model.active_layers == 2
        # The newly activated block is identity at insertion, then its gates
        # receive their first optimizer update while internal weights start from
        # fresh Adam state.
        assert abs(float(model.blocks[1].alpha_attn.data)) > 0.0
        assert abs(float(model.blocks[1].alpha_mlp.data)) > 0.0
