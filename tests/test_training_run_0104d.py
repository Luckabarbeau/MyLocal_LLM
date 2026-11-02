"""TrainingRun reuses existing model weights and production trainer logic."""
import numpy as np
import pytest

from mini_llm import Model
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.training_run import TrainingRun


def _model():
    config = ModelConfig(tokenizer_vocab_size=64, context_length=8,
                         n_layers=1, d_model=16, d_ff=32,
                         n_q_heads=2, n_kv_heads=1, d_head=8,
                         n_experts=1, top_k=1, ffn_mode="dense", dtype="float32")
    return Model.wrap_native(DecoderLanguageModel(config, rng_seed=16))


def test_native_adapter_runs_exact_existing_trainer(tmp_path):
    tokens = np.arange(64, dtype=np.uint16)
    shard = tmp_path / "shard.bin"
    tokens.tofile(shard)
    model = _model()
    run = model.train(train_shard_paths=[str(shard)], val_shard_paths=[str(shard)],
                      batch_size=1, seq_length=8, grad_accum_steps=1,
                      total_steps=2, warmup_steps=1, val_interval=100,
                      save_interval=100, val_steps=1,
                      checkpoint_dir=str(tmp_path / "checkpoint"))
    assert run.trainer.model is model._native_model
    assert run.step == 0
    before = {p.name: p.data.copy() for p in model._native_model.parameters()}
    losses = run.train_steps(1, log_interval=50)
    assert run.step == 1 and len(losses) == 1
    assert run.supervised_tokens > 0
    assert run.inspect()['grad_accum_steps'] == 1
    assert any(np.any(p.data != before[p.name]) for p in model._native_model.parameters())
    run.save_checkpoint()
    from mini_llm.checkpoint import load_checkpoint
    _, optimizer, state = load_checkpoint(tmp_path / "checkpoint")
    assert optimizer['step'] == 1
    assert state['step'] == 1
    assert TrainingRun.wrap_native(model, run.trainer).trainer is run.trainer
    with pytest.raises(ValueError, match='nonnegative'):
        run.train_steps(-1)


def test_run_rejects_dataset_omissions_and_mismatched_model(tmp_path):
    model = _model()
    with pytest.raises(ValueError, match='train_shard_paths'):
        model.train(train_shard_paths=[], val_shard_paths=[])
    class Dummy:
        model = object()
    with pytest.raises(ValueError, match='share'):
        TrainingRun.wrap_native(model, Dummy())
