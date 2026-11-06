"""Interrupted-versus-uninterrupted AR continuation for the Model B training path."""

import numpy as np

from mini_llm.checkpoint import load_checkpoint
from mini_llm.config import DualResolutionConfig, ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.hybrid_muon import HybridMuonAdamW
from mini_llm.train_extended import ExtendedTrainer


def _model(seed=117):
    config = ModelConfig(
        tokenizer_vocab_size=128, context_length=16, n_layers=1,
        d_model=16, d_ff=32, n_q_heads=2, n_kv_heads=1, d_head=8,
        n_experts=1, top_k=1, ffn_mode="dense", tie_embeddings=True,
        dtype="float32", dual_resolution=DualResolutionConfig(
            enabled=True, block_size=4, local_window=8, compression_dim=8,
            compression_heads=2, compression_mode="matrix_pool",
            summaries_per_block=2, global_heads=1, global_head_dim=8,
            max_blocks=16, global_query_chunk=4,
        ),
    )
    return DecoderLanguageModel(config, rng_seed=seed, dtype="float32")


def _trainer(model, shard, checkpoint_dir):
    return ExtendedTrainer(
        model=model, train_shard_paths=[str(shard)], val_shard_paths=[str(shard)],
        eos_token_id=127, batch_size=1, seq_length=16, grad_accum_steps=4,
        gradient_accumulation_weighting="supervised_tokens",
        optimizer_kind="hybrid-muon", muon_matmul_precision="fp32",
        total_steps=6, warmup_steps=2, peak_lr=1.5e-4,
        grad_clip=1.0, weight_decay=0.1, val_interval=100,
        save_interval=100, val_steps=1, rng_seed=42,
        checkpoint_dir=str(checkpoint_dir),
    )


def _restore(trainer, checkpoint):
    parameters = trainer.model.parameters()
    loaded, optimizer_state, state = load_checkpoint(
        checkpoint, param_names=[parameter.name for parameter in parameters],
    )
    assert optimizer_state is not None and state is not None
    for parameter in parameters:
        parameter.data[...] = loaded[parameter.name]
    for block in trainer.model.blocks:
        block.refresh_compute_buffers()
    trainer.optimizer.restore_checkpoint(optimizer_state, from_hybrid=True)
    trainer.optimizer.restore_parameter_birth_steps(
        state["optimizer_parameter_birth_steps"]
    )
    trainer.step = int(state["step"])
    trainer.tokens_processed = int(state["tokens_processed"])
    trainer.supervised_tokens_total = float(state["supervised_tokens_total"])
    trainer.supervised_tokens_provenance = state["supervised_tokens_provenance"]
    trainer.current_train_shard_idx = int(state["current_train_shard_idx"])
    trainer.current_val_shard_idx = int(state["current_val_shard_idx"])
    trainer.train_rng.bit_generator.state = state["train_rng_state"]
    trainer.val_rng.bit_generator.state = state["val_rng_state"]
    return state


def _capture_next_step(trainer):
    batches = []
    original = trainer.get_train_batch

    def capture(*args, **kwargs):
        inputs, targets, metadata = original(*args, **kwargs)
        batches.append((
            np.asarray(inputs).copy(), np.asarray(targets).copy(),
            np.asarray(metadata["target_loss_mask"]).copy(),
        ))
        return inputs, targets, metadata

    trainer.get_train_batch = capture
    try:
        result = trainer.train_step()
    finally:
        trainer.get_train_batch = original
    return result, batches


def _optimizer_arrays(optimizer):
    state = optimizer.checkpoint_state()
    return {
        group + ":" + name: np.asarray(value).copy()
        for group in ("master_weights", "m", "v", "muon_m")
        for name, value in state[group].items()
    }


def test_interrupted_resume_matches_uninterrupted_hybrid_ar(tmp_path, monkeypatch):
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_SHAPE_BUCKET", "fine")
    monkeypatch.setenv("MINI_LLM_OPTIMIZER_OFFLOAD", "none")
    shard = tmp_path / "train.bin"
    np.array([
        1, 2, 3, 4, 127,
        6, 7, 8, 9, 10, 127,
        15, 16, 17, 18, 19, 20, 21, 22, 127,
        30, 31, 32, 33, 34, 35, 127,
    ], dtype=np.uint16).tofile(shard)

    checkpoint = tmp_path / "checkpoint"
    uninterrupted = _trainer(_model(), shard, checkpoint)
    uninterrupted.train_step()
    uninterrupted.save()

    resumed = _trainer(_model(), shard, checkpoint)
    saved = _restore(resumed, checkpoint)
    assert isinstance(resumed.optimizer, HybridMuonAdamW)
    assert saved["muon_hyperparameters"]["group_names"] == sorted(
        resumed.optimizer.muon_names
    )
    assert uninterrupted.step == resumed.step == uninterrupted.optimizer.step_index
    assert uninterrupted.scheduler(uninterrupted.step) == resumed.scheduler(resumed.step)

    live_result, live_batches = _capture_next_step(uninterrupted)
    resumed_result, resumed_batches = _capture_next_step(resumed)

    np.testing.assert_array_equal(live_result, resumed_result)
    assert len(live_batches) == len(resumed_batches) == 4
    for left, right in zip(live_batches, resumed_batches):
        for left_array, right_array in zip(left, right):
            np.testing.assert_array_equal(left_array, right_array)
    for left, right in zip(uninterrupted.model.parameters(), resumed.model.parameters()):
        assert left.name == right.name
        np.testing.assert_array_equal(left.data, right.data)
    for name, left in _optimizer_arrays(uninterrupted.optimizer).items():
        np.testing.assert_array_equal(left, _optimizer_arrays(resumed.optimizer)[name])
    assert uninterrupted.step == resumed.step == uninterrupted.optimizer.step_index
    assert uninterrupted.supervised_tokens_total == resumed.supervised_tokens_total
