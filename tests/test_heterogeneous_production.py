"""CPU native production-peer integration, using only temporary packed shards."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
import sys

import numpy as np
import pytest

from test_train_extended import create_dummy_shard


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setenv("MINI_LLM_BACKEND", "numpy")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    from mini_llm.backend import BACKEND_NAME
    from mini_llm.model_registry import registry

    assert BACKEND_NAME == "numpy", "run these tests with MINI_LLM_BACKEND=numpy"
    base = registry.config("model_b_348m_sparse16")
    model_config = replace(
        base, dtype="float32", tokenizer_vocab_size=64, context_length=32,
        n_layers=2, d_model=256, n_q_heads=4, n_kv_heads=2, d_head=64, d_ff=128,
        dual_resolution=replace(
            base.dual_resolution, block_size=4, local_window=4, long_local_window=8,
            long_local_heads=1, max_blocks=8, summaries_per_block=2,
            global_heads=2, global_head_dim=64, sparse_retrieval_layers=(0, 1)),
    )
    shards = []
    for name in ("train", "val"):
        path = tmp_path / f"{name}_shard_00000.bin"
        create_dummy_shard(path, num_tokens=1024)
        # The repository helper writes headerless uint16 tokens. Keep IDs inside
        # the tiny vocabulary and reserve EOS=63, so every sampled target is valid.
        tokens = np.fromfile(path, dtype=np.uint16) % 63
        tokens.tofile(path)
        shards.append(str(path))
    config = {
        "python": sys.executable, "backend": "numpy", "batch_size": 2,
        "context_length": 5, "max_microbatches": 1, "warmup_microbatches": 1,
        "seed": 104729, "timeout_s": 30, "work_dir": str(tmp_path),
        "log_file": str(tmp_path / "peer.log"),
    }
    config_path = tmp_path / "peer.json"
    config_path.write_text(json.dumps(config))
    return model_config, shards, config_path


def _trainer(native, tmp_path, *, peer=False):
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.train_extended import ExtendedTrainer

    config, shards, _ = native
    return ExtendedTrainer(
        DecoderLanguageModel(config, rng_seed=42, dtype="float32"),
        train_shard_paths=[shards[0]], val_shard_paths=[shards[1]],
        batch_size=2 if peer else 1, seq_length=5 if peer else 9,
        grad_accum_steps=1 if peer else 2,
        gradient_accumulation_weighting="supervised_tokens",
        optimizer_kind="hybrid-muon", optimizer_enabled=not peer,
        rng_seed=104729 if peer else 42, eos_token_id=63,
        total_steps=2000, warmup_steps=0, grad_clip=0.1,
        checkpoint_dir=str(tmp_path / "checkpoint"),
        log_file=str(tmp_path / "owner.csv"),
    )


@contextmanager
def _session(trainer, native, training_state=None):
    from mini_llm.heterogeneous_production import ProductionPeerSession

    config_path = native[2]
    log_path = json.loads(config_path.read_text())["log_file"]
    session = None
    try:
        session = ProductionPeerSession(
            trainer, config_path=config_path, training_state=training_state or {},
            acknowledge_fork=True, tokenizer_path=None,
        )
        trainer.accumulation_peer = session
        yield session
    except Exception as exc:
        from pathlib import Path
        log = Path(log_path)
        if log.exists():
            exc.add_note(f"native peer subprocess log:\n{log.read_text()}")
        raise
    finally:
        if session is not None:
            session.close()
            trainer.accumulation_peer = None


def _weights(trainer):
    return {p.name: p.data.copy() for p in trainer.model.parameters()}


def _assert_tree_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


def _serial_update(owner, peer):
    """Independent full-logit backward, globally normalized before one clip/update."""
    from mini_llm.optim.grad_clip import clip_grad_global_norm

    batches = []
    for sampler, count in ((owner, 2), (peer, 1)):
        for _ in range(count):
            batches.append(sampler.get_train_batch(return_metadata=True))
    total = sum(float(meta["target_loss_mask"].sum()) for _, _, meta in batches)
    owner.model.zero_grad()
    loss = 0.0
    for inputs, targets, meta in batches:
        logits, cache = owner.model.forward(
            inputs, dual_resolution_metadata=meta["dual_resolution_metadata"],
        )
        ce, loss_cache = owner.model.compute_loss(
            logits, targets, loss_mask=meta["target_loss_mask"],
        )
        weight = float(meta["target_loss_mask"].sum()) / total
        loss += float(ce) * weight
        owner.model.backward(owner.model.backward_loss(loss_cache) * weight, cache)
    norm, _, finite = clip_grad_global_norm(owner.optimization_parameters, owner.grad_clip)
    assert finite
    assert float(norm) > owner.grad_clip, "exercise clipping after global reduction"
    owner.optimizer.step(lr=owner.scheduler(owner.step))
    owner.model.refresh_compute_buffers()
    owner.optimizer.zero_grad()
    return loss, float(norm), total


def test_shared_update_matches_serial_native_global_token_mean(native, tmp_path):
    owner = _trainer(native, tmp_path)
    serial = _trainer(native, tmp_path / "serial")
    peer_sampler = _trainer(native, tmp_path / "serial-peer", peer=True)
    expected_loss, expected_norm, total = _serial_update(serial, peer_sampler)

    with _session(owner, native) as session:
        loss, norm = owner.train_step()
        assert session.process.pid != __import__("os").getpid()
        assert session.process.poll() is None
        assert session.last_report["microbatches"] == 1
        assert session.last_report["supervised_tokens"] == 10
        assert owner.step == owner.optimizer.step_index == 1
        assert owner._last_step_supervised_tokens == total == 28
        assert owner.tokens_processed == 28
        assert sum(r["supervised_tokens"] for r in owner.source_metrics.values()) == total
        assert sum(r["microbatches"] for r in owner.source_metrics.values()) == 3
        assert loss == pytest.approx(expected_loss, rel=2e-6, abs=2e-6)
        assert norm == pytest.approx(expected_norm, rel=3e-5, abs=3e-6)
        for name, value in _weights(owner).items():
            np.testing.assert_allclose(
                value, _weights(serial)[name], rtol=3e-4, atol=3e-6, err_msg=name,
            )
    assert session.process.poll() == 0
    assert not list(tmp_path.glob("production-peer-*"))


def test_checkpoint_restores_peer_membership_and_next_full_update(native, tmp_path):
    from mini_llm.checkpoint import load_checkpoint
    from mini_llm.heterogeneous_production import restore_sampler, sampler_state

    owner = _trainer(native, tmp_path)
    with _session(owner, native) as session:
        owner.train_step()
        owner.save()
        names = [p.name for p in owner.model.parameters()]
        params, optimizer_state, saved = load_checkpoint(owner.checkpoint_dir, param_names=names)
        heterogeneous = saved["heterogeneous_state"]
        assert heterogeneous["contract"] == session.contract
        assert heterogeneous["contract"]["normalization"] == "global_supervised_tokens"
        assert heterogeneous["step"] == 1
        assert heterogeneous["parent_step"] == 0
        assert heterogeneous["history"][0]["optimizer_state_preserved"] is True
        peer_before = heterogeneous["peer_sampler"]
        assert peer_before["tokens_processed"] == 10
        assert sum(peer_before["train_source_supervised_tokens"].values()) == 10
        assert "train_rng_state" in peer_before
        assert optimizer_state["step"] == 1
        assert all(optimizer_state[group] for group in ("master_weights", "m", "v", "muon_m"))
        owner_before = sampler_state(owner)
        uninterrupted_result = owner.train_step()
        uninterrupted_weights = _weights(owner)
        uninterrupted_optimizer = deepcopy(owner.optimizer.checkpoint_state())
        uninterrupted_sampler = sampler_state(owner)
        uninterrupted_peer = session.state_dict(owner)["peer_sampler"]
        assert session.last_report["microbatches"] == 1

    restored = _trainer(native, tmp_path / "restored")
    for parameter in restored.model.parameters():
        parameter.data[...] = params[parameter.name]
    restored.model.refresh_compute_buffers()
    restored.optimizer.restore_checkpoint(optimizer_state, from_hybrid=True)
    restored.step = saved["step"]
    restore_sampler(restored, saved)
    restored.restore_source_metrics(saved["source_metrics"])
    _assert_tree_equal(sampler_state(restored), owner_before)

    # Reuse exactly the saved config, including log/work paths, sequentially.
    with _session(restored, native, saved) as session:
        _assert_tree_equal(session.state_dict(restored)["peer_sampler"], peer_before)
        resumed_result = restored.train_step()
        assert session.last_report["microbatches"] == 1
        assert restored.step == restored.optimizer.step_index == 2
        assert resumed_result == pytest.approx(uninterrupted_result, rel=1e-7, abs=1e-7)
        _assert_tree_equal(_weights(restored), uninterrupted_weights)
        _assert_tree_equal(restored.optimizer.checkpoint_state(), uninterrupted_optimizer)
        _assert_tree_equal(sampler_state(restored), uninterrupted_sampler)
        _assert_tree_equal(session.state_dict(restored)["peer_sampler"], uninterrupted_peer)
        assert restored.tokens_processed == 56
        assert sum(r["supervised_tokens"] for r in restored.source_metrics.values()) == 56

    changed = json.loads(native[2].read_text())
    changed["max_microbatches"] = 2
    native[2].write_text(json.dumps(changed))
    from mini_llm.heterogeneous_production import ProductionPeerSession
    with pytest.raises(ValueError, match="topology/data contract changed"):
        ProductionPeerSession(
            restored, config_path=native[2], training_state=saved,
            acknowledge_fork=True, tokenizer_path=None,
        )
    assert not list(tmp_path.glob("production-peer-*"))


def test_warmup_preserves_sampler_weights_and_owner_optimizer_peer_has_no_masters(native, tmp_path):
    from mini_llm.heterogeneous_production import sampler_state, warmup
    from mini_llm.train_extended import _AccumulationOnlyOptimizer

    owner = _trainer(native, tmp_path)
    peer = _trainer(native, tmp_path / "peer", peer=True)
    assert isinstance(peer.optimizer, _AccumulationOnlyOptimizer)
    assert not peer.optimizer_enabled
    for name in ("master_weights", "m", "v", "muon_m", "adam", "muon"):
        assert not hasattr(peer.optimizer, name)
    for trainer in (owner, peer):
        before_sampler = sampler_state(trainer)
        before_weights = _weights(trainer)
        before_metrics = deepcopy(trainer.source_metrics)
        before_optimizer = (
            deepcopy(trainer.optimizer.checkpoint_state()) if trainer is owner else None
        )
        warmup(trainer, 1)
        assert trainer.step == trainer.optimizer.step_index == 0
        assert trainer.optimizer_enabled is (trainer is owner)
        assert getattr(trainer, "accumulation_peer", None) is None
        _assert_tree_equal(sampler_state(trainer), before_sampler)
        _assert_tree_equal(_weights(trainer), before_weights)
        _assert_tree_equal(trainer.source_metrics, before_metrics)
        assert all(not np.any(p.grad) for p in trainer.model.parameters())
        if trainer is owner:
            _assert_tree_equal(trainer.optimizer.checkpoint_state(), before_optimizer)
    with pytest.raises(RuntimeError, match="accumulation-only peers cannot update"):
        peer.optimizer.step()
