"""0078I: bounded direct-history shapes and route-aware GPU memory profiling."""
import numpy as np
import pytest

from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train_extended import ExtendedTrainer
from test_document_first_sampling_0078e import _make_trainer


def test_direct_history_bucket_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MINI_LLM_DIRECT_HISTORY_BUCKET_TOKENS", raising=False)
    model = DecoderLanguageModel.__new__(DecoderLanguageModel)
    model.direct_history_bucket_tokens = 0
    assert [model._direct_history_length(n, 2048) for n in (1, 127, 150, 2048, 4000)] == [
        1, 127, 150, 2048, 2048
    ]


def test_bucket_bounds_shapes_without_padding():
    model = DecoderLanguageModel.__new__(DecoderLanguageModel)
    model.direct_history_bucket_tokens = 128
    lengths = [model._direct_history_length(n, 2048) for n in range(1, 2200)]
    assert lengths[0] == 0
    assert lengths[127] == 128
    assert lengths[2047] == 2048
    assert max(lengths) == 2048  # 4096+2048 is the existing maximum active length.
    assert set(lengths) == set(range(0, 2049, 128))
    assert all(a <= n for a, n in zip(lengths, range(1, 2200)))


def test_short_direct_history_falls_back_to_dense(monkeypatch, tmp_path):
    from mini_llm.backend import xp
    monkeypatch.setenv("MINI_LLM_DIRECT_HISTORY_BUCKET_TOKENS", "2")
    trainer = _make_trainer(tmp_path, documents=[[1, 2, 3, 4, 5]], retrieval_probability=0.0)
    x, y, meta = trainer.get_train_batch(return_metadata=True)
    assert meta["history_mode_requested"] == "direct_history"
    assert trainer.model.direct_history_bucket_tokens == 2
    active, cache = trainer.model._hierarchical_context_forward(xp.asarray(x), memory_metadata=meta)
    assert cache["prefix_mode"] == "none"
    assert cache["prefix_length"] == 0
    assert active.target_start == 0
    assert active.target_end == 4
    np.testing.assert_array_equal(np.asarray(active.token_ids), x[:, -4:])


def test_bucketed_prefix_uses_only_genuine_recent_tokens(monkeypatch, tmp_path):
    from mini_llm.backend import xp
    monkeypatch.setenv("MINI_LLM_DIRECT_HISTORY_BUCKET_TOKENS", "2")
    trainer = _make_trainer(tmp_path, documents=[[1, 2, 3, 4, 5, 6]], retrieval_probability=0.0)
    x, y, meta = trainer.get_train_batch(return_metadata=True)
    active, cache = trainer.model._hierarchical_context_forward(xp.asarray(x), memory_metadata=meta)
    assert cache["prefix_mode"] == "direct_history"
    assert cache["prefix_length"] == 2
    np.testing.assert_array_equal(np.asarray(active.token_ids)[:, :2], x[:, 10:12])
    np.testing.assert_array_equal(np.asarray(active.token_ids)[:, 2:], x[:, 12:16])
    np.testing.assert_array_equal(np.asarray(active.position_ids)[:, :2], [[10, 11]])
    assert active.target_start == 2
    assert active.target_end == 6


def test_invalid_bucket_env_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_LLM_DIRECT_HISTORY_BUCKET_TOKENS", "-5")
    with pytest.raises(ValueError, match="must be >= 0"):
        _make_trainer(tmp_path, documents=[[1, 2, 3, 4, 5]])


def test_memory_profile_schedules_are_periodic_and_bounded():
    trainer = ExtendedTrainer.__new__(ExtendedTrainer)
    trainer.memory_profile_steps = 3
    trainer.memory_profile_interval = 20
    trainer._memory_profiled_steps = 0
    trainer.step = 50001
    assert not trainer._memory_profile_due()
    trainer.step = 50019
    assert trainer._memory_profile_due()
    trainer._memory_profiled_steps = 3
    assert not trainer._memory_profile_due()


def test_history_shapes_are_cpu_only_and_reset(capsys):
    trainer = ExtendedTrainer.__new__(ExtendedTrainer)
    trainer.gpu_memory_audit = True
    trainer.gpu_memory_audit_train_interval = 20
    trainer._gpu_memory_history_shapes = {}
    for mode, lengths in [
        ("none", (0, 0)),
        ("direct_history", (128, 128, 256)),
        ("routed_blocks", (2048,)),
    ]:
        for length in lengths:
            trainer._record_gpu_history_shape({
                "memory_context_cache": {"prefix_mode": mode, "prefix_length": length}
            })
    trainer._print_gpu_history_shapes()
    out = capsys.readouterr().out
    assert "direct_history: n=3, distinct_lengths=2, min=128, max=256" in out
    assert "routed_blocks: n=1, distinct_lengths=1, min=2048, max=2048" in out
    assert trainer._gpu_memory_history_shapes == {}


def test_mode_peak_reporting_uses_existing_snapshot(capsys):
    peaks = {}
    snap = {"device_used": 900 * 2**20, "pool_used": 300 * 2**20,
            "pool_reserved": 800 * 2**20, "device_free": 100 * 2**20}
    for phase in ("forward", "backward", "released"):
        ExtendedTrainer._update_mode_memory_peak(peaks, "direct_history", 128, phase, snap)
    trainer = ExtendedTrainer.__new__(ExtendedTrainer)
    trainer.step = 50020
    trainer._print_memory_report({"step_start": snap}, peaks)
    assert "direct_history: n=1, prefix_lengths=[128]" in capsys.readouterr().out


def test_bucketed_direct_history_full_training_step(monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_LLM_DIRECT_HISTORY_BUCKET_TOKENS", "2")
    trainer = _make_trainer(tmp_path, documents=[[1, 2, 3, 4, 5, 6]], retrieval_probability=0.0)
    loss, grad_norm = trainer.train_step()
    assert np.isfinite(loss)
    assert np.isfinite(grad_norm)
    assert trainer.step == 1
