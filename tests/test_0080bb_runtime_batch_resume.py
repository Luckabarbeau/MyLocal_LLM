"""0080BB: runtime batch/accum changes are not checkpoint shape contracts."""

import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME
from test_block_proposal_0080o2_hybrid_muon import _hybrid
from test_block_proposal_0080n_production import _docs, _close

pytestmark = pytest.mark.skipif(BACKEND_NAME != "numpy", reason="CPU contract tests")


def test_can_resume_with_changed_accumulation_and_batch(tmp_path, capsys):
    shard = _docs(tmp_path)
    live = _hybrid(shard)
    resumed = _hybrid(shard)
    try:
        original = live.block_proposal_resume_contract()
        assert original["batch_size"] == 2
        assert original["grad_accum_steps"] == 2
        # Reproduce a real parameter change: 2x2 -> 1x16.
        resumed.batch_size = 1
        resumed.grad_accum_steps = 16
        resumed.effective_batch_size = 16
        resumed.training_objective.block_group_size = 32
        resumed.validate_block_proposal_resume_contract(
            {"block_proposal_resume_contract": original})
        out = capsys.readouterr().out
        assert "batch_size: 2 -> 1" in out
        assert "grad_accum_steps: 2 -> 16" in out
        assert "Effective batch: 4 -> 16" in out
        assert "bitwise reproducibility" in out
        assert "block group size changed" in out
        # Checkpoint still records the NEW values for future resume.
        updated = resumed.block_proposal_resume_contract()
        assert updated["batch_size"] == 1
        assert updated["grad_accum_steps"] == 16
        assert updated["block_group_size"] == 32
    finally:
        _close(live)
        _close(resumed)


def test_strict_runtime_mode_is_opt_in(tmp_path, monkeypatch):
    shard = _docs(tmp_path)
    live = _hybrid(shard)
    resumed = _hybrid(shard)
    try:
        saved = live.block_proposal_resume_contract()
        resumed.grad_accum_steps = 16
        monkeypatch.setenv("MINI_LLM_STRICT_BATCH_ACCUM_RESUME", "1")
        with pytest.raises(ValueError, match="grad_accum_steps"):
            resumed.validate_block_proposal_resume_contract(
                {"block_proposal_resume_contract": saved})
        monkeypatch.delenv("MINI_LLM_STRICT_BATCH_ACCUM_RESUME")
        resumed.validate_block_proposal_resume_contract(
            {"block_proposal_resume_contract": saved})
    finally:
        _close(live)
        _close(resumed)


def test_optimizer_and_schedule_incompatibility_still_rejected(tmp_path):
    shard = _docs(tmp_path)
    live = _hybrid(shard)
    resumed = _hybrid(shard)
    try:
        saved = live.block_proposal_resume_contract()
        resumed.grad_accum_steps = 16
        for change in ("adam_beta2", "seq_length", "optimizer_kind"):
            bad = dict(saved)
            if change == "adam_beta2":
                bad[change] -= 0.01
            elif change == "seq_length":
                bad[change] += 1
            else:
                bad[change] = "adamw"
            with pytest.raises(ValueError, match=change):
                resumed.validate_block_proposal_resume_contract(
                    {"block_proposal_resume_contract": bad})
    finally:
        _close(live)
        _close(resumed)


def test_legacy_contract_without_runtime_provenance_warns(tmp_path, capsys):
    shard = _docs(tmp_path)
    live = _hybrid(shard)
    try:
        saved = live.block_proposal_resume_contract()
        saved.pop("grad_accum_steps")
        live.validate_block_proposal_resume_contract(
            {"block_proposal_resume_contract": saved})
        assert "lacks grad_accum_steps" in capsys.readouterr().out
    finally:
        _close(live)
