"""Failure-only norm audits retain evidence before clearing a rejected round."""
from copy import deepcopy
import json

import numpy as np
import pytest

from test_heterogeneous_production import native, _trainer, _session, _weights, _assert_tree_equal


@pytest.mark.parametrize("finite_elements", [True, False], ids=["norm-overflow", "inf-element"])
def test_rejected_round_audits_before_clear_preserves_optimizer_and_releases_peer(
    native, tmp_path, monkeypatch, finite_elements,
):
    import mini_llm.train_extended as training
    monkeypatch.setenv("NATIVE_GPU_LOG", "1")
    directory = tmp_path / "diagnostics"
    monkeypatch.setenv("NATIVE_GPU_LOG_DIR", str(directory))
    monkeypatch.delenv("MINI_LLM_GRAD_NORM_AUDIT", raising=False)
    owner = _trainer(native, tmp_path)
    before_weights = _weights(owner)
    before_optimizer = deepcopy(owner.optimizer.checkpoint_state())
    original_clip, original_audit = training.clip_grad_global_norm, training.audit_gradients
    audit_calls = []
    injected = []

    def bad_round(parameters, **kwargs):
        if not injected:
            parameter = parameters[0]
            injected.append(parameter.name)
            parameter.grad.reshape(-1)[0] = 3e20 if finite_elements else np.inf
            if finite_elements:
                parameter.grad.reshape(-1)[1] = 4e20
        return original_clip(parameters, **kwargs)

    def audit(parameters):
        result = original_audit(parameters)
        audit_calls.append(result)
        return result

    monkeypatch.setattr(training, "clip_grad_global_norm", bad_round)
    monkeypatch.setattr(training, "audit_gradients", audit)
    with _session(owner, native) as session:
        with np.errstate(over="ignore", invalid="ignore"):
            loss, norm = owner.train_step()
        assert np.isfinite(loss) and np.isinf(norm)
        assert owner.step == owner.optimizer.step_index == 0
        _assert_tree_equal(_weights(owner), before_weights)
        _assert_tree_equal(owner.optimizer.checkpoint_state(), before_optimizer)
        assert all(not np.any(p.grad) for p in owner.model.parameters())
        assert session.process.poll() is None
        assert len(audit_calls) == 1
        # A new round proves the peer was released without advancing its weights.
        loss, norm = owner.train_step()
        assert np.isfinite(loss) and np.isfinite(norm)
        assert owner.step == owner.optimizer.step_index == 1
        assert len(audit_calls) == 1, "successful rounds must not scan gradients"

    path, = directory.glob("owner-numpy.*.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    rejected = [r for r in records if r["event"] == "gradient_rejected"]
    assert len(rejected) == 1
    row = rejected[0]
    assert row["attempted_step"] == 1 and row["completed_step"] == 0
    assert row["ordinary_norm"] == "inf"
    report = row["gradient_audit"]
    assert report["all_finite"] is finite_elements
    if finite_elements:
        assert row["failure_kind"] == "norm_reduction_overflow"
        assert report["stable_norm"] == pytest.approx(5e20, rel=2e-6)
        assert report["nonfinite"] == []
        assert report["top"][0]["name"] == injected[0]
    else:
        assert row["failure_kind"] == "nonfinite_elements"
        assert report["stable_norm"] is None
        assert report["nonfinite"][0]["name"] == injected[0]
        assert report["nonfinite"][0]["inf"] == 1
    syncs = [r for r in records if r["event"] == "weights_synced"]
    assert [r["updated"] for r in syncs] == [False, True]
