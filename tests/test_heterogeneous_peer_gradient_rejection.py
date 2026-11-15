"""A non-finite peer export must reject the round, not abort the run, and must
replay identical data so the recurrence can be attributed.

The 5080 owner already treats a non-finite combined gradient as a skipped update
whose weights, optimizer and step counters are preserved, and then retries. A bad
HIP export receives that same treatment, with parameter-level evidence taken from
the shared export buffer (the combined owner audit cannot recover it once the Inf
has been folded in). Unlike the owner's own skip, the peer round is replayed on
the SAME documents and the SAME peer membership: reproducing on identical data
means a data/code-dependent defect, while a clean identical replay means the export
was not reproducible and points at non-deterministic execution on the peer GPU.
"""
from copy import deepcopy
import json

import numpy as np
import pytest

from test_heterogeneous_production import native, _trainer, _session, _weights, _assert_tree_equal
from mini_llm.heterogeneous_production import sampler_state

# (bad exports, expected verdict sequence, queued replays, budget-exhausted reports)
CASES = [
    (1, ["clean_replay"], 1, 0),
    (2, ["reproduced_same_parameters", "clean_replay"], 2, 0),
    (3, ["reproduced_same_parameters", "reproduced_same_parameters"], 2, 1),
]


@pytest.mark.parametrize("bad_exports,expected_verdicts,expected_queued,expected_exhausted", CASES)
def test_nonfinite_peer_export_rejects_round_and_replays_identical_data(
    native, tmp_path, monkeypatch, bad_exports, expected_verdicts, expected_queued, expected_exhausted,
):
    monkeypatch.setenv("NATIVE_GPU_LOG", "1")
    directory = tmp_path / "diagnostics"
    monkeypatch.setenv("NATIVE_GPU_LOG_DIR", str(directory))
    owner = _trainer(native, tmp_path)
    before_weights = _weights(owner)
    before_optimizer = deepcopy(owner.optimizer.checkpoint_state())

    with _session(owner, native) as session:
        round_start_sampler = sampler_state(owner)
        offsets = {}
        offset = 0
        for parameter in session.parameters:
            offsets[parameter.name] = offset
            offset += int(parameter.size)
        assert offset == sum(int(p.size) for p in session.parameters)
        first, second = session.parameters[0], session.parameters[-1]
        original_expect = session._expect
        injected = []

        def corrupting_expect(event):
            message = original_expect(event)
            # The peer writes the export buffer before announcing "exported",
            # so corrupting here is exactly what a defective HIP kernel would
            # leave behind.
            if message.get("event") == "exported" and len(injected) < bad_exports:
                injected.append(message["step"])
                buffer = session.buffers["peer-gradient"]
                buffer[offsets[first.name]] = np.inf
                buffer[offsets[second.name]] = np.nan
            return message

        monkeypatch.setattr(session, "_expect", corrupting_expect)

        with np.errstate(over="ignore", invalid="ignore"):
            loss, norm = owner.train_step()
        assert len(injected) == 1
        assert np.isfinite(loss) and np.isnan(norm)
        # Nothing about the round may advance: weights, optimizer, counters.
        assert owner.step == owner.optimizer.step_index == 0
        _assert_tree_equal(_weights(owner), before_weights)
        _assert_tree_equal(owner.optimizer.checkpoint_state(), before_optimizer)
        assert all(not np.any(p.grad) for p in owner.model.parameters())
        assert session.process.poll() is None
        _assert_tree_equal(sampler_state(owner), round_start_sampler)

        # Each queued replay re-runs the same round on identical data, so a
        # recurrence classifies as deterministic rather than unstable. Once the
        # replay budget is spent, the sampler intentionally advances again.
        for attempt in range(bad_exports - 1):
            with np.errstate(over="ignore", invalid="ignore"):
                loss, norm = owner.train_step()
            assert np.isfinite(loss) and np.isnan(norm)
            assert owner.step == owner.optimizer.step_index == 0
            assert session.process.poll() is None
            if attempt + 2 <= expected_queued:
                _assert_tree_equal(sampler_state(owner), round_start_sampler)
            else:
                assert sampler_state(owner)["tokens_processed"] > round_start_sampler["tokens_processed"]

        loss, norm = owner.train_step()
        assert np.isfinite(loss) and np.isfinite(norm)
        assert owner.step == owner.optimizer.step_index == 1
        assert len(injected) == bad_exports

    path, = directory.glob("owner-numpy.*.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    rejected = [row for row in records if row["event"] == "gradient_rejected"]
    assert len(rejected) == bad_exports
    row = rejected[0]
    assert row["completed_step"] == 0 and row["attempted_step"] == 1
    assert row["failure_kind"] == "peer_nonfinite_export"
    report = row["peer_gradient_audit"]
    assert report["all_finite"] is False
    assert [item["name"] for item in report["nonfinite"]] == [first.name, second.name]
    assert report["nonfinite"][0]["inf"] == 1 and report["nonfinite"][0]["nan"] == 0
    assert report["nonfinite"][1]["nan"] == 1 and report["nonfinite"][1]["inf"] == 0
    assert all(item["max_finite_abs"] > 0.0 for item in report["nonfinite"])
    # Rejected exports, then a normal update: the peer saw the same protocol the
    # owner uses for its own skipped rounds.
    syncs = [row for row in records if row["event"] == "weights_synced"]
    assert [row["updated"] for row in syncs] == [False] * bad_exports + [True]
    assert session.process.poll() == 0

    queued = [row for row in records if row["event"] == "peer_replay_queued"]
    verdicts = [row for row in records if row["event"] == "peer_replay_verdict"]
    exhausted = [row for row in records if row["event"] == "peer_replay_exhausted"]
    assert len(queued) == expected_queued
    assert len(exhausted) == expected_exhausted
    assert [row["verdict"] for row in verdicts] == expected_verdicts
    assert all(row["membership_match"] for row in verdicts), "peer membership must be pinned for a clean verdict"

    # The retry must be the same data, on both GPUs: identical sampled inputs per
    # attempt, and a different sample set once the replay budget is spent.
    width = 2
    for pattern, step_width in (("owner-numpy", 2), ("peer-numpy", 1)):
        path, = directory.glob(f"{pattern}.*.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        shas = [row["inputs"]["sha256"] for row in rows
                if row["event"] == "batch_ready" and row["update_step"] == 1 and row["phase"] == "train"]
        attempts = len(shas) // step_width
        assert attempts == bad_exports + 1, pattern
        identical = expected_queued + 1
        groups = [shas[i * step_width:(i + 1) * step_width] for i in range(attempts)]
        assert all(group == groups[0] for group in groups[:identical]), pattern
        if attempts > identical:
            assert groups[identical] != groups[0], pattern
