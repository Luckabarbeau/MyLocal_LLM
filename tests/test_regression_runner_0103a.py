"""Regression suite routing must never silently report GPU tests as passed."""

import os

import pytest

from scripts.run_0103a_regressions import (
    CPU_SUITES, GPU_SUITES, environment_for, suites_for,
)


def test_cpu_suite_includes_baseline_sparse_head_and_checkpoint():
    paths = set(sum((list(v) for v in CPU_SUITES.values()), []))
    assert "tests/test_sparse_stride_history_0102z.py" in paths
    assert "tests/test_baseline_contract_0103a.py" in paths
    assert "tests/test_checkpoint.py" in paths
    assert "tests/test_runtime_defaults.py" in paths


def test_runtime_profile_is_always_its_own_process():
    for suite_name, selectors in CPU_SUITES.items():
        if suite_name == "runtime_profile_isolated":
            assert selectors == ("tests/test_runtime_defaults.py",)
        else:
            assert "tests/test_runtime_defaults.py" not in selectors


def test_environment_is_pinned_and_does_not_leak_external_optimization_flags():
    original = {"MINI_LLM_BACKEND": "cupy", "MINI_LLM_RUNTIME_PROFILE": "gpu-fast",
                "MINI_LLM_NATIVE_CUDA_GRAPH_DECODE": "0", "MINI_LLM_SOMETHING": "1",
                "OTHER_VARIABLE": "keep"}
    before = dict(original)
    cpu = environment_for("cpu", original)
    gpu = environment_for("gpu", original)
    assert original == before
    assert cpu["MINI_LLM_BACKEND"] == "numpy"
    assert gpu["MINI_LLM_BACKEND"] == "cupy"
    assert {k for k in cpu if k.startswith("MINI_LLM_")} == {"MINI_LLM_BACKEND"}
    assert {k for k in gpu if k.startswith("MINI_LLM_")} == {"MINI_LLM_BACKEND"}
    assert cpu["OTHER_VARIABLE"] == "keep"
    assert os.path.isabs(cpu["PYTHONPATH"].split(os.pathsep)[0])


def test_gpu_requires_explicit_run_and_contains_graph_capture_assertion():
    assert all(tier == "cpu" for tier, *_ in suites_for("cpu"))
    assert all(tier == "gpu" for tier, *_ in suites_for("gpu"))
    assert len(suites_for("all")) == len(CPU_SUITES) + len(GPU_SUITES)
    assert any("test_model_b_native_graph_activates_and_matches_fallback" in target
               for suite in GPU_SUITES.values() for target in suite)
    with pytest.raises(ValueError):
        environment_for("invalid", {})


def test_gpu_memory_headroom_uses_isolated_backend_process(monkeypatch):
    from scripts import run_0103a_regressions as regressions

    calls = []

    def fake_run(command, *, cwd, env, capture_output, text):
        from types import SimpleNamespace
        calls.append((command, cwd, env, capture_output, text))
        return SimpleNamespace(returncode=0,
                               stdout='{"free_gib": 5.25, "total_gib": 15.99}\n',
                               stderr="")

    monkeypatch.setattr(regressions.subprocess, "run", fake_run)
    memory = regressions.gpu_memory_headroom({"MINI_LLM_BACKEND": "cupy"})
    assert memory == {"free_gib": 5.25, "total_gib": 15.99}
    assert len(calls) == 1
    assert calls[0][0][0] == regressions.sys.executable
    assert calls[0][2]["MINI_LLM_BACKEND"] == "cupy"


def test_gpu_memory_headroom_reports_cuda_query_failure(monkeypatch):
    from types import SimpleNamespace
    from scripts import run_0103a_regressions as regressions

    monkeypatch.setattr(
        regressions.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="no GPU"),
    )
    with pytest.raises(RuntimeError, match="no GPU"):
        regressions.gpu_memory_headroom({"MINI_LLM_BACKEND": "cupy"})


def test_low_gpu_headroom_is_blocked_not_counted_as_passed(monkeypatch, tmp_path):
    import json
    from scripts import run_0103a_regressions as regressions

    monkeypatch.setattr(regressions, "gpu_preflight", lambda env: None)
    monkeypatch.setattr(
        regressions, "gpu_memory_headroom",
        lambda env: {"free_gib": 0.75, "total_gib": 15.99},
    )
    report = tmp_path / "regression.json"
    assert regressions.run("gpu", report=report) == 1
    payload = json.loads(report.read_text())
    assert not payload["complete"]
    assert payload["suites"][0]["status"] == "blocked"
    assert "0.75 GiB" in payload["suites"][0]["reason"]
