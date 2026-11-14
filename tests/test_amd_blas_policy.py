"""CPU-only policy and isolated process-start tests; no GPU packages required."""

import importlib.util
import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "amd_blas_policy", ROOT / "mini_llm" / "amd_blas_policy.py"
)
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


@pytest.mark.parametrize("policy", [None, "auto", "tensile", "hipblaslt"])
@pytest.mark.parametrize("raw", [None, "0", "1", "external-value"])
def test_precedence_and_logging(monkeypatch, caplog, policy, raw):
    for name, value in [("MINI_LLM_AMD_BLAS", policy),
                        ("ROCBLAS_USE_HIPBLASLT", raw)]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with caplog.at_level(logging.INFO):
        POLICY.apply_amd_blas_policy()
    expected = {"tensile": "0", "hipblaslt": "1"}.get(policy, raw)
    assert os.environ.get("ROCBLAS_USE_HIPBLASLT") == expected
    if policy in (None, "auto"):
        assert not caplog.records
    else:
        assert f"MINI_LLM_AMD_BLAS={policy}" in caplog.text
        assert f"ROCBLAS_USE_HIPBLASLT={expected}" in caplog.text


@pytest.mark.parametrize("policy", ["", "invalid", "AUTO", " tensile "])
def test_invalid_is_nonmutating(monkeypatch, caplog, policy):
    monkeypatch.setenv("MINI_LLM_AMD_BLAS", policy)
    monkeypatch.setenv("ROCBLAS_USE_HIPBLASLT", "external-value")
    with pytest.raises(ValueError, match="MINI_LLM_AMD_BLAS must be"):
        POLICY.apply_amd_blas_policy()
    assert os.environ["ROCBLAS_USE_HIPBLASLT"] == "external-value"
    assert not caplog.records


STARTUP = r'''
import builtins
import importlib
import os
from pathlib import Path
import sys
import types

# Load the production backend without unrelated package __init__ imports.
package = types.ModuleType("mini_llm")
package.__path__ = [str(Path.cwd() / "mini_llm")]
sys.modules["mini_llm"] = package
original_import = builtins.__import__
seen = []
def mock_import(name, *args, **kwargs):
    if name == "cupy":
        seen.append(os.environ.get("ROCBLAS_USE_HIPBLASLT"))
        return types.ModuleType("cupy")
    if name == "torch":
        return types.ModuleType("torch")
    return original_import(name, *args, **kwargs)
builtins.__import__ = mock_import
raw = os.environ.get("ROCBLAS_USE_HIPBLASLT")
try:
    backend = importlib.import_module("mini_llm.backend")
except ValueError as exc:
    assert os.environ["MINI_LLM_BACKEND"] == "cupy"
    assert "MINI_LLM_AMD_BLAS must be" in str(exc)
    assert not seen
    assert os.environ.get("ROCBLAS_USE_HIPBLASLT") == raw
else:
    assert os.environ["MINI_LLM_AMD_BLAS"] != "invalid" or backend.BACKEND_NAME != "cupy"
    expected = raw
    if backend.BACKEND_NAME == "cupy":
        expected = {"tensile": "0", "hipblaslt": "1"}.get(
            os.environ["MINI_LLM_AMD_BLAS"], raw)
        assert seen == [expected]  # Policy was applied before CuPy import.
    else:
        assert not seen
        assert "mini_llm.amd_blas_policy" not in sys.modules
    assert os.environ.get("ROCBLAS_USE_HIPBLASLT") == expected
    os.environ["MINI_LLM_AMD_BLAS"] = "tensile"
    importlib.import_module("mini_llm.backend")
    assert os.environ.get("ROCBLAS_USE_HIPBLASLT") == expected
    assert len(seen) == (1 if backend.BACKEND_NAME == "cupy" else 0)
'''


@pytest.mark.parametrize("backend", ["numpy", "cupy", "pytorch"])
@pytest.mark.parametrize("policy", ["auto", "tensile", "hipblaslt", "invalid"])
@pytest.mark.parametrize("raw", [None, "external-value"])
def test_process_start(backend, policy, raw):
    env = dict(os.environ, MINI_LLM_BACKEND=backend, MINI_LLM_AMD_BLAS=policy)
    env.pop("ROCBLAS_USE_HIPBLASLT", None)
    if raw is not None:
        env["ROCBLAS_USE_HIPBLASLT"] = raw
    result = subprocess.run(
        [sys.executable, "-c", STARTUP], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
