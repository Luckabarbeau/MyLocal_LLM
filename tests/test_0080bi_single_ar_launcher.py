"""0080BI: both AR references use one self-contained Bash launcher."""
from pathlib import Path
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_0080BI_ar_reference.sh"


def _run(tmp_path, variant="moe", mode="smoke", *, extra_env=None):
    mix = tmp_path / "shards"
    mix.mkdir(exist_ok=True)
    (mix / "tokenizer.json").write_text("{}")
    (mix / "mixture_manifest.json").write_text("{}")
    fakebin = tmp_path / "bin"
    fakebin.mkdir(exist_ok=True)
    fake_python = fakebin / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf 'MOCK_TRAIN:'\n"
        "printf ' <%s>' \"$@\"\n"
        "printf '\\n'\n"
    )
    fake_python.chmod(0o755)
    env = dict(os.environ, PATH=str(fakebin) + os.pathsep + os.environ["PATH"],
               MODEL_VARIANT=variant, MIX_ROOT=str(mix),
               CKPT=str(tmp_path / "ckpt"), LOG=str(tmp_path / "log.csv"))
    if extra_env:
        env.update(extra_env)
    return subprocess.run(["bash", str(SCRIPT), mode], cwd=ROOT,
                          env=env, capture_output=True, text=True)


@pytest.mark.parametrize("variant,model", [
    ("moe", "wide24-500m-memory-64k"),
    ("dense", "wide24-500m-dense-memory-64k"),
])
def test_independent_variant_smoke_runs_python_directly(tmp_path, variant, model):
    r = _run(tmp_path, variant)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "MOCK_TRAIN: <train_model.py>" in r.stdout
    assert f"<--model> <{model}>" in r.stdout
    assert "<--objective> <ar>" in r.stdout
    assert "<--batch-size> <1>" in r.stdout
    assert "<--grad-accum-steps> <1>" in r.stdout
    assert "<--max-run-steps> <2>" in r.stdout
    assert "<--resume-from>" not in r.stdout


def test_modes_and_overrides(tmp_path):
    r = _run(tmp_path, "dense", "fresh", extra_env={
        "TRAIN_BATCH_SIZE": "2", "GRAD_ACCUM_STEPS": "32", "MINI_LLM_LM_HEAD_CHUNK_TOKENS": "256"
    })
    assert r.returncode == 0, r.stderr
    assert "<--batch-size> <2>" in r.stdout
    assert "<--grad-accum-steps> <32>" in r.stdout
    assert "<--max-run-steps>" not in r.stdout
    assert "ar_lm_head_tile=256" in r.stdout


def test_rejects_invalid_variant(tmp_path):
    r = _run(tmp_path, "foobar")
    assert r.returncode != 0
    assert "MODEL_VARIANT must be" in r.stderr


def test_dense_prevents_moe_cpu_offload(tmp_path):
    r = _run(tmp_path, "dense", extra_env={"MINI_LLM_MOE_OFFLOAD_GATE_UP": "1"})
    assert r.returncode != 0
    assert "Dense FFN" in r.stderr


def test_does_not_chain_scripts():
    content = SCRIPT.read_text()
    assert 'exec python train_model.py "${ARGS[@]}"' in content
    assert 'run_0080BF_wide24_ar_reference.sh' not in content
    assert 'exec bash ' not in content
    assert not (ROOT / 'run_0080BG_wide24_ar_dense_reference.sh').exists()
    assert not (ROOT / 'run_0080BG_wide24_ar_moe_reference.sh').exists()
