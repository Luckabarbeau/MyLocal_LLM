"""Input fingerprints and selected/loaded library provenance, without model execution."""

import hashlib
from pathlib import Path
import subprocess

import pytest

from benchmarks import benchmark_0106q_reference as probe


def test_streaming_fingerprint_and_input_change_detection(tmp_path):
    path = tmp_path / 'model.gguf'
    payload = b'bounded-reference-test' * 1024
    path.write_bytes(payload)
    identity = probe.file_identity(path)
    assert identity['sha256'] == hashlib.sha256(payload).hexdigest()
    assert identity['bytes'] == len(payload)
    probe.verify_input_unchanged(identity)
    path.write_bytes(payload + b'changed')
    with pytest.raises(ValueError, match='input changed'):
        probe.verify_input_unchanged(identity)


def test_loaded_library_identity_is_bound_to_selected_files(tmp_path):
    selected = tmp_path / 'libllama.so'
    selected.write_bytes(b'selected-library')
    provenance = {'linked_library_files': {
        selected.name: {'path': str(selected.resolve()), 'sha256': probe.sha256(selected)}}}
    metadata = {'loaded_reference_libraries': [str(selected)]}
    probe.validate_loaded_libraries(metadata, provenance)
    other = tmp_path / 'other.so'
    other.write_bytes(b'other-library')
    with pytest.raises(ValueError, match='does not match selected build'):
        probe.validate_loaded_libraries({'loaded_reference_libraries': [str(other)]}, provenance)
    selected.write_bytes(b'changed-library')
    with pytest.raises(ValueError, match='does not match selected build'):
        probe.validate_loaded_libraries(metadata, provenance)
    with pytest.raises(ValueError, match='did not identify'):
        probe.validate_loaded_libraries({}, provenance)


def test_selected_build_cannot_use_unrelated_default_build_info(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    default = root / 'build-cuda/common'
    default.mkdir(parents=True)
    (default / 'build-info.cpp').write_text(probe.REVISION)
    selected = tmp_path / 'different-build/bin'
    selected.mkdir(parents=True)
    (selected / 'libllama.so').write_bytes(b'not-executed')
    monkeypatch.setattr(probe.subprocess, 'check_output', lambda *a, **k: probe.REVISION + '\n')
    monkeypatch.setattr(probe.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(probe.shutil, 'which', lambda *a: '/usr/bin/c++')
    with pytest.raises(ValueError, match='Selected build must provide'):
        probe.build_exporter(root, selected, tmp_path, probe.REVISION)
