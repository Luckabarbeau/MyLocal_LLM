"""Diagnostic entrypoints fail closed before model/CUDA imports or output creation."""
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', ['probe_0291_resident_host', 'probe_0292_miss_split',
                                  'probe_0266_critical_path'])
def test_no_work_before_isolation(name, tmp_path, monkeypatch):
    probe = load(name)
    import run_0290_isolated_benchmark as guard
    output = tmp_path / 'new'
    def refuse():
        raise RuntimeError('test refused isolation')
    monkeypatch.setattr(guard, 'require_isolation', refuse)
    monkeypatch.setattr(sys, 'argv', ['probe', '--output', str(output)])
    with pytest.raises(RuntimeError, match='test refused isolation'):
        probe.main()
    assert not output.exists()


@pytest.mark.parametrize('options', [('--source-mib', '0'), ('--source-mib', '513'),
                                      ('--workers', '0'), ('--workers', '24')])
def test_resident_probe_rejects_unbounded_budget(options, tmp_path, monkeypatch):
    probe = load('probe_0291_resident_host')
    monkeypatch.setattr(sys, 'argv', ['probe', '--output', str(tmp_path / 'new'), *options])
    with pytest.raises(SystemExit):
        probe.main()
    assert not (tmp_path / 'new').exists()
