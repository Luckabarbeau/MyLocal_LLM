"""CPU-only safety checks for the opt-in WP9 probe; no model/library loading."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from scripts import profile_native_qwen_decode as probe
from scripts.build_native_decode_diagnostic import replace_once


@pytest.mark.parametrize('mask', ['', '0', '1', probe.GPU + ',GPU-other'])
def test_reject_other_masks(monkeypatch, tmp_path, mask):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', mask)
    monkeypatch.setattr(sys, 'argv', ['probe', '--output', str(tmp_path / 'run')])
    execute = Mock()
    monkeypatch.setattr(probe.subprocess, 'check_output', execute)
    with pytest.raises(SystemExit):
        probe.main()
    execute.assert_not_called()


def test_busy_preflight_never_launches(monkeypatch, tmp_path):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', probe.GPU)
    monkeypatch.setattr(sys, 'argv', ['probe', '--output', str(tmp_path / 'run')])
    query = Mock(side_effect=['index, uuid, name, driver, used, free\n1, uuid, gpu, 580, 16 MiB, 15825 MiB\n',
                             probe.GPU + ', 1234, external, 1024 MiB\n'])
    monkeypatch.setattr(probe.subprocess, 'check_output', query)
    launch = Mock()
    monkeypatch.setattr(probe.subprocess, 'Popen', launch)
    with pytest.raises(RuntimeError, match='externally occupied'):
        probe.main()
    launch.assert_not_called()
    assert all(call.args[0][1:3] == ['-i', probe.GPU] for call in query.call_args_list)


def test_detached_owned_pid_cleanup(monkeypatch, tmp_path):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', probe.GPU)
    # Use our receipt but mock signals: verifies start-time matching without
    # actually creating, signaling, or terminating any subprocess.
    receipt = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21])
    (tmp_path / 'profile_pid.json').write_text(json.dumps(receipt))
    child = Mock(pid=987654, returncode=-9)
    child.wait.side_effect = [subprocess.TimeoutExpired('probe', 1), None]
    monkeypatch.setattr(probe.subprocess, 'Popen', Mock(return_value=child))
    kill, killpg = Mock(), Mock()
    monkeypatch.setattr(probe.os, 'kill', kill)
    monkeypatch.setattr(probe.os, 'killpg', killpg)
    with pytest.raises(RuntimeError, match='Owned command failed'):
        probe.run_owned(['fake-probe'], tmp_path / 'profile_launch.json', time.monotonic()+1)
    kill.assert_called_once_with(os.getpid(), probe.signal.SIGKILL)
    assert all(call.args[0] == child.pid for call in killpg.call_args_list)
    record = json.loads((tmp_path / 'profile_launch.json').read_text())
    assert record['timeout'] and record['detached_worker_cleanup'] == receipt


def test_diagnostic_edits_refuse_ambiguous_source():
    assert replace_once('abc', 'b', 'B') == 'aBc'
    with pytest.raises(AssertionError):
        replace_once('bb', 'b', 'B')
    with pytest.raises(AssertionError):
        replace_once('abc', 'missing', 'B')
