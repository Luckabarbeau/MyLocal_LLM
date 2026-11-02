"""Checkpoint publication must not corrupt the last complete version."""
import json
import os

import numpy as np
import pytest

from mini_llm.checkpoint import save_checkpoint, load_checkpoint


def test_atomic_overwrite_and_nested_scalars(tmp_path):
    path = tmp_path / 'model'
    path.mkdir()
    (path / 'config.json').write_text('{"architecture": "test"}')
    save_checkpoint(path, {'w': np.array([1., 2.])}, {
        'step': 7, 'lr': 0.001, 'beta': 0.95,
        'm': {'w': np.array([3.]), 'iteration': 9},
        'private': {'some_tensor': np.array([4.]), 'count': 5},
    }, {'supervised_tokens': 123})
    save_checkpoint(path, {'w': np.array([5., 6.])}, {'step': 8}, {'supervised_tokens': 456})
    assert (path / 'config.json').read_text() == '{"architecture": "test"}'
    params, optimizer, training = load_checkpoint(path)
    np.testing.assert_array_equal(params['w'], [5, 6])
    assert optimizer == {'step': 8}
    assert training['supervised_tokens'] == 456
    previous_params, previous_optimizer, _ = load_checkpoint(tmp_path / 'model.previous')
    np.testing.assert_array_equal(previous_params['w'], [1, 2])
    assert previous_optimizer['step'] == 7 and previous_optimizer['lr'] == 0.001
    assert previous_optimizer['beta'] == 0.95
    assert previous_optimizer['m']['iteration'] == 9
    np.testing.assert_array_equal(previous_optimizer['private']['some_tensor'], [4])
    assert previous_optimizer['private']['count'] == 5


def test_failed_payload_does_not_touch_active_checkpoint(tmp_path, monkeypatch):
    import mini_llm.checkpoint as checkpoint
    path = tmp_path / 'model'
    save_checkpoint(path, {'w': np.array([1])}, training_state={'step': 1})
    def fail(*args, **kwargs):
        raise OSError('simulated disk write failure')
    monkeypatch.setattr(checkpoint, '_write_payload', fail)
    with pytest.raises(OSError, match='disk write'):
        save_checkpoint(path, {'w': np.array([2])})
    params, _, state = load_checkpoint(path)
    assert params['w'][0] == 1 and state['step'] == 1
    assert not list(tmp_path.glob('model.pending-*'))


def test_failed_publish_restores_previous(tmp_path, monkeypatch):
    import mini_llm.checkpoint as checkpoint
    path = tmp_path / 'model'
    save_checkpoint(path, {'w': np.array([1])}, training_state={'step': 1})
    original_replace = checkpoint.os.replace
    def fail_publish(src, dst):
        if '.pending-' in str(src) and dst == path:
            raise OSError('simulated publish interruption')
        return original_replace(src, dst)
    monkeypatch.setattr(checkpoint.os, 'replace', fail_publish)
    with pytest.raises(OSError, match='publish interruption'):
        save_checkpoint(path, {'w': np.array([2])})
    params, _, state = load_checkpoint(path)
    assert params['w'][0] == 1 and state['step'] == 1


def test_recovery_after_interrupted_rename(tmp_path):
    path = tmp_path / 'model'
    save_checkpoint(path, {'w': np.array([1])})
    os.replace(path, tmp_path / 'model.previous')
    params, _, _ = load_checkpoint(path)
    assert params['w'][0] == 1
