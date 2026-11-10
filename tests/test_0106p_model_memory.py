"""Header-only allocation accounting and mocked free-VRAM reservation checks."""

import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

if os.environ.get('MINI_LLM_BACKEND', 'numpy').lower() != 'numpy':
    pytest.skip('Resource-only mocks and fixtures run on CPU', allow_module_level=True)

from mini_llm.external import qwen3next_model as module

FIXTURE = runpy.run_path(str(Path(__file__).with_name('test_0106p_qwen3next_model.py')))


def test_free_vram_includes_pending_weights_after_state_allocation(monkeypatch):
    # Mock only the resource guard, never switch the numerical process backend.
    model = object.__new__(module.Qwen3NextGGUFModel)
    model.device_memory_budget_bytes = 4 * 1024**3
    model.device_reserve_bytes = 2 * 1024**3
    model._pending_device_bytes = lambda: 900 * 1024**2
    monkeypatch.setattr(module, 'BACKEND_NAME', 'cupy')
    monkeypatch.setattr(module, 'xp', SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(
        memGetInfo=lambda: (int(3.2 * 1024**3), 16 * 1024**3)))))
    plan = dict(estimated_total_bytes=2200 * 1024**2, workspace_bytes=256 * 1024**2)
    with pytest.raises(MemoryError, match='safety reserve'):
        model._check_memory(plan, 800 * 1024**2)
    model._pending_device_bytes = lambda: 0  # Already-uploaded weights must not be counted twice.
    model._check_memory(plan, 800 * 1024**2)


@pytest.mark.parametrize('suffix', ['attn_norm.weight', 'post_attention_norm.weight',
                                     'attn_q_norm.weight', 'attn_k_norm.weight'])
def test_packed_norm_rejected_during_header_inspection(suffix):
    index, _, _, reads = FIXTURE['fixture']()
    index.tensors['blk.3.' + suffix].tensor_type.name = 'Q8_0'
    with pytest.raises(NotImplementedError, match='normalization'):
        module.inspect_qwen3next_model(index)
    assert reads == []


def test_packed_recurrent_small_weight_budget_is_expanded_fp32():
    index, _, _, reads = FIXTURE['fixture']()
    _, _, _, before = module.inspect_qwen3next_model(index)
    tensor = index.tensors['blk.0.ssm_norm.weight']
    tensor.tensor_type.name = 'Q8_0'
    tensor._data = np.concatenate((np.array([.125], '<f2').view(np.uint8),
                                   np.ones(32, np.int8).view(np.uint8)))
    _, _, _, after = module.inspect_qwen3next_model(index)
    assert after == before  # Both variants retain a 32-element FP32 norm gamma.
    assert reads == []
