"""Synthetic metadata tests; no 50GB model or GGUF library required."""

import numpy as np
import pytest

from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
from mini_llm.ops.expert_residency import ExpertKey, HeterogeneousExpertExecutor
from mini_llm.ops.quantized_weight import PackedGGUFWeight


class Field:
    def __init__(self, value): self.value = value
    def contents(self, index): return self.value


class Kind:
    def __init__(self, name): self.name = name


class Tensor:
    def __init__(self, name, shape, kind='Q4_0'):
        self.name, self.shape, self.tensor_type = name, tuple(reversed(shape)), Kind(kind)
        from mini_llm.ops.quantized_weight import BLOCK_FORMATS
        qk, qb = BLOCK_FORMATS[kind]
        self.data = np.zeros(int(np.prod(shape)) // qk * qb, dtype=np.uint8)


class Reader:
    def __init__(self, fused=False, kind='Q4_0'):
        self.fields = {k: Field(v) for k, v in {
            'general.architecture': 'qwen3next',
            'qwen3next.block_count': 2,
            'qwen3next.embedding_length': 256,
            'qwen3next.expert_count': 3,
            'qwen3next.expert_used_count': 2,
        }.items()}
        self.tensors = []
        for i in range(2):
            stem = f'blk.{i}.'
            self.tensors += [Tensor(stem + 'ffn_down_exps.weight', (3, 256, 256), kind),
                             Tensor(stem + 'ffn_gate_inp.weight', (3, 256), kind)]
            if fused:
                self.tensors.append(Tensor(stem + 'ffn_gate_up_exps.weight', (3, 512, 256), kind))
            else:
                self.tensors += [Tensor(stem + 'ffn_gate_exps.weight', (3, 256, 256), kind),
                                 Tensor(stem + 'ffn_up_exps.weight', (3, 256, 256), kind)]


@pytest.mark.parametrize('fused', [True, False])
@pytest.mark.parametrize('format', ['Q4_0', 'Q3_K'])
def test_index_mmap_expert_provider(fused, format):
    idx = Qwen3NextGGUFIndex(Reader(fused, format))
    w = idx.provider(ExpertKey('test', 1, 2))
    assert len(w) == 3
    assert all(isinstance(weight, PackedGGUFWeight) for weight in w)
    assert [weight.shape for weight in w] == [(256, 256)] * 3
    assert all(not weight.data.flags.owndata for weight in w)
    stats = idx.report(device_budget_gib=0.001)
    assert stats['layers'] == 2
    assert stats['estimated_device_cache_experts'] > 0
    assert not stats['full_model_decode_available']
    executor = HeterogeneousExpertExecutor(idx.provider, model_id='test', d_model=256,
                                           d_ff=256, device_budget_bytes=0)
    x = np.ones((1, 256), dtype=np.float32)
    y = executor.forward(1, 2, x)
    assert y.shape == (1, 256)
    assert np.isfinite(y).all()
    executor.close()


def test_bad_architecture_and_missing_layers():
    reader = Reader()
    reader.fields['general.architecture'] = Field('llama')
    with pytest.raises(ValueError, match='Expected Qwen3-Next'):
        Qwen3NextGGUFIndex(reader)
    reader = Reader()
    reader.tensors = reader.tensors[4:]
    with pytest.raises(ValueError, match='Missing blk.0'):
        Qwen3NextGGUFIndex(reader)


def test_no_indexing_unrecognized_formats():
    reader = Reader()
    down = next(t for t in reader.tensors if t.name == 'blk.0.ffn_down_exps.weight')
    down.tensor_type = Kind('IQ3_S')
    idx = Qwen3NextGGUFIndex(reader)
    assert idx.report()['unsupported_routed_formats']['IQ3_S'] == 1
    with pytest.raises(NotImplementedError, match='IQ3_S'):
        idx.expert_weights(0, 0)


def test_bad_expert_indices():
    idx = Qwen3NextGGUFIndex(Reader())
    with pytest.raises(IndexError):
        idx.expert_weights(-1, 0)
    with pytest.raises(IndexError):
        idx.expert_weights(0, 3)
