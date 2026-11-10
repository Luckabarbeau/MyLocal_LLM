"""CPU full-model logits oracle, hybrid session lifecycle and loader integration."""

import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

if os.environ.get('MINI_LLM_BACKEND', 'numpy').lower() != 'numpy':
    pytest.skip('Model FP64 fixtures are CPU-only; GPU uses an isolated CPU export', allow_module_level=True)

from mini_llm.external.qwen3next_model import Qwen3NextGGUFModel, inspect_qwen3next_model
from mini_llm.generation import GenerationSession
from mini_llm.serving_models import ServingModel

ROOT = Path(__file__).resolve().parents[1]
LAYER = runpy.run_path(str(ROOT / 'tests/test_0106o_qwen3next_decoder.py'))
Field, Tensor = LAYER['Field'], LAYER['Tensor']


def fixture(*, tied=False, embedding_kind='Q8_0', head_kind='Q8_0'):
    index, weights, reads = LAYER['fixture'](packed_shared=True)
    index.reader.fields.update({'qwen3next.context_length': Field(32),
                               'qwen3next.vocab_size': Field(64)})
    globals_ = {}
    rng = np.random.default_rng(10616)
    for name, shape, kind in (
        ('token_embd.weight', (64, 32), embedding_kind),
        ('output_norm.weight', (32,), 'F32'),
        *(([] if tied else [('output.weight', (64, 32), head_kind)])),
    ):
        values = rng.normal(0, .2, shape).astype(np.float32)
        if name == 'output_norm.weight':
            values = np.linspace(.7, 1.3, 32, dtype=np.float32)
        if kind == 'Q8_0':
            q = rng.integers(-7, 8, (values.size // 32, 32), dtype=np.int8)
            scale = np.full((len(q), 1), .03125, dtype='<f2')
            payload = np.concatenate((scale.view(np.uint8), q.view(np.uint8)), axis=1).reshape(-1)
            values = (scale.astype(np.float32) * q).reshape(shape)
        elif kind == 'BF16':
            words = (values.view(np.uint32) >> 16).astype('<u2')
            payload = words.view(np.uint8).reshape(-1)
            values = (words.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        elif kind == 'F16':
            payload = values.astype(np.float16)
            values = payload.astype(np.float32)
        else:
            payload = values.copy()
        tensor = Tensor(name, shape, payload, reads, kind)
        index.reader.tensors.append(tensor)
        index.tensors[name] = tensor
        globals_[name] = values
    assert reads == []
    return index, weights, globals_, reads


def reference(ids, model, weights, globals_):
    hidden = globals_['token_embd.weight'][ids].astype(np.float64)
    for block in model.blocks:
        hidden, _, _ = LAYER['reference_layer'](hidden, weights[block.layer_id], block)
    hidden = LAYER['rms'](hidden, globals_['output_norm.weight'], model.attention_spec.rms_eps)
    head = globals_['token_embd.weight' if model.config.tie_word_embeddings else 'output.weight']
    return hidden @ head.astype(np.float64).T


@pytest.mark.parametrize('tied', [False, True])
@pytest.mark.parametrize('embedding_kind', ['Q8_0', 'F32', 'F16', 'BF16'])
def test_full_causal_logits_prefill_chunk_and_incremental_fp64(tied, embedding_kind):
    index, weights, globals_, reads = fixture(tied=tied, embedding_kind=embedding_kind)
    with Qwen3NextGGUFModel.from_index(index, cpu_workers=2) as model:
        assert not any('_exps.weight' in name for name in reads)
        assert 'token_embd.weight' not in reads and 'output.weight' not in reads
        ids = np.array([[1, 4, 2, 7, 1], [3, 2, 9, 3, 4]], np.int32)
        expected = reference(ids, model, weights, globals_)
        full = model.forward(ids)
        np.testing.assert_allclose(full, expected, rtol=4e-4, atol=8e-6)
        state = model.create_generation_state(2, 8)
        np.testing.assert_allclose(model.prefill(ids[:, :2], state), expected[:, 1], rtol=4e-4, atol=8e-6)
        fork = model.copy_generation_state(state)
        for t in range(2, 5):
            np.testing.assert_allclose(model.decode_one(ids[:, t], state), expected[:, t], rtol=4e-4, atol=8e-6)
        np.testing.assert_allclose(model.prefill(ids[:, 2:], fork), expected[:, -1], rtol=4e-4, atol=8e-6)
        assert state.length == fork.length == 5
        assert all(layer.length == 5 for layer in state.layer_states)
        state.reset()
        assert state.length == 0 and all(layer.length == 0 for layer in state.layer_states)
        np.testing.assert_allclose(model.prefill(ids, state), expected[:, -1], rtol=4e-4, atol=8e-6)
        # Causal outputs do not depend on a changed suffix.
        other = ids.copy()
        other[:, 3:] = 23
        np.testing.assert_array_equal(model.forward(other)[:, :3], full[:, :3])
        assert model.embedding.statistics()['used_bytes'] <= 4 * 1024**2


def test_invalid_inputs_horizons_and_state_owner_do_not_mutate():
    index, _, _, _ = fixture()
    with Qwen3NextGGUFModel.from_index(index) as model:
        with pytest.raises(ValueError, match='Explicit max_length'):
            model.create_generation_state()
        with pytest.raises(ValueError, match='context_length'):
            model.create_generation_state(1, 33)
        state = model.create_generation_state(1, 2)
        for ids, error in (([[1, 64]], (ValueError, IndexError)), ([[-1]], (ValueError, IndexError)),
                           ([[1.2]], TypeError), ([[True]], TypeError), ([[]], ValueError)):
            with pytest.raises(error):
                model.prefill(ids, state)
            assert state.length == 0 and not state.failed
            assert all(layer.length == 0 for layer in state.layer_states)
        model.prefill([[1, 2]], state)
        with pytest.raises(ValueError, match='capacity'):
            model.decode_one([3], state)
        state.layer_states[0].mixer_state.conv.length += 1
        with pytest.raises(ValueError, match='positions differ'):
            model.decode_one([3], state)
        state.reset()
        with Qwen3NextGGUFModel.from_index(index) as other:
            with pytest.raises(ValueError, match='another model'):
                other.prefill([[1]], state)
    with pytest.raises(RuntimeError, match='closed'):
        model.create_generation_state(1, 2)


def test_partial_failure_poison_reset_and_recovery(monkeypatch):
    index, _, _, _ = fixture()
    with Qwen3NextGGUFModel.from_index(index) as model:
        state = model.create_generation_state(1, 5)
        def fail(*args, **kwargs):
            raise RuntimeError('simulated expert failure')
        with monkeypatch.context() as patch:
            patch.setattr(model.blocks[1].moe, 'forward', fail)
            with pytest.raises(RuntimeError, match='simulated expert failure'):
                model.prefill([[1, 2]], state)
        assert state.failed and state.length == 0 and state.layer_states[0].length == 2
        with pytest.raises(RuntimeError, match='reset or discard'):
            model.decode_one([3], state)
        state.reset()
        result = model.prefill([[1, 2]], state)
        np.testing.assert_array_equal(result, model.forward([[1, 2]])[:, -1])


@pytest.mark.parametrize('mutation,error', [
    ('vocab', ValueError), ('head', ValueError), ('missing_norm', ValueError),
    ('unknown_tensor', NotImplementedError), ('expert_format', NotImplementedError),
])
def test_header_rejections_before_any_payload(mutation, error):
    index, _, _, reads = fixture()
    if mutation == 'vocab':
        index.reader.fields['qwen3next.vocab_size'] = Field(63)
    elif mutation == 'head':
        index.tensors['output.weight'].shape = (31, 64)
    elif mutation == 'missing_norm':
        del index.tensors['output_norm.weight']
    elif mutation == 'unknown_tensor':
        index.tensors['ngram_embedding.weight'] = Tensor('ngram_embedding.weight', (64, 32), None, reads)
    else:
        index.tensors['blk.3.ffn_down_exps.weight'].tensor_type.name = 'IQ1_M'
    with pytest.raises(error):
        Qwen3NextGGUFModel.from_index(index)
    assert reads == []


def test_memory_plan_header_only_and_executor_ownership():
    index, _, _, reads = fixture()
    config, _, _, resident = inspect_qwen3next_model(index)
    assert config.vocab_size == 64 and resident > 0 and reads == []
    external = LAYER['executor'](index, workers=2)
    model = Qwen3NextGGUFModel.from_index(index, expert_executor=external)
    plan = model.memory_plan(2, 8)
    assert plan['state_bytes'] > 0 and plan['resident_weight_bytes'] == resident
    model.close()
    assert external.overlaps_cpu_gpu  # Caller retains its shared executor.
    external.close()


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[1, 2])
    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(65 + int(i) % 26) for i in ids)
    def token_to_id(self, text):
        return None


def test_existing_generation_session_and_loader_dispatch(monkeypatch, tmp_path):
    import gguf
    from mini_llm.external.gguf_loader import load_gguf_model
    from mini_llm.external import gguf_tokenizer
    from mini_llm.model_loading import load_pretrained_weights
    index, _, _, _ = fixture()
    path = tmp_path / 'synthetic.gguf'
    path.write_bytes(b'fixture reader supplied by test')
    monkeypatch.setattr(gguf, 'GGUFReader', lambda path: index.reader)
    monkeypatch.setattr(gguf_tokenizer, 'load_gguf_tokenizer', lambda reader, **kwargs: TinyTokenizer())
    with load_gguf_model(path, cpu_workers=2)[0] as model:
        adapter = ServingModel(model, TinyTokenizer(), 16)
        session = GenerationSession(adapter)
        expected = []
        state = model.create_generation_state(1, 5)
        logits = model.prefill([[1, 2]], state)
        for step in range(3):
            token = int(np.argmax(logits[0]))
            expected.append(token)
            if step < 2:
                logits = model.decode_one([token], state)
        outcome = session.generate('raw completion', max_tokens=3, temperature=0)
        assert list(outcome.token_ids) == expected and outcome.reason == 'length'
        assert session.state.length == 4
        assert session.execution_info().backend == 'numpy'
    loaded = load_pretrained_weights(path, format='gguf', dtype='float32')
    assert isinstance(loaded.engine, Qwen3NextGGUFModel) and loaded.context_length == 32
    loaded.engine.close()
    with pytest.raises(ValueError, match='float32'):
        load_gguf_model(path, dtype='float16')
