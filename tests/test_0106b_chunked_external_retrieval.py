"""Chunked GGUF retrieval prefill must preserve exact causal last-token logits."""
import numpy as np
import pytest

from benchmarks.evaluate_external_long_retrieval import _prefill_last
from mini_llm.external import ExternalLlamaConfig, ExternalLlamaModel


def tiny_model():
    cfg = ExternalLlamaConfig.from_dict({
        'model_type': 'llama', 'architectures': ['LlamaForCausalLM'],
        'hidden_size': 24, 'intermediate_size': 48, 'num_hidden_layers': 2,
        'num_attention_heads': 3, 'num_key_value_heads': 1, 'head_dim': 8,
        'vocab_size': 49, 'max_position_embeddings': 128,
        'rms_norm_eps': 1e-5, 'rope_theta': 100000.0,
        'hidden_act': 'silu', 'attention_bias': False,
        'tie_word_embeddings': True,
    })
    model = ExternalLlamaModel(cfg)
    rng = np.random.default_rng(61)
    weights = {}
    for name, shape in model.expected_shapes().items():
        if name.endswith('norm.weight') or 'layernorm.weight' in name:
            weights[name] = (np.ones(shape) + rng.normal(size=shape)*0.025).astype('float32')
        else:
            weights[name] = (rng.normal(size=shape)*0.07).astype('float32')
    model.set_weights(weights)
    return model


@pytest.mark.parametrize('length,chunk', [(2,1),(7,1),(7,3),(13,4),(25,9)])
def test_external_chunked_matches_dense_prefill(length,chunk):
    model = tiny_model()
    ids = ((np.arange(length)*3+5) % 49).tolist()
    expected = _prefill_last(model, ids, chunk_threshold=128)
    actual = _prefill_last(model, ids, chunk_size=chunk, chunk_threshold=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-5)


def test_invalid_chunk_settings_fail_early():
    model = tiny_model()
    with pytest.raises(ValueError, match='positive'):
        _prefill_last(model, [1,2,3], chunk_size=0)
    with pytest.raises(ValueError, match='positive'):
        _prefill_last(model, [1,2,3], chunk_threshold=0)
