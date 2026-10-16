"""Production-only benchmark regression: frozen, deterministic, horizon-safe evaluation."""
import numpy as np
import pytest

from benchmarks.evaluate_block_proposal_production import (
    eval_metadata, heldout_benchmark, sample_heldout, summarize_ablation,
    text_windows, wikitext_benchmark,
)
from mini_llm.backend import BACKEND_NAME
from mini_llm.block_proposal.diagnostics import SCOPES
from mini_llm.config import ModelConfig, MemoryContextConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel


def _build(learned=False):
    memory = MemoryContextConfig(
        enabled=True, memory_length=48, recent_length=16, target_length=16,
        block_size=4, top_k_blocks=2, router_query_length=8, router_dim=4,
        integration_mode='routed_prefix', memory_training='joint',
        router_gumbel_noise=False, read_heads=1, read_kv_heads=1)
    config = ModelConfig(tokenizer_vocab_size=67, context_length=memory.active_length,
        n_layers=2, d_model=16, n_q_heads=2, n_kv_heads=1, d_head=8,
        d_ff=32, n_experts=2, top_k=1, memory_context=memory, dtype='float64')
    model = DecoderLanguageModel(config, rng_seed=98, dtype='float64')
    head = HorizonProposalHead(model, block_size=4, hidden_width=12, rng_seed=70)
    dec = BlockPrefixDecoder(model, block_size=4, width=12,
                             n_layers=2, n_heads=3, rng_seed=71)
    return SinglePassBlockTrainer(TwoStageBlockModel(head, dec, lambda_proposal=0.5))


def _sample(batch=2):
    src = np.mod(np.arange(batch*48).reshape(batch, 48)*7+3, 67)
    targets = np.mod(src[:, 32:]+1, 67)
    valid = np.ones((batch, 16), dtype=bool)
    valid[1, 7:] = False
    src[1, 32+7:] = 0
    targets[1, 7:] = -100
    metadata = dict(history_valid_starts=np.asarray([0,20]),
                    target_valid_lengths=np.asarray([16,7]), router_stochastic=False,
                    causal_prefix_query=True, use_retrieval=True)
    return src, targets, valid, metadata
from mini_llm.block_proposal.training import SinglePassBlockTrainer
from mini_llm.block_proposal.diagnostics import score_window

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='CPU correctness tests')


def test_wikitext_windows_score_every_next_token_once_and_never_cross_articles():
    from mini_llm.config import MemoryContextConfig
    memory = MemoryContextConfig(enabled=True, memory_length=12, recent_length=4,
             target_length=4, block_size=2, top_k_blocks=1,
             router_query_length=4, router_dim=2, integration_mode="routed_prefix")
    tokens = list(range(1, 11))
    records = list(text_windows(tokens, memory=memory, eos=99))
    assert len(records) == 3
    flat = [int(x) for _, y, mask, _ in records for x in y[mask]]
    assert flat == tokens[1:]
    for i, (source, labels, valid, md) in enumerate(records):
        assert source.shape == (1, 12)
        assert labels.shape == valid.shape == (1, 4)
        assert int(md['target_valid_lengths'][0]) == int(valid.sum())
        assert int(md['history_valid_starts'][0]) == 8 - min(i*4, 8)


def test_fixed_packed_heldout_sampling(tmp_path):
    data = np.asarray(list(range(1, 46)) + [99] + list(range(12, 75)) + [99], dtype='<u2')
    path = tmp_path / 'val_shard_00000.bin'
    data.tofile(path)
    memory = _build().model.config.memory_context
    def get():
        return list(sample_heldout([path], eos=99, memory=memory,
                     batches=2, batch_size=2, seed=901))
    a, b = get(), get()
    for x, y in zip(a, b):
        for i in range(3):
            np.testing.assert_array_equal(x[i], y[i])
        for k in x[3]:
            np.testing.assert_array_equal(x[3][k], y[3][k])
    assert sum(int(x[2].sum()) for x in a) > 0


def test_ablation_all_equals_production_two_stage_on_same_targets():
    trainer = _build(learned=True)
    source, targets, valid, metadata = _sample()
    reference = trainer.train_window(source, targets, valid_target_mask=valid,
                     memory_metadata=metadata, compute_gradients=False,
                     block_group_size=1)
    result = score_window(trainer, source, targets, valid, metadata, group_size=1)
    scored = summarize_ablation(result['counts'], result['proposal_sums'],
                                 result['final_sums'], 0.5)
    assert sum(scored['count_by_horizon']) == reference['unique_supervised_tokens']
    assert scored['conditions']['all']['proposal_ce'] == pytest.approx(float(reference['proposal_ce']), abs=3e-6)
    assert scored['conditions']['all']['final_ce'] == pytest.approx(float(reference['final_ce']), abs=3e-6)
    assert scored['conditions']['past_and_current']['final_ce_by_horizon'][-1] == pytest.approx(
        scored['conditions']['all']['final_ce_by_horizon'][-1], abs=1e-9)
    assert set(scored['conditions']) == set(SCOPES)


def test_wikitext_real_objective_counts_all_positions(tmp_path):
    class FakeTokenizer:
        def encode(self, text):
            return [1 + (ord(c) % 50) for c in text]
    path = tmp_path / 'wiki.test.raw'
    path.write_text('= A =\n' + 'a'*50 + '\n= B =\n' + 'b'*50)
    trainer = _build()
    scored = wikitext_benchmark(trainer.net, FakeTokenizer(), eos=1, path=path,
                   max_tokens=90, document_mode='article', group_size=2)
    assert scored['block_unique_supervised_tokens'] == scored['tokenized_tokens']-scored['documents']
    assert sum(scored[f'h{i}_count'] for i in range(1, 5)) == scored['block_unique_supervised_tokens']
    assert np.isfinite(scored['final_perplexity'])


def test_load_real_production_layout_requires_complete_two_stage_weights(tmp_path, monkeypatch):
    import dataclasses
    import json
    from mini_llm.checkpoint import save_checkpoint
    from mini_llm.block_proposal.checkpoint import architecture_metadata
    from benchmarks import evaluate_block_proposal_production as b

    original = _build()
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text(json.dumps(dataclasses.asdict(original.model.config)))
    (checkpoint / 'tokenizer.json').write_text('{}')

    class StubTokenizer:
        eos_token = '<|eos|>'
        def __len__(self):
            return original.model.config.vocab_size
        class _Inner:
            def token_to_id(self, text):
                return 1
        _tokenizer = _Inner()

    monkeypatch.setattr(b, 'load_tokenizer', lambda checkpoint: (
        StubTokenizer(), checkpoint / 'tokenizer.json'))
    save_checkpoint(checkpoint, {p.name:p.data for p in original.parameters()},
        training_state={'step': 23, 'prediction_objective': 'block-proposal',
                        'architecture': architecture_metadata(original.net),
                        'tokenizer_hash': b.sha256(checkpoint / 'tokenizer.json')})
    net, tokenizer, eos, state, sources, tokenizer_path = b.load_production(checkpoint)
    assert eos == 1 and not sources and state['step'] == 23
    assert tokenizer_path == checkpoint / 'tokenizer.json'
    for a, z in zip(original.parameters(), net.parameters()):
        assert a.name == z.name
        np.testing.assert_array_equal(a.data, z.data)
    # A trunk-only checkpoint must NEVER produce plausible-looking CE metrics.
    (checkpoint / 'model_params' / f'{original.net.decoder.parameters()[0].name}.npy').unlink()
    with pytest.raises(ValueError, match='missing production weights'):
        b.load_production(checkpoint)


def test_fast_heldout_matches_full_ablation_all_condition(tmp_path):
    from types import SimpleNamespace
    from mini_llm.config import MemoryContextConfig
    from benchmarks.evaluate_block_proposal_production import heldout_benchmark
    tokens = np.asarray(list(range(1, 46)) + [66] + list(range(1, 46)) + [66], dtype='<u2')
    path = tmp_path / 'val_shard_00000.bin'
    tokens.tofile(path)
    source = SimpleNamespace(name='test_corpus', val_paths=[path])
    model = _build()
    full = heldout_benchmark(model.net, [source], 66,
                            batches=1, batch_size=1, seed=78078, group_size=2,
                            ablations=True)
    fast = heldout_benchmark(model.net, [source], 66,
                            batches=1, batch_size=1, seed=78078, group_size=2,
                            ablations=False)
    assert full['test_corpus']['validation_sample_sha256'] == fast['test_corpus']['validation_sample_sha256']
    assert fast['test_corpus']['conditions'].keys() == {'all'}
    np.testing.assert_allclose(
        fast['test_corpus']['conditions']['all']['final_ce_by_horizon'],
        full['test_corpus']['conditions']['all']['final_ce_by_horizon'],
        atol=5e-6)
    np.testing.assert_allclose(
        fast['test_corpus']['conditions']['all']['proposal_ce_by_horizon'],
        full['test_corpus']['conditions']['all']['proposal_ce_by_horizon'],
        atol=5e-6)


def test_generation_uses_trained_two_stage_causal_commit():
    from benchmarks.evaluate_block_proposal_production import generation_benchmark

    class StubTokenizer:
        eos_token = '<|eos|>'
        token_to_id = {'<|eos|>': 66}

        def encode(self, text):
            return [3, 4, 5, 6]

        def decode(self, ids):
            return ' '.join(str(i) for i in ids)

    result = generation_benchmark(_build().net, StubTokenizer(),
                                  prompt='smoke', max_new_tokens=4, commit_stride=2)
    assert 1 <= len(result['generated_token_ids']) <= 4
    assert result['generation']['causal_main_updates'] == len(result['generated_token_ids'])
    assert result['generation']['proposal_horizon'] == 4
    assert result['generation']['implementation'] == 'sequential_main_kv'


def test_validation_retrieval_policy_matches_production_eligibility():
    memory = _build().model.config.memory_context
    # Full target + genuine history requests retrieval.
    full = eval_metadata([16, 16], [0, 10], memory=memory)
    assert full['use_retrieval'] is True
    # Production disables the routed prefix for partial-document targets.
    partial = eval_metadata([16, 7], [0, 10], memory=memory)
    assert partial['use_retrieval'] is False
    # Original 0080W forced retrieval even for these partial targets.
    forced = eval_metadata([16, 7], [0, 10], memory=memory,
                           retrieval_policy='force')
    assert forced['use_retrieval'] is True
    # No genuine history: production cannot route.
    empty_history = eval_metadata([16, 16], [48, 0], memory=memory)
    assert empty_history['use_retrieval'] is False


def test_stochastic_router_comparison_is_sample_paired(tmp_path):
    data = np.asarray(list(range(1, 46)) + [66] + list(range(1, 46)) + [66], dtype='<u2')
    path = tmp_path / 'val_shard_00000.bin'
    data.tofile(path)
    memory = _build().model.config.memory_context
    args = dict(eos=66, memory=memory, batches=2, batch_size=2, seed=402)
    a = list(sample_heldout([path], **args))
    b = list(sample_heldout([path], **args, stochastic=True, checkpoint_step=123))
    for aa, bb in zip(a, b):
        for j in range(3):
            np.testing.assert_array_equal(aa[j], bb[j])
        assert aa[3]['use_retrieval'] == bb[3]['use_retrieval']
        assert aa[3]['router_stochastic'] is False
        assert bb[3]['router_stochastic'] is True
        assert isinstance(bb[3]['router_gumbel_seed'], int)
        assert memory.router_temperature_min <= bb[3]['router_temperature'] <= memory.router_temperature


def test_production_vs_forced_retrieval_changes_only_metadata(tmp_path):
    data = np.asarray(list(range(1, 13)) + [66] + list(range(1, 13)) + [66], dtype='<u2')
    path = tmp_path / 'val_shard_00000.bin'
    data.tofile(path)
    memory = _build().model.config.memory_context
    args = dict(eos=66, memory=memory, batches=1, batch_size=2, seed=35)
    a = next(sample_heldout([path], **args))
    b = next(sample_heldout([path], **args, retrieval_policy='force'))
    for j in range(3):
        np.testing.assert_array_equal(a[j], b[j])
    assert a[3]['use_retrieval'] is False
    assert b[3]['use_retrieval'] is True
