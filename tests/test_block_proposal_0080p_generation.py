"""0080P1: actual two-stage token generation, not teacher-forced text."""

import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.inference_model import InferenceModel
from mini_llm.block_proposal.generation import (
    SequentialCommitBlockGenerator, DirectCachedBlockGenerator)
from tests.test_block_proposal_0080c import _joint


def test_sequential_commit_matches_validated_chunk_generator_dense():
    net, _, _, _ = _joint()
    model = net.proposal_head.model
    a = InferenceModel(model.config, dtype=model.dtype)
    b = InferenceModel(model.config, dtype=model.dtype)
    a.set_weights(model)
    b.set_weights(model)
    sa = a.create_generation_state(1, 16)
    sb = b.create_generation_state(1, 16)
    prompt = xp.asarray([[1, 2, 3, 4]], dtype=xp.int32)
    a.prefill(prompt, sa)
    b.prefill(prompt, sb)
    one = DirectCachedBlockGenerator(a, net.proposal_head, net.decoder)
    two = SequentialCommitBlockGenerator(b, net.proposal_head, net.decoder,
                                         chooser=lambda logits: int(xp.argmax(logits)))
    expected = one.generate(sa, 6)
    actual = two.generate(sb, 6)
    assert actual['tokens'] == expected['tokens']
    assert actual['proposal_blocks'] == expected['main_updates'] == 2
    assert actual['causal_main_updates'] == 6
    assert sa.length == sb.length == 10
    np.testing.assert_allclose(asnumpy(a.last_hidden), asnumpy(b.last_hidden), rtol=1e-10, atol=1e-10)


def test_eos_partial_block_and_invalid_sampler():
    net, _, _, _ = _joint()
    model = net.proposal_head.model
    inf = InferenceModel(model.config, dtype=model.dtype)
    inf.set_weights(model)
    state = inf.create_generation_state(1, 12)
    inf.prefill(xp.asarray([[1,2,3]], dtype=xp.int32), state)
    gen = SequentialCommitBlockGenerator(inf, net.proposal_head, net.decoder,
                                          chooser=lambda logits: 7, eos_token_id=7)
    result = gen.generate(state, 8)
    assert result['tokens'] == [7]
    assert result['eos'] and result['proposal_blocks'] == 1
    assert state.length == 4
    gen_bad = SequentialCommitBlockGenerator(inf, net.proposal_head, net.decoder,
                                              chooser=lambda logits: 19)
    with pytest.raises(ValueError, match='invalid token ID'):
        gen_bad.generate(state, 1)
    assert state.length == 4


def test_short_sparse_routed_model_proposals_update_causal_cache():
    from types import SimpleNamespace
    from examples.train_block_proposal_routed_text import _model_config
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.block_proposal.proposals import HorizonProposalHead
    from mini_llm.block_proposal.decoder import BlockPrefixDecoder
    cfg = _model_config(SimpleNamespace(model_preset='cpu_tiny', dtype='float32'), 33)
    large = DecoderLanguageModel(cfg, rng_seed=1, dtype='float32')
    head = HorizonProposalHead(large, 4, hidden_width=16, rng_seed=2)
    decoder = BlockPrefixDecoder(large, 4, width=16, n_layers=1, n_heads=2, rng_seed=3)
    inference = InferenceModel(cfg, dtype='float32')
    inference.set_weights(large)
    state = inference.create_generation_state(1, 40)
    assert state.long_memory_enabled  # activates original routed inference
    inference.prefill(xp.asarray([[1,2,3,4,5]], dtype=xp.int32), state)
    gen = SequentialCommitBlockGenerator(inference, head, decoder,
                                         chooser=lambda logits: int(xp.argmax(logits)))
    result = gen.generate(state, 36)
    assert len(result['tokens']) == 36
    assert state.length == 41  # sliding routed horizon permits continual generation
    assert result['proposal_blocks'] == 9
    assert result['causal_main_updates'] == 36
    assert state.routed_prefix_refresh_count >= 2


def test_checkpoint_loader_uses_both_model_parts_and_checks_tokenizer(tmp_path, monkeypatch):
    """Rebuild an 0080I-style artifact without requiring the original shard data."""
    from types import SimpleNamespace
    from mini_llm.block_proposal.benchmark_adapter import load_0080i_block_checkpoint
    from mini_llm.block_proposal.checkpoint import save_two_stage_checkpoint
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.block_proposal.proposals import HorizonProposalHead
    from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
    from mini_llm.predictive_refinement.real_data import sha256
    from examples.train_block_proposal_routed_text import _model_config

    options = dict(seed=13, batch_size=1, grad_accum=1, block_size=4,
                   block_group_size=1, proposal_width=16, decoder_width=16,
                   decoder_layers=1, decoder_heads=2, lambda_proposal=0.5,
                   learning_rate=0.0003, weight_decay=0.1, clip_grad_norm=1.0)
    tok_file = tmp_path/'tokenizer.json'
    tok_file.write_text('{"synthetic":true}')
    class FakeTokenizer:
        eos_token = '<|eos|>'
        def __init__(self):
            self._tokenizer = self
        def __len__(self): return 33
        def token_to_id(self, token): return 1 if token == self.eos_token else None
        def encode(self, text): return [3 + (ord(c) % 30) for c in text]
        def decode(self, ids): return ''.join(chr(97 + int(i)%26) for i in ids)
    monkeypatch.setattr('mini_llm.block_proposal.benchmark_adapter.FastBPETokenizer.load',
                        lambda path: FakeTokenizer())
    config = _model_config(SimpleNamespace(model_preset='cpu_tiny', dtype='float32'),33)
    main = DecoderLanguageModel(config,rng_seed=13,dtype='float32')
    head = HorizonProposalHead(main,4,hidden_width=16,rng_seed=14)
    decoder = BlockPrefixDecoder(main,4,width=16,n_layers=1,n_heads=2,rng_seed=15)
    net = TwoStageBlockModel(head,decoder,lambda_proposal=0.5)
    signature = dict(version='0080I-v1', model_config=repr(config),
                     **options)
    path = tmp_path/'checkpoint'
    save_two_stage_checkpoint(path,net,training_step=500,
                              tokenizer_hash=sha256(tok_file),data_provenance=signature)
    loaded, token, info = load_0080i_block_checkpoint(path,tok_file)
    assert info['step']==500 and info['block_size']==4
    from mini_llm.block_proposal.benchmark_adapter import BlockProposalTextGenerator
    generator = BlockProposalTextGenerator(loaded, token, max_context=config.memory_context.memory_length)
    generated, timing = generator.generate('math', max_new_tokens=9,
                                           strategy='greedy', seed=42,
                                           log_speed=True, return_token_ids=True)
    assert timing['block_proposal_implementation'] == 'sequential_main_kv'
    assert 1 <= len(timing['generated_token_ids']) <= 9
    assert len(generated) == len(timing['generated_token_ids'])
    assert len(loaded.parameters()) == len(net.parameters())
    for first, second in zip(loaded.parameters(),net.parameters()):
        np.testing.assert_array_equal(asnumpy(first.data),asnumpy(second.data))
    tok_file.write_text('{"synthetic":false}')
    with pytest.raises(ValueError, match='checksum differs'):
        load_0080i_block_checkpoint(path,tok_file)
