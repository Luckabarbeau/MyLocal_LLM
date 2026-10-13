"""0079I real-data, 65K vocabulary and optimizer/resume smoke checks."""
import json
from pathlib import Path
import numpy as np
import pytest

from mini_llm.predictive_refinement.real_data import PackedRefinementDocuments


def make_shards(tmp_path):
    # Three documents with EOS at the end, including one very short document.
    values = np.array([4, 5, 6, 1, 7, 1, 8, 9, 10, 11, 12, 1], dtype=np.uint16)
    for split in ('train', 'val'):
        values.tofile(tmp_path / f'{split}_shard_0000.bin')
    (tmp_path / 'tokenizer.json').write_text('{"test":true}')
    return values


def test_document_first_short_documents_and_eos(tmp_path):
    make_shards(tmp_path)
    source = PackedRefinementDocuments(shard_dir=tmp_path, eos_token_id=1)
    rng = np.random.default_rng(123)
    observed_short = False
    docs = {(4, 5, 6, 1), (7, 1), (8, 9, 10, 11, 12, 1)}
    for _ in range(75):
        batch, valid, prefix, counts = source.sample(
            rng, batch_size=3, prefix_length=4, block_size=2, window_blocks=2,
        )
        assert prefix >= 1
        assert sum(counts.values()) == 3
        assert np.all(valid[:, :prefix])
        assert not np.any(valid[:, 1:] & ~valid[:, :-1])
        for row, mask in zip(batch, valid):
            sequence = tuple(row[mask])
            assert len(sequence) >= 2
            assert any(tuple(doc[i:i+len(sequence)]) == sequence
                       for doc in docs for i in range(len(doc)))
            assert 1 not in sequence[:-1]  # EOS can only be the last token
            observed_short |= sequence == (7, 1)
    assert observed_short  # short docs are not discarded for failing prefix=4


def test_shard_validation(tmp_path):
    make_shards(tmp_path)
    with pytest.raises(ValueError, match='exactly one'):
        PackedRefinementDocuments(eos_token_id=1)
    with pytest.raises(ValueError, match='exactly one'):
        PackedRefinementDocuments(shard_dir=tmp_path, mixed_root=tmp_path, eos_token_id=1)
    with pytest.raises(ValueError, match='dtype'):
        PackedRefinementDocuments(shard_dir=tmp_path, eos_token_id=1, token_dtype='float32')
    with pytest.raises(ValueError, match='split'):
        PackedRefinementDocuments(shard_dir=tmp_path, eos_token_id=1, split='bad')


def test_real_65k_one_layer_step_checkpoint_resume(tmp_path, monkeypatch, capsys):
    make_shards(tmp_path)
    from examples import train_predictive_refinement_text as script

    class FakeExistingTokenizer:
        eos_token = '<|eos|>'
        def __len__(self):
            return 65280
        def encode(self, text):
            return [4, 5]
        def decode(self, ids):
            return 'decoded:' + str(ids)

    monkeypatch.setattr(script, '_load_tokenizer', lambda path: (FakeExistingTokenizer(), 1))
    checkpoint = tmp_path / 'saved'
    command = [
        '--shard-dir', str(tmp_path), '--checkpoint-dir', str(checkpoint),
        '--steps', '1', '--batch-size', '1', '--grad-accum', '2',
        '--prefix-length', '2', '--block-size', '2', '--window-blocks', '2',
        '--refinement-group-size', '2', '--layers', '1', '--d-model', '16',
        '--n-q-heads', '2', '--n-kv-heads', '1', '--d-ff', '32',
        '--experts', '1', '--log-every', '1', '--save-every', '1',
        '--val-every', '1', '--val-batches', '1', '--generate', '0',
    ]
    script.main(command)
    saved = json.loads((checkpoint / 'training_state.json').read_text())
    assert saved['architecture']['model_config']['vocab_size'] == 65280
    assert saved['architecture']['model_config']['n_layers'] == 1
    assert saved['step'] == 1
    state = json.loads((checkpoint / 'refinement_data_rng.json').read_text())
    assert state['step'] == 1
    assert 'validation step=1' in capsys.readouterr().out
    # Continue exactly one optimizer step without creating a new model/data stream.
    script.main(command + ['--resume', str(checkpoint), '--steps', '2', '--generate', '4'])
    assert json.loads((checkpoint / 'training_state.json').read_text())['step'] == 2


def test_global_accumulation_denominator_validation():
    from mini_llm.backend import xp
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.predictive_refinement.training_reference import RefinementReferenceConfig
    from mini_llm.predictive_refinement.window_training import WindowRefinementTrainer

    cfg = ModelConfig(tokenizer_vocab_size=32, context_length=32, n_layers=1,
                      d_model=16, n_q_heads=2, n_kv_heads=1, d_head=8,
                      n_experts=1, top_k=1, d_ff=32, dtype='float64')
    ref = WindowRefinementTrainer(DecoderLanguageModel(cfg, rng_seed=7, dtype='float64'),
                                  RefinementReferenceConfig(block_size=2,
                                      enable_shared_context_training=True))
    tokens = xp.asarray([[4, 5, 6, 7, 8]], dtype=xp.int64)
    mask = xp.ones_like(tokens, dtype=xp.bool_)
    ref.zero_grad()
    single = ref.train_window(tokens, prefix_length=1, valid_token_mask=mask,
                              group_size=2)
    ref.zero_grad()
    doubled = ref.train_window(tokens, prefix_length=1, valid_token_mask=mask,
                               group_size=2, loss_normalization_tokens=8)
    assert int(single['supervised_tokens']) == int(doubled['supervised_tokens']) == 4
    assert float(doubled['total_loss']) == pytest.approx(float(single['total_loss']) / 2)
    with pytest.raises(ValueError, match='>= local valid tokens'):
        ref.train_window(tokens, prefix_length=1, valid_token_mask=mask,
                         loss_normalization_tokens=2)
