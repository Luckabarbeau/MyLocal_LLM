"""0082S benchmark must keep windows, labels, and baseline metrics aligned."""

import numpy as np
import pytest

from benchmarks.evaluate_probability_fusion_checkpoint_0082s import (
    iter_corpus_windows, iter_lambada_windows, parse_members, score_rows,
    segment_windows, export_one_member,
)
from examples.train_probability_fusion_0082a import _evaluate
from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner


def test_corpus_windows_match_exact_reference_suffix_scoring():
    docs = [list(range(23)), list(range(15))]
    for context in (4, 8):
        for stride in (1, 3, context):
            windows = list(iter_corpus_windows(docs, context=context, stride=stride))
            positions = [[int(y) for y in w[2]] for w in windows]
            flattened = [y for target in positions for y in target]
            expected = [y for doc in docs for y in doc[1:]]
            assert flattened == expected
            for seq, pos, target, _ in windows:
                assert len(seq) <= context
                assert all(0 <= p < len(seq) for p in pos)


def test_member_specs_reject_ambiguous_identities():
    with pytest.raises(ValueError):
        parse_members(['0:a', '0:b'])
    with pytest.raises(ValueError):
        parse_members(['-1:a'])
    with pytest.raises(ValueError):
        parse_members(['not-a-spec'])


def test_segment_windows_preserve_order_and_partial_tail():
    rows = list(range(19))
    assert list(segment_windows(rows, 8)) == [rows[:8], rows[8:16], rows[16:]]


def test_0082s_ce_matches_existing_offline_fusion_evaluator():
    rng = np.random.default_rng(2026)
    vocab = 41
    ids = np.asarray([0, 1], dtype=np.int64)
    logits = [rng.normal(size=(13, vocab)).astype(np.float32) * 2 for _ in range(2)]
    targets = rng.integers(0, vocab, size=13).astype(np.int64)
    model = ProbabilityResidualCombiner(FusionConfig(vocab_size=vocab))
    # Nonzero learned reference and residual test the actual trained-model path.
    model.reference_logits.data[0] = -0.35
    model.reference_logits.data[1] = +0.35
    model.output_bias.data[:] = rng.normal(0, .03, size=vocab)
    measured, _ = score_rows(model, ids, logits, targets, batch_rows=3)
    source = ids, logits, targets
    fused, arithmetic, geometric, members, weighted = _evaluate(
        model, source, batch_rows=3, return_weighted=True)
    assert measured['fusion'] / 13 == pytest.approx(fused, abs=1e-5)
    assert measured['arithmetic'] / 13 == pytest.approx(arithmetic, abs=1e-5)
    assert measured['geometric'] / 13 == pytest.approx(geometric, abs=1e-5)
    assert measured['weighted'] / 13 == pytest.approx(weighted, abs=1e-5)
    for j, mid in enumerate(ids):
        assert measured[f'model_{mid}'] / 13 == pytest.approx(members[j], abs=1e-5)


def test_lambada_exact_word_requires_every_target_token_correct():
    vocab = 17
    ids = np.asarray([0, 1], dtype=np.int64)
    targets = np.asarray([2, 3, 4, 5], dtype=np.int64)
    example_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)
    # Member0 is correct on 3/4 tokens, but only word1 is exactly correct.
    # Member1 correct on every token. Fused initial arithmetic same as member1.
    first = np.full((4, vocab), -3., dtype=np.float32)
    second = np.full((4, vocab), -3., dtype=np.float32)
    for j, target in enumerate(targets):
        first[j, target] = 3.
        second[j, target] = 6.
    first[0, 11] = 4.
    model = ProbabilityResidualCombiner(FusionConfig(vocab_size=vocab))
    sums, exact = score_rows(model, ids, [first, second], targets,
                             batch_rows=1, example_ids=example_ids)
    assert exact['model_0'] == 1
    assert exact['model_1'] == 2
    assert exact['fusion'] == 2
    assert all(np.isfinite(nll) for nll in sums.values())


class FakeMember:
    def forward_body(self, batch, return_cache=False):
        return np.asarray(batch, dtype=np.float32)[:, :, None]
    class Head:
        def forward(self, tile):
            # Return scalar logits for the exact input positions.
            return np.repeat(tile, 5, axis=-1), None
    output_proj = Head()


def test_export_selects_only_scored_positions_and_keeps_window_order(tmp_path):
    # Different window lengths and different selected suffix positions: the
    # reference must not reuse output rows or accidentally score the padding.
    windows = [
        (np.asarray([1,2,3], dtype=np.int64), np.asarray([1,2]), np.asarray([3,4]), 0),
        (np.asarray([3,4,5], dtype=np.int64), np.asarray([1,2]), np.asarray([5,6]), 1),
        (np.asarray([7,8], dtype=np.int64), np.asarray([1]), np.asarray([9]), 2),
    ]
    path = tmp_path / 'model_0.npy'
    export_one_member(FakeMember(), windows, output_file=path, vocab=5,
                      lm_head_rows=2, batch_windows=2)
    saved = np.load(path)
    assert saved.shape == (5, 5)
    assert np.all(saved[:, 0] == [2,3,4,5,8])


def test_end_to_end_benchmark_streaming_with_fake_members(tmp_path, monkeypatch):
    import argparse
    import json
    import benchmarks.evaluate_probability_fusion_checkpoint_0082s as bench

    models = []
    for mid in (0, 1):
        ckpt = tmp_path / f'ckpt_{mid}'
        ckpt.mkdir()
        (ckpt / 'config.json').write_text('{}')
        (ckpt / 'tokenizer.json').write_text('{"equal":true}')
        models.append(f'{mid}:{ckpt}')
    model = ProbabilityResidualCombiner(FusionConfig(vocab_size=5))
    fusion_path = tmp_path / 'fusion.npz'
    model.save(fusion_path)
    source_path = tmp_path / 'corpus.txt'
    source_path.write_text('dummy')
    json_path = tmp_path / 'report.json'

    class Config:
        vocab_size = 5

    class Tok:
        pass

    monkeypatch.setattr(bench, 'load_tokenizer', lambda path: (Tok(), path / 'tokenizer.json'))
    monkeypatch.setattr(bench, 'load_model',
                        lambda path, **kw: (FakeMember(), Config()))
    monkeypatch.setattr(bench, 'tokenize_documents',
                        lambda *args, **kw: ([[0,1,2,3,4,0,1,2,3,4,0]], 11, 1, 1, 5))
    args = argparse.Namespace(member=models, data=source_path,
        data_format='text', fusion_checkpoint=fusion_path, context_length=4,
        stride=4, text_field='text', max_examples=None, max_tokens=None,
        max_documents=None, segment_windows=2, lm_head_rows=3,
        batch_windows=2, batch_rows=2, dataset_name='smoke', temp_root=tmp_path,
        json_output=json_path)
    bench.run(args)
    report = json.loads(json_path.read_text())
    assert report['scored_tokens'] == 10
    assert report['segments'] == 2
    assert report['cross_entropy']['fusion'] == pytest.approx(np.log(5), abs=1e-5)
    assert report['cross_entropy']['arithmetic'] == pytest.approx(np.log(5), abs=1e-5)
    assert report['cross_entropy']['weighted'] == pytest.approx(np.log(5), abs=1e-5)
    assert not list(tmp_path.glob('0082s_fusion_*'))
