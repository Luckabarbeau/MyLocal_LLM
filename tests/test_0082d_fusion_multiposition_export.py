"""0082D: causal multi-position logits export, memmap alignment, epoch sampler."""
import dataclasses
import json

import numpy as np
import pytest

from mini_llm.checkpoint import save_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from benchmarks.evaluate_wikitext import load_model
from examples.export_probability_fusion_logits_0082a import (
    _scored_positions, main as export,
)
from examples.train_probability_fusion_0082a import _sample_epoch_rows, _evaluate, _open_members
from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner


@pytest.fixture()
def fixture(tmp_path):
    cfg = ModelConfig(
        tokenizer_vocab_size=17, context_length=6, n_layers=2,
        d_model=8, n_q_heads=1, n_kv_heads=1, d_head=8,
        n_experts=1, top_k=1, d_ff=16, ffn_mode='dense',
    )
    sources = []
    for mid in (0, 4):
        ckpt = tmp_path / f'model_{mid}'
        ckpt.mkdir()
        (ckpt / 'config.json').write_text(json.dumps(dataclasses.asdict(cfg)))
        (ckpt / 'tokenizer.json').write_text('{"tokenizer-test": true}')
        model = DecoderLanguageModel(cfg, rng_seed=101 + mid, dtype='float32')
        save_checkpoint(ckpt, {p.name: p.data for p in model.parameters()})
        sources.append((mid, ckpt))
    windows = np.random.default_rng(822).integers(0, 17, size=(7, 7), dtype=np.int64)
    np.save(tmp_path / 'windows.npy', windows)
    return tmp_path, sources, windows


def _run(fixture, name, extra=(), batch=3):
    tmp, members, _ = fixture
    opts = []
    for mid, ckpt in members:
        opts += ['--member', f'{mid}:{ckpt}']
    path = tmp / name
    export(opts + ['--windows', str(tmp / 'windows.npy'), '--output-dir', str(path),
                   '--context-length', '6', '--batch-windows', str(batch),
                   '--lm-head-rows', '3'] + list(extra))
    return path


@pytest.mark.parametrize('last_k,stride,expected', [
    (6, 1, [0, 1, 2, 3, 4, 5]),
    (6, 2, [1, 3, 5]),
    (4, 2, [3, 5]),
    (3, 1, [3, 4, 5]),
    (1, 1, [5]),
])
def test_selected_positions_and_full_logits_against_full_forward(fixture, last_k, stride, expected):
    _, models, windows = fixture
    out = _run(fixture, f'k{last_k}_s{stride}',
               ('--score-last-k', str(last_k), '--score-stride', str(stride)))
    meta = json.loads((out/'metadata.json').read_text())
    assert meta['prediction_positions'] == expected
    assert meta['rows'] == len(windows)*len(expected)
    np.testing.assert_array_equal(np.load(out/'targets.npy'),
                                  windows[:, np.asarray(expected)+1].reshape(-1))
    for mid, ckpt in models:
        model, _ = load_model(ckpt, context_length=6, memory_mode='plain')
        expected_logits = []
        for window in windows:
            scores = np.asarray(model.forward(window[None, :-1], return_cache=False))[0]
            expected_logits.extend(scores[expected])
        got = np.load(out/f'model_{mid}_logits.npy')
        assert got.dtype == np.float16
        np.testing.assert_allclose(got, np.asarray(expected_logits).astype(np.float16),
                                   rtol=0.002, atol=0.002)


def test_all_default_last_only_and_model_batch_parity(fixture):
    _, sources, windows = fixture
    old_full = _run(fixture, 'all_default', batch=1)
    new_full = _run(fixture, 'all_explicit', ('--score-last-k', '6'), batch=3)
    last = _run(fixture, 'last', ('--last-only',), batch=4)
    for mid, _ in sources:
        np.testing.assert_allclose(np.load(old_full/f'model_{mid}_logits.npy'),
                                   np.load(new_full/f'model_{mid}_logits.npy'), atol=0.002)
        last_logits = np.load(last/f'model_{mid}_logits.npy')
        full = np.load(old_full/f'model_{mid}_logits.npy').reshape(len(windows), 6, -1)
        np.testing.assert_allclose(last_logits, full[:, -1], atol=0.002)
    np.testing.assert_array_equal(np.load(last/'targets.npy'), windows[:, -1])


def test_rejected_invalid_selection(fixture):
    for opts in [('--last-only', '--score-last-k', '4'),
                 ('--score-last-k', '0'), ('--score-last-k', '7'),
                 ('--score-stride', '0'), ('--lm-head-rows', '0')]:
        with pytest.raises(SystemExit):
            _run(fixture, 'invalid', opts)


def test_epoch_sampler_covers_all_rows_before_reshuffling():
    rng = np.random.default_rng(823)
    state = [rng.permutation(13), 0]
    taken = np.concatenate([_sample_epoch_rows(rng, 13, 5, state),
                            _sample_epoch_rows(rng, 13, 5, state),
                            _sample_epoch_rows(rng, 13, 3, state)])
    np.testing.assert_array_equal(np.sort(taken), np.arange(13))
    newer = _sample_epoch_rows(rng, 13, 18, state)
    np.testing.assert_array_equal(np.sort(newer[:13]), np.arange(13))


def test_limited_validation_is_deterministic(tmp_path):
    rng = np.random.default_rng(824)
    for mid in (0, 1):
        np.save(tmp_path/f'{mid}.npy', rng.normal(size=(25, 11)).astype(np.float16))
    np.save(tmp_path/'labels.npy', rng.integers(0, 11, size=25, dtype=np.int32))
    source = _open_members([f'0:{tmp_path}/0.npy', f'1:{tmp_path}/1.npy'],
                           tmp_path/'labels.npy')
    net = ProbabilityResidualCombiner(FusionConfig(vocab_size=11))
    one = _evaluate(net, source, 3, max_rows=7)
    two = _evaluate(net, source, 2, max_rows=7)
    np.testing.assert_allclose(one[:3], two[:3], atol=1e-6)
    np.testing.assert_allclose(one[3], two[3], atol=1e-6)
    with pytest.raises(ValueError):
        _evaluate(net, source, 2, max_rows=0)
