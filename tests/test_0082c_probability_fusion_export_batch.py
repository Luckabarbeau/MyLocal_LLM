"""0082C: independent-window batching and last-token-only LM projection."""

import dataclasses
import json

import numpy as np
import pytest

from mini_llm.checkpoint import save_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from benchmarks.evaluate_wikitext import load_model
from examples.export_probability_fusion_logits_0082a import main as export


@pytest.fixture()
def fixture(tmp_path):
    cfg = ModelConfig(
        tokenizer_vocab_size=17, context_length=6, n_layers=2,
        d_model=8, n_q_heads=1, n_kv_heads=1, d_head=8,
        n_experts=1, top_k=1, d_ff=16, ffn_mode='dense',
    )
    ckpts = []
    for mid in (0, 4):
        ckpt = tmp_path / f'model_{mid}'
        ckpt.mkdir()
        (ckpt / 'config.json').write_text(json.dumps(dataclasses.asdict(cfg)))
        (ckpt / 'tokenizer.json').write_text('{"tokenizer-test": true}')
        model = DecoderLanguageModel(cfg, rng_seed=101 + mid, dtype='float32')
        save_checkpoint(ckpt, {p.name: p.data for p in model.parameters()})
        ckpts.append((mid, ckpt))
    rng = np.random.default_rng(821)
    windows = rng.integers(0, 17, size=(7, 7), dtype=np.int64)
    np.save(tmp_path / 'windows.npy', windows)
    return tmp_path, ckpts, windows


def run_export(tmp_path, ckpts, *, batch, last_only, name):
    args = []
    for mid, ckpt in ckpts:
        args += ['--member', f'{mid}:{ckpt}']
    out = tmp_path / name
    export(args + ['--windows', str(tmp_path / 'windows.npy'),
                   '--output-dir', str(out), '--context-length', '6',
                   '--batch-windows', str(batch)] + (['--last-only'] if last_only else []))
    return out


@pytest.mark.parametrize('batch', [1, 3, 4])
def test_last_only_matches_full_forward_for_both_models(fixture, batch):
    tmp_path, ckpts, windows = fixture
    out = run_export(tmp_path, ckpts, batch=batch, last_only=True,
                     name=f'last_{batch}')
    np.testing.assert_array_equal(np.load(out / 'targets.npy'), windows[:, -1])
    for mid, ckpt in ckpts:
        model, _ = load_model(ckpt, context_length=6, memory_mode='plain')
        expected = []
        for row in windows:
            expected.append(np.asarray(model.forward(row[:-1][None, :], return_cache=False))[0, -1])
        actual = np.load(out / f'model_{mid}_logits.npy')
        np.testing.assert_allclose(actual, np.asarray(expected).astype(np.float16),
                                   rtol=0.002, atol=0.002)
    meta = json.loads((out / 'metadata.json').read_text())
    assert meta['batch_windows'] == batch
    assert meta['rows'] == len(windows)


def test_all_position_batched_export_identical_to_unbatched(fixture):
    tmp_path, ckpts, windows = fixture
    single = run_export(tmp_path, ckpts, batch=1, last_only=False, name='all_one')
    batched = run_export(tmp_path, ckpts, batch=4, last_only=False, name='all_four')
    for mid, _ in ckpts:
        a = np.load(single / f'model_{mid}_logits.npy')
        b = np.load(batched / f'model_{mid}_logits.npy')
        assert a.shape == (len(windows) * (windows.shape[1] - 1), 17)
        np.testing.assert_allclose(a, b, atol=0.002, rtol=0.002)
    np.testing.assert_array_equal(np.load(batched / 'targets.npy'), windows[:, 1:].ravel())


def test_invalid_batch_windows_is_rejected(fixture):
    tmp_path, ckpts, _ = fixture
    with pytest.raises(SystemExit):
        run_export(tmp_path, ckpts[:1], batch=0, last_only=True, name='bad')
