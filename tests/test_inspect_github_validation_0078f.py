"""Ensure GitHub document inspector reproduces production validator fingerprints."""
import importlib.util
from pathlib import Path

import numpy as np

from mini_llm.config import MemoryContextConfig, ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train_extended import ExtendedTrainer


def test_replays_exact_production_sample_hashes(tmp_path, monkeypatch):
    monkeypatch.setenv('MINI_LLM_LM_HEAD_CHUNK_TOKENS', '4')
    path = Path(__file__).parents[1] / 'benchmarks' / 'inspect_github_validation_0078f.py'
    spec = importlib.util.spec_from_file_location('inspect_github_validation_0078f', path)
    inspect = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspect)
    mem = MemoryContextConfig(
        enabled=True, memory_length=16, recent_length=4, target_length=4,
        block_size=2, top_k_blocks=1, router_query_length=4, router_dim=4,
        read_heads=2, integration_mode='routed_prefix', memory_training='joint',
        retrieval_batch_probability=0, read_kv_heads=1, read_query_chunk=2,
    )
    config = ModelConfig(
        tokenizer_vocab_size=32, context_length=mem.active_length,
        n_layers=1, d_model=8, n_q_heads=2, n_kv_heads=1, d_head=4,
        n_experts=2, top_k=1, d_ff=16, dtype='float32', memory_context=mem,
    )
    docs = ([4, 5, 6, 2] * 130) + (list(range(4, 21)) + [2]) + ([10, 11, 12, 13, 14, 15, 16, 2] * 12)
    data = np.asarray(docs, dtype=np.uint16)
    p = tmp_path / 'val_shard_00000.bin'
    data.tofile(p)
    sources = {'github': [str(p)]}
    trainer = ExtendedTrainer(
        model=DecoderLanguageModel(config, rng_seed=1),
        train_shard_paths=sources['github'], val_shard_paths=sources['github'],
        train_source_shards=sources, val_source_shards=sources,
        source_weights={'github':1}, batch_size=2, seq_length=mem.source_input_length,
        grad_accum_steps=1, total_steps=2, warmup_steps=1,
        val_steps=1, val_interval=100, save_interval=100,
        val_per_source_steps=2, val_representative_steps=2,
        eos_token_id=2, log_file=str(tmp_path/'training.csv'),
    )
    for policy, actual in (
        ('legacy', trainer.compute_val_loss_by_source()['github']),
        ('document_first', trainer.compute_val_loss_by_source_representative()['github']),
    ):
        audit = inspect.replay_source(
            paths=[p], eos_token_id=2, source='github', policy=policy,
            source_seed=78078, batch_size=2, steps=2,
            memory_length=mem.distant_memory_length, target_length=4,
            min_history_tokens=3,
        )
        assert audit['sample_sha256'] == actual['sample_sha256']
        assert audit['total_supervised_tokens'] == actual['supervised_tokens']
        assert audit['target_fill'] == actual['target_fill']
        assert len(audit['samples']) == 4
        assert all(sample['document_eos_terminated'] for sample in audit['samples'])


def test_repetition_stats_and_headers(tmp_path):
    path = Path(__file__).parents[1] / 'benchmarks' / 'inspect_github_validation_0078f.py'
    spec = importlib.util.spec_from_file_location('inspect_github_validation_0078f', path)
    inspect = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspect)
    assert inspect._ngram_repeat_fraction(np.array([5] * 20), 8) > 0.9
    assert inspect._ngram_repeat_fraction(np.arange(20), 8) == 0
    assert inspect._line_metrics('aaaabbbbb\naaaabbbbb\nccccccccc')['duplicated_line_share'] == 2/3


def test_optional_reference_missing_does_not_crash(tmp_path, capsys):
    import csv
    path = Path(__file__).parents[1] / 'benchmarks' / 'inspect_github_validation_0078f.py'
    spec = importlib.util.spec_from_file_location('inspect_github_validation_0078f', path)
    inspect = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspect)
    absent = tmp_path / 'training.val_sources_representative.csv'
    assert inspect._parse_csv_expected(absent, 'github') is None
    assert 'UNVERIFIED' in capsys.readouterr().out
    present = tmp_path / 'training.val_sources.csv'
    with present.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'sample_sha256'])
        writer.writeheader()
        writer.writerow({'source': 'github', 'sample_sha256': '123abc'})
    assert inspect._parse_csv_expected(present, 'github') == '123abc'
