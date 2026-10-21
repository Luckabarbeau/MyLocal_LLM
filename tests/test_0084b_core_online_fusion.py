"""Core on-the-fly fusion behavior, not dependent on examples scripts."""
import json
import numpy as np
import pytest
from types import SimpleNamespace
from pathlib import Path
from mini_llm.collective import ProbabilityResidualCombiner, FusionConfig
from mini_llm.collective import online_training as online
from mini_llm.collective.online_data import split_fusion_shards, OnlineShardSampler
from mini_llm.optim.adamw import AdamW

class FakeProj:
    def __init__(self, vocab, offset):
        self.vocab = vocab
        self.offset = offset
    def forward(self, x):
        z = np.arange(self.vocab, dtype=np.float32)[None, None, :]
        x = np.asarray(x, dtype=np.float32)
        return (np.cos(x[..., :1] + z * 0.3 + self.offset)).astype(np.float32), None

class FakeTeacher:
    def __init__(self, vocab, offset):
        self.output_proj = FakeProj(vocab, offset)
        self.calls = 0
        self.offset = offset
    def forward_body(self, ids, *, return_cache):
        assert not return_cache
        self.calls += 1
        return np.asarray(ids, dtype=np.float32)[..., None] * .01 + self.offset

class FakeSampler:
    def __init__(self):
        self.calls = 0
    def batch(self, n):
        self.calls += 1
        return np.tile(np.array([[0, 1, 2, 3, 4]], dtype=np.int64), (n, 1)), ['test'] * n


def test_aligned_logits_match_teacher_and_labels():
    windows = np.array([[2, 3, 4, 5, 6], [1, 3, 2, 4, 5]])
    positions = online.prediction_positions(4, 3, 1)
    assert positions.tolist() == [1, 2, 3]
    targets, contexts, lengths = online.aligned_training_rows(windows, positions)
    assert targets.tolist() == [4, 5, 6, 2, 4, 5]
    assert lengths.tolist() == [2, 3, 4, 2, 3, 4]
    assert contexts[1, :3].tolist() == [2, 3, 4]
    teachers = [FakeTeacher(9, 0.0), FakeTeacher(9, 0.25)]
    z = online.frozen_teacher_logits(teachers, windows, positions, head_rows=2)
    assert z.shape == (6, 2, 9)
    np.testing.assert_allclose(z[0, 0], np.cos(.03 + np.arange(9)*.3), rtol=1e-6, atol=1e-6)
    assert [t.calls for t in teachers] == [1, 1]


def test_online_step_updates_combiner_not_frozen_teachers():
    teachers = [FakeTeacher(11, 0.0), FakeTeacher(11, 0.25)]
    sampler = FakeSampler()
    model = ProbabilityResidualCombiner(FusionConfig(vocab_size=11, max_models=10,
                 context_heads=2, context_head_dim=4, context_hidden=6, seed=31))
    opt = AdamW(model.parameters(), lr=3e-3)
    original = model.output_bias.data.copy()
    result = online.train_online_step(model, opt, teachers, sampler,
                positions=online.prediction_positions(4, 4, 1), rows_per_update=7,
                teacher_batch_windows=1, batch_rows=3, lm_head_rows=2,
                grad_clip=1.0, member_ids=[1, 4])
    assert result['supervised_rows'] == 7
    assert np.isfinite(result['fusion_ce'])
    assert not np.array_equal(original, model.output_bias.data)
    assert [teacher.calls for teacher in teachers] == [2, 2]
    assert opt.step_index == 1
    assert np.isfinite(result['weighted_ce'])


def test_online_evaluation_is_read_only():
    model = ProbabilityResidualCombiner(FusionConfig(vocab_size=11, max_models=6,
                 context_heads=2, context_head_dim=4, context_hidden=6, seed=3))
    original = {name: x.data.copy() for name, x in model.parameters_by_name.items()}
    windows = np.array([[0, 1, 2, 3, 4], [0, 5, 2, 6, 7]])
    ev = online.evaluate_online(model, [FakeTeacher(11, 0), FakeTeacher(11, 0.4)],
              windows, online.prediction_positions(4, 3, 1), member_ids=[2, 3],
              teacher_batch_windows=1, batch_rows=2, lm_head_rows=2)
    assert ev['rows'] == 6
    assert len(ev['member_ce']) == 2
    assert all(np.isfinite(v) for v in (ev['fusion_ce'], ev['arithmetic_ce'], ev['weighted_ce']))
    for name, p in model.parameters_by_name.items():
        np.testing.assert_array_equal(p.data, original[name])


def test_fusion_resume_retains_optimizer_and_guard(tmp_path):
    cfg = FusionConfig(vocab_size=11, context_heads=2, context_head_dim=4,
                       context_hidden=6, seed=13)
    m = ProbabilityResidualCombiner(cfg)
    opt = AdamW(m.parameters(), lr=2e-3)
    m.output_bias.grad[:] = 1.0
    opt.step()
    rng = np.random.default_rng(82)
    original_rng = rng.bit_generator.state
    expected = {'teacher_fingerprints': {'0': 'abc'}, 'context_length': 4}
    path = tmp_path / 'last.npz'
    online.save_last_state(path, m, opt, rng, {**expected, 'step': 1,
                            'best_ce': 2.2, 'stale': 0})
    n = ProbabilityResidualCombiner(cfg)
    o2 = AdamW(n.parameters(), lr=2e-3)
    info = online.restore_last_state(path, n, o2, expected)
    assert info['step'] == 1 and o2.step_index == 1
    for name, p in m.parameters_by_name.items():
        np.testing.assert_array_equal(n.parameters_by_name[name].data, p.data)
    with pytest.raises(ValueError, match='teacher_fingerprints'):
        online.restore_last_state(path, n, o2, {'teacher_fingerprints': {'0': 'changed'}})


def test_tokenizer_mismatch_fails_before_loading_teachers(tmp_path):
    data = tmp_path / 'data'; data.mkdir()
    (data / 'tokenizer.json').write_text('a')
    teacher = tmp_path / 't'; teacher.mkdir()
    (teacher / 'tokenizer.json').write_text('b')
    with pytest.raises(ValueError, match='tokenizer SHA256'):
        online.validate_teacher_tokenizers([(0, teacher)], data, {})


def test_disjoint_partition_and_eos_sampler(tmp_path):
    paths = []
    for i in range(8):
        p = tmp_path / f'val_shard_{i}.bin'
        np.array([99] + list(range(1, 20)) + [99, 0], dtype=np.uint16).tofile(p)
        paths.append(p)
    source = SimpleNamespace(name='test', weight=1.0, val_paths=tuple(paths))
    parts = split_fusion_shards([source], seed=82)
    assert set(parts['train']['test']).isdisjoint(parts['val']['test'])
    assert set(parts['train']['test']).isdisjoint(parts['test']['test'])
    sampler = OnlineShardSampler([source], parts['train'],
                {'eos_token_id':99, 'token_dtype':'uint16'}, 4, np.random.default_rng(3))
    w, _ = sampler.batch(12)
    assert w.shape == (12, 5)
    assert not np.any(w[:, :-1] == 99)


def test_core_entrypoint_help_executes(monkeypatch):
    parser = online.build_parser()
    options = parser.parse_args(['--member','0:/tmp/deep', '--member','1:/tmp/wide',
             '--mixed-shard-root','/tmp/dataset', '--output','/tmp/fusion.npz'])
    assert options.context_heads == 2
    assert options.grad_accum_steps == 128


def test_complete_online_training_and_resume(monkeypatch, tmp_path):
    teacher_paths = [tmp_path / 'deep', tmp_path / 'wide']
    for p in teacher_paths:
        p.mkdir()
    mixed = tmp_path / 'shards'
    mixed.mkdir()
    (mixed / 'tokenizer.json').write_text('{}')
    class FakeSamples:
        def __init__(self, seed):
            self.rng = np.random.default_rng(seed)
        def batch(self, n):
            windows = np.tile(np.array([[0, 1, 2, 3, 4]]), (n, 1))
            return windows, ['src'] * n
    def samplers(*args):
        return {'train': FakeSamples(6), 'val': FakeSamples(7)}, \
               {'tokenizer_sha256': None}, {}
    monkeypatch.setattr(online, 'make_online_samplers', samplers)
    monkeypatch.setattr(online, 'validate_teacher_tokenizers',
                        lambda members, root, manifest: ('hash', {'0': 'a', '1': 'b'}))
    monkeypatch.setattr(online, 'load_frozen_teacher',
                        lambda directory, length: (FakeTeacher(11, .2 if directory.name == 'wide' else .0),
                                                   SimpleNamespace(vocab_size=11)))
    output = tmp_path / 'fusion.npz'
    cli = ['--member', f'0:{teacher_paths[0]}', '--member', f'1:{teacher_paths[1]}',
           '--mixed-shard-root', str(mixed), '--output', str(output),
           '--context-length', '4', '--score-last-k', '4', '--batch-windows','1',
           '--batch-rows', '3', '--grad-accum-steps', '2', '--lm-head-rows','2',
           '--context-head-dim', '4', '--context-hidden', '6',
           '--val-windows', '2', '--steps','2', '--save-interval','1',
           '--eval-interval','1', '--log-interval','1',
           '--log-file', str(tmp_path / 'fusion.csv')]
    online.main(cli)
    assert output.exists()
    assert (tmp_path / 'fusion.last.npz').exists()
    fusion = ProbabilityResidualCombiner.load(output)
    assert fusion.config.context_heads == 2
    assert len((tmp_path / 'fusion.csv').read_text().splitlines()) == 3
    # Resuming must preserve teacher identities/config and optimizer step.
    online.main(cli[:-6] + ['--steps','3', '--save-interval','1',
                            '--eval-interval','1', '--log-interval','1',
                            '--log-file', str(tmp_path / 'fusion.csv'), '--resume'])
    with np.load(tmp_path / 'fusion.last.npz', allow_pickle=False) as archive:
        metadata = json.loads(str(archive['__meta_json__']))
    assert metadata['step'] == 3


def test_real_tiny_teacher_checkpoints_and_packed_shards(tmp_path):
    """End to end without mocking teacher inference or shard partitioning."""
    import dataclasses
    import hashlib
    from mini_llm.config import ModelConfig
    from mini_llm.checkpoint import save_checkpoint
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    tokenizer = b'{"test": "shared"}'
    root = tmp_path / 'mixed'
    src = root / 'source'
    src.mkdir(parents=True)
    (root / 'tokenizer.json').write_bytes(tokenizer)
    np.asarray([16] + list(range(1, 15)) + [16], dtype=np.uint16).tofile(src / 'train_shard_000.bin')
    for j in range(6):
        np.asarray([16] + list(range(1, 15)) + [16], dtype=np.uint16).tofile(
            src / f'val_shard_{j:03d}.bin')
    manifest = {
        'format_version': 'weighted_packed_mixture_v1',
        'eos_token_id': 16,
        'token_dtype': 'uint16',
        'tokenizer_sha256': hashlib.sha256(tokenizer).hexdigest(),
        'sources': {'source': {'weight': 1.0, 'relative_shard_dir': 'source',
                    'train_shards': 1, 'val_shards': 6}},
    }
    (root / 'mixture_manifest.json').write_text(json.dumps(manifest))
    cfg = ModelConfig(tokenizer_vocab_size=17, context_length=4, n_layers=2,
                      d_model=8, n_q_heads=1, n_kv_heads=1, d_head=8,
                      n_experts=1, top_k=1, d_ff=16, ffn_mode='dense')
    members = []
    for mid in (0, 1):
        directory = tmp_path / f'teacher_{mid}'
        directory.mkdir()
        (directory / 'tokenizer.json').write_bytes(tokenizer)
        (directory / 'config.json').write_text(json.dumps(dataclasses.asdict(cfg)))
        teacher = DecoderLanguageModel(cfg, rng_seed=11 + mid, dtype='float32')
        save_checkpoint(directory, {p.name: p.data for p in teacher.parameters()})
        members += ['--member', f'{mid}:{directory}']
    out = tmp_path / 'online.npz'
    online.main(members + [
        '--mixed-shard-root', str(root), '--output', str(out),
        '--context-length','4', '--score-last-k','3', '--batch-rows','2',
        '--grad-accum-steps','2', '--batch-windows','1', '--lm-head-rows','2',
        '--val-windows','2', '--steps','2', '--eval-interval','1',
        '--save-interval','1', '--log-interval','1', '--context-head-dim','4',
        '--context-hidden','6', '--runtime-profile','reference',
    ])
    assert out.is_file()
    assert out.with_name('online.last.npz').is_file()
