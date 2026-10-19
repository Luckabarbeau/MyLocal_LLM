"""0082F: bounded streaming, aligned teacher outputs, and resumable optimizer."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from examples import train_probability_fusion_streaming_0082f as streaming
from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner
from mini_llm.optim.adamw import AdamW


def test_segment_budget_scales_with_model_count():
    assert streaming._segment_windows_for_budget(1024, members=2, vocab=32768,
                                                   scored=256, budget_gib=1) == 32
    assert streaming._segment_windows_for_budget(1024, members=10, vocab=32768,
                                                   scored=256, budget_gib=1) == 6
    with pytest.raises(ValueError, match='too small'):
        streaming._segment_windows_for_budget(2, members=10, vocab=32768,
                                               scored=256, budget_gib=0.01)


def test_checkpoint_full_optimizer_rng_roundtrip(tmp_path):
    config = FusionConfig(vocab_size=17, seed=82)
    model = ProbabilityResidualCombiner(config)
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    rng = np.random.default_rng(123)
    # Train one actual step so Adam state is nonzero.
    logits = np.random.default_rng(8).normal(size=(3, 2, 17)).astype('float32')
    optimizer.zero_grad()
    model.loss_and_backward(logits, np.array([0, 1]), np.array([2, 3, 5]))
    optimizer.step()
    baseline_weights = model.state_dict()
    baseline_m = [streaming._to_host(m).copy() for m in optimizer.m]
    baseline_v = [streaming._to_host(v).copy() for v in optimizer.v]
    baseline_master = [streaming._to_host(m).copy() for m in optimizer.master_weights]
    state = {'root':'example', 'step':1, 'segment':1, 'best_val_ce':3.2, 'stale_evals':0}
    path = tmp_path/'last.npz'
    streaming._write_last_state(path, model, optimizer, rng, state=state)
    expected_draw = rng.integers(0, 10000, size=4)
    clone = ProbabilityResidualCombiner(config)
    clone_opt = AdamW(clone.parameters(), lr=2e-4, weight_decay=0.01)
    loaded = streaming._read_last_state(path, clone, clone_opt, expected={'root':'example'})
    rng2 = np.random.default_rng()
    rng2.bit_generator.state = loaded['rng_state']
    np.testing.assert_array_equal(rng2.integers(0,10000,size=4), expected_draw)
    assert clone_opt.step_index == 1
    for j, (name, weight) in enumerate(baseline_weights.items()):
        np.testing.assert_array_equal(streaming._to_host(clone.parameters_by_name[name].data), weight)
        np.testing.assert_array_equal(streaming._to_host(clone_opt.m[j]), baseline_m[j])
        np.testing.assert_array_equal(streaming._to_host(clone_opt.v[j]), baseline_v[j])
        np.testing.assert_array_equal(streaming._to_host(clone_opt.master_weights[j]), baseline_master[j])
    with pytest.raises(ValueError, match='cannot resume'):
        streaming._read_last_state(path, clone, clone_opt, expected={'root':'changed'})


class _FakeSampler:
    def __init__(self, rng, context):
        self.rng, self.context = rng, context

    def sample(self):
        tokens = self.rng.integers(0, 17, size=self.context+1, dtype=np.int32)
        return tokens, Path('fictional_val_shard.bin'), int(self.rng.integers(0, 1000))


def test_online_train_generates_discards_and_resumes(tmp_path, monkeypatch, capsys):
    vocab, context = 17, 4
    root = tmp_path/'shards'
    root.mkdir()
    win_meta = tmp_path/'window_metadata.json'
    win_meta.write_text('{}')
    token_bytes = b'{"test":"shared"}'
    token_hash = hashlib.sha256(token_bytes).hexdigest()
    members = []
    for mid in (0, 1):
        ckpt = tmp_path / f'ckpt{mid}'
        ckpt.mkdir()
        (ckpt/'tokenizer.json').write_bytes(token_bytes)
        members += [f'{mid}:{ckpt}']
    val = []
    rng = np.random.default_rng(11)
    for mid in (0, 1):
        path = tmp_path/f'val_{mid}.npy'
        np.save(path, rng.normal(size=(8, vocab)).astype(np.float16))
        val += [f'{mid}:{path}']
    ypath = tmp_path/'val_targets.npy'
    np.save(ypath, rng.integers(0, vocab, size=8))

    def fake_samplers(root, metadata, partition_seed, win_context, local_rng):
        src = type('Source', (), {'name':'test', 'weight':1.0})()
        return [src], {'vocab_size':vocab, 'tokenizer_sha256':token_hash}, {
            'test': _FakeSampler(local_rng, win_context)}, np.array([1.0])

    calls = []

    def fake_export(argv):
        calls.append(list(argv))
        path = Path(argv[argv.index('--windows')+1])
        output = Path(argv[argv.index('--output-dir')+1])
        output.mkdir(parents=True, exist_ok=True)
        windows = np.load(path)
        positions = streaming._scored_positions(context)
        labels = windows[:, positions+1].reshape(-1)
        np.save(output/'targets.npy', labels)
        for mid in (0,1):
            logits = np.zeros((len(labels), vocab), dtype=np.float16)
            logits[np.arange(len(labels)), labels] = 1.0 + mid*0.1
            np.save(output/f'model_{mid}_logits.npy', logits)
        (output/'metadata.json').write_text(json.dumps({
            'tokenizer_sha256':token_hash, 'rows':len(labels),
            'token_windows_sha256':'synthetic',
        }))

    monkeypatch.setattr(streaming, '_stream_samplers', fake_samplers)
    monkeypatch.setattr(streaming, 'export_main', fake_export)
    arguments = ['--mixed-shard-root', str(root), '--windows-metadata', str(win_meta),
                 '--member', members[0], '--member', members[1],
                 '--val-member', val[0], '--val-member', val[1],
                 '--val-targets', str(ypath),
                 '--output', str(tmp_path/'best.npz'),
                 '--work-dir', str(tmp_path/'scratch'),
                 '--context-length', str(context), '--score-last-k', str(context),
                 '--batch-windows','2', '--segment-windows','2',
                 '--batch-rows','2', '--max-cache-gib','0.01',
                 '--eval-interval','2', '--log-interval','2',
                 '--early-stop-patience','0', '--steps','6']
    streaming.main(arguments)
    assert len(calls) == 2
    log = capsys.readouterr().out
    for key in ('Step 2/6', 'Validation CE', 'Models', 'vs best model',
                'vs best reference', 'arithmetic=', 'geometric=', 'weighted=',
                'fusion=', 'Best checkpoint'):
        assert key in log
    assert not (tmp_path/'scratch/active_segment').exists()
    assert (tmp_path/'best.npz').exists()
    last = tmp_path/'best_last.npz'
    with np.load(last) as state:
        assert json.loads(str(state['__meta_json__']))['step'] == 6
    lines = (tmp_path/'scratch/sample_trace.jsonl').read_text().strip().splitlines()
    assert len(lines) == 2
    streaming.main(arguments[:-2] + ['--steps','10','--resume'])
    with np.load(last) as state:
        assert json.loads(str(state['__meta_json__']))['step'] == 10
    assert len(calls) == 3
    assert len((tmp_path/'scratch/sample_trace.jsonl').read_text().strip().splitlines()) == 3
    # 0082R: switch from 2 rows/update to effective 4 rows/update on
    # an existing optimizer checkpoint without discarding optimizer moments.
    # A segment has 8 supervised rows, hence two updates per segment.
    streaming.main(arguments[:-2] + ['--steps','12', '--grad-accum-steps','2', '--resume'])
    with np.load(last, allow_pickle=False) as state:
        info = json.loads(str(state['__meta_json__']))
        assert info['step'] == 12
        assert info['grad_accum_steps'] == 2
        assert info['telemetry']['total_rows'] == 28
    assert len(calls) == 4
    # A non-divisible final accumulation is row-weighted and retained:
    # 8 available labels / (2 rows * 3 microbatches) = 6 + 2 labels.
    streaming.main(arguments[:-2] + ['--steps','14', '--grad-accum-steps','3', '--resume'])
    with np.load(last, allow_pickle=False) as state:
        info = json.loads(str(state['__meta_json__']))
        assert info['step'] == 14
        assert info['telemetry']['total_rows'] == 36
        assert info['grad_accum_steps'] == 3
    assert len(calls) == 5


def test_actual_tiny_models_and_packed_shards_end_to_end(tmp_path):
    """Exercise the REAL exporter plus actual held-out shard partitioning."""
    import dataclasses
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.checkpoint import save_checkpoint
    from examples.prepare_probability_fusion_windows_0082b import prepare_windows
    from examples.export_probability_fusion_logits_0082a import main as export

    root = tmp_path/'mix'
    (root/'tiny').mkdir(parents=True)
    tokenizer = b'{"tiny": "same vocabulary"}'
    (root/'tokenizer.json').write_bytes(tokenizer)
    digest = hashlib.sha256(tokenizer).hexdigest()
    def tokens():
        # EOS only at valid document boundaries; plenty of complete windows.
        result = [1]
        for i in range(15):
            result.extend([2 + ((i+j)%15) for j in range(12)])
            result.append(1)
        return np.asarray(result, dtype=np.uint16)
    (root/'tiny/train_shard_000.bin').write_bytes(tokens().tobytes())
    for i in range(6):
        (root/f'tiny/val_shard_{i:03d}.bin').write_bytes(tokens().tobytes())
    manifest = {'format_version':'weighted_packed_mixture_v1',
                'complete':True, 'tokenizer_sha256':digest,
                'vocab_size':17, 'token_dtype':'uint16', 'eos_token_id':1,
                'sources':{'tiny':{'weight':1.0, 'relative_shard_dir':'tiny',
                                   'train_shards':1,'val_shards':6}}}
    (root/'mixture_manifest.json').write_text(json.dumps(manifest))
    window_dir = tmp_path/'windows'
    prepare_windows(root, window_dir, 4, (3, 2, 2), 82)
    members = []
    for mid in (0, 1):
        ckpt = tmp_path/f'member_{mid}'
        ckpt.mkdir()
        (ckpt/'tokenizer.json').write_bytes(tokenizer)
        config = ModelConfig(tokenizer_vocab_size=17, context_length=4, n_layers=2,
                             d_model=8, n_q_heads=1, n_kv_heads=1, d_head=8,
                             n_experts=1, top_k=1, d_ff=16, ffn_mode='dense')
        (ckpt/'config.json').write_text(json.dumps(dataclasses.asdict(config)))
        model = DecoderLanguageModel(config, rng_seed=200+mid, dtype='float32')
        save_checkpoint(ckpt, {p.name:p.data for p in model.parameters()})
        members += [f'{mid}:{ckpt}']
    val_dir = tmp_path/'val_logits'
    export(['--member', members[0], '--member', members[1],
            '--windows', str(window_dir/'fusion_val_windows.npy'),
            '--context-length','4', '--score-last-k','4',
            '--batch-windows','2','--lm-head-rows','3',
            '--output-dir',str(val_dir)])
    streaming.main([
        '--member', members[0], '--member', members[1],
        '--mixed-shard-root',str(root),
        '--windows-metadata',str(window_dir/'windows_metadata.json'),
        '--val-member', f'0:{val_dir}/model_0_logits.npy',
        '--val-member', f'1:{val_dir}/model_1_logits.npy',
        '--val-targets',str(val_dir/'targets.npy'),
        '--output',str(tmp_path/'fusion_best.npz'),
        '--work-dir',str(tmp_path/'scratch'), '--context-length','4',
        '--score-last-k','4', '--batch-rows','2','--batch-windows','2',
        '--segment-windows','2','--lm-head-rows','3',
        '--steps','5','--eval-interval','2', '--early-stop-patience','0',
        '--log-interval','1',
    ])
    assert (tmp_path/'fusion_best.npz').exists()
    with np.load(tmp_path/'fusion_best_last.npz', allow_pickle=False) as obj:
        assert json.loads(str(obj['__meta_json__']))['step'] == 5
    trace = (tmp_path/'scratch/sample_trace.jsonl').read_text().strip().splitlines()
    assert len(trace) == 2
    meta = json.loads((window_dir/'windows_metadata.json').read_text())
    val_shards = set(meta['splits']['val']['validation_shards']['tiny'])
    test_shards = set(meta['splits']['test']['validation_shards']['tiny'])
    for line in trace:
        for _, shard, _ in json.loads(line)['sample_locations']:
            assert str(Path(shard).relative_to(root)) not in val_shards | test_shards
