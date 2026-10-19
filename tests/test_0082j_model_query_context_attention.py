"""0082J: exact model-query attention gradients, causal alignment, and stream integration."""
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner
from mini_llm.collective.context_windows import AlignedContextWindows
from examples import train_probability_fusion_streaming_0082f as streaming


def _network(seed=7):
    return ProbabilityResidualCombiner(FusionConfig(vocab_size=13, max_models=4,
        embed_dim=6, member_width=4, decoder_width=7,
        context_heads=2, context_head_dim=4, context_hidden=5, seed=seed))


def _sample():
    rng = np.random.default_rng(19)
    logits = rng.normal(size=(3, 2, 13)).astype('float32')
    ids = np.array([1, 0])
    seq = rng.integers(0, 13, size=(3, 7), dtype=np.int64)
    lengths = np.array([2, 5, 7])
    y = np.array([4, 2, 8])
    return logits, ids, seq, lengths, y


def _nll(m, values):
    z, ids, seq, lens, targets = values
    p = m.forward(z, ids, context_tokens=seq, context_lengths=lens)
    return -float(np.mean(np.log(np.maximum(np.asarray(p)[np.arange(len(targets)), targets], 1e-37))))


def test_zero_context_adjustment_reproduces_weighted_reference_and_old_checkpoint(tmp_path):
    z, ids, seq, lengths, y = _sample()
    m = _network()
    original = ProbabilityResidualCombiner(FusionConfig(
        vocab_size=13, max_models=4, embed_dim=6, member_width=4,
        decoder_width=7, seed=7))
    for name in original.parameters_by_name:
        np.testing.assert_array_equal(m.parameters_by_name[name].data,
                                      original.parameters_by_name[name].data)
    m.load_state_dict(original.state_dict())
    np.testing.assert_allclose(m.forward(z, ids, context_tokens=seq, context_lengths=lengths),
                               original.forward(z, ids), rtol=1e-6, atol=1e-7)
    # Upgrade a trained pre-context model: preserve original residual and weights.
    original.reference_logits.data[1] = 1.2
    original.output_weight.data[0] = 0.3
    ckpt = tmp_path/'previous.npz'
    original.save(ckpt)
    m.load_state_dict(ProbabilityResidualCombiner.load(ckpt).state_dict())
    np.testing.assert_allclose(m.forward(z, ids, context_tokens=seq, context_lengths=lengths),
                               original.forward(z, ids), rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match='context_tokens must match'):
        m.forward(z, ids)
    with pytest.raises(ValueError, match='requires context_tokens'):
        m.member_weights(ids)
    saved = tmp_path/'with_context.npz'
    m.save(saved)
    np.testing.assert_allclose(ProbabilityResidualCombiner.load(saved).forward(
        z, ids, context_tokens=seq, context_lengths=lengths), original.forward(z, ids), atol=1e-7)


def test_context_changes_model_weights_and_model_pair_permutation():
    z, ids, seq, lengths, y = _sample()
    m = _network()
    m.ctx_score.data[:] = np.array([0.8, -0.6, 0.5, 0.3, -0.4])
    m.ctx_q.data *= 15.0
    m.ctx_k.data *= 15.0
    m.ctx_v.data *= 12.0
    m.model_embedding.data *= 5.0
    m.token_embedding.data *= 5.0
    w = np.asarray(m.member_weights(ids, context_tokens=seq, context_lengths=lengths))
    assert w.shape == (3,2)
    assert np.max(np.abs(w - w[0])) > 1e-6
    np.testing.assert_allclose(w.sum(axis=1), np.ones(3), atol=1e-6)
    pred = m.forward(z, ids, context_tokens=seq, context_lengths=lengths)
    flip = m.forward(z[:, ::-1, :], ids[::-1], context_tokens=seq, context_lengths=lengths)
    np.testing.assert_allclose(pred, flip, rtol=2e-6, atol=2e-7)
    # Replacing only future (masked) tokens must not change earlier predictions.
    modified = seq.copy()
    modified[0, lengths[0]:] = (modified[0, lengths[0]:] + 7) % 13
    modified[1, lengths[1]:] = (modified[1, lengths[1]:] + 7) % 13
    new_pred = m.forward(z, ids, context_tokens=modified, context_lengths=lengths)
    np.testing.assert_allclose(pred, new_pred, rtol=2e-6, atol=2e-7)


def test_handwritten_attention_gradients_match_finite_difference():
    values = _sample()
    m = _network()
    m.ctx_score.data[:] = np.array([0.9, -0.4, 0.5, 0.25, -0.6], dtype=np.float32)
    m.output_weight.data[:] = np.linspace(-0.04, 0.04, m.config.decoder_width)
    m.zero_grad()
    m.loss_and_backward(values[0], values[1], values[4], context_tokens=values[2], context_lengths=values[3])
    checks = [
        ('reference_logits', (1,)),
        ('ctx_score', (0,)),
        ('ctx_hidden_weight', (0, 0)),
        ('ctx_q', (0, 0)),
        ('ctx_k', (0, 0)),
        ('ctx_v', (0, 0)),
        ('ctx_recency', (1, 0)),
        ('model_embedding', (1, 0)),
        ('token_embedding', (int(values[2][0,0]), 0)),
    ]
    for name, idx in checks:
        param = m.parameters_by_name[name]
        before = float(param.data[idx])
        h = 1e-3
        param.data[idx] = before + h
        plus = _nll(m, values)
        param.data[idx] = before - h
        minus = _nll(m, values)
        param.data[idx] = before
        numeric = (plus - minus) / (2*h)
        analytic = float(param.grad[idx])
        assert np.isclose(analytic, numeric, rtol=0.055, atol=1.1e-4), (name, idx, analytic, numeric)


def test_context_input_validation():
    m = _network()
    z, ids, tokens, lengths, _ = _sample()
    with pytest.raises(ValueError, match='invalid causal'):
        m.forward(z, ids, context_tokens=tokens, context_lengths=[1,8,4])
    with pytest.raises(ValueError, match='outside vocabulary'):
        bad = tokens.copy()
        bad[1,5] = 100
        m.forward(z, ids, context_tokens=bad, context_lengths=lengths)
    with pytest.raises(ValueError, match='match logit rows'):
        m.forward(z, ids, context_tokens=tokens[:2], context_lengths=lengths[:2])


def test_context_alignment_rejects_mismatch(tmp_path):
    windows = np.array([[1,2,3,4,5],[1,3,4,5,6]], dtype=np.int32)
    file = tmp_path/'windows.npy'
    np.save(file, windows)
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    positions = [1,3]
    y = windows[:, np.array(positions)+1].reshape(-1)
    metadata = tmp_path/'metadata.json'
    info = {'token_windows_sha256':digest, 'prediction_positions':positions,
            'rows':len(y), 'tokenizer_sha256':'test'}
    metadata.write_text(json.dumps(info))
    obj = AlignedContextWindows(file,metadata,y,tokenizer_sha256='test')
    x,lengths = obj.batch([0,1,3])
    np.testing.assert_array_equal(lengths,[2,4,4])
    np.testing.assert_array_equal(x[0],windows[0,:4])
    with pytest.raises(ValueError,match='misaligned'):
        AlignedContextWindows(file, metadata, y[::-1], tokenizer_sha256='test')
    with pytest.raises(ValueError,match='tokenizer mismatch'):
        AlignedContextWindows(file, metadata, y, tokenizer_sha256='wrong')
    tampered = windows.copy()
    tampered[0,0] = 12
    np.save(file, tampered)
    with pytest.raises(ValueError,match='SHA256'):
        AlignedContextWindows(file, metadata, y, tokenizer_sha256='test')


def test_streaming_context_real_tiny_export(tmp_path):
    """Use genuine frozen LMs rather than mocking context/logit alignment."""
    import hashlib
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.checkpoint import save_checkpoint
    from examples.prepare_probability_fusion_windows_0082b import prepare_windows
    from examples.export_probability_fusion_logits_0082a import main as export
    root = tmp_path/'mix'
    (root/'tiny').mkdir(parents=True)
    tokenizer = b'{"tiny":"same"}'
    (root/'tokenizer.json').write_bytes(tokenizer)
    digest = hashlib.sha256(tokenizer).hexdigest()
    tokens = [1]
    for i in range(15):
        tokens.extend([2+(i+j)%15 for j in range(12)])
        tokens.append(1)
    for folder, names in [('train', [0]), ('val', range(6))]:
        for i in names:
            (root/f'tiny/{folder}_shard_{i:03d}.bin').write_bytes(np.asarray(tokens,dtype='uint16').tobytes())
    manifest = {'format_version':'weighted_packed_mixture_v1','complete':True,
                'tokenizer_sha256':digest,'vocab_size':17,'token_dtype':'uint16',
                'eos_token_id':1,'sources':{'tiny':{'weight':1.0,
                'relative_shard_dir':'tiny','train_shards':1,'val_shards':6}}}
    (root/'mixture_manifest.json').write_text(json.dumps(manifest))
    window_dir=tmp_path/'windows'
    prepare_windows(root,window_dir,4,(3,2,2),82)
    members=[]
    for mid in (0,1):
        ck=tmp_path/f'member_{mid}'
        ck.mkdir()
        (ck/'tokenizer.json').write_bytes(tokenizer)
        cfg=ModelConfig(tokenizer_vocab_size=17,context_length=4,n_layers=2,
                        d_model=8,n_q_heads=1,n_kv_heads=1,d_head=8,
                        n_experts=1,top_k=1,d_ff=16,ffn_mode='dense')
        (ck/'config.json').write_text(json.dumps(dataclasses.asdict(cfg)))
        m=DecoderLanguageModel(cfg,rng_seed=400+mid,dtype='float32')
        save_checkpoint(ck,{p.name:p.data for p in m.parameters()})
        members.append(f'{mid}:{ck}')
    val_dir=tmp_path/'val_logits'
    export(['--member',members[0],'--member',members[1],
            '--windows',str(window_dir/'fusion_val_windows.npy'),
            '--context-length','4','--score-last-k','4','--batch-windows','2',
            '--output-dir',str(val_dir)])
    best=tmp_path/'best.npz'
    commands = ['--member',members[0],'--member',members[1],
      '--mixed-shard-root',str(root),'--windows-metadata',str(window_dir/'windows_metadata.json'),
      '--val-member',f'0:{val_dir}/model_0_logits.npy',
      '--val-member',f'1:{val_dir}/model_1_logits.npy',
      '--val-targets',str(val_dir/'targets.npy'),
      '--val-context-windows',str(window_dir/'fusion_val_windows.npy'),
      '--val-context-metadata',str(val_dir/'metadata.json'),
      '--context-heads','2','--context-head-dim','4','--context-hidden','8',
      '--output',str(best),'--work-dir',str(tmp_path/'cache'),
      '--context-length','4','--score-last-k','4','--batch-rows','2',
      '--batch-windows','2','--segment-windows','2','--lm-head-rows','3',
      '--steps','5','--eval-interval','2','--early-stop-patience','0',
      '--log-interval','1']
    streaming.main(commands)
    assert best.is_file()
    with np.load(tmp_path/'best_last.npz',allow_pickle=False) as s:
        assert 'weight/ctx_q' in s and 'moment1/ctx_q' in s
        assert json.loads(str(s['__meta_json__']))['step']==5
    streaming.main(commands + ['--resume', '--steps', '7'])
    with np.load(tmp_path/'best_last.npz',allow_pickle=False) as s:
        assert json.loads(str(s['__meta_json__']))['step']==7


def test_contextual_attention_learns_opposite_model_specializations():
    """A global mixture cannot solve this; only the text-dependent path can.

    Train only attention/global weights, freezing residual and its vocabulary
    decoder. Equal populations of topics 0/1 have opposite best teachers.
    """
    from mini_llm.optim.adamw import AdamW
    rng = np.random.default_rng(1)
    model = ProbabilityResidualCombiner(FusionConfig(
        vocab_size=17, max_models=2, embed_dim=16, member_width=8,
        decoder_width=24, context_heads=2, context_head_dim=8,
        context_hidden=16, seed=82))
    parameters = [p for name, p in model.parameters_by_name.items()
                  if name.startswith('ctx_') or name in (
                      'reference_logits', 'model_embedding', 'token_embedding')]
    opt = AdamW(parameters, lr=0.01, weight_decay=0.0)

    def sample(n):
        topic = rng.integers(0, 2, size=n)
        y = rng.integers(2, 4, size=n)
        tokens = np.full((n, 12), 4, dtype=np.int64)
        tokens[:, [0, 5, 11]] = topic[:, None] + 10
        z = np.full((n, 2, 17), -6, dtype=np.float32)
        for j in range(n):
            z[j, topic[j], y[j]] = 4.0
            z[j, 1 - topic[j], 5 - y[j]] = 4.0
        return z, tokens, y, topic

    for _ in range(120):
        z, tokens, y, _ = sample(32)
        opt.zero_grad()
        model.loss_and_backward(z, [0, 1], y, context_tokens=tokens,
                                context_lengths=np.full(len(y), 12))
        opt.step()
    z, tokens, y, topic = sample(128)
    mixture = np.asarray(model.member_weights(
        [0, 1], context_tokens=tokens, context_lengths=np.full(len(y), 12)))
    assert mixture[topic == 0, 0].mean() > 0.90
    assert mixture[topic == 1, 1].mean() > 0.90
    prediction = model.forward(z, [0, 1], context_tokens=tokens,
                               context_lengths=np.full(len(y), 12))
    assert -np.mean(np.log(prediction[np.arange(len(y)), y])) < 0.1
