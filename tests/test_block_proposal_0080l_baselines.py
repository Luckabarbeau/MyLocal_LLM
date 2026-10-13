"""0080L: frozen NumPy baselines from the unmodified 0080K archive.

No objective adapter is introduced in 0080L.  ``_production_ar_step`` is a
literal invocation of the existing model forward/loss/backward interface, used
to pin the future 0080M AR adapter without changing production training.
The JSON reference was captured before any production source was modified.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.config import (
    AttentionLayerConfig, ContextRouterConfig, LocalAttentionConfig,
    MemoryContextConfig, ModelConfig, RetrievalAttentionConfig,
)
from mini_llm.block_proposal.baseline import MatchedAutoregressiveTrainer
from mini_llm.block_proposal.checkpoint import (
    save_two_stage_checkpoint, load_two_stage_checkpoint,
)
from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.real_data import PackedRoutedBlockProposalDocuments
from mini_llm.block_proposal.training import SinglePassBlockTrainer
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW

REF = Path(__file__).parent / 'fixtures' / '0080l_cpu_reference.json'
pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='frozen CPU float64 reference')


def _configuration():
    memory = MemoryContextConfig(
        enabled=True, memory_length=24, recent_length=8, target_length=8,
        block_size=4, top_k_blocks=2, router_query_length=8, router_dim=4,
        integration_mode='routed_prefix', memory_training='joint',
        router_gumbel_noise=False, read_heads=1, read_kv_heads=1,
    )
    router = ContextRouterConfig(
        history_block_size=2, routing_stride=4, query_window=4,
        router_dim=2, top_k_blocks=2, exclude_recent_tokens=2,
        query_pooling='mean', history_pooling='mean', num_queries=1,
    )
    heads = AttentionLayerConfig(heads=(
        LocalAttentionConfig(window=8), RetrievalAttentionConfig(router),
    ))
    return ModelConfig(
        tokenizer_vocab_size=101, context_length=memory.active_length,
        n_layers=2, d_model=16, n_q_heads=2, n_kv_heads=1, d_head=8,
        d_ff=32, n_experts=2, top_k=1, memory_context=memory,
        attention_layers=(heads, heads), dtype='float64',
    )


def _make(k=8):
    main = DecoderLanguageModel(_configuration(), rng_seed=8080, dtype='float64')
    net = TwoStageBlockModel(
        HorizonProposalHead(main, k, hidden_width=12, rng_seed=8081),
        BlockPrefixDecoder(main, k, width=12, n_layers=1, n_heads=3, rng_seed=8082),
        lambda_proposal=0.5,
    )
    return SinglePassBlockTrainer(net)


def _window():
    src = np.mod(np.arange(48).reshape(2, 24) * 7 + 3, 101).astype(np.int64)
    labels = np.mod(src[:, 16:24] * 13 + 1, 101)
    mask = np.ones((2, 8), dtype=np.bool_)
    mask[1, 5:] = False
    labels[1, 5:] = -100
    src[1, 16 + 5:] = 0
    meta = dict(history_valid_starts=np.array([0, 0], dtype=np.int64),
                target_valid_lengths=np.array([8, 5], dtype=np.int64),
                causal_prefix_query=True, causal_masked_target_refinement=True,
                router_stochastic=False, use_retrieval=True)
    return src, labels, mask, meta


def _slice_window(row):
    src, labels, mask, meta = _window()
    md = {key: (val[row:row+1].copy() if isinstance(val, np.ndarray) else val)
          for key, val in meta.items()}
    return src[row:row+1], labels[row:row+1], mask[row:row+1], md


def _gradients(parameters):
    return {p.name: np.asarray(asnumpy(p.grad), dtype=np.float64).copy()
            for p in parameters}


def _sample_parameters(parameters):
    """Deterministic numeric signature for every trainable parameter group."""
    signatures = {}
    grads = _gradients(parameters)
    for name, grad in grads.items():
        if name.startswith('embedding'):
            group = 'embedding'
        elif name.startswith('output_proj'):
            group = 'output_proj'
        elif name.startswith('memory_router'):
            group = 'top_level_router'
        elif 'retriev' in name or 'router' in name:
            group = 'layer_retrieval'
        elif name.startswith('block_proposal'):
            group = 'proposal'
        elif name.startswith('block_decoder'):
            group = 'decoder'
        else:
            group = 'main_trunk'
        entry = signatures.setdefault(group, {'sum_sq': 0.0, 'max_abs': 0.0,
                                             'parameter_count': 0})
        entry['sum_sq'] += float(np.dot(grad.ravel(), grad.ravel()))
        entry['max_abs'] = max(entry['max_abs'], float(np.max(np.abs(grad))))
        entry['parameter_count'] += 1
    return {key: {'l2': float(np.sqrt(value['sum_sq'])),
                  'max_abs': value['max_abs'],
                  'parameter_count': value['parameter_count']}
            for key, value in sorted(signatures.items())}


def _numeric_snapshot(trainer, k, group_size):
    source, labels, mask, meta = _window()
    trainer.zero_grad()
    metrics = trainer.train_window(source, labels, valid_target_mask=mask,
                                   memory_metadata=meta, block_group_size=group_size)
    counts = np.asarray(asnumpy(metrics['position_counts']), dtype=np.int64)
    proposal = np.asarray(asnumpy(metrics['proposal_position_ce_sums']), dtype=np.float64)
    final = np.asarray(asnumpy(metrics['final_position_ce_sums']), dtype=np.float64)
    params = trainer.parameters()
    signature = {
        'k': k,
        'supervised_tokens': int(metrics['unique_supervised_tokens']),
        'main_forward_passes': int(metrics['main_forward_passes']),
        'main_backward_passes': int(metrics['main_backward_passes']),
        'total_loss': float(metrics['total_loss']),
        'final_ce': float(metrics['final_ce']),
        'proposal_ce': float(metrics['proposal_ce']),
        'horizon_counts': counts.tolist(),
        'horizon_proposal_ce': (proposal / np.maximum(counts, 1)).tolist(),
        'horizon_final_ce': (final / np.maximum(counts, 1)).tolist(),
        'gradients': _sample_parameters(params),
    }
    optimizer = AdamW(params, lr=1e-4, weight_decay=0.1)
    before = {p.name: asnumpy(p.data).copy() for p in params}
    optimizer.step()
    signature['optimizer_delta_l2'] = float(np.sqrt(sum(
        np.sum((asnumpy(p.data) - before[p.name]) ** 2) for p in params)))
    signature['optimizer_step'] = int(optimizer.step_index)
    return signature


def _production_ar_step(main, source, labels, mask, md):
    """Exact original LM path used by production's non-chunked CE branch."""
    main.zero_grad()
    logits, cache = main.forward(source, memory_metadata=md)
    loss, loss_cache = main.compute_loss(logits, np.where(mask, labels, 0),
                                         loss_mask=mask)
    main.backward(main.backward_loss(loss_cache), cache)
    return float(loss), _gradients(main.parameters())


def _read_reference():
    return json.loads(REF.read_text(encoding='utf8'))


def _close_dict(actual, expected, *, rtol=2e-8, atol=2e-10):
    assert actual.keys() == expected.keys()
    for key in actual:
        a, b = actual[key], expected[key]
        if isinstance(a, dict):
            _close_dict(a, b, rtol=rtol, atol=atol)
        elif isinstance(a, list):
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=key)
        elif isinstance(a, float):
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=key)
        else:
            assert a == b, key


def _sample_trace(tmp_path):
    # Include short documents with incomplete blocks and multiple EOS boundaries.
    documents = [
        [11, 12, 13, 99],
        [21, 22, 23, 24, 25, 26, 99],
        [31, 32, 33, 34, 35, 36, 37, 38, 39, 99],
        [41, 42, 99],
    ]
    arr = np.asarray([v for doc in documents for v in doc], dtype='<u2')
    file = tmp_path / 'train_shard_000.bin'
    arr.tofile(file)
    arr.tofile(tmp_path / 'val_shard_000.bin')
    tokenizer_bytes = b'{"vocab_size":101,"eos_token_id":99,"fixture":"0080L"}\n'
    (tmp_path / 'tokenizer.json').write_bytes(tokenizer_bytes)
    tokenizer_sha256 = hashlib.sha256(tokenizer_bytes).hexdigest()
    manifest = {
        'format_version': 'weighted_packed_mixture_v1',
        'tokenizer_sha256': tokenizer_sha256,
        'token_dtype': 'uint16',
        'sources': {
            'synthetic': {
                'weight': 1.0, 'relative_shard_dir': '.',
                'train_shards': 1, 'val_shards': 1,
                'train_tokens': int(arr.size), 'val_tokens': int(arr.size),
            }
        },
    }
    # Hashes cover the raw manifest bytes.  write_text() translates LF to
    # CRLF on Windows, changing the hash without changing the JSON data.
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(',', ':'))
                      + '\n').encode('utf8')
    (tmp_path / 'mixture_manifest.json').write_bytes(manifest_bytes)
    sampler = PackedRoutedBlockProposalDocuments(
        mixed_root=tmp_path, split='train', eos_token_id=99,
        token_dtype='uint16', write_sidecars=False)
    rng = np.random.default_rng(8080)
    records = []
    for _ in range(3):
        x, y, valid, metadata, obs = sampler.sample_causal(
            rng, batch_size=3, memory_length=24, target_length=8)
        # All targets are the true next token within the sampled document.
        for row in range(3):
            n = int(valid[row].sum())
            assert n > 0
            np.testing.assert_array_equal(y[row, :n - 1], x[row, 25:24+n])
            if n < 8:
                assert y[row, n - 1] == 99
            assert int(metadata['target_valid_lengths'][row]) == n
            assert int(metadata['history_valid_starts'][row]) >= 0
        records.append(dict(x=x.tolist(), y=y.tolist(), valid=valid.tolist(),
                            starts=metadata['history_valid_starts'].tolist(),
                            lengths=metadata['target_valid_lengths'].tolist(),
                            observed=obs))
    payload = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    return dict(
        shard_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
        tokenizer_sha256=tokenizer_sha256,
        manifest_sha256=sampler.provenance()['manifest_sha256'],
        sampler_trace_sha256=hashlib.sha256(payload).hexdigest(),
        supervised_tokens=sum(sum(record['lengths']) for record in records),
    )


def test_frozen_document_first_sampling_and_tokenizer_fingerprint(tmp_path):
    assert _sample_trace(tmp_path) == _read_reference()['data']


@pytest.mark.parametrize('k', [1, 8])
def test_frozen_0080_loss_per_horizon_gradients_and_adamw(k):
    recorded = _numeric_snapshot(_make(k), k, group_size=2)
    _close_dict(recorded, _read_reference()['two_stage'][str(k)])
    assert recorded['supervised_tokens'] == 13
    assert recorded['main_forward_passes'] == recorded['main_backward_passes'] == 1
    assert recorded['gradients']['embedding']['l2'] > 0
    assert recorded['gradients']['output_proj']['l2'] > 0
    assert recorded['gradients']['decoder']['l2'] > 0
    assert recorded['gradients']['top_level_router']['l2'] > 0
    if k > 1:
        assert recorded['gradients']['proposal']['l2'] > 0


def test_original_production_ar_matches_existing_research_ar_helper():
    source, labels, mask, md = _window()
    baseline = DecoderLanguageModel(_configuration(), rng_seed=8080, dtype='float64')
    adapter = DecoderLanguageModel(_configuration(), rng_seed=8080, dtype='float64')
    loss, grad = _production_ar_step(baseline, source, labels, mask, md)
    ref = _read_reference()['ar']
    np.testing.assert_allclose(loss, ref['final_ce'], rtol=1e-10, atol=1e-10)
    _close_dict(_sample_parameters(baseline.parameters()), ref['gradients'])
    trainer = MatchedAutoregressiveTrainer(adapter)
    trainer.zero_grad()
    got = trainer.train_window(source, labels, valid_target_mask=mask,
                               memory_metadata=md)
    np.testing.assert_allclose(got['final_ce'], loss, rtol=0, atol=1e-12)
    for param in adapter.parameters():
        np.testing.assert_allclose(asnumpy(param.grad), grad[param.name],
                                   rtol=1e-9, atol=1e-10, err_msg=param.name)
    opt1 = AdamW(baseline.parameters(), lr=1e-4, weight_decay=0.1)
    opt2 = AdamW(adapter.parameters(), lr=1e-4, weight_decay=0.1)
    before = {p.name: asnumpy(p.data).copy() for p in baseline.parameters()}
    opt1.step(); opt2.step()
    delta_l2 = float(np.sqrt(sum(np.sum((asnumpy(p.data) - before[p.name])**2)
                                 for p in baseline.parameters())))
    np.testing.assert_allclose(delta_l2, ref['optimizer_delta_l2'],
                               rtol=1e-10, atol=1e-10)
    for p, q in zip(baseline.parameters(), adapter.parameters()):
        np.testing.assert_allclose(asnumpy(p.data), asnumpy(q.data), rtol=0, atol=1e-11)


@pytest.mark.parametrize('k', [1, 8])
def test_chunk_sizes_and_microbatch_weighting_preserve_every_gradient(k):
    source, labels, valid, md = _window()
    def run(group_size):
        trainer = _make(k)
        trainer.zero_grad()
        metrics = trainer.train_window(source, labels, valid_target_mask=valid,
                                        memory_metadata=md, block_group_size=group_size)
        return metrics, _gradients(trainer.parameters())
    full, ref_grads = run(64)
    for group in [1, 2]:
        metric, grads = run(group)
        for name in ('proposal_ce', 'final_ce', 'total_loss'):
            np.testing.assert_allclose(metric[name], full[name], rtol=1e-8, atol=1e-9)
        for name in grads:
            np.testing.assert_allclose(grads[name], ref_grads[name],
                                       rtol=1e-5, atol=1e-7, err_msg=name)
    micro = _make(k)
    micro.zero_grad()
    for row in [0, 1]:
        s, y, m, info = _slice_window(row)
        micro.train_window(s, y, valid_target_mask=m, memory_metadata=info,
                           block_group_size=1, grad_scale=float(m.sum()) / 13)
    for name, grad in _gradients(micro.parameters()).items():
        np.testing.assert_allclose(grad, ref_grads[name],
                                   rtol=1e-5, atol=1e-7, err_msg=name)


def test_future_targets_and_source_suffix_do_not_modify_first_proposal():
    main = _make(k=8).model
    src, labels, valid, md = _window()
    hidden1 = main.forward_body(src, memory_metadata=md, return_cache=False)
    changed = src.copy()
    changed[:, 17:] = (changed[:, 17:] + 19) % 101
    hidden2 = main.forward_body(changed, memory_metadata=md, return_cache=False)
    prefix = hidden1.shape[1] - labels.shape[1]
    np.testing.assert_allclose(hidden1[:, prefix, :], hidden2[:, prefix, :],
                               rtol=0, atol=1e-12)
    head = HorizonProposalHead(main, 8, hidden_width=12, rng_seed=8081)
    logits1 = head.forward(hidden1[:, prefix, :], return_cache=False)
    standard_lm_logits = main.forward(src, memory_metadata=md, return_cache=False)
    np.testing.assert_allclose(logits1[:, 0], standard_lm_logits[:, 0],
                               rtol=0, atol=1e-12)
    logits2 = head.forward(hidden2[:, prefix, :], return_cache=False)
    np.testing.assert_allclose(logits1, logits2, rtol=0, atol=1e-12)
    # Retain the stricter pre-window route requirement even if source changes.
    disallowed = dict(md, causal_prefix_query=False)
    with pytest.raises(ValueError, match='noncausal'):
        _make(k=8).train_window(src, labels, valid_target_mask=valid,
                                memory_metadata=disallowed)


def test_two_stage_full_window_checkpoint_resume_with_fixed_validation(tmp_path):
    a, b = _make(k=8), _make(k=8)
    s, y, mask, metadata = _window()
    opt_a = AdamW(a.parameters(), lr=1e-4, weight_decay=0.1)
    opt_b = AdamW(b.parameters(), lr=1e-4, weight_decay=0.1)
    def step(trainer, optimizer):
        trainer.zero_grad()
        stats = trainer.train_window(s, y, valid_target_mask=mask,
                                     memory_metadata=metadata, block_group_size=2)
        optimizer.step()
        return stats
    step(a, opt_a)
    path = tmp_path / 'baseline_checkpoint'
    data_provenance = {'manifest_sha256': '0080L-fixture', 'source': 'synthetic-eos'}
    save_two_stage_checkpoint(
        path, a.net, optimizer=opt_a, training_step=1,
        supervised_tokens=13, tokenizer_hash='0080L-fixture-tokenizer',
        data_provenance=data_provenance, rng_state={'seed': 8080},
    )
    state = load_two_stage_checkpoint(
        path, b.net, optimizer=opt_b,
        expected_tokenizer_hash='0080L-fixture-tokenizer',
        expected_data_provenance=data_provenance,
    )
    assert state['step'] == 1 and state['supervised_tokens'] == 13
    assert opt_b.step_index == 1
    va = a.train_window(s, y, valid_target_mask=mask,
                        memory_metadata=metadata, compute_gradients=False,
                        block_group_size=2)
    vb = b.train_window(s, y, valid_target_mask=mask,
                        memory_metadata=metadata, compute_gradients=False,
                        block_group_size=2)
    np.testing.assert_allclose(va['total_loss'], vb['total_loss'], rtol=0, atol=0)
    first, second = step(a, opt_a), step(b, opt_b)
    np.testing.assert_allclose(first['total_loss'], second['total_loss'], rtol=0, atol=0)
    for p, q in zip(a.parameters(), b.parameters()):
        assert p.name == q.name
        np.testing.assert_allclose(p.data, q.data, rtol=0, atol=0, err_msg=p.name)
