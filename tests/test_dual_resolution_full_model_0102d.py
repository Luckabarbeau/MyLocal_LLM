"""0102D full decoder+head+optimizer smoke and gradient-recompute contracts (CPU)."""
import numpy as np

from benchmarks.benchmark_dual_resolution_full_model_0102d import parse_args, run
from mini_llm.config import DualResolutionConfig, ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata


def _model(checkpoint):
    config = ModelConfig(
        tokenizer_vocab_size=47, context_length=24, n_layers=2,
        d_model=32, d_ff=64, n_q_heads=4, n_kv_heads=2, d_head=8,
        n_experts=1, top_k=1, ffn_mode='dense', dtype='float64',
        dual_resolution=DualResolutionConfig(
            enabled=True, block_size=8, local_window=16,
            compression_dim=16, compression_heads=2, max_blocks=8,
            global_query_chunk=16, recompute_attention_probs=True,
        ),
    )
    return DecoderLanguageModel(config, rng_seed=101, dtype='float64')


def _training_result(checkpoint):
    model = _model(checkpoint)
    tokens = np.array([[3, 7, 2, 9, 8, 14, 16, 10, 5, 6, 12, 10, 9,
                        4, 2, 5, 10, 21, 11, 6, 14, 17, 3, 8, 21]], dtype=np.int64)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:][:, -5:]
    metadata = build_dual_resolution_metadata(inputs, eos_token_id=46, block_size=8)
    model.zero_grad()
    hidden, cache = model.forward_body(
        inputs, activation_checkpoint=checkpoint,
        dual_resolution_metadata=metadata,
    )
    loss, loss_cache = model.chunked_lm_head_loss_forward(
        hidden, targets, chunk_tokens=3, target_slice=(19, 24),
    )
    dx = model.chunked_lm_head_backward(loss_cache)
    model.backward_body(dx, cache)
    gradients = {p.name: np.asarray(p.grad).copy() for p in model.parameters()}
    return loss, gradients


def test_full_model_checkpoint_replay_matches_no_checkpoint():
    direct_loss, direct_grad = _training_result(False)
    replay_loss, replay_grad = _training_result(True)
    assert abs(direct_loss - replay_loss) < 1e-11
    assert direct_grad.keys() == replay_grad.keys()
    for name in direct_grad:
        np.testing.assert_allclose(direct_grad[name], replay_grad[name],
                                   atol=1e-10, rtol=1e-10, err_msg=name)
    assert any('.compress.' in name and np.any(values != 0)
               for name, values in direct_grad.items())
    assert any('.global.' in name and np.any(values != 0)
               for name, values in direct_grad.items())


def test_full_model_smoke_real_adamw_update_and_json_output(tmp_path):
    output = tmp_path / 'smoke.json'
    args = parse_args([
        '--tiny', '--layers', '2', '--context', '24',
        '--score-last-k', '5', '--activation-checkpoint', 'block',
        '--optimizer-step', '--output', str(output),
    ])
    report = run(args)
    assert output.exists()
    assert np.isfinite(report['loss'])
    assert report['parameters'] > 0
    assert report['gradients']['compress']['nonzero']
    assert report['gradients']['global']['nonzero']
    assert report['gradients']['ffn']['nonzero']
    assert report['optimizer_ms'] is not None
    assert any(point['stage'] == 'after_optimizer_update'
               for point in report['memory_samples'])


def test_matrix_pool_local_only_ablation_reports_disabled_global_branch():
    args = parse_args([
        '--tiny', '--architecture', 'matrix-pool', '--disable-global',
        '--layers', '2', '--context', '24', '--score-last-k', '4',
        '--activation-checkpoint', 'block',
    ])
    report = run(args)
    assert report['global_enabled'] is False
    assert not report['gradients']['compress']['nonzero']
    assert not report['gradients']['global']['nonzero']
    assert report['gradients']['ffn']['nonzero']


def test_full_twenty_layer_shape_contract_with_tiny_width():
    args = parse_args(['--tiny', '--layers', '20', '--context', '24',
                       '--score-last-k', '4', '--activation-checkpoint', 'block'])
    report = run(args)
    assert report['layers'] == 20
    assert report['gradients']['compress']['nonzero']
    assert report['gradients']['global']['nonzero']
    assert report['gradients']['ffn']['nonzero']
