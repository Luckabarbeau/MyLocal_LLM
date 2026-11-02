"""Compare the public TrainingRun adapter to its untouched legacy trainer.

A tiny dual-resolution document-first configuration checks numerical updates,
source/target accounting and optimizer states on the same token stream. This is
NOT a claim of full 348M GPU or throughput parity.
"""
import numpy as np

from mini_llm import Model
from mini_llm.config import ModelConfig, DualResolutionConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train_extended import ExtendedTrainer


def _model(seed=17):
    cfg = ModelConfig(tokenizer_vocab_size=128, context_length=16,
        n_layers=1, d_model=16, d_ff=32, n_q_heads=2, n_kv_heads=1,
        d_head=8, n_experts=1, top_k=1, ffn_mode='dense', dtype='float32',
        dual_resolution=DualResolutionConfig(enabled=True, block_size=4,
            local_window=8, compression_dim=8, compression_heads=2,
            compression_mode='matrix_pool', global_heads=1,
            global_head_dim=8, max_blocks=16, global_query_chunk=4))
    return DecoderLanguageModel(cfg, rng_seed=seed)


def test_training_run_is_step_exact_legacy_with_dual_document_sampling(tmp_path, monkeypatch):
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_SHAPE_BUCKET', 'fine')
    monkeypatch.setenv('MINI_LLM_OPTIMIZER_OFFLOAD', 'none')
    shard = tmp_path / 'train.bin'
    # EOS-bounded short documents, including an incomplete tail.
    np.array([1, 2, 3, 4, 127, 6, 7, 8, 9, 10, 127, 15, 16, 17, 18,
              19, 20, 21, 22, 127], dtype=np.uint16).tofile(shard)
    common = dict(train_shard_paths=[str(shard)], val_shard_paths=[str(shard)],
                  eos_token_id=127, batch_size=1, seq_length=16,
                  grad_accum_steps=2, gradient_accumulation_weighting='supervised_tokens',
                  optimizer_kind='hybrid-muon', muon_matmul_precision='fp32',
                  total_steps=6, warmup_steps=2, val_interval=100,
                  save_interval=100, val_steps=1, rng_seed=42)
    direct_model = _model()
    public_model = Model.wrap_native(_model())
    legacy = ExtendedTrainer(model=direct_model, **common)
    public = public_model.train(**common)
    for _ in range(2):
        direct_losses = legacy.train(num_steps=1, log_interval=100)
        public_losses = public.train_steps(1, log_interval=100)
        np.testing.assert_allclose(direct_losses, public_losses, atol=1e-7, rtol=1e-7)
        assert legacy.supervised_tokens_total == public.supervised_tokens
        assert legacy.step == public.step
        assert legacy.sample_trace_sha256 == public.trainer.sample_trace_sha256
        for p, q in zip(legacy.model.parameters(), public_model._native_model.parameters()):
            assert p.name == q.name
            np.testing.assert_allclose(p.data, q.data, rtol=0, atol=0)
        assert legacy.optimizer.step_index == public.trainer.optimizer.step_index
    assert legacy.supervised_tokens_total > 0
