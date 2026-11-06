"""Incremental decoding parity for the optional PyTorch backend."""

from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from mini_llm.backends.pytorch.inference import TorchGenerationEngine
from mini_llm.backends.pytorch.model import TorchDecoderLanguageModel
from mini_llm.config import ModelConfig


def _config():
    base = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    return replace(
        base, tokenizer_vocab_size=48, context_length=32, n_layers=2,
        d_model=32, n_q_heads=4, n_kv_heads=2, d_head=8, d_ff=64,
        dual_resolution=replace(
            base.dual_resolution, block_size=4, local_window=4,
            long_local_window=8, long_local_heads=1, max_blocks=8,
            summaries_per_block=2, global_heads=2, global_head_dim=8,
            sparse_retrieval_stride=16, sparse_retrieval_head=1,
            sparse_retrieval_layers=(0, 1),
        ),
    )


def test_incremental_model_b_matches_full_prefix_logits():
    torch.manual_seed(9)
    config = _config()
    model = TorchDecoderLanguageModel(
        config, dtype="float32", device="cpu", activation_checkpointing=False
    ).eval()
    ids = torch.randint(0, config.vocab_size, (1, 11))
    engine = TorchGenerationEngine(model)
    state = engine.new_state(batch_size=1, max_length=ids.shape[1])
    with torch.no_grad():
        for index in range(ids.shape[1]):
            incremental = engine.decode_one(ids[:, index:index + 1], state)
            full = model(ids[:, :index + 1])[:, -1:]
            torch.testing.assert_close(incremental, full, rtol=2e-5, atol=2e-5)
    assert state.position == ids.shape[1]
    assert all(layer.summary_count == 4 for layer in state.layers)
