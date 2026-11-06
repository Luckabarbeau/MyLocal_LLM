"""Focused correctness tests for the optional PyTorch model backend."""

from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from mini_llm.backends.pytorch.model import TorchDecoderLanguageModel
from mini_llm.config import ModelConfig


def tiny_model_b_config():
    base = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    return replace(
        base,
        tokenizer_vocab_size=64,
        context_length=32,
        n_layers=2,
        d_model=32,
        n_q_heads=4,
        n_kv_heads=2,
        d_head=8,
        d_ff=64,
        dual_resolution=replace(
            base.dual_resolution,
            block_size=4,
            local_window=4,
            long_local_window=8,
            long_local_heads=1,
            max_blocks=8,
            summaries_per_block=2,
            global_heads=2,
            global_head_dim=8,
            sparse_retrieval_stride=16,
            sparse_retrieval_head=1,
            sparse_retrieval_layers=(0, 1),
        ),
    )


def test_model_b_parameter_names_shapes_and_count_match_native():
    from mini_llm.model.decoder_lm import DecoderLanguageModel as NativeModel

    config = tiny_model_b_config()
    native = NativeModel(config, rng_seed=3, dtype="float32")
    model = TorchDecoderLanguageModel(
        config, dtype="float32", device="cpu", activation_checkpointing=False
    )
    native_shapes = {parameter.name: tuple(parameter.shape) for parameter in native.parameters()}
    torch_shapes = {
        name: tuple(parameter.shape) for name, parameter in model.canonical_named_parameters()
    }
    assert torch_shapes == native_shapes
    assert sum(parameter.numel() for _, parameter in model.canonical_named_parameters()) == sum(
        parameter.data.size for parameter in native.parameters()
    )


def test_model_b_forward_backward_preserves_pool_and_sparse_gradient_paths():
    torch.manual_seed(4)
    config = tiny_model_b_config()
    model = TorchDecoderLanguageModel(
        config, dtype="float32", device="cpu", activation_checkpointing=False
    )
    token_ids = torch.randint(0, config.vocab_size, (1, 9))
    targets = torch.randint(0, config.vocab_size, (1, 9))
    loss = model.forward_loss(token_ids, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for layer in range(config.n_layers):
        attention = model.blocks[layer].attention
        assert attention.matrix_pool.score.grad is not None
        assert torch.isfinite(attention.matrix_pool.score.grad).all()
        assert attention.global_module.Wq.grad is not None
        assert torch.isfinite(attention.Wq.grad).all()


def test_token_weighted_loss_ignores_unsupervised_rows():
    config = replace(tiny_model_b_config(), n_layers=1,
                     dual_resolution=replace(tiny_model_b_config().dual_resolution,
                                             sparse_retrieval_layers=(0,)))
    model = TorchDecoderLanguageModel(
        config, dtype="float32", device="cpu", activation_checkpointing=False
    )
    ids = torch.randint(0, config.vocab_size, (1, 5))
    targets = torch.randint(0, config.vocab_size, (1, 5))
    mask = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.float32)
    logits = model(ids)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :2].float().reshape(-1, config.vocab_size), targets[:, :2].reshape(-1)
    )
    actual = model.forward_loss(ids, targets, loss_mask=mask)
    torch.testing.assert_close(actual, expected)
