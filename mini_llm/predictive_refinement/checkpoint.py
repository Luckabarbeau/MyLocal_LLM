"""0079E strict, opt-in checkpoint metadata for predictive-refinement research.

Uses the existing MyLocal_LLM checkpoint serializer and AdamW state layout.
Legacy autoregressive checkpoints are *never* implicitly treated as
predictive-refinement checkpoints, even when the backbone tensor shapes match.
"""

import json
from pathlib import Path

from mini_llm.checkpoint import load_checkpoint, save_checkpoint


ARCHITECTURE = "block_causal_predictive_refinement_v1"
CACHE_VERSION = "dense_kv_provisional_v1"


def _architecture_metadata(reference, *, tokenizer_fingerprint=None, data_provenance=None):
    cfg = reference.config
    model = reference.model
    return {
        "type": ARCHITECTURE,
        "block_size": int(cfg.block_size),
        "initial_prediction_loss_weight": float(cfg.initial_loss_weight),
        "feedback_type": (
            "full_vocabulary_soft_token_expectation" if cfg.feedback_mode == "full"
            else "unknown_position_control"
        ),
        "feedback_temperature": float(cfg.feedback_temperature),
        "gradient_through_feedback": cfg.feedback_mode == "full",
        "refinement_passes_per_block": int(cfg.block_size) + 1,
        "block_attention": "committed_causal_provisional_bidirectional",
        "commitment_policy_training": "teacher_force_after_scoring",
        "cache_version": CACHE_VERSION,
        "model_config": {
            "vocab_size": int(model.config.vocab_size),
            "context_length": int(model.config.context_length),
            "rope_base": float(model.config.rope_base),
            "rms_eps": float(model.config.rms_eps),
            "progressive_depth": bool(model.config.progressive_depth),
            "active_layers": int(model.active_layers),
            "d_model": int(model.config.d_model),
            "n_layers": int(model.config.n_layers),
            "n_q_heads": int(model.config.n_q_heads),
            "n_kv_heads": int(model.config.n_kv_heads),
            "d_head": int(model.config.d_head),
            "d_ff": int(model.config.d_ff),
            "n_experts": int(model.config.n_experts),
            "top_k": int(model.config.top_k),
            "dtype": str(model.dtype),
        },
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "data_provenance": data_provenance,
    }


def save_refinement_checkpoint(
    path, reference, *, optimizer=None, training_step=0,
    supervised_tokens=0, tokenizer_fingerprint=None, data_provenance=None,
):
    """Save shared Transformer + state embeddings, opt. exact AdamW moments."""
    metadata = _architecture_metadata(
        reference,
        tokenizer_fingerprint=tokenizer_fingerprint,
        data_provenance=data_provenance,
    )
    params = reference.parameters()
    if len({p.name for p in params}) != len(params):
        raise ValueError("duplicate parameter names in refinement model")
    optimizer_state = None
    if optimizer is not None:
        if [id(p) for p in optimizer.parameters] != [id(p) for p in params]:
            raise ValueError("optimizer parameter set/order differs from refinement model")
        optimizer_state = {
            "step": int(optimizer.step_index),
            "m": {p.name: v for p, v in zip(params, optimizer.m)},
            "v": {p.name: v for p, v in zip(params, optimizer.v)},
            "master_weights": {
                p.name: v for p, v in zip(params, optimizer.master_weights)
            },
        }
    save_checkpoint(
        path,
        {p.name: p.data for p in params},
        optimizer_state=optimizer_state,
        training_state={
            "architecture": metadata,
            "step": int(training_step),
            "supervised_tokens": int(supervised_tokens),
            "optimizer_step": (
                None if optimizer is None else int(optimizer.step_index)
            ),
            "optimizer_hyperparameters": (
                None if optimizer is None else {
                    key: float(getattr(optimizer, key))
                    for key in ("lr", "beta1", "beta2", "eps", "weight_decay")
                }
            ),
        },
    )


def load_refinement_checkpoint(path, reference, *, optimizer=None):
    """Strictly restore a matching experiment; reject AR and incompatible K."""
    state_path = Path(path) / "training_state.json"
    if not state_path.is_file():
        raise ValueError("missing refinement architecture metadata")
    with open(state_path, encoding="utf8") as source:
        state = json.load(source)
    metadata = state.get("architecture")
    if not isinstance(metadata, dict) or metadata.get("type") != ARCHITECTURE:
        raise ValueError("not a predictive-refinement checkpoint")
    expected = _architecture_metadata(reference)
    for key in (
        "type", "block_size", "initial_prediction_loss_weight",
        "feedback_type", "feedback_temperature", "gradient_through_feedback",
        "refinement_passes_per_block", "block_attention",
        "commitment_policy_training", "cache_version", "model_config",
    ):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"incompatible refinement checkpoint metadata: {key}")
    saved_params, saved_opt, _ = load_checkpoint(
        path, skip_optimizer=(optimizer is None), skip_training=True,
    )
    params = reference.parameters()
    if set(saved_params) != {p.name for p in params}:
        raise ValueError("refinement checkpoint parameter names do not match")
    # Validate all parameter shapes before changing any model weights.
    for p in params:
        if tuple(saved_params[p.name].shape) != tuple(p.shape):
            raise ValueError(f"shape mismatch for {p.name}")
    if optimizer is not None:
        if saved_opt is None or any(
            key not in saved_opt for key in ("step", "m", "v", "master_weights")
        ):
            raise ValueError("refinement checkpoint lacks full AdamW state")
        if [id(p) for p in optimizer.parameters] != [id(p) for p in params]:
            raise ValueError("AdamW parameter list differs from checkpoint model")
        for key in ("m", "v", "master_weights"):
            if set(saved_opt[key]) != {p.name for p in params}:
                raise ValueError(f"incomplete optimizer state: {key}")
        settings = state.get("optimizer_hyperparameters") or {}
        for key in ("lr", "beta1", "beta2", "eps", "weight_decay"):
            if key not in settings or float(settings[key]) != float(getattr(optimizer, key)):
                raise ValueError(f"incompatible AdamW option: {key}")
    for p in params:
        p.data[...] = saved_params[p.name]
    for block in reference.model.blocks:
        if hasattr(block, "refresh_compute_buffers"):
            block.refresh_compute_buffers()
    if optimizer is not None:
        for i, p in enumerate(params):
            if not optimizer.restore_master_weight(
                i, saved_opt["master_weights"][p.name]
            ) or not optimizer.restore_moments(
                i, saved_opt["m"][p.name], saved_opt["v"][p.name]
            ):
                raise ValueError(f"optimizer tensor shape mismatch: {p.name}")
        optimizer.step_index = int(saved_opt["step"])
    return state
