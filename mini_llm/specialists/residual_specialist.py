"""Phase 2 reference prototype: independently trainable causal residual blocks.

The zero-initialized final projection guarantees exact foundation parity. Only
specialist parameters receive gradients. Historical information in Phase 2 is
inherited through the foundation's contextual hidden states; private readers
are a separate unimplemented future adapter, not silently disabled.
"""

from dataclasses import dataclass

from mini_llm.backend import RandomStream, xp, resolve_dtype
from mini_llm.blocks.transformer_block import TransformerBlock
from mini_llm.config import AttentionLayerConfig, LocalAttentionConfig
from mini_llm.ops.linear import Linear


@dataclass(frozen=True)
class SpecialistConfig:
    specialist_id: str = "python"
    n_layers: int = 2  # implementation smoke test; 12 is the practical target
    d_model: int = 1024
    d_ff: int = 4096
    n_q_heads: int = 16
    n_kv_heads: int = 4
    d_head: int = 64
    local_window: int = 512
    full_window: int = 4096
    full_heads: int = 1
    dtype: str = "bfloat16"
    init_std: float = 0.02
    rms_eps: float = 1e-6
    rope_base: float = 10000.0
    memory_mode: str = "foundation_only"
    # Future `private` readers will use this independently of the foundation:
    history_horizon: int | None = None
    history_block_size: int = 32
    history_retrieval_blocks: int = 8

    def __post_init__(self):
        if not self.specialist_id or not all(c.isalnum() or c in "-_" for c in self.specialist_id):
            raise ValueError("specialist_id must be alphanumeric, '-' or '_'")
        if self.n_layers <= 0 or self.d_model <= 0 or self.d_ff <= 0:
            raise ValueError("specialist size must be positive")
        if self.n_q_heads * self.d_head != self.d_model:
            raise ValueError("n_q_heads * d_head must equal d_model")
        if self.n_q_heads % self.n_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if not 1 <= self.full_heads <= self.n_q_heads:
            raise ValueError("full_heads outside available attention heads")
        if self.local_window <= 0 or self.full_window <= 0:
            raise ValueError("attention windows must be positive")
        if self.memory_mode not in {"foundation_only", "private"}:
            raise ValueError("unknown specialist memory_mode")
        if self.history_horizon is not None and self.history_horizon <= 0:
            raise ValueError("history_horizon must be positive")
        if self.history_block_size <= 0 or self.history_retrieval_blocks <= 0:
            raise ValueError("history block sizes must be positive")


class ResidualSpecialist:
    """Trainable ΔH = W_out(Transformer(H_frozen)), sharing foundation LM head."""

    def __init__(self, config: SpecialistConfig, *, rng_seed=0):
        if config.memory_mode != "foundation_only":
            raise NotImplementedError(
                "private specialist historical readers need their own retrieval "
                "schedule and causal source-history integration"
            )
        self.config = config
        self.dtype = resolve_dtype(config.dtype)
        rng = RandomStream(rng_seed)
        name = f"specialists.{config.specialist_id}"
        # Head zero is full-current; other heads are strictly causal local-512.
        # Reuse the optimized per-head attention kernels, not dense 16-head attention.
        attention = AttentionLayerConfig(heads=(
            *(LocalAttentionConfig(window=config.full_window)
              for _ in range(config.full_heads)),
            *(LocalAttentionConfig(window=config.local_window)
              for _ in range(config.n_q_heads - config.full_heads)),
        ))
        self.blocks = [TransformerBlock(
            d_model=config.d_model, n_q_heads=config.n_q_heads,
            n_kv_heads=config.n_kv_heads, d_head=config.d_head,
            d_ff=config.d_ff, n_experts=1, top_k=1,
            input_std=config.init_std,
            output_std=config.init_std / (2 * config.n_layers)**0.5,
            rope_base=config.rope_base, rng=rng, eps=config.rms_eps,
            name=f"{name}.blocks.{i}", dtype=config.dtype,
            attention_config=attention, ffn_mode="dense",
        ) for i in range(config.n_layers)]
        self.correction_proj = Linear(
            config.d_model, config.d_model,
            config.init_std, rng, name=f"{name}.correction_proj", dtype=config.dtype
        )
        self.correction_proj.W.data[...] = 0

    def parameters(self):
        return [*(p for block in self.blocks for p in block.parameters()),
                *self.correction_proj.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def refresh_compute_buffers(self):
        for block in self.blocks:
            block.refresh_compute_buffers()

    def forward_delta(self, frozen_hidden, *, position_ids=None,
                      return_cache=True, activation_checkpoint=False):
        """Calculate correction. The foundation hidden tensor is never modified."""
        if frozen_hidden.shape[-1] != self.config.d_model:
            raise ValueError("specialist/foundation hidden width mismatch")
        # First no-history implementation operates on a contiguous causal
        # working sequence. `position_ids` can preserve its absolute source
        # coordinates when provided by the foundation training batch.
        x = frozen_hidden
        checkpoint = bool(return_cache and activation_checkpoint)
        checkpoints = []
        caches = []
        for idx, block in enumerate(self.blocks):
            if checkpoint:
                checkpoints.append(x)
                x = block.forward(x, layer_idx=idx, position_ids=position_ids,
                                  return_cache=False)
            elif return_cache:
                x, cache = block.forward(x, layer_idx=idx, position_ids=position_ids)
                caches.append(cache)
            else:
                x = block.forward(x, layer_idx=idx, position_ids=position_ids,
                                  return_cache=False)
        # Tensor-core GEMMs require inputs in the configured low-precision type.
        projection_input = x.astype(self.dtype, copy=False)
        delta, projection_cache = self.correction_proj.forward(projection_input)
        if not return_cache:
            return delta
        return delta, {
            "block_caches": caches, "checkpoints": checkpoints,
            "activation_checkpoint": checkpoint, "position_ids": position_ids,
            "projection_cache": projection_cache,
        }

    def backward_delta(self, ddelta, cache):
        """Accumulate specialist-only gradients; return dH0 for diagnostics."""
        dx = self.correction_proj.backward(ddelta, cache.pop("projection_cache"))
        if cache["activation_checkpoint"]:
            for idx in range(len(self.blocks) - 1, -1, -1):
                block_input = cache["checkpoints"].pop()
                _, replay_cache = self.blocks[idx].forward(
                    block_input, layer_idx=idx, position_ids=cache["position_ids"]
                )
                dx = self.blocks[idx].backward(dx, replay_cache)
        else:
            for idx in range(len(self.blocks) - 1, -1, -1):
                dx = self.blocks[idx].backward(dx, cache["block_caches"].pop())
        return dx

    @staticmethod
    def corrected_hidden(frozen_hidden, delta):
        """No post-addition normalization: zero delta exactly recovers H0."""
        return frozen_hidden + delta
