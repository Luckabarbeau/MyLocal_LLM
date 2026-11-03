"""Phase 2 reference prototype: independently trainable causal residual blocks.

The zero-initialized final projection guarantees exact foundation parity. Only
specialist parameters receive gradients. Historical information in Phase 2 is
inherited through the foundation's contextual hidden states; private readers
are a separate unimplemented future adapter, not silently disabled.
"""

from dataclasses import dataclass, replace
import os

from mini_llm.backend import RandomStream, xp, resolve_dtype
from mini_llm.blocks.transformer_block import TransformerBlock
from mini_llm.config import (
    AttentionLayerConfig, DualResolutionConfig, LocalAttentionConfig,
)
from mini_llm.ops.linear import Linear
from mini_llm.ops.rope import prepare_rope_positions


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
    residual_init_std: float | None = None
    dual_resolution: DualResolutionConfig | dict | None = None
    memory_mode: str = "foundation_only"
    # Future `private` readers will use this independently of the foundation:
    history_horizon: int | None = None
    history_block_size: int = 32
    history_retrieval_blocks: int = 8

    def __post_init__(self):
        dual = self.dual_resolution
        if isinstance(dual, dict):
            dual = DualResolutionConfig(**dual)
            object.__setattr__(self, "dual_resolution", dual)
        elif dual is not None and not isinstance(dual, DualResolutionConfig):
            raise TypeError("dual_resolution must be DualResolutionConfig-compatible")
        if dual is not None and not dual.enabled:
            raise ValueError("specialist dual_resolution must be enabled when provided")
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
        if self.residual_init_std is not None and self.residual_init_std <= 0:
            raise ValueError("residual_init_std must be positive")

    @classmethod
    def from_foundation(
            cls, foundation_config, *, specialist_id, n_layers,
            recompute_attention_probs: bool | None = None):
        """Build a truncated stack with the exact foundation block geometry."""
        dual = foundation_config.dual_resolution
        if not dual.enabled or dual.compression_mode != "matrix_pool":
            raise ValueError("Model B specialist requires matrix-pool dual resolution")
        if not 0 < int(n_layers) <= int(foundation_config.n_layers):
            raise ValueError("specialist layers must be within the foundation depth")
        # A 16-layer specialist represents Model B's complete sparse phase bank
        # (foundation layers 2..17), without the four non-sparse boundary layers.
        if (int(n_layers) == 16 and dual.sparse_retrieval_stride == 16
                and len(dual.sparse_retrieval_layers) == 16):
            dual = replace(
                dual, sparse_retrieval_phase=-1,
                sparse_retrieval_layers=tuple(range(16)))
        if recompute_attention_probs is not None:
            dual = replace(
                dual, recompute_attention_probs=bool(recompute_attention_probs))
        return cls(
            specialist_id=specialist_id, n_layers=int(n_layers),
            d_model=foundation_config.d_model, d_ff=foundation_config.d_ff,
            n_q_heads=foundation_config.n_q_heads,
            n_kv_heads=foundation_config.n_kv_heads,
            d_head=foundation_config.d_head,
            local_window=dual.local_window,
            full_window=dual.long_local_window,
            full_heads=dual.long_local_heads,
            dtype=foundation_config.dtype,
            init_std=foundation_config.init_std,
            rms_eps=foundation_config.rms_eps,
            rope_base=foundation_config.rope_base,
            residual_init_std=foundation_config.residual_init_std,
            dual_resolution=dual,
        )


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
        attention = None
        if config.dual_resolution is None:
            # Legacy prototype geometry retained for small reference tests and
            # old checkpoints. Production specialists use from_foundation().
            attention = AttentionLayerConfig(heads=(
                *(LocalAttentionConfig(window=config.full_window)
                  for _ in range(config.full_heads)),
                *(LocalAttentionConfig(window=config.local_window)
                  for _ in range(config.n_q_heads - config.full_heads)),
            ))
        output_std = (
            config.residual_init_std if config.residual_init_std is not None
            else config.init_std / (2 * config.n_layers)**0.5
        )
        sparse_phases = (0, 8, 4, 12, 2, 10, 6, 14,
                         1, 9, 5, 13, 3, 11, 7, 15)
        dual_base = config.dual_resolution
        if dual_base is not None and not isinstance(dual_base, DualResolutionConfig):
            raise TypeError("dual_resolution was not canonicalized")
        self.blocks = []
        for i in range(config.n_layers):
            dual = dual_base
            if (dual is not None and dual.sparse_retrieval_stride
                    and i in dual.sparse_retrieval_layers):
                dual = replace(
                    dual,
                    sparse_retrieval_phase=sparse_phases[
                        dual.sparse_retrieval_layers.index(i)],
                )
            self.blocks.append(TransformerBlock(
                d_model=config.d_model, n_q_heads=config.n_q_heads,
                n_kv_heads=config.n_kv_heads, d_head=config.d_head,
                d_ff=config.d_ff, n_experts=1, top_k=1,
                input_std=config.init_std, output_std=output_std,
                rope_base=config.rope_base, rng=rng, eps=config.rms_eps,
                name=f"{name}.blocks.{i}", dtype=config.dtype,
                attention_config=attention, ffn_mode="dense",
                dual_resolution_config=dual,
            ))
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
                      dual_resolution_metadata=None,
                      return_cache=True, activation_checkpoint=False):
        """Calculate correction. The foundation hidden tensor is never modified."""
        if frozen_hidden.shape[-1] != self.config.d_model:
            raise ValueError("specialist/foundation hidden width mismatch")
        uses_dual_resolution = self.config.dual_resolution is not None
        if uses_dual_resolution and dual_resolution_metadata is None:
            raise ValueError("Model B specialist requires dual-resolution metadata")
        if dual_resolution_metadata is not None and not uses_dual_resolution:
            # The legacy GQA specialist uses optimized causal attention kernels
            # without document-ID masks. Fail closed instead of leaking packed
            # neighboring documents. Single-document rows remain fully supported.
            from mini_llm.backend import asnumpy
            import numpy as np
            docs = np.asarray(asnumpy(dual_resolution_metadata["document_ids"]))
            if docs.shape != frozen_hidden.shape[:2]:
                raise ValueError("specialist document IDs must have shape [B,T]")
            lengths = dual_resolution_metadata.get("valid_lengths")
            if lengths is None:
                lengths = np.full(docs.shape[0], docs.shape[1], dtype=np.int64)
            else:
                lengths = np.asarray(asnumpy(lengths), dtype=np.int64)
                if lengths.shape != (docs.shape[0],) or np.any(lengths < 1) or np.any(lengths > docs.shape[1]):
                    raise ValueError("invalid specialist valid_lengths")
            # Padding EOS creates artificial document identifiers *after* the
            # genuine sample. Future padding cannot influence causal predictions
            # on genuine positions, and its CE mask is zero. Never permit an
            # actual packed multi-document input to pass this gate.
            if any(np.any(docs[row, :length] != docs[row, 0])
                   for row, length in enumerate(lengths)):
                raise NotImplementedError(
                    "specialist multi-document attention needs document-aware "
                    "optimized masks; use document-first, single-document rows"
                )
            if position_ids is None:
                position_ids = dual_resolution_metadata["position_ids"]
            else:
                if not np.array_equal(np.asarray(asnumpy(position_ids)),
                                      np.asarray(asnumpy(dual_resolution_metadata["position_ids"]))):
                    raise ValueError("specialist position_ids disagree with foundation metadata")
        # First no-history implementation operates on a contiguous causal
        # working sequence. `position_ids` can preserve its absolute source
        # coordinates when provided by the foundation training batch. Prepare
        # their RoPE factors once for all specialist layers and checkpoint
        # replays, matching DecoderLanguageModel.forward_body.
        if (position_ids is not None and
                os.environ.get("MINI_LLM_ROPE_PREPARED_POSITIONS", "1").strip().lower()
                not in {"0", "false", "off", "no", ""}):
            position_ids = prepare_rope_positions(
                position_ids, d_head=self.config.d_head,
                base=self.config.rope_base, dtype=self.config.dtype,
            )
        x = frozen_hidden
        checkpoint = bool(return_cache and activation_checkpoint)
        checkpoints = []
        caches = []
        block_kwargs = {"position_ids": position_ids}
        if uses_dual_resolution:
            block_kwargs["dual_resolution_metadata"] = dual_resolution_metadata
        for idx, block in enumerate(self.blocks):
            if checkpoint:
                checkpoints.append(x)
                x = block.forward(
                    x, layer_idx=idx, return_cache=False, **block_kwargs)
            elif return_cache:
                x, cache = block.forward(x, layer_idx=idx, **block_kwargs)
                caches.append(cache)
            else:
                x = block.forward(
                    x, layer_idx=idx, return_cache=False, **block_kwargs)
        # Tensor-core GEMMs require inputs in the configured low-precision type.
        projection_input = x.astype(self.dtype, copy=False)
        delta, projection_cache = self.correction_proj.forward(projection_input)
        if not return_cache:
            return delta
        return delta, {
            "block_caches": caches, "checkpoints": checkpoints,
            "activation_checkpoint": checkpoint, "position_ids": position_ids,
            "dual_resolution_metadata": dual_resolution_metadata,
            "projection_cache": projection_cache,
        }

    def backward_delta(self, ddelta, cache):
        """Accumulate specialist-only gradients; return dH0 for diagnostics."""
        dx = self.correction_proj.backward(ddelta, cache.pop("projection_cache"))
        if cache["activation_checkpoint"]:
            for idx in range(len(self.blocks) - 1, -1, -1):
                block_input = cache["checkpoints"].pop()
                replay_kwargs = {"position_ids": cache["position_ids"]}
                if self.config.dual_resolution is not None:
                    replay_kwargs["dual_resolution_metadata"] = cache[
                        "dual_resolution_metadata"]
                _, replay_cache = self.blocks[idx].forward(
                    block_input, layer_idx=idx, **replay_kwargs)
                dx = self.blocks[idx].backward(dx, replay_cache)
        else:
            for idx in range(len(self.blocks) - 1, -1, -1):
                dx = self.blocks[idx].backward(dx, cache["block_caches"].pop())
        return dx

    @staticmethod
    def corrected_hidden(frozen_hidden, delta):
        """No post-addition normalization: zero delta exactly recovers H0."""
        return frozen_hidden + delta
