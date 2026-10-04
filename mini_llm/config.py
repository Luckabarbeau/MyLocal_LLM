from dataclasses import dataclass, field
import math

@dataclass(frozen=True)
class MemoryContextConfig:
    """Hierarchical long-memory configuration.

    In the 0058C ``terminal_landmark`` path ``memory_length`` is the total
    causal horizon ending at the current working window. ``target_length``
    rows form the ordinary dense Transformer working window; the preceding
    ``memory_length - target_length`` rows are a same-document external store.
    The router pools the recent working context once, selects old blocks, and
    exact tokens from those blocks become K/V only for the terminal query of
    one configurable attention layer. They never become deep Transformer query
    rows. Dense LM training still supervises the complete working window in
    ``joint`` mode; ``router_only`` post-training projects and backpropagates
    only the terminal next-token objective.

    ``pretransformer_read`` remains solely as the 0058A/B comparison path.
    """

    enabled: bool = False
    memory_length: int = 65_536
    recent_length: int = 4_096
    target_length: int = 4_096
    block_size: int = 128
    top_k_blocks: int = 16
    router_query_length: int = 4_096
    router_dim: int = 64
    query_pooling: str = "mean"
    history_pooling: str = "mean"
    router_weight_scale: float = 1.0

    # 0058C memory integration.  ``pretransformer_read`` preserves the 0058A/B
    # comparison path. ``terminal_landmark`` keeps the 4k working sequence
    # untouched and exposes exact routed historical K/V only to the terminal
    # query in one configurable Transformer attention layer.
    integration_mode: str = "pretransformer_read"
    memory_attention_layer: int = -1
    memory_training: str = "joint"
    min_router_history_blocks: int = 1

    # Exact-token memory adapter.  In terminal-landmark mode these are the
    # historical K/V heads used by the retrieval heads in the memory-aware
    # layer.  The old pre-Transformer reader also reuses these sizing fields.
    # these heads; GQA keeps the memory K/V projection deliberately small.
    read_heads: int = 4
    read_kv_heads: int = 1
    read_query_chunk: int = 64
    reader_weight_eps: float = 1e-8
    reader_residual_scale: float = 1.0

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("memory context enabled must be a bool")
        positive = {
            "memory_length": self.memory_length,
            "recent_length": self.recent_length,
            "target_length": self.target_length,
            "block_size": self.block_size,
            "top_k_blocks": self.top_k_blocks,
            "router_query_length": self.router_query_length,
            "router_dim": self.router_dim,
            "read_heads": self.read_heads,
            "read_kv_heads": self.read_kv_heads,
            "read_query_chunk": self.read_query_chunk,
            "min_router_history_blocks": self.min_router_history_blocks,
        }
        for name, value in positive.items():
            if int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.recent_length > self.memory_length:
            raise ValueError("recent_length must not exceed memory_length")
        if self.integration_mode == "terminal_landmark" and self.target_length >= self.memory_length:
            raise ValueError(
                "terminal_landmark requires target_length < memory_length"
            )
        if self.target_length > self.recent_length:
            raise ValueError(
                "0058A requires target_length <= recent_length so target rows "
                "never age into the fixed historical memory store within one sample"
            )
        if self.router_query_length > self.recent_length:
            raise ValueError("router_query_length must not exceed recent_length")
        if self.read_heads % self.read_kv_heads != 0:
            raise ValueError("read_heads must be divisible by read_kv_heads")
        if self.integration_mode not in {"pretransformer_read", "terminal_landmark"}:
            raise ValueError(
                "integration_mode must be 'pretransformer_read' or 'terminal_landmark'"
            )
        if self.memory_training not in {"joint", "router_only", "disabled"}:
            raise ValueError("memory_training must be 'joint', 'router_only', or 'disabled'")
        if int(self.min_router_history_blocks) > int(self.searchable_blocks):
            raise ValueError("min_router_history_blocks exceeds searchable memory blocks")
        if self.query_pooling not in {"mean", "last", "learned"}:
            raise ValueError("query_pooling must be 'mean', 'last', or 'learned'")
        if self.history_pooling not in {"mean", "first", "last", "learned"}:
            raise ValueError("history_pooling must be 'mean', 'first', 'last', or 'learned'")
        for name, value in {
            "router_weight_scale": self.router_weight_scale,
            "reader_weight_eps": self.reader_weight_eps,
            "reader_residual_scale": self.reader_residual_scale,
        }.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(self.reader_weight_eps) <= 0.0:
            raise ValueError("reader_weight_eps must be positive")
        if self.enabled:
            if self.memory_length < self.block_size:
                raise ValueError("enabled memory context requires at least one memory block")
            if self.distant_memory_length % self.block_size != 0:
                raise ValueError("external searchable memory must be divisible by block_size")
            if self.searchable_blocks < self.top_k_blocks:
                raise ValueError("top_k_blocks exceeds the number of complete memory blocks")

    @property
    def distant_memory_length(self) -> int:
        """Number of tokens physically stored outside the 4k working window."""
        if self.integration_mode == "terminal_landmark":
            return int(self.memory_length) - int(self.target_length)
        return int(self.memory_length)

    @property
    def searchable_blocks(self) -> int:
        return int(self.distant_memory_length) // int(self.block_size)

    @property
    def retrieved_length(self) -> int:
        return int(self.block_size) * int(self.top_k_blocks)

    @property
    def active_length(self) -> int:
        # 0058A external memory is read into each current token; retrieved
        # history is never appended to the deep Transformer sequence.
        return int(self.target_length)

    @property
    def source_input_length(self) -> int:
        """Input rows consumed by one hierarchical-memory sample.

        In 0058C terminal-landmark mode ``memory_length`` is the *total causal
        horizon*, including the current working window.  The sampler still reads
        one additional corpus token to provide the terminal next-token label.
        """
        if self.integration_mode == "terminal_landmark":
            return int(self.memory_length)
        return int(self.memory_length) + int(self.target_length)


def _coerce_memory_context(value):
    if value is None:
        return MemoryContextConfig(enabled=False)
    if isinstance(value, MemoryContextConfig):
        return value
    if isinstance(value, dict):
        return MemoryContextConfig(**value)
    raise TypeError("memory_context must be MemoryContextConfig-compatible")


@dataclass(frozen=True)
class ModelConfig:
    # Tokenizer vocabulary size (separate from model vocab for flexibility)
    tokenizer_vocab_size: int = 16_384
    
    # Model configuration
    context_length: int = 1_024
    
    # Backward compatibility: vocab_size is an alias for tokenizer_vocab_size
    @property
    def vocab_size(self) -> int:
        """Backward compatible alias for tokenizer_vocab_size."""
        return self.tokenizer_vocab_size

    n_layers: int = 8
    d_model: int = 384
    n_q_heads: int = 6
    n_kv_heads: int = 2
    d_head: int = 64

    n_experts: int = 6
    top_k: int = 2
    d_ff: int = 1_024

    rope_base: float = 10_000.0
    rms_eps: float = 1e-6

    init_std: float = 0.02
    init_cutoff: float = 3.0
    router_logit_std: float = 0.01

    dtype: str = "float32"

    # ``None`` preserves the original full-causal GQA implementation on every
    # layer. Otherwise provide one AttentionLayerConfig per Transformer layer.
    attention_layers: tuple | None = None

    # Optional top-level memory selector.  This is deliberately separate from
    # per-layer sparse/retrieval attention: long addressable history is reduced
    # to a bounded active sequence *before* the expensive Transformer trunk.
    memory_context: MemoryContextConfig = field(default_factory=MemoryContextConfig)

    def __post_init__(self):
        object.__setattr__(
            self, "memory_context", _coerce_memory_context(self.memory_context)
        )
        if self.n_q_heads * self.d_head != self.d_model:
            raise ValueError("n_q_heads * d_head must equal d_model.")
        if self.n_q_heads % self.n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads.")
        if not (1 <= self.top_k <= self.n_experts):
            raise ValueError("top_k must satisfy 1 <= top_k <= n_experts.")

        if self.attention_layers is not None:
            layers = tuple(
                _coerce_attention_layer(layer) for layer in self.attention_layers
            )
            if len(layers) != self.n_layers:
                raise ValueError(
                    "attention_layers must contain exactly one entry per model layer"
                )
            for layer_index, layer in enumerate(layers):
                if len(layer.heads) != self.n_q_heads:
                    raise ValueError(
                        f"attention layer {layer_index} must define exactly "
                        f"{self.n_q_heads} query heads"
                    )
                retrieval_groups = {}
                for head in layer.heads:
                    if not isinstance(head, RetrievalAttentionConfig):
                        continue
                    previous = retrieval_groups.setdefault(
                        head.group, head.context_router
                    )
                    if previous != head.context_router:
                        raise ValueError(
                            f"retrieval group {head.group!r} in layer {layer_index} "
                            "must use one shared ContextRouterConfig"
                        )
                for group_name, router_config in retrieval_groups.items():
                    group_size = sum(
                        isinstance(head, RetrievalAttentionConfig)
                        and head.group == group_name
                        for head in layer.heads
                    )
                    if router_config.num_queries not in {1, group_size}:
                        raise ValueError(
                            f"retrieval group {group_name!r} in layer {layer_index} "
                            "requires num_queries=1 or num_queries equal to the "
                            f"group head count ({group_size})"
                        )
            object.__setattr__(self, "attention_layers", layers)

    def estimated_parameter_count(self) -> int:
        """Return the exact trainable-array element count for this config.

        The calculation mirrors the explicit modules without allocating model
        tensors.  This is useful for preflighting large MoE presets before
        committing several GiB of GPU/host memory.
        """
        d = int(self.d_model)
        kv_width = int(self.n_kv_heads) * int(self.d_head)

        # Untied token embedding + output projection + final RMSNorm.
        total = 2 * int(self.vocab_size) * d + d

        for layer_index in range(int(self.n_layers)):
            # Two RMSNorm scales per Transformer block.
            total += 2 * d

            # Q, K, V and output attention projections.  Q width equals d_model.
            total += 2 * d * d + 2 * d * kv_width

            # MoE router (weight+bias) and SwiGLU expert matrices.
            total += d * int(self.n_experts) + int(self.n_experts)
            total += int(self.n_experts) * 3 * d * int(self.d_ff)

            if self.attention_layers is not None:
                # Retrieval heads with the same group share one ContextRouter.
                groups = {}
                for head in self.attention_layers[layer_index].heads:
                    if isinstance(head, RetrievalAttentionConfig):
                        groups.setdefault(head.group, head.context_router)
                for router in groups.values():
                    total += d * int(router.num_queries) * int(router.router_dim)
                    total += d * int(router.router_dim)
                    if router.query_pooling == "learned":
                        total += d
                    if router.history_pooling == "learned":
                        total += d

        memory = self.memory_context
        if memory.enabled:
            total += 2 * d * int(memory.router_dim)
            if memory.query_pooling == "learned":
                total += d
            if memory.history_pooling == "learned":
                total += d
            read_q = int(memory.read_heads) * int(self.d_head)
            read_kv = int(memory.read_kv_heads) * int(self.d_head)
            if memory.integration_mode == "terminal_landmark":
                # RMSNorm scale + compact historical K/V adapter. The terminal
                # query reuses the selected Transformer's existing Q projection.
                total += d + 2 * d * read_kv
            else:
                total += d * read_q + 2 * d * read_kv + read_q * d

        return int(total)

    @property
    def residual_init_std(self) -> float:
        return self.init_std / (2.0 * self.n_layers) ** 0.5

    @property
    def router_init_std(self) -> float:
        return self.router_logit_std / (self.d_model ** 0.5)

    @classmethod
    def tiny_inspection(cls):
        return cls(
            tokenizer_vocab_size=256,
            context_length=16,
            n_layers=1,
            d_model=16,
            n_q_heads=2,
            n_kv_heads=1,
            d_head=8,
            n_experts=3,
            top_k=2,
            d_ff=32,
        )

    @classmethod
    def micro_debug(cls):
        """Micro model: ~500K params, for quick debugging."""
        return cls(
            tokenizer_vocab_size=8192*2,
            context_length=32,
            n_layers=2,
            d_model=32,
            n_q_heads=2,
            n_kv_heads=1,
            d_head=16,
            n_experts=2,
            top_k=1,
            d_ff=64,
        )

    @classmethod
    def mini(cls):
        """Mini model: ~12M params, good for initial experiments."""
        return cls(
            tokenizer_vocab_size=8192*2,
            context_length=512,
            n_layers=8,
            d_model=256,
            n_q_heads=4,
            n_kv_heads=2,
            d_head=64,
            n_experts=6,
            top_k=2,
            d_ff=768,
        )

    @classmethod
    def small(cls):
        """Small model: ~53M params (current default)."""
        return cls(
            tokenizer_vocab_size=8192*2,
            context_length=512,
            n_layers=8,
            d_model=384,
            n_q_heads=6,
            n_kv_heads=2,
            d_head=64,
            n_experts=6,
            top_k=2,
            d_ff=1_024,
        )

    @classmethod
    def medium(cls):
        """Medium model: ~120M params."""
        return cls(
            tokenizer_vocab_size=8192*2,
            context_length=512,
            n_layers=8,
            d_model=512,
            n_q_heads=8,
            n_kv_heads=2,
            d_head=64,
            n_experts=6,
            top_k=2,
            d_ff=1_536,
        )
    @classmethod
    def medium_context_4k(cls):
        """Medium long-context experiment with all sparse head types.

        This is the first end-to-end context-expansion training preset.  It
        keeps the existing medium trunk/MoE dimensions while using a 65,280
        token vocabulary and a 4k training context.  Every layer uses:

        - 4 exact local heads (1k window)
        - 1 dilated head (4k span, dilation 4)
        - 1 deterministic global sparse head (stride 128)
        - 2 learned retrieval heads sharing one two-query context router

        The shorter windows are intentional for the initial 4k validation
        stage.  They keep retrieval active over a substantial fraction of the
        sequence while exercising the same mechanisms that will later be
        scaled to 8k/16k/32k contexts.
        """
        router = ContextRouterConfig(
            history_block_size=128,
            routing_stride=128,
            query_window=512,
            router_dim=32,
            top_k_blocks=4,
            exclude_recent_tokens=1_024,
            query_pooling="learned",
            history_pooling="mean",
            num_queries=2,
            router_weight_mode="logit_bias",
            router_weight_scale=1.0,
        )
        layer_attention = AttentionLayerConfig(
            heads=(
                LocalAttentionConfig(window=1_024),
                LocalAttentionConfig(window=1_024),
                LocalAttentionConfig(window=1_024),
                LocalAttentionConfig(window=1_024),
                DilatedAttentionConfig(window=4_096, dilation=4, offset=0),
                GlobalSparseAttentionConfig(
                    stride=128, offset=0, include_current=True
                ),
                RetrievalAttentionConfig(router, group="far"),
                RetrievalAttentionConfig(router, group="far"),
            )
        )
        return cls(
            tokenizer_vocab_size=65_280,
            context_length=4_096,
            n_layers=8,
            d_model=512,
            n_q_heads=8,
            n_kv_heads=2,
            d_head=64,
            n_experts=6,
            top_k=2,
            d_ff=1_536,
            attention_layers=tuple(layer_attention for _ in range(8)),
        )

    @classmethod
    def moe_525m_context_4k(cls):
        """~525.6M-parameter 4k MoE scaling preset.

        The active trunk is deliberately identical to ``medium_context_4k``:
        8 layers, d_model=512, d_ff=1536 and top-2 routing.  Capacity is scaled
        by increasing the expert pool from 6 to 24, so activation geometry and
        per-token expert FLOPs remain close to the validated 185M model while
        total parameter/state memory grows by ~2.8x.
        """
        base = cls.medium_context_4k()
        return cls(
            tokenizer_vocab_size=base.tokenizer_vocab_size,
            context_length=base.context_length,
            n_layers=base.n_layers,
            d_model=base.d_model,
            n_q_heads=base.n_q_heads,
            n_kv_heads=base.n_kv_heads,
            d_head=base.d_head,
            n_experts=24,
            top_k=base.top_k,
            d_ff=base.d_ff,
            rope_base=base.rope_base,
            rms_eps=base.rms_eps,
            init_std=base.init_std,
            init_cutoff=base.init_cutoff,
            router_logit_std=base.router_logit_std,
            dtype=base.dtype,
            attention_layers=base.attention_layers,
        )

    @classmethod
    def moe_1b_context_4k(cls):
        """~978.7M-parameter near-1B 4k MoE stress-test preset.

        As with the 525M preset, the active top-2 path stays 512-wide and
        8 layers deep.  Forty-eight experts raise total capacity to just under
        one billion parameters without multiplying the routed activation size.
        This preset is intended first as a memory/throughput scaling test; long
        training should also monitor expert utilization because the current MoE
        does not yet add an explicit load-balancing auxiliary loss.
        """
        base = cls.medium_context_4k()
        return cls(
            tokenizer_vocab_size=base.tokenizer_vocab_size,
            context_length=base.context_length,
            n_layers=base.n_layers,
            d_model=base.d_model,
            n_q_heads=base.n_q_heads,
            n_kv_heads=base.n_kv_heads,
            d_head=base.d_head,
            n_experts=48,
            top_k=base.top_k,
            d_ff=base.d_ff,
            rope_base=base.rope_base,
            rms_eps=base.rms_eps,
            init_std=base.init_std,
            init_cutoff=base.init_cutoff,
            router_logit_std=base.router_logit_std,
            dtype=base.dtype,
            attention_layers=base.attention_layers,
        )

    @classmethod
    def _wide_500m_sparse_context(cls, context_length: int):
        """~500M wide/deep sparse-context model with only six experts.

        Unlike :meth:`moe_525m_context_4k`, this family spends its parameter
        budget on a wider/deeper shared trunk instead of a larger expert pool:

        - 12 Transformer layers (vs. 8)
        - d_model=768 / 12 query heads / 3 KV heads
        - d_ff=2304
        - 6 experts, top-2 routing

        The five context presets share *identical parameter shapes* and use the
        same RoPE base so model weights are architecture-compatible across the
        4k -> 8k -> 16k -> 32k -> 64k ladder.  Sparse attention work per token
        is kept bounded as context grows: local and retrieval spans stay fixed,
        while dilation and global-anchor stride grow with the requested context.
        """
        context_length = int(context_length)
        if context_length not in {4_096, 8_192, 16_384, 32_768, 65_536}:
            raise ValueError(
                "wide 500M sparse-context presets support 4k/8k/16k/32k/64k"
            )

        # Bound *implementation* complexity as well as mathematical sparsity.
        #
        # The current dilated kernel decomposes work into one residue phase per
        # dilation value.  Letting dilation grow to 256 at 64k therefore turns
        # the 4-phase 4k fast path into hundreds of small launches even though
        # the number of useful key slots stays small.  Keep the medium-range
        # dilated span at <=8k and cap dilation at 32; the deterministic global
        # and learned retrieval heads provide the truly long-range 16k-64k
        # connectivity.
        local_window = 1_024 if context_length <= 8_192 else (
            512 if context_length <= 32_768 else 256
        )
        if context_length <= 4_096:
            dilated_window, dilation = 4_096, 4
        elif context_length <= 8_192:
            dilated_window, dilation = 8_192, 8
        elif context_length <= 16_384:
            dilated_window, dilation = 8_192, 16
        else:
            dilated_window, dilation = 8_192, 32

        # Keep ~64 global anchors across the full sequence.  Retrieval routing
        # is refreshed progressively less often as context grows so the number
        # of route decisions stays O(10^2), not proportional to every token.
        global_stride = max(128, context_length // 64)
        if context_length <= 8_192:
            routing_stride = 128
        elif context_length <= 16_384:
            routing_stride = 256
        elif context_length <= 32_768:
            routing_stride = 512
        else:
            routing_stride = 1_024

        # The learned query pool previously grew to 2048 tokens at 64k and
        # materialized a large [B,R,W,D] gather in every layer (and again during
        # checkpoint replay).  A bounded recent summary is sufficient for the
        # route-selection role and avoids that superlinear orchestration cost.
        query_window = 512 if context_length <= 16_384 else 1_024
        exclude_recent = max(1_024, context_length // 16)

        router = ContextRouterConfig(
            history_block_size=128,
            routing_stride=routing_stride,
            query_window=query_window,
            router_dim=32,
            top_k_blocks=4,
            exclude_recent_tokens=exclude_recent,
            query_pooling="learned",
            history_pooling="mean",
            num_queries=2,
            router_weight_mode="logit_bias",
            router_weight_scale=1.0,
        )

        # 12 heads: 6 local, 2 dilated, 2 deterministic global, 2 learned
        # retrieval.  The two global heads use complementary anchor phases.
        # Dilated heads intentionally share one phase so the current grouped
        # fast path can process both heads together efficiently.
        layer_attention = AttentionLayerConfig(
            heads=(
                LocalAttentionConfig(window=local_window),
                LocalAttentionConfig(window=local_window),
                LocalAttentionConfig(window=local_window),
                LocalAttentionConfig(window=local_window),
                LocalAttentionConfig(window=local_window),
                LocalAttentionConfig(window=local_window),
                DilatedAttentionConfig(
                    window=dilated_window, dilation=dilation, offset=0
                ),
                DilatedAttentionConfig(
                    window=dilated_window, dilation=dilation, offset=0
                ),
                GlobalSparseAttentionConfig(
                    stride=global_stride, offset=0, include_current=True
                ),
                GlobalSparseAttentionConfig(
                    stride=global_stride,
                    offset=global_stride // 2,
                    include_current=True,
                ),
                RetrievalAttentionConfig(router, group="far"),
                RetrievalAttentionConfig(router, group="far"),
            )
        )

        return cls(
            tokenizer_vocab_size=65_280,
            context_length=context_length,
            n_layers=12,
            d_model=768,
            n_q_heads=12,
            n_kv_heads=3,
            d_head=64,
            n_experts=6,
            top_k=2,
            d_ff=2_304,
            # Use one long-context-friendly base throughout the ladder.  This
            # project trains these models from scratch; no post-hoc RoPE scaling
            # is being applied.
            rope_base=1_000_000.0,
            attention_layers=tuple(layer_attention for _ in range(12)),
        )

    @classmethod
    def wide_500m_context_4k(cls):
        return cls._wide_500m_sparse_context(4_096)

    @classmethod
    def wide_500m_context_8k(cls):
        return cls._wide_500m_sparse_context(8_192)

    @classmethod
    def wide_500m_context_16k(cls):
        return cls._wide_500m_sparse_context(16_384)

    @classmethod
    def wide_500m_context_32k(cls):
        return cls._wide_500m_sparse_context(32_768)

    @classmethod
    def wide_500m_context_64k(cls):
        return cls._wide_500m_sparse_context(65_536)

    @classmethod
    def _wide_500m_memory_context(cls, memory_length: int):
        """Wide ~500M trunk with terminal Landmark-gated external memory.

        ``memory_length`` is the total causal horizon.  The final 4k tokens are
        the ordinary dense Transformer working window; preceding same-document
        tokens form the searchable store.  Only the terminal prediction invokes
        learned long-memory routing, while the base LM remains densely trained.
        """
        memory_length = int(memory_length)
        if memory_length not in {16_384, 32_768, 65_536}:
            raise ValueError("hierarchical memory presets support 16k/32k/64k")

        memory = MemoryContextConfig(
            enabled=True,
            memory_length=memory_length,
            recent_length=4_096,
            target_length=4_096,
            block_size=128,
            top_k_blocks=16,
            router_query_length=4_096,
            router_dim=64,
            query_pooling="mean",
            history_pooling="mean",
            router_weight_scale=1.0,
            integration_mode="terminal_landmark",
            memory_attention_layer=-1,
            memory_training="joint",
            read_heads=2,
            read_kv_heads=1,
            read_query_chunk=64,
            reader_weight_eps=1e-8,
            reader_residual_scale=1.0,
        )

        # 0058C keeps only the dense neighboring training window in the deep
        # trunk. Long history is external K/V memory, so attention geometry is
        # based on 4k regardless of the 16k/32k/64k addressable horizon.
        base = cls._wide_500m_sparse_context(4_096)
        return cls(
            tokenizer_vocab_size=base.tokenizer_vocab_size,
            context_length=memory.active_length,
            n_layers=base.n_layers,
            d_model=base.d_model,
            n_q_heads=base.n_q_heads,
            n_kv_heads=base.n_kv_heads,
            d_head=base.d_head,
            n_experts=base.n_experts,
            top_k=base.top_k,
            d_ff=base.d_ff,
            rope_base=base.rope_base,
            rms_eps=base.rms_eps,
            init_std=base.init_std,
            init_cutoff=base.init_cutoff,
            router_logit_std=base.router_logit_std,
            dtype=base.dtype,
            attention_layers=base.attention_layers,
            memory_context=memory,
        )

    @classmethod
    def wide_500m_memory_16k(cls):
        return cls._wide_500m_memory_context(16_384)

    @classmethod
    def wide_500m_memory_32k(cls):
        return cls._wide_500m_memory_context(32_768)

    @classmethod
    def wide_500m_memory_64k(cls):
        return cls._wide_500m_memory_context(65_536)

    @classmethod
    def large(cls):
        """Large model: ~120M params."""
        return cls(
            tokenizer_vocab_size=8192*2,
            context_length=1024,
            n_layers=16,
            d_model=1024,
            n_q_heads=16,
            n_kv_heads=2,
            d_head=64,
            n_experts=6,
            top_k=2,
            d_ff=1_536,
        )


@dataclass(frozen=True)
class ContextRouterConfig:
    """Configuration for learned long-context block retrieval.

    ``history_block_size`` controls how distant history is summarized and
    stored. ``routing_stride`` independently controls how often a new retrieval
    decision is made, so retrieval frequency can be changed without changing
    the historical memory granularity.
    """

    history_block_size: int = 256
    routing_stride: int = 256
    query_window: int = 2_048
    router_dim: int = 32
    top_k_blocks: int = 8
    exclude_recent_tokens: int = 4_096

    query_pooling: str = "learned"
    history_pooling: str = "mean"
    score_function: str = "dot"
    selection_strategy: str = "topk"
    num_queries: int = 1

    # How selected routing probabilities influence exact retrieved-token
    # attention.  ``logit_bias`` adds beta*log(p+eps) to every token from the
    # corresponding selected block.  ``none`` keeps selection discrete and is
    # useful as an ablation/auxiliary-loss-only mode.
    router_weight_mode: str = "logit_bias"
    router_weight_scale: float = 1.0
    router_weight_eps: float = 1e-8

    def __post_init__(self):
        positive_ints = {
            "history_block_size": self.history_block_size,
            "routing_stride": self.routing_stride,
            "query_window": self.query_window,
            "router_dim": self.router_dim,
            "top_k_blocks": self.top_k_blocks,
            "num_queries": self.num_queries,
        }
        for name, value in positive_ints.items():
            if int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.exclude_recent_tokens < 0:
            raise ValueError("exclude_recent_tokens must be non-negative")
        if self.query_pooling not in {"mean", "last", "learned"}:
            raise ValueError("query_pooling must be 'mean', 'last', or 'learned'")
        if self.history_pooling not in {"mean", "first", "last", "learned"}:
            raise ValueError(
                "history_pooling must be 'mean', 'first', 'last', or 'learned'"
            )
        if self.score_function != "dot":
            raise ValueError("only score_function='dot' is currently implemented")
        if self.selection_strategy != "topk":
            raise ValueError(
                "only selection_strategy='topk' is currently implemented"
            )
        if self.router_weight_mode not in {"none", "logit_bias"}:
            raise ValueError(
                "router_weight_mode must be 'none' or 'logit_bias'"
            )
        if not math.isfinite(float(self.router_weight_scale)):
            raise ValueError("router_weight_scale must be finite")
        if (
            not math.isfinite(float(self.router_weight_eps))
            or self.router_weight_eps <= 0
        ):
            raise ValueError("router_weight_eps must be positive and finite")

@dataclass(frozen=True)
class DenseAttentionConfig:
    """Full causal attention for one query head."""

    kind: str = field(default="dense", init=False)


@dataclass(frozen=True)
class LocalAttentionConfig:
    """Exact causal attention restricted to the most recent ``window`` keys."""

    window: int = 4_096
    kind: str = field(default="local", init=False)

    def __post_init__(self):
        if int(self.window) != self.window or self.window <= 0:
            raise ValueError("local attention window must be a positive integer")


@dataclass(frozen=True)
class DilatedAttentionConfig:
    """Exact causal attention sampled at a fixed dilation phase.

    For query position ``t``, visible keys are
    ``t-offset-n*dilation`` while they remain inside the trailing ``window``
    token span. ``offset=0`` includes the current token; non-zero offsets let
    different heads/layers cover complementary phases.
    """

    window: int = 32_768
    dilation: int = 8
    offset: int = 0
    kind: str = field(default="dilated", init=False)

    def __post_init__(self):
        if int(self.window) != self.window or self.window <= 0:
            raise ValueError("dilated attention window must be a positive integer")
        if int(self.dilation) != self.dilation or self.dilation <= 0:
            raise ValueError("dilation must be a positive integer")
        if int(self.offset) != self.offset or self.offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if self.offset >= self.dilation:
            raise ValueError("offset must satisfy 0 <= offset < dilation")
        if self.offset >= self.window:
            raise ValueError("offset must be smaller than window")


@dataclass(frozen=True)
class GlobalSparseAttentionConfig:
    """Deterministic causal anchors spanning the complete available prefix.

    Visible global anchor tokens satisfy ``key % stride == offset``. When
    ``include_current`` is true, the query token itself is also visible even
    when it is not on the anchor phase. Different heads can use complementary
    offsets while retaining the same inexpensive whole-history connectivity.
    """

    stride: int = 256
    offset: int = 0
    include_current: bool = True
    kind: str = field(default="global_sparse", init=False)

    def __post_init__(self):
        if int(self.stride) != self.stride or self.stride <= 0:
            raise ValueError("global sparse stride must be a positive integer")
        if int(self.offset) != self.offset or self.offset < 0:
            raise ValueError("global sparse offset must be a non-negative integer")
        if self.offset >= self.stride:
            raise ValueError("global sparse offset must satisfy 0 <= offset < stride")
        if not isinstance(self.include_current, bool):
            raise TypeError("include_current must be a bool")


@dataclass(frozen=True)
class RetrievalAttentionConfig:
    """Learned distant-block retrieval for one query head.

    Heads carrying the same ``group`` string share one
    :class:`ContextRetrievalAttention` router. ``context_router.num_queries``
    must therefore be either 1 (one shared retrieval query) or the number of
    heads in that group (one learned retrieval query per head).
    """

    context_router: ContextRouterConfig
    group: str = "retrieval"
    kind: str = field(default="retrieval", init=False)

    def __post_init__(self):
        if isinstance(self.context_router, dict):
            object.__setattr__(
                self, "context_router", ContextRouterConfig(**self.context_router)
            )
        if not isinstance(self.context_router, ContextRouterConfig):
            raise TypeError("context_router must be a ContextRouterConfig")
        if not isinstance(self.group, str) or not self.group:
            raise ValueError("retrieval group must be a non-empty string")


def _coerce_attention_head(value):
    if isinstance(
        value,
        (
            DenseAttentionConfig,
            LocalAttentionConfig,
            DilatedAttentionConfig,
            GlobalSparseAttentionConfig,
            RetrievalAttentionConfig,
        ),
    ):
        return value
    if not isinstance(value, dict):
        raise TypeError("attention heads must be attention configs or dictionaries")

    data = dict(value)
    kind = data.pop("kind", None)
    if kind == "dense":
        return DenseAttentionConfig(**data)
    if kind == "local":
        return LocalAttentionConfig(**data)
    if kind == "dilated":
        return DilatedAttentionConfig(**data)
    if kind == "global_sparse":
        return GlobalSparseAttentionConfig(**data)
    if kind == "retrieval":
        return RetrievalAttentionConfig(**data)
    raise ValueError(f"unsupported attention head kind: {kind!r}")


@dataclass(frozen=True)
class AttentionLayerConfig:
    """Ordered per-query-head attention topology for one Transformer layer."""

    heads: tuple

    def __post_init__(self):
        heads = tuple(_coerce_attention_head(head) for head in self.heads)
        if not heads:
            raise ValueError("attention layer must contain at least one head")
        object.__setattr__(self, "heads", heads)


def _coerce_attention_layer(value):
    if isinstance(value, AttentionLayerConfig):
        return value
    if isinstance(value, dict):
        return AttentionLayerConfig(**value)
    if isinstance(value, (list, tuple)):
        return AttentionLayerConfig(heads=tuple(value))
    raise TypeError("attention layer entries must be AttentionLayerConfig-compatible")
