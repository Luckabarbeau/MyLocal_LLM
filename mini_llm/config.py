from dataclasses import dataclass, field
import math


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

    def __post_init__(self):
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
        (DenseAttentionConfig, LocalAttentionConfig, RetrievalAttentionConfig),
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
