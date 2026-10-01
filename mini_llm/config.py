from dataclasses import dataclass


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

    def __post_init__(self):
        if self.n_q_heads * self.d_head != self.d_model:
            raise ValueError("n_q_heads * d_head must equal d_model.")
        if self.n_q_heads % self.n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads.")
        if not (1 <= self.top_k <= self.n_experts):
            raise ValueError("top_k must satisfy 1 <= top_k <= n_experts.")

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
