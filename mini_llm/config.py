from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16_384
    context_length: int = 1_024

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
        if self.dtype is None:
            self.dtype = "float32"
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
            vocab_size=256,
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
