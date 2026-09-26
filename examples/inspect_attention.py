import numpy as np

from mini_llm.backend import asnumpy, BACKEND_NAME, RandomStream
from mini_llm.config import ModelConfig
from mini_llm.ops.attention import GQAAttention


np.set_printoptions(precision=4, suppress=True, linewidth=140)

cfg = ModelConfig.tiny_inspection()
rng = RandomStream(12345)

attn = GQAAttention(
    d_model=cfg.d_model,
    n_q_heads=cfg.n_q_heads,
    n_kv_heads=cfg.n_kv_heads,
    d_head=cfg.d_head,
    input_std=cfg.init_std,
    output_std=cfg.residual_init_std,
    rng=rng,
    rope_base=cfg.rope_base,
    dtype=cfg.dtype,
)

x = rng.normal((1, 5, cfg.d_model), std=0.25, dtype=cfg.dtype)
y, cache = attn.forward(x)

print("backend:", BACKEND_NAME)
for name, p in [("Wq", attn.Wq), ("Wk", attn.Wk), ("Wv", attn.Wv), ("Wo", attn.Wo)]:
    print(f"\n{name} = {p.shape}")
    print(asnumpy(p.data))

print("\nQ after RoPE [B,T,Hq,Dh]:", cache["q"].shape)
print(asnumpy(cache["q"]))

print("\nK before GQA expansion [B,T,Hkv,Dh]:", cache["k"].shape)
print(asnumpy(cache["k"]))

print("\nK after GQA expansion [B,T,Hq,Dh]:", cache["k_exp"].shape)
print(asnumpy(cache["k_exp"]))

print("\nHead-0 unmasked QK^T/sqrt(dh):")
print(asnumpy(cache["scores"][0, 0]))

print("\nHead-0 causal attention probabilities:")
print(asnumpy(cache["probs"][0, 0]))

print("\nOutput Y:")
print(asnumpy(y))
