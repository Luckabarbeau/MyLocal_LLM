"""Pinpoint: IQ4_NL single-block decode, kernel vs reference."""
import os
os.environ.setdefault('MINI_LLM_BACKEND', 'cupy')
import numpy as np
from mini_llm.backend import xp
from mini_llm.ops.quantized_weight import PackedGGUFWeight
import mini_llm.ops.triton_prefill_experts as tpe

rng = np.random.default_rng(1)
row = rng.integers(0, 256, (18,), dtype=np.uint8)
row[0:2] = np.frombuffer(np.float16(0.02).tobytes(), np.uint8)
w = PackedGGUFWeight(np.ascontiguousarray(row), (1, 32), 'IQ4_NL')
ref = w.dequantize_reference()[0]

G2S, G3S, G3X, KSG, KV = tpe._codebooks()
dev = xp.asarray(row)
out = xp.zeros(32, xp.float16)
tpe._dequant_kernel[(1, 1, 1)](tpe._ptr(dev), tpe._ptr(out), G2S, G3S, G3X, KSG,
                               KV, 18, N=1, K=32, QK=32, QB=18, FMT=3)
got = xp.asnumpy(out).astype(np.float32)
print('ref', ref[:8])
print('got', got[:8])
print('nibbles', [(int(b) & 15, int(b) >> 4) for b in row[2:10]])
print('kvalues', np.asarray(__import__('mini_llm.ops.gguf_codebooks',
                                        fromlist=['kvalues_iq4nl']
                                        ).kvalues_iq4nl))
