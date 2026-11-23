"""A/B the real _dequant_kernel against the verified dbg3 copy, same data."""
import os
import sys
sys.path.insert(0, 'benchmarks')
os.environ.setdefault('MINI_LLM_BACKEND', 'cupy')
import numpy as np
import triton
import triton.language as tl
from mini_llm.backend import xp
import mini_llm.ops.triton_prefill_experts as tpe

rng = np.random.default_rng(1)
row = rng.integers(0, 256, (18,), dtype=np.uint8)
row[0:2] = np.frombuffer(np.float16(0.02).tobytes(), np.uint8)


@triton.jit
def copy_kernel(W, OUT16, G2S, G3S, G3X, KSG, KV,
                stride, N: tl.constexpr, K: tl.constexpr, QK: tl.constexpr,
                QB: tl.constexpr, FMT: tl.constexpr):
    n = tl.program_id(0)
    kb = tl.program_id(1)
    slot = tl.program_id(2)
    rowbytes: tl.constexpr = (K // QK) * QB
    j0 = kb * 32
    B = W + slot.to(tl.int64) * stride + n.to(tl.int64) * rowbytes + (j0 // QK) * QB
    j = j0 + tl.arange(0, 32)
    d = (tl.load(B).to(tl.uint16)
         | (tl.load(B + 1).to(tl.uint16) << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    if FMT == 5:
        jj = j % 64
        v = (tl.load(B + 2 + jj // 4) >> (2 * (jj % 4))) & 3
        w = d * (v.to(tl.float32) - 1.0)
    elif FMT == 3:
        jj = j % 32
        nib = (tl.load(B + 2 + jj % 16) >> (4 * (jj // 16))) & 15
        w = d * tl.load(KV + nib).to(tl.float32)
    tl.store(OUT16 + slot.to(tl.int64) * (N * K) + n.to(tl.int64) * K + j,
             w.to(tl.float16))


dev = xp.asarray(row)
G2S, G3S, G3X, KSG, KV = tpe._codebooks()
out1 = xp.zeros(32, xp.float32)
out2 = xp.zeros(32, xp.float32)
args = (tpe._ptr(dev), tpe._ptr(out1), G2S, G3S, G3X, KSG, KV, 18)
tpe._dequant_kernel[(1, 1, 1)](*args, N=1, K=32, QK=32, QB=18, FMT=3)
copy_kernel[(1, 1, 1)](tpe._ptr(dev), tpe._ptr(out2), G2S, G3S, G3X, KSG, KV,
                       18, N=1, K=32, QK=32, QB=18, FMT=3)
print('real', xp.asnumpy(out1).astype(np.float32)[:6])
print('copy', xp.asnumpy(out2).astype(np.float32)[:6])
out3 = xp.zeros(32, xp.float16)
copy_kernel[(1, 1, 1)](tpe._ptr(dev), tpe._ptr(out3), G2S, G3S, G3X, KSG, KV,
                       18, N=1, K=32, QK=32, QB=18, FMT=3)
print('f16 ', xp.asnumpy(out3).astype(np.float32)[:6])
from mini_llm.ops.quantized_weight import PackedGGUFWeight
print('ref ', PackedGGUFWeight(np.ascontiguousarray(row), (1, 32),
                               'IQ4_NL').dequantize_reference()[:6])
