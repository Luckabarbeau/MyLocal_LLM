"""Bisect: dbg3 copy, debug stores removed one at a time. fp32 out."""
import os
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
def kA(W, OUT16, G2S, G3S, G3X, KSG, KV,
       stride, N: tl.constexpr, K: tl.constexpr, QK: tl.constexpr,
       QB: tl.constexpr, FMT: tl.constexpr):
    # debug stores REMOVED vs dbg3; simple store expression kept
    n = tl.program_id(0)
    kb = tl.program_id(1)
    slot = tl.program_id(2)
    rowbytes: tl.constexpr = (K // QK) * QB
    j0 = kb * 32
    B = W + slot.to(tl.int64) * stride + n.to(tl.int64) * rowbytes + (j0 // QK) * QB
    j = j0 + tl.arange(0, 32)
    d = (tl.load(B).to(tl.uint16)
         | (tl.load(B + 1).to(tl.uint16) << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    jj = j % 32
    nib = (tl.load(B + 2 + jj % 16) >> (4 * (jj // 16))) & 15
    w = d * tl.load(KV + nib).to(tl.float32)
    tl.store(OUT16 + 32 + j, w.to(tl.float16))


@triton.jit
def kB(W, OUT16, G2S, G3S, G3X, KSG, KV,
       stride, N: tl.constexpr, K: tl.constexpr, QK: tl.constexpr,
       QB: tl.constexpr, FMT: tl.constexpr):
    # as kA but store at OUT16 + j
    n = tl.program_id(0)
    kb = tl.program_id(1)
    slot = tl.program_id(2)
    rowbytes: tl.constexpr = (K // QK) * QB
    j0 = kb * 32
    B = W + slot.to(tl.int64) * stride + n.to(tl.int64) * rowbytes + (j0 // QK) * QB
    j = j0 + tl.arange(0, 32)
    d = (tl.load(B).to(tl.uint16)
         | (tl.load(B + 1).to(tl.uint16) << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    jj = j % 32
    nib = (tl.load(B + 2 + jj % 16) >> (4 * (jj // 16))) & 15
    w = d * tl.load(KV + nib).to(tl.float32)
    tl.store(OUT16 + j, w.to(tl.float16))


@triton.jit
def kC(W, OUT16, G2S, G3S, G3X, KSG, KV,
       stride, N: tl.constexpr, K: tl.constexpr, QK: tl.constexpr,
       QB: tl.constexpr, FMT: tl.constexpr):
    # exact dbg3 kernel, ALL debug stores present
    n = tl.program_id(0)
    kb = tl.program_id(1)
    slot = tl.program_id(2)
    rowbytes: tl.constexpr = (K // QK) * QB
    j0 = kb * 32
    B = W + slot.to(tl.int64) * stride + n.to(tl.int64) * rowbytes + (j0 // QK) * QB
    j = j0 + tl.arange(0, 32)
    d = (tl.load(B).to(tl.uint16)
         | (tl.load(B + 1).to(tl.uint16) << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    jj = j % 32
    nib = (tl.load(B + 2 + jj % 16) >> (4 * (jj // 16))) & 15
    tl.store(OUT16 + j, nib.to(tl.uint8))
    w = d * tl.load(KV + nib).to(tl.float32)
    tl.store(OUT16 + 32 + j, w.to(tl.float16))
    tl.store(OUT16 + 64 + j, d)


dev = xp.asarray(row)
G2S, G3S, G3X, KSG, KV = tpe._codebooks()
for name, kern, off in (('A no-dbg-stores off+32', kA, 32), ('B no-dbg at0', kB, 0),
                        ('C full-dbg off+32', kC, 32)):
    out = xp.zeros(96, xp.float32)
    kern[(1, 1, 1)](tpe._ptr(dev), tpe._ptr(out), G2S, G3S, G3X, KSG, KV,
                    18, N=1, K=32, QK=32, QB=18, FMT=3)
    print(name, xp.asnumpy(out)[off:off + 6])
