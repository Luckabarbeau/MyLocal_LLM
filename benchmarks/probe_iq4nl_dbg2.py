"""Dump per-element nibble/d as the Triton kernel computes them (IQ4_NL)."""
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
def dbg(W, NIB, DOUT, KV):
    j = tl.arange(0, 32)
    d = (tl.load(W).to(tl.uint16)
         | (tl.load(W + 1).to(tl.uint16) << 8)).to(tl.float16, bitcast=True)
    nib = (tl.load(W + 2 + j % 16) >> (4 * (j // 16))) & 15
    tl.store(NIB + j, nib.to(tl.uint8))
    tl.store(DOUT + j, d.to(tl.float32))
    tl.store(DOUT + j, tl.load(KV + nib).to(tl.float32))


dev = xp.asarray(row)
nib = xp.zeros(32, xp.uint8)
kvv = xp.zeros(32, xp.float32)
KV = tpe._codebooks()[4]
dbg[(1,)](tpe._ptr(dev), tpe._ptr(nib), tpe._ptr(kvv), KV)
print('row', row)
print('nib', xp.asnumpy(nib))
print('kv ', xp.asnumpy(kvv))
print('host nibbles', [(int(b) & 15, int(b) >> 4) for b in row[2:10]])
import mini_llm.ops.gguf_codebooks as books
print('kvalues', books.kvalues_iq4nl.astype(int))
