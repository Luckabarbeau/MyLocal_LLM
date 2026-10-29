# Patch 0102V — memory-bounded packed projection backward (Model B)

Context: the 65,536-token full-model benchmark intermittently runs out of VRAM inside
`_PackedProjection.backward()` at `xp.concatenate(pieces, axis=-1)`, which
requests a 192-MiB BF16 QKV gradient tensor at `[65536,1536]`.

The new optional path splits backward QKV packing and the two GEMMs into
row chunks, while accumulating weight gradients into the normal persistent
FP32 packed-gradient storage. It also casts input/gradient slices rather than
casting the entire sequence before packing. It does not alter the model
architecture, checkpoint layout, or the unchunked fast path.

Enable on the RTX 5080 with:

```bash
export MINI_LLM_DUAL_RESOLUTION_PACKED_BACKWARD_CHUNK_TOKENS=8192
```

Unset or use `0` for the original behavior.  At 65K, an 8192-row packed BF16
buffer is 24 MiB, versus 192 MiB without chunking. This only describes the
packed-gradient temporary, not the complete backward peak, and does not
promise that the entire 65K run will fit.

**Tradeoff:** four to eight or more smaller GEMMs may be slower than one large
GEMM, and BF16 summation order changes. Test gradient parity and measure
full-model throughput + memory before enabling this in production.

Tests:

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q tests/test_packed_projection_chunk_0102v.py
MINI_LLM_BACKEND=cupy python -m pytest -v tests/test_packed_projection_chunk_0102v.py
```

Use matched 65K benchmarks with `0` and `8192` for the env variable. The
benchmark's `--score-last-k 4096` **does not** provide 65K full supervision;
run separate all-position scoring tests after the memory path is stable.
