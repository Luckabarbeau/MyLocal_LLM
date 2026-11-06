# 0106G — Q5_K packed GGUF support for Qwen3-Next (incremental after 0106F)

## Why

The Qwen3-Coder-Next UD-Q4_K_XL GGUF can mix quantization formats. In one
user-reported run, `blk.0.ffn_down_exps.weight` uses GGUF `Q5_K`. The 0106F
index correctly rejected this rather than silently converting an 80B model
to FP32. 0106G enables Q5_K in the *existing* shared backend.

## GGML block layout

Each Q5_K block represents 256 elements with 176 packed bytes: 2 bytes
FP16 d, 2 bytes FP16 dmin, 12 scale/min bytes, 32 high-bit bytes, 128
low-nibble bytes. Each of eight 32-value groups gets its own six-bit
scale and six-bit min. Dequantization is `d * scale * (low4 | high1 << 4)
- dmin * min`. This matches upstream ggml-common.h and gguf-py quants.py.

## Paths covered

- Lazy GGUF expert indexing and slicing (no whole-file dequantization).
- Small CPU reference oracle and locally compiled native packed C++ GEMV.
- Native packed C++ complete SwiGLU experts, including mixed Q4_K / Q5_K.
- Triton packed GEMV and fused gate/up/SwiGLU for both Q5_K and mixed formats.
- Existing adaptive expert cache and CPU fallback; no training-path changes.
- Benchmark helpers initialize both Q5_K half-float scales to finite values.

## Local validation

Run from a repo with 0106A–F already applied:

```bash
.venv/bin/python -m pytest -q tests/test_0106g_q5k_packed.py \
    tests/test_0106f_qwen3next_gguf.py \
    tests/test_0106e_cpu_packed_and_gpu_swaps.py
```

Then inspect and test your actual GGUF (file is not on the test machine):

```bash
MODEL=/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf
MINI_LLM_BACKEND=numpy .venv/bin/python scripts/probe_0106f_qwen3next_gguf.py \
    --model "$MODEL" > qwen3next_inventory.json
MINI_LLM_BACKEND=numpy .venv/bin/python scripts/probe_0106f_qwen3next_gguf.py \
    --model "$MODEL" --smoke-expert --layer 0 --expert 0 --cpu-backend native
MINI_LLM_BACKEND=cupy .venv/bin/python -m pytest -q tests/test_0106g_q5k_packed.py
MINI_LLM_BACKEND=cupy .venv/bin/python scripts/probe_0106f_qwen3next_gguf.py \
    --model "$MODEL" --gpu-smoke --layer 0 --expert 0 --cpu-backend native
```

`Q5_K` support does not imply full-model text decoding. Model-specific
DeltaNet and attention are still missing; additional GGUF quantization types
may also be present. Consult `unsupported_routed_formats` and
`tensor_counts_by_format` in the inventory before further implementation.

The GPU route has not been compiled/tested on the user's RTX 5080 here.
