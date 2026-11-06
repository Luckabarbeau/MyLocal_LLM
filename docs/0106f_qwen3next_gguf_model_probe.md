# 0106F: Qwen3-Coder-Next GGUF incremental integration

This is a **weight/index and single-expert execution milestone**, not an end-to-end Qwen3-Next model decoder. It reuses `PackedGGUFWeight`, `HeterogeneousExpertExecutor`, the 0106B–C Triton quantized kernels, and the 0106E CPU backend. It does not install or vendor a separate inference engine.

## Local model on the RTX 5080 workstation

```bash
MODEL=/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf
ls -lh "$MODEL"
# gguf is optional in the project and only used to inspect directory/metadata:
.venv/bin/python -m pip install 'gguf>=0.16'
MINI_LLM_BACKEND=numpy .venv/bin/python scripts/probe_0106f_qwen3next_gguf.py \
  --model "$MODEL" --device-budget-gib 6 --ram-budget-gib 32 \
  | tee qwen3next_0106f_inventory.json
```

The first pass reports actual tensor types, parameter layouts, their approximate compressed memory consumption, and whether some projection types lack packed kernels. **It does not dequantize the 49+ GB model**.

Smoke test a *real* routed expert from the GGUF with the shared CPU backend (native packed backend auto-compiles a local shared library if supported; no root privileges):

```bash
MINI_LLM_BACKEND=numpy .venv/bin/python scripts/probe_0106f_qwen3next_gguf.py \
  --model "$MODEL" --smoke-expert --layer 0 --expert 0 --cpu-backend auto \
  | tee qwen3next_0106f_expert_cpu.json
```

Optional **first GPU correctness test** (will compile Triton kernels, compare against CPU; CUDA correctness is not established until run on the RTX 5080):

```bash
MINI_LLM_BACKEND=cupy .venv/bin/python scripts/probe_0106f_qwen3next_gguf.py \
  --model "$MODEL" --gpu-smoke --layer 0 --expert 0 \
  | tee qwen3next_0106f_expert_gpu.json
```

**Do not execute the smoke test if inventory lists unsupported routed expert formats**; those must have new kernels or a deliberate mixed-format fallback first.

## Remaining steps for complete inference

1. Implement/validate the exact Qwen3-Next Gated DeltaNet, causal convolution state, gated full-attention layers, and recurrent state; use layer-by-layer reference parity.
2. Wire all quantized dense projections, RMSNorm, softmax routing with 10/512 experts and the gated shared expert into the existing model runtime.
3. Support all weight formats **actually present** in the UD-Q4_K_XL file, without full-model dequantization; validate fused gate/up order.
4. Implement tokenizer, prompt template, output head, and end-to-end token/logit parity; only then expose OpenAI-compatible serving.
5. Benchmark token-level end-to-end latency and miss cost versus llama.cpp with identical GGUF, context and prompt.

The cache budgets in the CLI are *planning inputs*, not allocations; the probe reports metadata only, and the optional one-expert smoke does not allocate a model-scale GPU cache.
