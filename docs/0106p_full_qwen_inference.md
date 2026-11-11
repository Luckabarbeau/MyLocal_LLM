# 0106P — Complete Qwen3-Next inference through the existing backend

Date: 2026-11-10. Implementation baseline:
`f133d3f2bafd47d7edc21b3fdb452a90a347a2db`.

**Verified:** a complete 48-layer forward/incremental path for the installed
Qwen3-Coder-Next checkpoint on NumPy and CuPy, including raw text tokenization,
embeddings, hybrid decoder layers, final normalization, packed vocabulary head,
and the existing generation session. An eight-token CUDA completion exactly
matched CPU greedy IDs. Synthetic full-model logits also match an independent
FP64 graph within the documented tolerance.

**Not established:** independent HF/llama.cpp full-model logits parity, chat
framing, long-context quality, steady-state throughput, or the requested Qwen3.8
pruned/IQ1_M variants. Those files were not needed or examined for this increment.

The implementation was saved in the baseline commit before this report was
written. The real-model measurements below are preserved prior-session
artifacts, not reruns performed while writing this report. CPU and focused CUDA
regressions were rerun successfully against that saved implementation.

## 1. Exact model and forward graph

Checkpoint:
`/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf`

| Quantity | Header/runtime contract |
|---|---|
| Architecture | `qwen3next` |
| Layers / model width | 48 / 2048 |
| Vocabulary | 151,936 |
| Metadata context ceiling | 262,144; not a validated allocation/performance target |
| Routed experts / selected per token | 512 / 10 |
| Routed / shared FFN width | 512 / 512 |
| Full attention | layers 3,7,...,47; the other 36 are recurrent |
| Runtime hidden/expert compute | FP32 |
| Embedding / output head | separate Q8_0 matrices |
| Verified real-model state horizon | 16 tokens |

```text
integer token IDs
  -> selected mmap embedding rows -> bounded host row cache -> backend hidden
  -> 48 existing decoder-layer adapters with independent KV/recurrent state
  -> final direct-gamma RMSNorm
  -> packed output projection
  -> existing ServingModel / GenerationSession sampling and stopping
```

There is no new attention, recurrence, expert dispatcher, sampler, training
backend, optimizer, or foundation-model checkpoint format. Existing native and
fully resident inference defaults remain unchanged.

## 2. Components and compatibility boundaries

### General row lookup and sparse GGUF access

`mini_llm/ops/embedding_inference.py` adds
`RowEmbeddingInference(n_rows, embedding_dim, row_provider, *, cache_bytes=0,
dtype="float32")`.

- Validates all integer IDs/ranges before provider calls or cache changes.
- Deduplicates rows within a call, even with persistent caching disabled.
- Retains owned CPU rows in a byte-bounded LRU, not a full GPU embedding table.
- Supports FP16/FP32/FP64 output storage and empty inputs without row reads.
- Nonempty GPU ID input incurs one explicit host readback per lookup; this is
  not a synchronization-free GPU embedding implementation.

`decode_gguf_row` in `external/gguf_tensor.py` slices packed matrices or BF16
bytes before row decoding. F32/F16 rows can remain views. It reuses established
packed decoding rather than defining another quantization format. The Qwen
adapter defaults to a 4 MiB host row cache. No complete embedding or expert bank
is dequantized to implement lookup.

### Whole-model adapter and state

`mini_llm/external/qwen3next_model.py` contains:

- `inspect_qwen3next_model(index)`: full supported-header validation and a
  retained-backbone estimate without tensor payload reads.
- `Qwen3NextGGUFModel.from_index(...)`: thin model-level orchestration over the
  verified 0106O layer adapters and shared executor.
- `create_generation_state(B, max_length)`: requires an explicit bounded horizon.
- `prefill(ids,state)`: appends a prompt chunk and returns last-position logits.
- `decode_one(ids,state)`: one token per batch row.
- `forward(ids)`: all causal `[B,T,V]` logits for bounded parity/benchmark work.
- `copy_generation_state` and in-place `reset`: independent native layer storage,
  explicit owner/position/capacity checks.

A runtime error after layers begin executing poisons that state until reset;
partial progress is never silently retried. Sessions sharing one managed
executor must execute serially. The model closes only executors it owns.
Cross-request speculative-state transactions are not introduced.

The packed head uses the existing lazy projection helper. Tied-head fixtures
reuse the embedding matrix for output, while the installed file has an explicit
separate `output.weight`.

### Existing loading and generation interfaces

`external/gguf_loader.py` dispatches Qwen3-Next headers to this adapter and
preserves the legacy Llama mapping. `model_loading.load_pretrained_weights`
also supports Qwen without making a second model mmap for its tokenizer.
Qwen requires **explicit FP32**; the shared pretrained loader's historical
FP16 default is not silently changed for other models.

The existing `ServingModel` and `GenerationSession` provide encoding, request
state, sampling, stopping, and text generation. `Qwen3NextGGUFIndex.report()` now
reports whether headers are compatible with the complete runtime; that boolean
is not proof of hardware fit or independent numerical parity.

### Offline Qwen2 tokenization

`external/gguf_tokenizer.py` adds the explicit `gpt2/qwen2` variant while
preserving `gpt2/smollm`. Installed metadata has 151,936 tokens, 151,387 merges,
EOS 151645, no BOS ID, and no implicit BOS insertion.

The implementation follows the inspected llama.cpp Qwen2 split behavior,
followed by byte encoding without a second GPT-2 regex pass. Tests cover code
whitespace, CRLF, individual digits, contractions, Unicode/combining marks,
special IDs, unchanged vocabulary IDs and an independent Python rank-BPE
reference. Unverified tokenizer variants remain rejected. Qwen `chat_prompt`
explicitly rejects unvalidated chat framing; raw completion works.

Reference revision retained from source inspection:
`171e8846b4af9766c354064cb776cb34a50f053f`, notably `src/llama-vocab.cpp`'s
`LLAMA_VOCAB_PRE_TYPE_QWEN2` and `src/unicode.cpp`'s
`unicode_regex_split_custom_qwen2`. Decoder provenance is recorded in 0106M–O.
No Strata binary, external inference runtime, or new dependency was used here.

## 3. Allocation safety and large-vocabulary correction

The model's default CUDA planning parameters are explicit:

- 4 GiB device allocation budget for backbone/cache/state/workspace planning.
- 2 GiB free-VRAM safety reserve.
- 256 MiB base workspace allowance, plus shape-dependent scratch/logit estimates.
- Zero expert-cache budget unless the caller opts in.
- No automatic use of all free VRAM.

The benchmark additionally uses a private CuPy pool capped at 4 GiB. Its pool
measurements exclude CUDA context/library allocations outside that pool and
are not process-wide peak VRAM measurements. Model-level estimates cover a
planned session; arbitrary concurrent forks or other GPU consumers still need
caller resource discipline. Host RSS/SSD traffic are not measured by this smoke.

Review and regressions identified and localized three accounting/validation
issues before the large CUDA run:

1. Free-VRAM checks now include remaining lazy backbone/cache allocations, not
   just the next state/workspace, without counting already-resident uploads twice.
2. Dense-only decoder and Q/K norm formats are rejected during header inspection.
3. Recurrent control/norm/convolution tensors retained in FP32 are estimated at
   their expanded size even if their source format is packed.

The first real CUDA run then reached the vocabulary head and failed with
`CUDA invalid argument`. The old packed GEMV grid was `(tokens, output_rows)`;
151,936 output rows exceeded the hardware grid-Y limit of 65,535. The correction
flattens token/output-row pairs onto grid-X. Weight decoding, equations,
reductions, dtype, and explicit CuPy stream launches are unchanged. Exact-output
CUDA regressions cover 65,536 and 151,936 rows with one/two input tokens. The
existing mock's warmup/launch-grid expectations were updated while preserving
its pointer and stream assertions.

## 4. Validation

Latest revalidation at the implementation baseline:

| Selection | Result |
|---|---:|
| Broad CPU native/external/MoE/generation regression | **1,004 passed, 78 skipped** |
| Model benchmark and large-output selection | **8 passed** |
| Packed-kernel / asynchronous execution CUDA selection | **51 passed, 56 deselected** |

Exact commands and durations are in `0106p_validation.json`. The eight-test
selection includes CPU-safe CLI checks, isolated-reference synthetic CUDA
execution, allocation-budget refusal, and four large-output kernel cases.
GPU-only tests are intentionally skipped in the broad NumPy selection.

### Numerical results from saved artifacts

| Run | Maximum reported logit absolute error | Maximum tolerance ratio | Private pool reserved bytes |
|---|---:|---:|---:|
| Synthetic CUDA vs CPU/FP64 | 2.145767e-6 | 0.076858 | 494,080 |
| Real, CPU experts, 2 generated tokens | 1.001358e-5 | 0.007762 | 2,673,235,968 |
| Real, 256 MiB promotion cache, 2 tokens | 9.655952e-6 | 0.008401 | 2,927,908,864 |
| Real, 256 MiB promotion cache, 8 tokens | 1.525879e-5 | 0.010134 | 2,927,908,864 |

The synthetic CPU independent-FP64 maximum was 2.322141e-6. For the eight-token
real run, CPU incremental/full-sequence error was at most 1.430511e-5.

Synthetic tolerances are rtol=4e-4, atol=8e-6; real full-model consistency uses
rtol=4e-4, atol=1e-3 across distinct FP32 reduction orders. These thresholds
were declared before the real run and were not loosened to pass it. The
large-output exact fixtures use exact array equality. Real-model comparisons
use the same packed weights and matched token inputs: these are consistency
checks, not estimates of quantization error versus original floating weights.

The real raw prompt was `def add`, token IDs `[750,912]`. CPU and CUDA produced
identical eight greedy IDs:
`[2877,11,293,982,262,470,264,488]`, yielding:

```python
def add(a, b):
    return a +
```

This is deliberately truncated at eight generated tokens, not a complete-function
quality result. CUDA checked the matched-input continuation and full-sequence
logits; the CPU reference generation itself uses the existing generation session.

### Observed timing and cache behavior — not controlled throughput

| Saved run | CUDA-path prefill wall ms | Mean decode wall ms |
|---|---:|---:|
| CPU experts, 2 tokens | 2763.181 | 1009.760, one decode step |
| Promotion cache, 2 tokens | 2109.079 | 637.286, one decode step |
| Promotion cache, 8 tokens | 1826.778 | 647.348, seven decode steps |

In the eight-token run, CPU-reference prefill was 2739.658 ms and mean decode
1564.975 ms. These are single smoke observations, not a fair worker/cache
speedup comparison: compilation, cache state, fixed execution order, mmap/page
cache and CPU scheduling are uncontrolled. CPU reference generation precedes
CUDA and warms file pages. CUDA event intervals also include stream idle time
while host work is scheduled; they are not isolated GPU kernel times.
Speculative decoding is disabled. No Strata throughput comparison was run.

Eight-token promotion-run counters, **including the final nine-token full-forward
validation replay**:

- 6,345 grouped expert requests; 599 hits; 812 promotions; 672 evictions.
- 1,411 GPU expert calls and 4,934 CPU expert calls.
- 8,640 routed assignments: 1,322 GPU-hit, 1,071 promoted, 6,247 CPU.
- 1,547,657,216 expert-transfer bytes, plus activation staging.
- End-of-run resident expert bytes: 267,280,384 within the 268,435,456-byte cap.

The reported request hit rate is 9.44%; it is not the assignment hit fraction.
The two-token promotion run had zero hits. Neither trace is a representative
conversation profile. Do not divide these replay-inclusive counters by eight
and call them bytes/uploads per generated token.

## 5. Reproduce and use the existing generation interface

Bounded real-model correctness smoke, including an isolated CPU reference:

```sh
MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python benchmarks/benchmark_0106p_qwen_model.py \
  --model /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --prompt 'def add' --max-new-tokens 8 --context 16 --cpu-workers 2 \
  --expert-cache-mib 256 --promote-after 1 \
  --output benchmarks/results/0106p_real_model_8token_gpu.json
```

The smoke CLI intentionally limits output to eight tokens and context to 128.
Ordinary generation uses the already-existing API, without the extra CPU oracle:

```python
from pathlib import Path
from mini_llm.model_sources import ModelSpec
from mini_llm.serving_models import load_model
from mini_llm.generation import GenerationSession

model = load_model(ModelSpec(
    id="qwen-coder-next",
    path=Path("/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"),
    format="gguf", dtype="float32", context_length=16,
))
try:
    output = GenerationSession(model).generate("def add", max_tokens=8, temperature=0)
    print(output.text)
finally:
    model.engine.close()
```

Set `MINI_LLM_BACKEND` before launching Python; never switch backends in one
process. The canonical loader uses safe default expert-cache/worker settings,
not the benchmark's explicit 256 MiB/two-worker settings. Advanced placement
options are available through `load_gguf_model` / `Qwen3NextGGUFModel.from_index`.

## 6. Files and next step

Core additions: `ops/embedding_inference.py`, `external/qwen3next_model.py`.
Extended: `external/gguf_tensor.py`, `gguf_tokenizer.py`, `gguf_loader.py`,
`qwen3next_gguf.py`, `model_loading.py`, and the packed GEMV launch in
`ops/quantized_linear.py`. Tests are the `test_0106p_*.py` series plus the
localized launch-grid expectation in `test_0106h_triton_cupy_pointer.py`.
Benchmark: `benchmarks/benchmark_0106p_qwen_model.py`.

Raw artifacts remain in ignored `benchmarks/results/`; their checksums and key
measurements are preserved in `0106p_validation.json`. No source/kernel change
was needed during this report-finalization pass, and no stash was applied or
new commit/branch created.

Recommended next work, in order:

1. Independent real-model logits/reference validation on identical token IDs,
   then explicit chat-template/stop behavior and longer streaming checks.
2. Controlled same-workload cold/warm measurements, with separate generation
   traces that exclude validation replays and compilation.
3. Profile CPU expert cost, host synchronization, routing/count readback and
   transfer pressure; only then tune cache budgets/policies or kernels.

Large-context allocation/performance, persistent concurrent sessions, profile
seeding/static hot regions, native IQ support and SSD/n-gram tiers remain
separate work. Continue with this installed Coder-Next checkpoint first.
