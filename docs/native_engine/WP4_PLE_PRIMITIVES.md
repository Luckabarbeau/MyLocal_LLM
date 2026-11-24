# WP4: isolated PLE primitives and integration handoff

## Scope and provenance

New internal C++ files only:

- `mini_llm/native_cpp/ple_hash.h/.cpp`: exact host hashes and explicit predecessor advancement.
- `mini_llm/native_cpp/ple.h/.cu`: optional dense FP32 projections and projected-input PLE apply.
- `tests/native_ple_primitives_harness.py`: temporary standalone host/CUDA libraries, not production bindings.
- `tests/test_native_engine_ple_primitives.py`: host test and opt-in CUDA pytest entry point.

No runtime, C ABI, Python production binding, CMake, codebook, manifest, or existing implementation changes. These sources are intentionally **not integrated** into the engine build yet.

Math source: `mini_llm/external/qwen4exp_ple.py`, `PLEConfig.rows` lines 70–85, `_norm` lines 201–203, `step` lines 212–229, and `apply_embedding` lines 231–257. The reference's stable `_sigmoid` comes from `qwen4exp_hyperconnection.py`. Converted gammas already contain their multiplicative scale: never add 1. Four residual streams are channels, not time.

## Exact host hash contract

`ml_ple_hash_rows(config, token, predecessors, rows, next_predecessors)` has no CUDA dependency or allocation. Returns `bool`; invalid arguments return false without writes.

- `token` and `config.eos`: nonnegative signed 32-bit IDs. This is **PLE EOS metadata**, not necessarily tokenizer EOS.
- `predecessors`: two `int32_t`, oldest first; `-1` means missing. Values below `-1` are invalid.
- Three `uint64_t` multipliers, sixteen offsets and sixteen positive vocabulary sizes. `1 <= table_rows <= 2^31`; every head range must fit. Validation subtracts offsets before comparing vocab sizes, avoiding unsigned addition overflow.
- Start context with current token. Visit immediate predecessor then older predecessor. A missing token or EOS cuts that predecessor and all older positions to PLE EOS. Current EOS itself is hashed normally; its presence cuts the context of **subsequent** tokens.
- Multiply each context ID by its multiplier using unsigned 64-bit arithmetic, wrapping **before XOR/modulo**. Heads 0–7 use product0 XOR product1; heads 8–15 also XOR product2. Per-head row is `hash % vocab[h] + offset[h]`.
- Output `rows`: sixteen `uint64_t`. Output next predecessors: `(old_predecessors[1], token)`. Both outputs must be disjoint from all inputs and one another. Inputs are unchanged.
- Explicit sequence reset is `(-1,-1)` **and** zero convolution history. EOS alone does **not** reset history.

## Device buffer/layout contract

Single token, single sequence; `D=2560`, `C=4`, `W=C*D=10240`. All floating buffers are contiguous FP32 on the caller's current CUDA device. Row-major layouts, no padding/strides, no implicit BF16 rounding. Offset notation below is in **float elements**, not bytes.

### Optional `ml_ple_project_launch`

| Buffer | Direction | Shape | Elements | Bytes |
|---|---|---:|---:|---:|
| embedding | input | `[D]` | 2560 | 10240 |
| key_weight | input | `[4D,D]` | 26214400 | 104857600 |
| value_weight | input | `[D,D]` | 6553600 | 26214400 |
| projected_key | output | `[4,D]` | 10240 | 40960 |
| value | output | `[D]` | 2560 | 10240 |

Output-first dense dot products, no bias. Weights use offset `output*D+input`. Dense projection weights total 125 MiB. The integration may instead pass outputs from existing **exact** dense/packed projection machinery to apply. This primitive does not decode or own weights.

Embedding construction is outside these sources: gather/dequantize the sixteen selected IQ4_NL rows of width 160, concatenate in **head order** (`embedding[h*160+j] = decoded_row[h,j]`). No averaging or summing heads, no full table materialization.

### `ml_ple_apply_launch`

| Buffer | Direction | Shape | Elements | Bytes |
|---|---|---:|---:|---:|
| streams | input | `[4,D]` | 10240 | 40960 |
| projected_key | input | `[4,D]` | 10240 | 40960 |
| value | input | `[D]` | 2560 | 10240 |
| gamma_key | input | `[4,D]` | 10240 | 40960 |
| gamma_query | input | `[4,D]` | 10240 | 40960 |
| gamma_conv | input | `[4,D]` | 10240 | 40960 |
| weights | input | `[4,D,4]` / `[4D,4]` | 40960 | 163840 |
| history | input | `[9,4,D]` | 92160 | 368640 |
| key_norm | scratch | `[4,D]` | 10240 | 40960 |
| query_norm | scratch | `[4,D]` | 10240 | 40960 |
| u | scratch | `[4,D]` | 10240 | 40960 |
| gates | output | `[4]` | 4 | 16 |
| z | output | `[4,D]` | 10240 | 40960 |
| updated | output | `[4,D]` | 10240 | 40960 |
| next_history | output | `[9,4,D]` | 92160 | 368640 |
| status | sticky in/out | device `int` | 1 | 4 |

Apply scratch + outputs total **573456 bytes**, plus four-byte status; inputs total 747520 bytes. Scratch/output buffers must be mutually disjoint and disjoint from inputs. Read-only inputs may share storage. No in-place update. Buffer allocation, device/size/contiguity validation, nonoverlap, and lifetime are caller obligations.

Element offsets:

- stream/gamma/key/z: `c*D+j`.
- history: `(t*4+c)*D+j`, oldest to newest `t=0..8`.
- convolution coefficient: `(c*D+j)*4+tap`, taps 0,1,2,3 correspond to history indices **0,3,6,current**.

Equations (all FP32):

```text
RMS(x,g)[c,j] = (x[c,j] / sqrt(mean_D(x[c,:]^2) + epsilon)) * g[c,j]
key_norm = RMS(projected_key, gamma_key)
query_norm = RMS(streams, gamma_query)
score[c] = sum_D(key_norm[c,:] * query_norm[c,:]) * float32(1/sqrt(D))
gates[c] = sigmoid(sign(score[c]) * sqrt(max(abs(score[c]), float32(1e-6))))
u[c,j] = gates[c] * value[j]
z = RMS(u, gamma_conv)
conv = history[0]*weights[...,0]
conv = conv + history[3]*weights[...,1]
conv = conv + history[6]*weights[...,2]
conv = conv + z*weights[...,3]
updated = streams + (u + conv*sigmoid(conv))
next_history = concatenate(history[1:], z[None,...])
```

`sign(0)=0`, so a zero score gives exactly 0.5, not sigmoid(0.001). Negative scores remain negative after signed square root. History stores **normalized z**, never `u`, residual streams, or convolution outputs. At step t the taps are t−9, t−6, t−3, t. Convolution addition order and residual parentheses match Python. Division precedes gamma multiplication. CUDA uses explicit round-to-nearest operations and a 256-thread reduction; no FMA, fast math, or lower-precision path. Reduction order can differ from Python, so floating results are tolerance-based, not promised bitwise identical. Hash results and history shifts are exact.

## Launch/completion/ownership handoff

1. For a new sequence, initialize host predecessors to `(-1,-1)` and device history to zero. Keep both sequence-owned, not layer-global.
2. Compute rows and next host predecessors. Gather/decode only selected rows; project embedding to key/value (optional provided dense launch or existing exact projection).
3. Initialize caller-owned device `status` to zero on the supplied stream. Launch apply with finite positive FP32 epsilon and all required pointers.
4. Launches allocate, copy and synchronize **nothing**. Every kernel uses the supplied stream; no hidden default-stream work. Keep all input/scratch/output storage alive and stable until that stream completes. Production functions have C++ linkage only.
5. Check returned `cudaError_t` for submission errors, asynchronous CUDA errors at the existing completion gate, and status. Invalid host arguments return `cudaErrorInvalidValue` before launching and do not change status. Nonfinite consumed inputs, intermediate arithmetic or outputs set sticky status=1; no kernel clears it. Results are invalid if status is nonzero. A late submission failure may leave partial output, also invalid.
6. On successful completion, publish `updated`, new predecessors and `next_history` together. Do not commit next host predecessors on failed device work. Ping-pong separate stream/history buffers; preserve old state until completion. On EOS, do exactly the same history update, with **no clearing**. A real new sequence requires explicit reset.

No batched requests, ring history, table ownership, weight decode/upload, runtime dispatch, or graph integration is introduced here. Those remain subsequent integration work.

## Validation

Standalone (host compiler, NumPy, CuPy, CUDA toolkit/GPU for full run):

```sh
MINI_LLM_BACKEND=numpy .venv/bin/python tests/native_ple_primitives_harness.py --hash-only
MINI_LLM_BACKEND=numpy .venv/bin/python tests/native_ple_primitives_harness.py
MINI_LLM_BACKEND=numpy MINI_LLM_TEST_PLE_CUDA=1 .venv/bin/python -m pytest -q tests/test_native_engine_ple_primitives.py
```

`NVCC` overrides compiler discovery; otherwise PATH then `/usr/local/cuda/bin/nvcc`. CUDA compilation is bounded to 180 seconds and uses the current device architecture, `--fmad=false`, and temporary test-only shim/shared library. Host build is bounded to 60 seconds with warnings as errors. Default pytest runs host checks and skips the opt-in CUDA compile.

Observed standalone validation on RTX 5080, CUDA toolkit at `/usr/local/cuda`, NumPy reference backend:

- **642 exact hash/state comparisons** against `PLEConfig.rows`, including uint64 overflow, zero/high-bit multipliers, distinct head vocabs/offsets, initial/missing context, current/repeated/predecessor EOS, tokenizer-like non-PLE EOS, maximum int32 token, random sequences, and invalid configs without writes.
- Actual D=2560 dense key/value projections compared to Python FP32 matrix products.
- Actual Python `Qwen4ExpPLE.apply_embedding` with pure projected-input callables (no codebook/store), random, zero, positive/negative/zero/tiny/saturated gates, each individual convolution tap, nonzero initial history, and twelve consecutive updates exceeding the nine-slot history length.
- Intermediate RMS/key/query, gates, u and z checked; exact input immutability/history shifts; invalid epsilon no-write behavior; nonfinite untapped history validation and sticky status.
- Opt-in pytest command above **passed: 2 passed in 1.99s** (host and CUDA).
- Full standalone command **passed**, maximum absolute numerical error **7.62939453e-06**, requested `atol=rtol=8e-6`, FP32, no finite differences. No production source corrections were needed after testing; harness capability parsing/indentation were corrected before the successful run.

Tests deliberately do not load a model/codebook or exercise runtime integration. The apply API has no token/EOS parameter: EOS cannot implicitly reset its history; token boundary handling is checked separately in the host hash tests.
