# WP4: bounded native PLE runtime integration

## Scope and authority

This integrates the validated `ple.h/.cu` and `ple_hash.h/.cpp` primitives into the **experimental native engine**, following `WP4_PLE_PRIMITIVES.md`. The primitive sources are unchanged. This is a measured PLE integration result, **not a claim that all WP4 or end-to-end model inference is complete**. Parent-owned status documents and the reference manifest are not updated.

Implementation:

- `mini_llm/native_cpp/ple_runtime.inc`: private native weight/table and per-sequence state owners, validation, budget admission, transactions, reset and diagnostics.
- `runtime.cpp`: primitive headers and private include only for this integration.
- `runtime.h`: handle-based C ABI; `mini_llm/native/__init__.py`: thin Python ownership/dispatch wrappers.
- `qsa_projection.h/.cu`: bounded F32/F16/BF16/IQ4_NL row decode reusing the existing GPU packed lane decoder and IQ4 lookup table. Existing projection math is unchanged.
- `CMakeLists.txt`: adds the two validated PLE translation units.
- `tests/test_native_engine_ple.py`: integration and actual-checkpoint oracle tests.

No production reference implementation is edited. No Python hash, embedding construction, projection, normalization, gate or convolution is used in the native execution path. Python numerical work in the tests is an independent oracle only.

## Actual descriptors, not an assumed dense table

Inspected using the native GGUF descriptor API against the checkpoint in `WP2_GGUF_INVENTORY.json`:

| Tensor | Python/output-first shape | Stored type | Stored bytes |
|---|---|---|---:|
| `per_layer_token_embd.weight` | `[320001536,160]` | IQ4_NL (20) | 28,800,138,240 |
| `blk.1.ple_key.weight` | `[10240,2560]` | BF16 (30) | 52,428,800 |
| `blk.1.ple_value.weight` | `[2560,2560]` | BF16 (30) | 13,107,200 |
| `blk.1.ple_norm_{key,query,conv}.weight` | each `[10240]` | F32 (0) | each 40,960 |
| `blk.1.ple_conv1d.weight` | `[10240,4]` | F16 (1) | 81,920 |

The table lives in shard 1 (zero-based), starting at byte 192. Each row contains five 32-element/18-byte IQ4_NL blocks: **90 bytes per row**. Each step:

1. Calls the validated host uint64-wrapping hash primitive with sequence-owned predecessors.
2. Copies exactly sixteen selected 90-byte rows, in head order, from the retained native read-only mmap into engine-budgeted pinned staging.
3. Uploads 1,440 packed bytes and decodes `[16,160]` directly on GPU into a 2,560-element embedding. IQ4_NL low nibbles decode the first sixteen values of a block, high nibbles the next sixteen, using the existing nonlinear codebook and FP16 scale.
4. Uses the existing direct packed `reference_fp32` projection launch for key and value, then the validated PLE apply launch.

There is no full-table upload, host expansion, GPU expansion, pin/register, or mutable Python table pointer. Native selected-row decoding is checked **exactly** against `Qwen4ExpPLE.embedding_reference`, which independently reads and dequantizes sixteen rows using `PackedGGUFWeight`.

## API and ownership

```python
weights = engine.ple_weights(
    table_lease, config, key, value, gamma_key, gamma_query, gamma_conv, conv,
    epsilon=epsilon, arithmetic='reference_fp32')
state = weights.state(capacity=16)
state.step(uploaded_f32_streams, token, position)  # F32 [4,2560]
# Or, without any stream readback/reupload:
state.step(hc_state, token, position)             # initialized native HC D=2560
state.reset()
```

The caller supplies the architecture-approved `PLEConfig` and epsilon. The ABI independently validates all hash ranges and shapes and copies the three multipliers, sixteen offsets and sixteen head vocabularies into native storage. The wrapper also checks config/table row-count agreement. This is not a second architecture parser.

- Native `PLEWeights` retains a `shared_ptr<PayloadLease>` for the **table**, plus shared owners for six uploaded weights/controls and their engine. Existing mappings use `O_RDONLY`, `PROT_READ`, `MAP_PRIVATE`.
- Closing the public store/table lease/upload/weight/engine handles does not invalidate a retained PLE state. Sources must remain immutable until all consumers release them, as with existing payload leases.
- Key/value accepted types are the existing F32/BF16/Q4_K/Q5_K/Q6_K/IQ4_XS projection types. The actual checkpoint exercises BF16. BF16 is decoded lane-locally during GPU matvec, without BF16 activation rounding or a full FP32 weight copy.
- Controls accept F32/F16/BF16. F32 aliases the immutable upload; F16/BF16 expands losslessly on GPU into budgeted FP32 storage, finite-checked before publication. Gammas accept flat `[10240]` or grouped `[4,2560]` and are already multiplicative scales: **no extra +1**.
- Native IDs and predecessor sentinel semantics are exactly the primitive contract: nonnegative int32 current token; `-1` only for missing predecessors; wrapping multiplication before XOR/modulo; independently validated positive head vocabularies and ranges. PLE EOS metadata, not tokenizer EOS, governs predecessor cutting.
- All kernels/copies use the engine-owned nonblocking stream. The parent initializes/checks device status and synchronizes completion; primitives do not own/clear status.

### Sequence state and transaction

Each `PLEState` owns its predecessors, committed row IDs and position, nine normalized-z history entries, output and staging/scratch. Two states sharing weights do not share history. Capacity counts successful steps, not token values; capacity does not scale memory allocation.

Step checks the explicit expected position and capacity. Predecessors, history and output publish together only after successful launch, stream completion and finite-status checks. A metadata or numerical failure preserves committed predecessors/history/position, invalidates output diagnostics and permits a corrected retry at the same position. CUDA failures quarantine the PLE state until reset; recovery from a poisoned CUDA context is not promised. Failed construction leaves no registered owner and unwinds allocated resources, including an invalid weight-handle lookup before state ownership is installed.

EOS is a normal convolution/history step. It affects future hash context but **does not clear history**. Reset zeroes storage/history, resets predecessors to `(-1,-1)` and position to zero, clears committed row IDs and invalidates output without reallocating.

Direct HC input requires same-engine, D=2560, initialized, uncollapsed, noninvalid HC state **without a pending read/write transaction**. PLE computes into its own staging first; after the finite gate it copies into HC's uncommitted buffer and synchronizes, then exchanges HC's own stream/staging pointers and invalidates its hidden snapshot. Allocation ownership is never exchanged between PLE and HC. The PLE output/history commit then publishes without a fallible numerical operation. No numerical readback is used to feed HC.

Diagnostics are explicit only: `readback_for_test('streams'|'history'|'embedding'|'gates', destination)` and `metadata` returning `(predecessors, rows, position)`. Diagnostic readbacks are not native graph inputs. History and metadata remain available after non-CUDA failed steps; output/embedding/gates require the last step to have succeeded.

## Bounded memory accounting

All requested device and pinned payloads compete in the existing engine budget. Creation admits the entire additional device+pinned amount before allocating; step/reset allocate nothing.

| Owner | Additional device bytes | Additional pinned bytes |
|---|---:|---:|
| Weight owner | 4 status + 4 per expanded control element | 4 |
| Each PLE sequence | 1,045,940 | 1,444 |

Per sequence: 261,124 float elements (two histories, two outputs, embedding, key/value, normalization scratch, u, four gates and z), 1,440 packed device bytes, four-byte device status; host staging is 1,440 bytes plus four-byte host status. Input upload bytes are counted by their existing upload owners, not double counted here. Actual F16 convolution expansion adds 163,840 device bytes; actual FP32 gammas are aliased.

Measured actual-checkpoint fixture, including retained uploads, controls, one sequence and uploaded input streams:

- Device payload: **66,991,544 bytes**.
- Pinned payload: **1,448 bytes**.
- Peak combined payload: **66,992,992 bytes (63.8895 MiB)**, below 512 MiB.
- Counters/peak remain unchanged across all steps and resets.

This is requested native payload accounting, **not** total GPU/process memory. CUDA context/allocator overhead, mmap virtual address space/page cache, host registry objects and independent Python oracle allocations are outside it. Mapping the sparse table does not charge its full file length as device or pinned memory.

## Measured validation

RTX 5080, CUDA toolkit `/usr/local/cuda`; exact checkpoint identified by the existing WP2 inventory. Library `/tmp/mini-llm-wp1-build/libmini_llm_native.so`, SHA256:

`23c4afd9ffd89cfe927c4a0ad29e20de29286557b1159a10b5a343c399cef1c6`

All comparisons are FP32, no finite differences. Requested `atol=rtol=8e-6` for numerical results; hashes/metadata and decoded actual embeddings are exact.

### Focused integration

- Sparse synthetic `2^31`-row IQ4_NL table, only selected rows physically written; high-bit/overflow multipliers, distinct head offsets, maximum int32 token, current/repeated/predecessor PLE EOS and a distinct tokenizer-like token.
- Three control formats F32/F16/BF16. Sixteen steps plus reset/replay, checking output, complete nine-entry history, embedding and exact metadata every step.
- Independently progressing second sequence feeds native HC streams directly; checks HC and PLE outputs/history and independent release of both allocation owners.
- Maximum synthetic output absolute error: **4.76837158203125e-7** for each control format.
- Public native store/lease/uploads/weight owner close before row fetch; public engine close followed by another valid state step; retained ownership and final allocation release checked.
- Wrong positions, exhausted capacity, cross-engine/wrong-shape/stale inputs, negative IDs via direct ABI, Python token overflow/bool, nonfinite streams, invalid hash vocabs/ranges, unsupported arithmetic/epsilon and diagnostic capacities.
- Failed step preserves complete committed history and metadata; a retry succeeds. HC uninitialized/pending transactions reject without PLE commit.
- Exact **one-byte-short** state and weight-owner budgets reject without counter changes; huge capacity costs the same constant allocation. Zero/stale handles and failed partial construction are checked.

### Actual checkpoint

Sixteen tokens (including repeated PLE EOS) followed by reset and sixteen-token replay, **32 successful steps and two resets**, compared with actual GPU `Qwen4ExpPLE` on the same stored weight bytes and Python row decoder. The separate reference store remains open; public native sources close before execution.

- Maximum output absolute error: **1.1920928955078125e-7**.
- Maximum normalized-history absolute error: **3.337860107421875e-6**.
- All selected-row embeddings, hash row IDs, predecessors and positions match exactly.
- Zero history/missing predecessors verified after each reset; stable allocation counters verified every step.

### Commands and final verified state

```sh
cmake -S mini_llm/native_cpp -B /tmp/mini-llm-wp1-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build /tmp/mini-llm-wp1-build -j 4

MINI_LLM_BACKEND=cupy NVIDIA_TF32_OVERRIDE=0 OPENBLAS_NUM_THREADS=1 MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q -s tests/test_native_engine_ple.py

MINI_LLM_BACKEND=numpy NVIDIA_TF32_OVERRIDE=0 OPENBLAS_NUM_THREADS=1 MINI_LLM_TEST_PLE_CUDA=1 MINI_LLM_QSA_LAYER_CHECKPOINT=1 MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q tests/test_native_engine*.py

MINI_LLM_BACKEND=cupy NVIDIA_TF32_OVERRIDE=0 OPENBLAS_NUM_THREADS=1 MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python -m pytest -q tests/test_native_engine_ple.py
```

Final build passed. Final full suite: **480 passed, 3 skipped in 66.96 s**, with QSA checkpoint enabled and both standalone PLE primitive tests enabled. Full-suite NumPy mode dispatches the actual PLE checkpoint oracle to a bounded CuPy subprocess; it is not skipped or replaced by CPU PLE math. Final memcheck: **6 passed in 21.73 s, ERROR SUMMARY: 0 errors**, including the actual checkpoint and direct-HC paths.

The first focused run passed five integration tests; ownership review subsequently identified a destructor dereference when an invalid weight handle fails lookup before installation. A localized null-owner guard and sixth regression test were added, followed by the successful final rebuild/full suite/memcheck above. No PLE mathematical correction was necessary.

## Remaining scope

No batching, graph capture, full model scheduler, logits/tokenizer/sampler integration, complete recurrent/FFN/HC block orchestration, concurrent session throughput, or performance claim is made here. Single-sequence PLE state is an explicit native owner, not implicitly attached to the existing synthetic transport session. Direct HC and uploaded-stream inputs are supported; no other internal stream-owner API is silently emulated. PLE integration correctness/memory is measured; complete WP4 status remains with the parent.
