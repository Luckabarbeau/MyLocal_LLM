# 0210: bounded independent full-model reference diagnostics

## Observed outcome

**Blocked before model load by the pinned reference loader's memory behavior. No independent full-model numerical claim is established.**

The existing actual-model smoke result (two tokens in 26.76 s, reported by the task owner) remains a smoke result only. This probe does not reinterpret it as numerical validation and did not rerun it.

The new probe was run once with the original first shard explicitly supplied. Results are in `0210_reference_validation.json`:

- Pinned source: `171e8846b4af9766c354064cb776cb34a50f053f`; Git status clean.
- Existing `build-cuda/bin/llama-cli --version`: `0.5.0-dev`, build 11192, commit `171e8846b`.
- New C++ harness compiled and linked successfully against the existing shared libraries with an explicit rpath. No llama.cpp source build was performed.
- Linked capability query succeeded: `GGML_TYPE_Q2_0 == 42`; `ggml_is_numa()` is false after NUMA initialization. This verifies API/type availability, **not execution of a Q2_0 model operation**.
- Full-model reference and production child processes were **not started**. No logits, layer vectors, numerical discrepancies, or greedy model results were obtained.
- Probe returned exit code **2** with `blocked_memory_safety_before_model_load` in approximately 1.22 s. This is an explicit unresolved reference blocker, not a passing model-comparison test.
- Both checkpoint shards' size/mtime pairs were unchanged; neither was hashed.

The CUDA-linked library initializes device discovery during the capability query and reports an RTX 5080. No model was loaded or offloaded. A successful CPU-only model run or zero GPU initialization allocation is **not** claimed.

## Evidence for refusing the model load

At this pinned source revision:

1. `src/llama-model.cpp`, `llama_model_base::load_tensors`, calls `ml.init_mappings(true, ...)`.
2. `src/llama-model-loader.cpp`, `init_mappings`, passes a prefetch size covering the file when mmap is enabled.
3. `src/llama-mmap.cpp` disables prefetch when its `numa` argument is true. Otherwise it uses `MAP_POPULATE` when the file has no lazy ranges and advises `POSIX_MADV_WILLNEED` for the complement of lazy ranges.
4. `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml_is_numa`, returns whether the detected NUMA-node count is greater than one. This host exposes only `/sys/devices/system/node/node0`; the linked query confirms false.
5. `src/models/qwen4exp.cpp` marks the large PLE embedding as `TENSOR_READ_LAZY`. This does not mark the expert banks lazy. The model can wrap CPU mmap storage without copying it, but its loader still requests eager residency/prefetch for the non-lazy regions.
6. This revision's public `llama_model_params` uses `load_mode = LLAMA_LOAD_MODE_MMAP` rather than `use_mmap`/`use_mlock` booleans. It exposes no independent no-prefetch option. `use_extra_bufts = false` avoids repacked CPU weight buffers but does not disable prefetch.

Thus mmap alone does not satisfy the requested **packed expert banks mapped, not eagerly resident** invariant. The roughly 46.8 GiB of packed expert banks cannot safely be treated as merely virtual by this loader configuration. The preflight deliberately does not experiment with that residency pressure. It neither modifies the pinned source nor interposes replacement loader functions.

**Single next experiment:** with separate approval, use a separately built independent reference exposing a no-prefetch loader option, verify lazy PLE/CPU mmap and bounded RSS first, then run the fixed `[1,2]` comparison. That requires expanding the present frozen-file/no-source-build scope. Do not edit the production model on the basis of this blocker.

## Implemented comparison contract (not yet model-validated)

Only these new files are in this task's write scope:

- `scripts/probe_0210_qwen4exp_reference.py`
- `tests/fixtures/qwen4exp_reference.cpp`
- `docs/qwen4exp/0210_reference_diagnostics.md`
- `docs/qwen4exp/0210_reference_validation.json`

The C++ harness is original driver code. It does not copy vendored kernels or transcribe the model graph: `llama_model_load_from_file`, `llama_init_from_model`, and `llama_decode` execute the independent linked qwen4exp implementation.

Configuration:

- Required explicit existing first-shard path; no download/default model discovery.
- Context 32, one sequence, one-token microbatches, explicit positions, fixed input IDs `[1,2]`, logits requested at **both** positions.
- `n_gpu_layers = 0`, empty device list, no KQV/op offload, mmap on, mlock off, lazy PLE on, extra CPU repacking off.
- FP32 K/V, FP32 saved logits and diagnostic arrays; flash attention disabled.
- One CPU thread/worker by default. Child environment sets NumPy backend and BLAS/OpenMP thread counts to one.
- Optional `--greedy-count 1..3` generates that many argmax IDs after `[1,2]`, without text decoding or EOS stopping. Default is just the two fixed input positions. Positions after the fixed prefix are labeled as not guaranteed to share inputs; comparisons there are trajectory diagnostics, not same-prefix parity.
- Reference process exits before the production child starts. No duplicate resident model processes.
- Compilation, preflight and both children share a 120-second deadline. Parent samples direct-child RSS every 50 ms and kills above 12 GiB; this is a sampling guard, not a hard RSS/cgroup limit and does not account for compiler descendant RSS. Child address space is limited to 112 GiB (including file mappings); core dumps disabled. The known eager-prefetch case is blocked before load rather than relying on the RSS sampler.
- Temporary executable, logs, raw FP32 vectors and NPY arrays are deleted after the attempt. Only a small JSON report persists. A result file is never overwritten by a later run.

### Layer diagnostics

The production child uses the existing callback without changing production code. The linked reference uses `cb_eval` and `ggml_backend_tensor_get`:

| Production stage | Linked reference boundary |
| --- | --- |
| `hc_attention_read` | First evaluated `hc_mixed-{layer}` |
| `post_mixer` | `hc_combine-{layer}` (attention residual write) |
| `hc_ffn_read` | Second evaluated `hc_mixed-{layer}` |
| `post_moe` | `l_last-{layer}` |
| `hc_output_read` | `result_norm` |

The scheduler's `ask=true` callback records names/selects boundaries only. Actual data copies occur only on evaluated selected tensors; FP32/contiguous layout and a 65,536-element cap are checked. GGML dimension-zero-fastest flattening corresponds to the production `[streams,width]` flattening at residual boundaries. Missing boundaries are reported, not synthesized. HC read occurrence ordering and coverage must still be verified on an actual reference run; compilation is not evidence that all requested callbacks will be present. The first **nonzero** boundary difference is reported descriptively, not declared a defect or a tolerance failure.

### Numerical reporting

For each full logit vector the comparator computes:

- maximum absolute difference;
- relative L2, `||actual-reference||2 / max(||reference||2, 1e-30)`;
- both argmax IDs, both top-10 lists and overlap;
- maximum absolute log-probability difference and differences on reference top-10 IDs, using stable FP64 log-softmax reductions.

Boundary vectors receive the same max-absolute and relative-L2 reductions. Nonfinite values and shape mismatches are explicit failures. No broad acceptance tolerance or equality assertion is applied.

Native production expert arithmetic uses FP32 activations, whereas linked GGML CPU quantized matmuls may quantize activations to Q8_K/Q8_0. A future observed discrepancy must retain that distinction; neither matching text/argmax nor small isolated codec errors proves full-model parity. A numerical mismatch would require first-divergence evidence before proposing any production correction.

## Validation performed

Passed:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/probe_0210_qwen4exp_reference.py --self-test
```

This checks metric reductions, log-softmax shift invariance and temporary synthetic logit/layer artifact comparison. **It is not an independent model test.**

The bounded actual-checkpoint attempt compiled/linked the harness and executed its capability query successfully, then reported the memory blocker described above:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/probe_0210_qwen4exp_reference.py \
  --model /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --llama-source /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/vendor/llama.cpp \
  --result docs/qwen4exp/0210_reference_validation.json
```

No retry, speculative loader workaround, production edit, or numerical success claim followed the initial blocker. Broader consolidated CPU validation is outside this probe and is being handled separately by the task owner.

## Authorized follow-up: separate CPU no-prefetch reference

The task owner subsequently authorized a separate reference source/build at `/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch`. **The initial memory blocker is now resolved for that separate build, but numerical parity remains unestablished.** The original result JSON above is preserved unchanged; the new run is recorded in `0210_noprefetch_reference_validation.json`.

### Source and build provenance

- Base SHA: `171e8846b4af9766c354064cb776cb34a50f053f`. Original source remained clean at this SHA after the run; original CUDA build was untouched.
- Copied using Python `shutil.copytree`, without `.git`, build directories, or model weights. Original licensing was retained byte-for-byte. The first copy filter also omitted source filenames starting with `build`; CMake correctly reported missing `cmake/build-info.cmake`. One localized copy correction restored omitted source files, without copying build directories. Configuration then succeeded. No reference mathematical implementation was changed.
- Compared 3,643 copied source files byte-for-byte against the clean base, with build directories excluded. The **only** differing file was `src/llama-model.cpp`, with precisely this one-line change:

```diff
-    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);
+    ml.init_mappings(false, use_mlock ? &pimpl->mlock_mmaps : nullptr);
```

The exact contextual unified diff is preserved in `0210_noprefetch_reference.patch` and embedded in the result JSON. The unchanged loader computes `prefetch_size = prefetch && use_mmap ? -1 : 0`; the unchanged mmap code gates `MAP_POPULATE` on nonzero prefetch and `POSIX_MADV_WILLNEED` on positive prefetch. Passing false therefore disables both, without changing graph or quantization arithmetic.

CMake configuration: Release, shared libraries, common/tests/tools/examples/server/app/MTMD off; CUDA/HIP/Vulkan/Metal/SYCL/OpenCL/RPC/BLAS off. Built only target `llama` using `cmake --build .../build-cpu --target llama --parallel 8`, with a 120-second tool timeout. The build completed successfully within that timeout. Since `.git` was deliberately absent, build-version Git metadata is unknown; provenance is established by source comparison, not a fabricated compiled revision string.

The probe received one localized correction pass adding `--llama-build`, exact-copy/diff verification, CPU-only CMake-cache checks, and linked-library checks. Before load, `ldd` confirmed libllama/libggml/libggml-base/libggml-cpu resolve to the new `build-cpu/bin` and showed no CUDA/HIP dependency. The existing repository C++ harness stayed unchanged. Only a temporary driver's obsolete NUMA-only guard was relaxed **after** the positive source/build checks; its exact guard-only diff is included in the JSON. The linked capability query still truthfully reports `numa_disables_prefetch: false`; loader patch verification is a separate recorded fact.

### Actual full-model measurements

One run used fixed IDs `[1,2]`, context 32, single sequence, single-token microbatches, FP32 KV, one CPU thread, CPU mmap weights and lazy PLE. Optional three-token greedy continuation was not requested. Both positions use exactly the same prefixes in the two implementations. Neither tokenization nor text decoding participates.

| Measurement | Position 0: input `[1]` | Position 1: input `[1,2]` |
| --- | ---: | ---: |
| Logit count | 248,320 | 248,320 |
| Maximum absolute logit difference | 0.958172082901001 | 1.006776511669159 |
| Relative L2 logit difference | 0.06164339321300017 | 0.05067542595511982 |
| Reference logit L2 norm | 1388.40126924956 | 1683.5608159287451 |
| Actual/reference argmax | 271 / 271 | 220 / 220 |
| Top-10 overlap | 9/10 | 9/10 |
| Maximum absolute log-probability difference | 1.0778488640567865 | 0.948035699819064 |

Top-10 IDs:

- Position 0 actual: `[271,198,524,303,11069,1358,3070,1892,1000,369]`.
- Position 0 reference: `[271,198,524,303,11069,1358,3070,1892,369,4831]`.
- Position 1 actual: `[220,1088,100561,14955,57491,10838,8313,2972,97634,5261]`.
- Position 1 reference: `[220,1088,57491,14955,100561,10838,8313,2972,14791,5261]`.

Full top-10 log-probabilities and signed differences are in the JSON. Argmax agreement does **not** establish numerical agreement. No equality/broad tolerance was asserted, and the process exit code 0 means diagnostic collection completed, not that parity passed.

Total probe elapsed time was **27.1565 s** including checks/harness compilation. Reference child: 1.5608 s, reported peak RSS 6,145,638,400 bytes (~5.72 GiB). Production child: 24.2231 s process elapsed, reported peak RSS 9,355,169,792 bytes (~8.71 GiB). Parent sampled peaks were 6,109,483,008 and 9,356,021,760 bytes respectively, both below 12 GiB. Reference exited before production started. The actual model's memory plan reports 83,606,638,080 canonical mmap bytes; this is virtual mapped file storage, not an allocation of resident expert banks. No full expert dequantization or duplicated resident model occurred. Both shards' size/mtime remained unchanged.

### Layer observations and unresolved diagnosis

386 FP32 boundary comparisons were collected (48 layers × four boundaries plus output collapse, at two positions), with no missing expected boundary filenames. The first **reported** nonzero difference is `p0_l0_hc_attention_read`: max absolute 4.845122471451759, relative L2 1.8329839493961013, reference L2 19.397197904191327, 2,560 values. However, **do not treat this as a verified first mathematical divergence**: the recorded reference attention-read and FFN-read norms are exactly identical within each layer, including layer 0. The harness associates duplicate `hc_mixed-{layer}` names by evaluation occurrence, and that association/data capture needs independent checking. Temporary raw vectors were deleted as designed, so equality or aliasing of those two arrays cannot be established from their norms alone.

The more clearly named first layer residual `p0_l0_post_mixer` has max absolute difference 0.0009640827775001526, relative L2 0.017725783533927172, reference L2 0.6005304899510213. First layer `post_moe`: max absolute 0.0011451542377471924, relative L2 0.009565881614235635, reference L2 1.3304051735091875. These are observations, not an identified incorrect equation. Requested arithmetic is FP32, but native-vs-GGML Q8 activation quantization remains a real distinction. No tolerance was imposed and no finite-difference test was involved.

The available evidence does not distinguish activation-quantization propagation, other numerical/graph differences, and an HC boundary-capture association problem. **No production code was edited and no rerun or speculative corrective cycle was performed.**

**Single cheapest next experiment:** capture only token 1, layer 0, recording each selected reference tensor's pointer identity, op, dimensions and source identities plus its immediate FP32 values; verify the attention/FFN HC boundary associations against the graph before interpreting first-divergence metrics. Keep the loader-only reference patch and frozen production implementation unchanged. This is proposed, not executed.

The metric/artifact self-test passed again after the localized probe edit. The task owner's separately reported 1,488 CPU tests/184 skips and CPU/GPU lifecycle greedy IDs `[220,100561,198]` are useful production context but were not independently rerun here and do not close this numerical discrepancy.

Reproduction (use no `--result` to preserve both existing reports):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/probe_0210_qwen4exp_reference.py \
  --model /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --llama-source /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/vendor/llama.cpp \
  --llama-build /home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu
```
