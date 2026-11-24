# WP5 isolated GPU MoE primitives

## Scope and ownership

New standalone sources only:

- `mini_llm/native_cpp/moe.h`: internal C++ launch contract and packed descriptors.
- `mini_llm/native_cpp/moe.cu`: routing, registered-descriptor expert execution,
  shared branch, and final combination.
- `mini_llm/native_cpp/moe_formats.cuh`: canonical licensed IQ lane decoder and
  constant codebooks, copied verbatim from `iq_packed_gpu._cuda_source('constant')`
  up to the first projection kernel; provenance comments precede the source.
- `tests/native_moe_primitives_harness.py`: standalone temporary native shim and
  unchanged Python oracle comparisons. Not a runtime binding or model dispatcher.

No edits to runtime, runtime headers, Python bindings, CMake, existing ops, or
existing tests. Link the existing `qsa_projection.cu` alongside `moe.cu` when the
parent integrates these primitives. The parent owns registration, immutable
weight leases, CUDA device/stream selection, budgets, status, workspace lifetimes,
and result validity. This is **not** runtime integration or a CPU/GPU dispatcher.

The only arithmetic contract is **reference_fp32**. There is no Q8-activation
mode, silent precision selection, fallback, CPU decoding, allocation, host-device
transfer, synchronization, or Python control in the launch implementation.
Stored Q8_0 shared-down **weights** do not imply activation quantization.

## Native API / handoff

All activation, workspace, route, and descriptor-array pointers are already on
the caller's current GPU. Matrix descriptor scalar arguments for the shared
branch are passed by value from host metadata; their packed payload pointers
are GPU addresses. There is no device descriptor download.

`MLMoeMatrix` is 24 bytes: payload pointer, uint64 physical row stride, uint32
GGML type, and uint32 reserved (zero). `MLMoeExpert` contains gate/up/down and is
72 bytes on the validated 64-bit ABI. The parent uploads a registered descriptor
array of **512** entries. Unselected entries may be null. Kernels access only the
10 selected entries; no host expert-list construction occurs during execution.
Payloads can be separate allocations or subranges of retained allocations.
Physical row padding is supported for routed matrices and shared down.

| Primitive | Inputs | Caller-owned outputs/workspace |
|---|---|---|
| `ml_moe_route_launch` | FP32 logits[512], finite nonnegative expert_scale | FP32 probabilities[512], int32 ascending IDs[10], FP32 scaled weights[10] |
| `ml_moe_experts_launch` | descriptors[512], IDs[10], already-scaled weights[10], FP32 x[D] | hidden[10,F], weighted terms[10,D], routed[D] |
| `ml_moe_shared_launch` | shared descriptors, F32 scalar gate[D], FP32 x[D] | gate[F], up[F], hidden[F], score[1], raw_shared[D], gated_shared[D] |
| `ml_moe_combine_launch` | routed[D], gated_shared[D] | combined[D] |

All primitives also take an explicit `cudaStream_t` and a caller-initialized
device `int* status`. They never clear status. Zero/nonzero, **not preservation
of arbitrary bit patterns on errors**, is the cross-primitive contract. New
kernels OR 1 on nonfinite values and OR 2 on invalid descriptors/routes; the
unchanged shared projection helper can replace a nonzero value with 1 on a
nonfinite value. A prior nonzero status remains nonzero. Host validation/launch
errors return `cudaError_t`; queued device failures are reported through status.
The parent must observe completion/status and discard failed outputs. A partial
launch failure is not transactional. Invalid raw GPU addresses cannot be
validated here: allocation ranges, sizes, alignment, and ownership are the
parent's responsibility.

Buffers are disjoint except combination output may alias either input.
Dimensions are single-token D=256..6144 and F=256..12288, multiples of 256.
Shared gate/up must have the exact contiguous row strides required by the
existing projection helper. Workspace arithmetic does not allocate internally:
expert scratch is `4 * (10*F + 10*D)` bytes, shared scratch excluding outputs is
`4 * (3*F + 1)` bytes. Caller supplies the separate raw/output buffers and status.

Routing starts from already-produced logits. A parent can use the unchanged
F32 `ml_qsa_projection_launch` for a [512,D] router matrix; router projection,
bias addition, and its registration are not wrapped in this API.

## Computational graph and ordering

1. Materialize FP32 softmax probabilities over **all 512** logits using a max
   shift. This is deliberately not a selected-logit softmax shortcut.
2. Select the 10 largest logits, lower ID first on ties. Normalize their full
   probabilities by `max(sum_selected, 2^-14)`. Multiply by `expert_scale`
   **once**, then sort IDs and associated scaled weights into ascending ID order.
   For finite logits with 512 experts and top10, selected mass is at least
   10/512; the required floor is consequently inactive, but explicitly retained.
3. Each selected expert performs `g = G*x`, `u = U*x`,
   `h = (g / (1 + exp(-g))) * u`, then `y = Down*h`. Products and accumulators
   are FP32, with lane-local packed weights and no floating weight materialization.
4. Store `term[slot] = y * scaled_weight[slot]`, then add the ten terms in
   **sequential ascending-ID order**, starting from zero. Multiplication and
   addition are separate round-to-nearest operations, not FMA.
5. Independently evaluate shared `g`, `u`, then
   `h_shared = (g * sigmoid(g)) * u`, `raw_shared = Down*h_shared`, and the
   separate scalar `score = scalar_gate*x`. Produce
   `gated_shared = sigmoid(score) * raw_shared`. Shared output is not multiplied
   by the routed expert scale.
6. `combined = routed + gated_shared`, one FP32 addition.

Routed gate/up/down use the unchanged `iq_grouped_warp_gpu.py` row arithmetic:
32-lane partial sums and shuffle tree with steps 16,8,4,2,1; four rows per CTA.
No atomic output accumulation or FMA-weighted expert summation is used.
The router's bounded selection/control and full-probability sum are serial on
one GPU thread. This is a correctness primitive, not a tuned router kernel.

Supported routed triples (same inventory as the unchanged grouped IQ tests):

| Gate | Up | Down |
|---|---|---|
| IQ3_XXS | IQ3_XXS | IQ4_NL |
| IQ2_S | IQ2_S | Q2_0 |
| IQ2_S | IQ2_S | IQ4_NL |
| IQ3_XXS | IQ3_XXS | Q2_0 |
| IQ3_S | IQ3_S | IQ4_NL |
| IQ3_S | IQ3_S | Q2_0 |
| IQ4_XS | IQ4_XS | IQ4_NL |

Shared gate/up support Q4_K/Q5_K/Q6_K/IQ4_XS through **calls to** the unchanged
native projection helper, not copied K-format equations. Shared down supports
IQ4_NL through the canonical IQ lane decoder and Q8_0 through the existing
`quantized_weight.py::dequantize_reference` rule (FP16 block scale times signed
byte). No new dequantization equation is introduced. GGML type IDs are listed
in `moe.h`; Q2_0 uses raw type 42.

The IQ table MIT notice and pinned llama.cpp attribution are retained in full.
The source also retains the IQ implementation's Strata layout-consultation
notice; no Strata kernel code is copied. Canonical decoder/codebook byte parity
is asserted by the harness before compiling.

## Build and tests

Compile **both** `moe.cu` and `qsa_projection.cu` with `--fmad=false`, no
`--use_fast_math`. C++17 and CUDA are required. Architecture selection belongs to
the parent build. The standalone harness currently explicitly targets sm_120.
It includes `moe.cu` in its temporary test shim to expose decoder/product/
projection probes without adding test-only exports to the production header.
A separate `nvcc -c moe.cu` build was also executed successfully.

```sh
MINI_LLM_BACKEND=cupy .venv/bin/python tests/native_moe_primitives_harness.py --report /tmp/wp5_moe_validation.json

MINI_LLM_BACKEND=cupy /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python tests/native_moe_primitives_harness.py --quick --report /tmp/wp5_moe_memcheck.json

/usr/local/cuda/bin/nvcc -std=c++17 -O2 --fmad=false -arch=sm_120 -c mini_llm/native_cpp/moe.cu -o /tmp/wp5_moe.o
```

Validation executed on NVIDIA GeForce RTX 5080 with nvcc 13.4.92:

| Check | Actual result |
|---|---|
| Full harness, D=768/F=512, all seven triples | Passed, 451 numerical/exact comparisons |
| Quick harness under CUDA memcheck, D=256/F=256, all seven triples | Passed, 315 comparisons; **ERROR SUMMARY: 0 errors** |
| Separate production translation-unit compile | Passed |
| Native IQ decoder vs unchanged CPU dequantizer | Bitwise equal for gate/up/down probes across all seven triples |
| Individual native FP32 products vs decoded NumPy products | Bitwise equal for gate/up probes |
| Routed weighted terms and ascending sequential sum vs unchanged Python warp oracle | **Bitwise equal** across all seven triples with identical route inputs |
| Largest overall absolute error, full harness | 0.00012969970703125 |
| Relative L2 at that largest-absolute-error comparison | 1.960420548884632e-7 (large synthetic shared output) |
| Largest overall relative L2 | 7.457750010158052e-6 (serial full512 router probability sum vs pairwise oracle) |

The unchanged oracles are `expert_topk_forward`, its selected-mass normalization
rule from `RouterInference`, `iq.grouped`/`iq.grouped_terms` in warp mode,
`iq.swiglu`, bounded CPU `PackedGGUFWeight` decoding and `dense_oracle`, and
`Qwen4ExpMoE._compute_shared`/`combine_gpu_branch` with existing packed projections.
The harness does not modify their source. Each of the seven triples uses ten
independent synthetic experts, including sparse IDs 0 and 511, padded physical
strides and a non-default nonblocking CUDA stream. All eight shared GU/down
format combinations are exercised with random, zero, positive-saturated and
negative-saturated scalar gates (quick mode uses zero gate). Shared down also
uses padded strides. Status retention, selected null descriptors, out-of-range,
duplicate and reversed IDs, nonfinite router logits, invalid host dimensions/
pointers/scale, and permitted combination aliasing are tested.

### Numerical tolerance and the observed failure/correction

Bitwise assertions apply to the same warp graph with identical route inputs,
individual packed lane decoding/products, and saturated/zero scalar-gate limits.
Different projection reductions (native shared 128-thread tree vs Python
projection/BLAS) and the dense FP64 oracle use relative L2 < 1e-5 (or absolute
error < 2e-6 near zero), plus max absolute < `2e-6 + 2e-5*max(abs(reference))`.
This follows the retained warp test's relative/scaled bounds with a 2e-6 absolute
allowance for shared projection/scalar reduction order. It is not a claim of
bitwise shared BLAS parity.

Router probability mass is checked within 8e-6 on these fixtures. A stricter
initial 2e-6 mass check failed on boundary ties: native mass 1.000003695487976,
reference mass 0.9999999403953552, max probability error 1.7695128917694092e-8,
relative L2 3.5049494868774912e-6, FP32, no finite-difference step. This is serial
FP32 summation roundoff, not a decoder or routing equation defect; the 511-add
forward error bound gamma_511 is approximately 3.05e-5. The production arithmetic
was not changed to satisfy that unnecessarily strict extra assertion.

A separate initial expert failure was real: hidden relative L2 approximately
5.88e-4. Decoder and product probes matched exactly, but emitted PTX contained
FP32-to-FP16 conversions around warp shuffles. Including the generated decoder's
`cuda_fp16.h` inside an anonymous namespace hid the global float shuffle
overload. The localized correction includes `cuda_fp16.h` at global scope before
the generated source. Final expert warp terms and outputs are bitwise equal;
the probes remain as regression coverage. No decoder equation was rewritten.

## Limits / unvalidated work

- Synthetic bounded weights only; the seven **format signatures**, not real
  checkpoint payloads, are validated. No checkpoint files are opened or mutated.
- No full-model, production D=2560 shape, long-sequence, CPU/GPU mixed dispatch,
  runtime registration, model-load, or end-to-end inference validation.
- No CMake/runtime/binding integration, native exported parent handles, weight
  staging, CPU pool, cache/admission, or cross-device synchronization changes.
- No latency/throughput claims. Status checks in inner loops and serial routing
  prioritize correctness over performance. No GPU graph-capture or concurrent
  multi-stream reentrancy test; callers must provide disjoint workspace/status
  and explicit dependency ordering.
- Only the RTX 5080/sm_120 toolchain above was exercised. Racecheck and synccheck
  were not run; memcheck is not evidence of broader runtime correctness.
- The parent must explicitly adopt reference_fp32 and enforce the compilation
  flags, valid registered memory ranges, descriptor ABI, and result status. These
  kernels do not replace a ggml_q8 implementation or silently match its arithmetic.
