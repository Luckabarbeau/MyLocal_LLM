#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#define ML_NOEXCEPT noexcept
#else
#define ML_NOEXCEPT
#endif
/* Synthetic transport runtime, NOT model inference. Handles are process-local.
 * All calls are serialized internally. Zero destroy is a no-op; stale handles
 * are rejected. Error text is thread-local, valid until the next ABI call.
 * Budget covers requested device+pinned payload, not CUDA context/allocator overhead.
 */
typedef uint64_t ml_handle;
/* WP8 independent sampler. All full-array copies are explicit test diagnostics.
 * top_k=0 means full vocabulary; temperature=0 ignores top_k. CuPy stable
 * ascending partition / reversed stable nucleus order; greedy ties lower ID.
 * Device result status: 1 nonfinite input, 2 scaled overflow,
 * 3 invalid mass. Only a compact result crosses D2H during sample. */
typedef struct ml_sample_result {
    int32_t token, status, kept, reserved;
    double uniform;
    float total, probability, cdf_low, cdf_high;
} ml_sample_result;
int ml_sampler_create(ml_handle engine, uint64_t vocab, uint64_t seed, ml_handle* out) ML_NOEXCEPT;
int ml_sampler_destroy(ml_handle sampler) ML_NOEXCEPT;
int ml_sampler_reset(ml_handle sampler, uint64_t seed) ML_NOEXCEPT;
int ml_sampler_sample(ml_handle sampler, ml_handle source, int decoder,
    double temperature, double top_p, uint64_t top_k, int supplied, double uniform,
    ml_sample_result* out) ML_NOEXCEPT;
int ml_sampler_readback_for_test(ml_handle sampler, int32_t* ids, float* probabilities,
    float* cdf, uint64_t count) ML_NOEXCEPT;
/* Decoder consumes all mutable handles on successful create; immutable weights
 * remain shareable. Failure leaves bindings untouched. All steps are native.
 * Any failed step poisons the session; committed global position never advances.
 * Diagnostic trace stages: attention read, post mixer, FFN read, post MoE. */
typedef struct ml_decoder_layer {
    ml_handle attention_hc, branch, ffn_hc, moe, ple;
    int32_t kind; /* 0 QSA, 1 GDN */
} ml_decoder_layer;
int ml_decoder_create(ml_handle engine, ml_handle embedding, ml_handle output,
    ml_handle final, ml_handle hc, const ml_decoder_layer* layers, uint64_t count,
    uint64_t capacity, float scale, int trace, ml_handle* out) ML_NOEXCEPT;
int ml_decoder_destroy(ml_handle decoder) ML_NOEXCEPT;
int ml_decoder_reset(ml_handle decoder) ML_NOEXCEPT;
int ml_decoder_state(ml_handle decoder, uint64_t* position, int* failed) ML_NOEXCEPT;
/* Null logits with count=0 explicitly suppresses full-vocabulary D2H. */
int ml_decoder_step(ml_handle decoder, int32_t token, float* logits, uint64_t count,
    uint64_t* committed) ML_NOEXCEPT;
/* Sequential baseline, not batched prefill. No logits D2H; committed is the
 * absolute successful length (also on partial execution failure). Empty is a no-op.
 * Input/capacity validation is atomic; execution errors require reset. */
int ml_decoder_consume(ml_handle decoder, const int32_t* tokens, uint64_t count,
    uint64_t* committed) ML_NOEXCEPT;
/* One ordinary-host-budgeted prompt boundary per decoder, no QSA prefix copy.
 * Capture replaces the previous checkpoint. Reset/close invalidate it. Both
 * operations reject worker-owned decoders. Partial restore errors poison state. */
int ml_decoder_checkpoint_capture(ml_handle decoder, uint64_t host_budget,
    ml_handle* out, uint64_t* host_bytes) ML_NOEXCEPT;
int ml_decoder_checkpoint_restore(ml_handle decoder, ml_handle checkpoint,
    uint64_t* committed) ML_NOEXCEPT;
int ml_decoder_checkpoint_destroy(ml_handle checkpoint) ML_NOEXCEPT;
int ml_decoder_trace(ml_handle decoder, uint64_t layer, int stage, float* out, uint64_t count) ML_NOEXCEPT;
/* Latest completed MoE epoch only, read-only, no extra step allocations.
 * which: 0 IDs(int32[10]), 1 weights[10], 2 logits[512], 3 probs[512],
 * 4 CPU unweighted[10,D], 5 weighted[10,D], 6 routed[D], 7 shared[D],
 * 8 output[D], 9 input[D], 10 gate[SF], 11 up[SF], 12 hidden[SF],
 * 13 shared score[1], 14 shared raw[D]. Others are FP32; count is elements.
 * CPU rows for GPU-resident slots are zero. Decoder requires enabled trace. */
int ml_moe_diagnostic_for_test(ml_handle owner,int which,void* out,uint64_t count) ML_NOEXCEPT;
int ml_decoder_moe_diagnostic_for_test(ml_handle decoder,uint64_t layer,int which,void* out,uint64_t count) ML_NOEXCEPT;
enum ml_status { ML_OK=0, ML_INVALID=1, ML_CAPACITY=2, ML_CUDA=3, ML_INTERNAL=4 };
typedef struct ml_stats {
    uint64_t device_bytes, pinned_bytes, peak_bytes, sessions, budget_bytes;
} ml_stats;
/* GGUF stores and payload leases. No payload pointers escape; shards unmap
 * after the store and its last consumer lease are destroyed.
 * Metadata values use the original little-endian GGUF value encoding.
 * Borrowed names/value pointers remain valid until handle destruction.
 * Callers must serialize borrowed-pointer use against destruction. */
typedef struct ml_gguf_tensor {
    uint64_t shard_index, rank, ggml_shape[4], relative_offset, data_offset;
    uint64_t nbytes, row_bytes, block_elements, block_bytes;
    uint32_t type_id;
} ml_gguf_tensor;
/* Exact descriptor schema supplied by the authoritative Python architecture validator.
 * Shapes here are GGML fastest-first, including resolved Python-approved aliases.
 * This is not a standalone Qwen metadata parser or a payload-consumer lease. */
typedef struct ml_tensor_requirement {
    const char* name;
    uint64_t rank, ggml_shape[4];
    uint32_t type_id;
} ml_tensor_requirement;
int ml_gguf_validate_schema(ml_handle handle, const ml_tensor_requirement* schema, uint64_t count) ML_NOEXCEPT;
int ml_gguf_open(const char* const* paths, uint64_t count, ml_handle* out) ML_NOEXCEPT;
int ml_gguf_destroy(ml_handle handle) ML_NOEXCEPT;
int ml_gguf_counts(ml_handle handle, uint64_t* tensors, uint64_t* metadata) ML_NOEXCEPT;
int ml_gguf_tensor_get(ml_handle handle, uint64_t index, const char** name, ml_gguf_tensor* out) ML_NOEXCEPT;
int ml_gguf_metadata_get(ml_handle handle, uint64_t index, const char** key, uint32_t* type,
                         const void** value, uint64_t* size, uint64_t* shard) ML_NOEXCEPT;
/* A lease retains shared GGUF ownership and immutable tensor identity. Store
 * destruction prevents new leases but does not invalidate existing leases.
 * Future asynchronous uploads must retain this same shared lease until completion.
 * Copy uses tensor-relative byte offsets, not decoded elements. size is explicit;
 * destination is caller-owned writable host memory of at least capacity bytes.
 * Null destination is allowed only for size=0. copied is required and zeroed on
 * failure; create outputs are likewise zeroed. No partial copies on errors.
 * All operations, including copy and destruction, share the runtime lock. */
int ml_gguf_lease_create(ml_handle store, const char* name, ml_handle* out) ML_NOEXCEPT;
int ml_payload_lease_destroy(ml_handle lease) ML_NOEXCEPT;
int ml_payload_lease_descriptor(ml_handle lease, ml_gguf_tensor* out) ML_NOEXCEPT;
int ml_payload_lease_copy(ml_handle lease, uint64_t offset, uint64_t size,
                          void* destination, uint64_t capacity, uint64_t* copied) ML_NOEXCEPT;
/* Exact stored-byte upload on the engine's nonblocking stream. Pageable mmap
 * sources may block the host inside CUDA; asynchronous host return is NOT promised.
 * Native upload ownership retains the lease and engine even after public close.
 * query returns ready=0 for not-ready; wait/query errors retain source ownership.
 * Destruction drains the stream before releasing buffers and source ownership.
 * Test-only readback uses explicit byte range/caller capacity; no device pointer
 * escapes. It waits for upload completion before copying. No implicit decoding. */
int ml_engine_upload(ml_handle engine, ml_handle lease, ml_handle* out) ML_NOEXCEPT;
int ml_uploaded_destroy(ml_handle upload) ML_NOEXCEPT;
int ml_uploaded_descriptor(ml_handle upload, ml_gguf_tensor* out) ML_NOEXCEPT;
int ml_uploaded_query(ml_handle upload, int* ready) ML_NOEXCEPT;
int ml_uploaded_wait(ml_handle upload) ML_NOEXCEPT;
int ml_uploaded_readback_for_test(ml_handle upload, uint64_t offset, uint64_t size,
                                void* destination, uint64_t capacity, uint64_t* copied) ML_NOEXCEPT;
/* WP5 CPU-only ggml_q8 caller, NOT reference_fp32 or model dispatch.
 * Paths explicitly select the existing bridge and compatible GGML libraries.
 * Registration retains immutable leases; offsets select whole 2D matrices in
 * rank-2/rank-3 tensors. One outstanding job per pool, native-owned FP32 input
 * and unweighted [count,D] output. Destroy drains; jobs survive public pool close.
 * Diagnostic copy capacities are float elements. Host memory is not GPU budgeted. */
int ml_cpu_experts_create(ml_handle engine, int d, int f, int threads, int capacity,
    const char* bridge_path, const char* ggml_path, ml_handle* out) ML_NOEXCEPT;
int ml_cpu_experts_destroy(ml_handle pool) ML_NOEXCEPT;
int ml_cpu_experts_register(ml_handle pool, const ml_handle* leases,
    const uint64_t* offsets, int32_t* index) ML_NOEXCEPT;
int ml_cpu_experts_submit(ml_handle pool, const int32_t* indices, uint64_t count,
    const float* input, uint64_t input_count, ml_handle* job) ML_NOEXCEPT;
int ml_cpu_job_wait(ml_handle job) ML_NOEXCEPT;
int ml_cpu_job_readback_for_test(ml_handle job, float* output, uint64_t capacity) ML_NOEXCEPT;
int ml_cpu_job_destroy(ml_handle job) ML_NOEXCEPT;
/* WP5 single-token static placement. weights[5]: uploaded router F32/F16/BF16 [512,D],
 * shared gate/up/down, separate scalar F32/F16/BF16 [D]. banks[3]: leased packed [512,F,D]
 * gate/up and [512,D,F] down. GPU IDs are explicit unique IDs; all others use
 * existing registered CPU pool. Required policy: cpu_ggml_q8_gpu_reference_fp32.
 * Sources/libraries/slots survive public handle close. One job until wait/close.
 * Submit synchronizes bounded route metadata, not CPU completion. Wait combines
 * ordered unweighted CPU rows with resident GPU terms and separate shared branch. */
int ml_moe_create(ml_handle engine, int d, int f, const ml_handle* weights,
    const ml_handle* banks, const int32_t* gpu_ids, uint64_t gpu_count, int threads,
    const char* bridge_path, const char* ggml_path, const char* policy, ml_handle* out) ML_NOEXCEPT;
int ml_moe_submit(ml_handle moe, const float* input, uint64_t count, float scale, ml_handle* out) ML_NOEXCEPT;
int ml_moe_reset(ml_handle moe) ML_NOEXCEPT;
int ml_moe_destroy(ml_handle moe) ML_NOEXCEPT;
/* Detached diagnostic of last completed native input; never a dispatch input. */
int ml_moe_input_readback_for_test(ml_handle moe, float* output, uint64_t count) ML_NOEXCEPT;
int ml_moe_job_wait(ml_handle job) ML_NOEXCEPT;
int ml_moe_job_readback_for_test(ml_handle job, float* output, uint64_t capacity,
    int32_t* ids, float* weights) ML_NOEXCEPT;
int ml_moe_job_destroy(ml_handle job) ML_NOEXCEPT;
/* WP3 bounded index selection, not attention. capacity is positive, divisible
 * by 4, <= INT32_MAX-3. GGML F32 shapes: query [128,4], pooled [128,capacity/4],
 * dead [128]. Inputs must belong to this engine; pooled includes any newly
 * completed pool already committed. Only consumed inputs must be finite.
 * n=1..capacity is explicitly uploaded to a stable device parameter. reset sets
 * n=0 and invalidates results without reallocating. Policy 0=ggml_cpu, 1=strata.
 * run synchronizes, checks scalar device status (rejects nonfinite arithmetic),
 * and leaves scores/ascending int64 IDs on GPU. No implicit score/ID copy.
 * All workspace payload allocations share the engine budget. Native ownership
 * retains engine; run retains its inputs until synchronized completion.
 * benchmark repeats scores+selection only, CUDA-event milliseconds per iteration;
 * excludes n upload, input uploads, diagnostics, and host synchronization.
 * Readback is explicit diagnostics: element capacities, no partial bounds copy. */
/* WP6 independent FP32 per-session/layer paged KV. capacity<=262144, page_size
 * 1..4096, fixed physical_pages<=ceil(capacity/page_size). No full raw/KV cache.
 * Device table + bounded physical pool + fixed selected gather/remap scratch are
 * engine-budgeted. All calls serialize and fence readers before mutation/free.
 * Generation starts at 1, increments on reset; stale generation is rejected.
 * append takes finite F32 [rows,2,256] uploads and commits consecutive history.
 * remap swaps occupied slots (or moves to an empty slot), preserving token data.
 * attention consumes an existing valid device selection, n=history length+1,
 * separate current k/v; no host IDs. Output uses the existing attention snapshot.
 * Readback is diagnostic only: interleaved [length,2,512] (K then V), F32 count.
 * Capacity is address space, not a promise that all pages fit or are populated. */
/* Explicit strata_int8_per64_fp16 policy: 64 signed bytes + FP16 scale per
 * group, 16 groups/token (1056 bytes). Separate bounded ordinary-RAM backing;
 * engine budget charges GPU only. Append <=page_size rows. All selected history
 * pages must fit GPU slots: preflight refuses before eviction/staging otherwise.
 * Native device->host page metadata transfer is allowed; no Python token IDs.
 * Calls fence completion; no async return, hidden full-resident fallback or FP32
 * substitution. Current K/V is quantized too. Packed diagnostics are bytes. */
typedef struct ml_kvi8_stats {
    uint64_t length, generation, device_bytes, host_bytes, host_budget,
        populated_bytes, resident_pages, hits, misses, h2d_bytes, metadata_bytes, pending;
} ml_kvi8_stats;
int ml_kvi8_create(ml_handle engine, uint64_t capacity, uint64_t page_size, uint64_t slots,
    uint64_t host_budget, const char* policy, ml_handle* out) ML_NOEXCEPT;
int ml_kvi8_destroy(ml_handle store) ML_NOEXCEPT;
int ml_kvi8_stats_get(ml_handle store, ml_kvi8_stats* out) ML_NOEXCEPT;
int ml_kvi8_reset(ml_handle store) ML_NOEXCEPT;
int ml_kvi8_append(ml_handle store, uint64_t generation, ml_handle keys, ml_handle values) ML_NOEXCEPT;
int ml_kvi8_remap(ml_handle store, uint64_t generation, uint64_t logical, uint64_t slot) ML_NOEXCEPT;
int ml_kvi8_attention(ml_handle store, uint64_t generation, ml_handle attention,
    ml_handle query, ml_handle gate, ml_handle key, ml_handle value, float scale) ML_NOEXCEPT;
int ml_kvi8_readback_for_test(ml_handle store, uint64_t generation, uint64_t start,
    uint64_t rows, void* output, uint64_t bytes) ML_NOEXCEPT;
int ml_kv_create(ml_handle engine, uint64_t capacity, uint64_t page_size,
    uint64_t physical_pages, ml_handle* out) ML_NOEXCEPT;
int ml_kv_destroy(ml_handle store) ML_NOEXCEPT;
int ml_kv_state(ml_handle store, uint64_t* length, uint64_t* generation) ML_NOEXCEPT;
int ml_kv_reset(ml_handle store) ML_NOEXCEPT;
int ml_kv_append(ml_handle store, uint64_t generation, ml_handle keys, ml_handle values) ML_NOEXCEPT;
int ml_kv_remap(ml_handle store, uint64_t generation, uint64_t logical_page, uint64_t slot) ML_NOEXCEPT;
int ml_kv_attention(ml_handle store, uint64_t generation, ml_handle attention,
    ml_handle query, ml_handle gate, ml_handle key, ml_handle value, float scale) ML_NOEXCEPT;
int ml_kv_readback_for_test(ml_handle store, uint64_t generation, float* output, uint64_t count) ML_NOEXCEPT;
int ml_qsa_create(ml_handle engine, uint64_t capacity, ml_handle* out) ML_NOEXCEPT;
int ml_qsa_destroy(ml_handle workspace) ML_NOEXCEPT;
int ml_qsa_set_n(ml_handle workspace, uint64_t n) ML_NOEXCEPT;
int ml_qsa_reset(ml_handle workspace) ML_NOEXCEPT;
int ml_qsa_run(ml_handle workspace, ml_handle query, ml_handle pooled, ml_handle dead,
               int policy, uint64_t repeats, float* milliseconds) ML_NOEXCEPT;
int ml_qsa_readback_for_test(ml_handle workspace, float* scores, uint64_t score_capacity,
                            int64_t* ids, uint64_t id_capacity, uint64_t* n) ML_NOEXCEPT;
/* WP3 selected GQA attention, no projection or cache mutation. Creation retains
 * selection+engine. Exact GGML F32 shapes: q/gate [256,24], k/v [256,2],
 * history keys/values [256,2,selection.capacity]. Gate is separate, not qg[512,24].
 * All q/gate/current and history[:n-1] sources must be finite; unused capacity
 * is ignored. run synchronizes scalar status before marking outputs valid.
 * index_query/pooled/dead must all be zero for attention-only (requires valid
 * selection); otherwise all must be valid uploads, and selection is rerun with
 * policy 0/1 before attention in every timed iteration. scale must be finite >0.
 * repeats 1..10000; timing excludes uploads, diagnostics and final sync/status.
 * Device payload 49156 bytes, pinned payload 8 bytes, fixed at creation.
 * readback capacities are F32 element counts, at least 6144 each. Output snapshots
 * remain valid independently of later selection updates; failed runs invalidate.
 * Inputs are shared-owned through stream completion, including error paths. */
int ml_qsa_attention_create(ml_handle selection, ml_handle* out) ML_NOEXCEPT;
int ml_qsa_attention_destroy(ml_handle attention) ML_NOEXCEPT;
int ml_qsa_attention_run(ml_handle attention, ml_handle q, ml_handle gate,
    ml_handle k, ml_handle v, ml_handle keys, ml_handle values,
    ml_handle index_query, ml_handle pooled, ml_handle dead, int policy,
    float scale, uint64_t repeats, float* milliseconds) ML_NOEXCEPT;
int ml_qsa_attention_readback_for_test(ml_handle attention, float* contexts,
    uint64_t context_capacity, float* gated, uint64_t gated_capacity) ML_NOEXCEPT;
/* reference_fp32 projection consumes the last valid native gated snapshot.
 * Exact GGML shape [6144, rows], rows 1..2560; F32/BF16/Q4_K/Q6_K only.
 * Separate snapshot ownership; failed runs invalidate projection results.
 * Payload 4*rows+4 device, 4 pinned bytes. No activation/output readback in graph.
 * Timing includes status clear and matvec; scalar status D2H/sync excluded. */
int ml_qsa_projection_create(ml_handle attention, ml_handle weight, ml_handle* out) ML_NOEXCEPT;
int ml_qsa_projection_destroy(ml_handle projection) ML_NOEXCEPT;
int ml_qsa_projection_run(ml_handle projection, uint64_t repeats, float* milliseconds) ML_NOEXCEPT;
int ml_qsa_projection_readback_for_test(ml_handle projection, float* output, uint64_t capacity) ML_NOEXCEPT;
/* Projected-input frontend only: shapes and equations in qsa_frontend.h.
 * Fixed payload: 54288 device + 16 pinned bytes, engine-budgeted. n >= 1;
 * set_n uploads a stable device uint64 scalar; reset sets n=0. No allocation
 * during run/reset/set_n. Every run attempt invalidates the previous snapshot.
 * Inputs are exact F32 GGML shapes, same engine, retained through completion.
 * run synchronizes and checks device status; no output pointer escapes.
 * Diagnostic readback: six F32 element capacities, pool flag and actual device n.
 * Pool storage is untouched when flag=0 and must not be consumed as a new pool.
 * No projections, cache/session commits, selection or attention integration. */
int ml_qsa_frontend_create(ml_handle engine, ml_handle* out) ML_NOEXCEPT;
int ml_qsa_frontend_destroy(ml_handle workspace) ML_NOEXCEPT;
int ml_qsa_frontend_set_n(ml_handle workspace, uint64_t n) ML_NOEXCEPT;
int ml_qsa_frontend_reset(ml_handle workspace) ML_NOEXCEPT;
int ml_qsa_frontend_run(ml_handle workspace, ml_handle qg, ml_handle k,
    ml_handle raw, ml_handle iq, ml_handle first_raw, ml_handle partial,
    ml_handle gamma_q, ml_handle gamma_k, ml_handle gamma_iq, ml_handle gamma_ik,
    float epsilon) ML_NOEXCEPT;
/* Hidden frontend: five packed [2560,rows] projections, rows Q=12288,
 * K/V/IQ=512, IK=128. F32/BF16/Q4_K/Q5_K/Q6_K/IQ4_XS only.
 * Hidden F32[2560], converted gammas and history as in projected API.
 * First hidden run adds persistent 55808 budgeted device bytes, no pinned bytes.
 * Inputs retained through synchronous completion; scratch belongs to frontend.
 * Failed run/set_n/reset/projected run invalidates hidden diagnostic snapshot. */
int ml_qsa_frontend_from_hidden(ml_handle workspace, ml_handle hidden,
    ml_handle qg, ml_handle k, ml_handle v, ml_handle iq, ml_handle raw,
    ml_handle gamma_q, ml_handle gamma_k, ml_handle gamma_iq, ml_handle gamma_ik,
    ml_handle first_raw, ml_handle partial, uint64_t n, float epsilon) ML_NOEXCEPT;
int ml_qsa_frontend_hidden_readback_for_test(ml_handle workspace,
    float* qg, uint64_t qg_capacity, float* k, uint64_t k_capacity,
    float* v, uint64_t v_capacity, float* raw, uint64_t raw_capacity,
    float* iq, uint64_t iq_capacity) ML_NOEXCEPT;
int ml_qsa_frontend_readback_for_test(ml_handle workspace,
    float* q, uint64_t q_capacity, float* k, uint64_t k_capacity,
    float* gate, uint64_t gate_capacity, float* iq, uint64_t iq_capacity,
    float* dead, uint64_t dead_capacity, float* pool, uint64_t pool_capacity,
    int* pool_ready, uint64_t* n) ML_NOEXCEPT;
/* Complete fixed-geometry QSA attention layer. tensors: Q,K,V,IQ,IK,O and
 * converted gamma Q,K,IQ,IK. State/output remain native device-owned.
 * Import accepts explicit reference history, validating exact shapes/F32/finite;
 * semantic consistency with weights/positions is the caller's responsibility.
 * Diagnostic which: output=0, raw=1, keys=2, values=3, pool=4, IDs=5, scores=6. */
int ml_qsa_layer_create(ml_handle engine, uint64_t capacity, const ml_handle* tensors,
    int policy, float epsilon, float scale, ml_handle* out) ML_NOEXCEPT;
/* Opt-in complete paged layer, same step/import/reset APIs and monolithic default.
 * capacity<=262144, divisible by4; explicit reference_fp32/strata_int8_per64_fp16.
 * Only first raw + four raw slots on GPU; pooled/staged index arrays stay GPU.
 * Lazy ordinary-RAM K/V pages and optional full raw diagnostics share host_budget.
 * Chunked gather supports any selected working set with >=1 fixed GPU slot.
 * Status is validated before append; failed CUDA commit poisons until reset.
 * Diagnostics keys/values return decoded storage; raw requires raw_diagnostics=1.
 * Extra diagnostics 7=first raw[128], 8=token%4 raw slots[4,128]. */
int ml_qsa_layer_create_paged(ml_handle engine, uint64_t capacity, const ml_handle* tensors,
    int policy, float epsilon, float scale, uint64_t page_size, uint64_t slots,
    uint64_t host_budget, const char* format, int raw_diagnostics, ml_handle* out) ML_NOEXCEPT;
int ml_qsa_layer_paged_stats(ml_handle layer, ml_kvi8_stats* out) ML_NOEXCEPT;
/* Paged-only consecutive prepared-state chunks, 1..4096 rows, position==length.
 * F32 raw[rows,128], K/V[rows,512]; pool rows=floor(end/4)-floor(start/4),
 * with zero pool handle iff no block completes. Caller supplies fixture provenance.
 * No model forward computation. Mutation failures poison until reset. */
int ml_qsa_layer_import_chunk(ml_handle layer, uint64_t position, ml_handle raw,
    ml_handle keys, ml_handle values, ml_handle pooled) ML_NOEXCEPT;
/* Detached bounded diagnostics: which=1 raw,2 keys,3 values,4 pools. start/rows
 * count tokens except pools count blocks; count is caller F32 element capacity. */
int ml_qsa_layer_paged_range_for_test(ml_handle layer, int which, uint64_t start,
    uint64_t rows, void* destination, uint64_t count) ML_NOEXCEPT;
int ml_qsa_layer_destroy(ml_handle layer) ML_NOEXCEPT;
int ml_qsa_layer_reset(ml_handle layer) ML_NOEXCEPT;
int ml_qsa_layer_position(ml_handle layer, uint64_t* length) ML_NOEXCEPT;
int ml_qsa_layer_step(ml_handle layer, ml_handle hidden, uint64_t position, uint64_t* committed) ML_NOEXCEPT;
int ml_qsa_layer_import(ml_handle layer, uint64_t length, ml_handle raw,
    ml_handle keys, ml_handle values, ml_handle pooled) ML_NOEXCEPT;
int ml_qsa_layer_readback_for_test(ml_handle layer, int which, void* dst, uint64_t count) ML_NOEXCEPT;
/* HC reference_fp32 (arithmetic=0 only). Weight owner retains same-engine exact
 * F32/BF16 uploads. BF16 expands losslessly on GPU into budgeted FP32 storage;
 * gamma is already converted, flat[4D] or grouped[4,D], no extra +1. Shapes:
 * down[R,4D], up[4D,R], optional inject[4,4D]. Epsilon finite positive FP32.
 * State owns stable streams/staging/normalized[4,D], hidden[D], low[rank_capacity],
 * gains[4], status; (13D+rank_capacity+5)*4 device +4 pinned bytes. No run allocation.
 * Initialize empty state from F32[D] (repeat four times) or F32[4,D]. Each read
 * requires injection and returns a process-unique ticket identifying these gains
 * and original streams; write consumes it exactly once with F32 block output[D].
 * No second read or collapse with a pending write. Final collapse requires weights
 * without injection and is terminal until reset. All arithmetic stays on GPU.
 * Mutating call failures poison state: reset then initialize before any resume.
 * Finite status is checked before stream commit; snapshots never expose poison.
 * Native ownership outlives public engine/upload/weight closes; streams are internal.
 * Calls synchronize scalar completion/status, not numerical intermediates.
 * Weight extra payload =4 status bytes +4 pinned +4 bytes per decoded BF16 element;
 * F32 uploads are retained/aliased read-only. Diagnostic which=0 streams,1 hidden,
 * 2 pending gains; count is F32 elements. Failed diagnostics do not poison state. */
int ml_hc_weights_create(ml_handle engine, ml_handle gamma, ml_handle down,
    ml_handle up, ml_handle inject, float epsilon, int arithmetic, ml_handle* out) ML_NOEXCEPT;
int ml_hc_weights_destroy(ml_handle weights) ML_NOEXCEPT;
int ml_hc_create(ml_handle engine, int d, int rank_capacity, ml_handle* out) ML_NOEXCEPT;
int ml_hc_destroy(ml_handle state) ML_NOEXCEPT;
int ml_hc_reset(ml_handle state) ML_NOEXCEPT;
int ml_hc_initialize(ml_handle state, ml_handle input) ML_NOEXCEPT;
int ml_hc_read(ml_handle state, ml_handle weights, uint64_t* snapshot) ML_NOEXCEPT;
int ml_hc_write(ml_handle state, uint64_t snapshot, ml_handle output) ML_NOEXCEPT;
int ml_hc_collapse(ml_handle state, ml_handle weights) ML_NOEXCEPT;
/* One native read -> branch -> write transaction. kind=0 QSA,1 recurrent,2 MoE.
 * No Python activation transport; weights must inject and match state.
 * Branch engine/width and exact position/capacity are checked before read.
 * committed is zero on error. Every failure poisons HC; execution failures also
 * poison branch (its cache may have advanced); reset both before replay.
 * Internal one-use ticket never escapes. PLE may precede this call via HC input;
 * MoE uses static explicit CPU/GPU arithmetic; only CPU misses stage activation
 * to native host storage. Its device output goes directly to HC write. MoE has
 * an HC-only commit position reset by ml_moe_reset; standalone jobs are stateless.
 * ml_hc_mix uses scale=1; scaled API accepts routed MoE scale (other kinds require 1). */
int ml_hc_mix(ml_handle state, ml_handle weights, ml_handle branch, int kind,
              uint64_t position, uint64_t* committed) ML_NOEXCEPT;
int ml_hc_mix_scaled(ml_handle state, ml_handle weights, ml_handle branch, int kind,
    uint64_t position, float scale, uint64_t* committed) ML_NOEXCEPT;
int ml_hc_readback_for_test(ml_handle state, int which, float* dst, uint64_t count) ML_NOEXCEPT;
/* Bounded Qwen4Exp recurrent layer; arithmetic=0 explicit reference_fp32 only.
 * d<=2560, Hk<=16, Hv<=48 divisible by Hk, dim<=128, taps<=16; batch=1.
 * tensors: QKV,Z,beta,alpha,O (F32/BF16/Q4_K/Q5_K/Q6_K/IQ4_XS), then
 * F32 conv[channels,taps], gamma[dim], converted negative A[Hv], dt[Hv].
 * F32 hidden[d]. Tiled value-head mapping h%Hk; memory [Hv,Dkey,Dvalue],
 * history [taps-1,channels] oldest first. No activation quantization.
 * Stable budgeted scratch/state, no step allocation. Native owner retains uploads.
 * Numerical/metadata failures preserve committed state, invalidate output; CUDA
 * failures poison all diagnostics until reset. Status synchronized before commit.
 * capacity bounds committed steps, independent of constant-size state allocation.
 * Diagnostic which=0 output,1 history,2 matrix; counts are F32 elements. */
int ml_recurrent_create(ml_handle engine,int d,int hk,int hv,int dim,int taps,
    uint64_t capacity,const ml_handle* tensors,float epsilon,int arithmetic,ml_handle* out) ML_NOEXCEPT;
int ml_recurrent_destroy(ml_handle layer) ML_NOEXCEPT;
int ml_recurrent_reset(ml_handle layer) ML_NOEXCEPT;
int ml_recurrent_position(ml_handle layer,uint64_t* length) ML_NOEXCEPT;
int ml_recurrent_step(ml_handle layer,ml_handle hidden,uint64_t position,uint64_t* committed) ML_NOEXCEPT;
int ml_recurrent_readback_for_test(ml_handle layer,int which,float* dst,uint64_t count) ML_NOEXCEPT;
/* PLE reference_fp32 only. Native immutable table lease (IQ4_NL[rows,160]),
 * uploads key[10240,2560],value[2560,2560],gamma_key/query/conv[10240]
 * (or [4,2560]),conv[10240,4]. Key/value use existing packed projection types;
 * controls F32/F16/BF16. config arrays are copied, never borrowed. IDs int32,
 * head vocabulary/ranges uint64 validated by exact wrapping hash primitive.
 * State owns bounded pinned 1440-byte row staging +4-byte status; no table copy.
 * Step commits output/history/predecessors together after completion/status.
 * Failures preserve committed state; CUDA failures require reset. EOS is a normal
 * history step, not reset. Capacity bounds successful steps; no step allocations.
 * HC step requires initialized uncollapsed D=2560 with no pending read; success
 * replaces streams and invalidates hidden. Failed PLE never publishes HC output.
 * Diagnostics: 0 streams,1 history,2 embedding,3 gates; output unavailable after
 * failed step/reset. Metadata returns committed predecessors, rows and position.
 */
int ml_ple_weights_create(ml_handle engine,ml_handle table,const ml_handle* tensors,
    const uint64_t* multipliers,const uint64_t* offsets,const uint64_t* vocabs,
    int32_t eos,float epsilon,int arithmetic,ml_handle* out) ML_NOEXCEPT;
int ml_ple_weights_destroy(ml_handle weights) ML_NOEXCEPT;
int ml_ple_create(ml_handle weights,uint64_t capacity,ml_handle* out) ML_NOEXCEPT;
int ml_ple_destroy(ml_handle state) ML_NOEXCEPT;
int ml_ple_reset(ml_handle state) ML_NOEXCEPT;
int ml_ple_step(ml_handle state,ml_handle input,int32_t token,uint64_t position,int hc) ML_NOEXCEPT;
int ml_ple_metadata(ml_handle state,int32_t* predecessors,uint64_t* rows,uint64_t* position) ML_NOEXCEPT;
int ml_ple_readback_for_test(ml_handle state,int which,float* dst,uint64_t count) ML_NOEXCEPT;
const char* ml_last_error(void) ML_NOEXCEPT;
int ml_engine_create(int device, uint64_t budget, uint64_t max_sessions, ml_handle* out) ML_NOEXCEPT;
int ml_engine_destroy(ml_handle engine) ML_NOEXCEPT;
int ml_engine_stats(ml_handle engine, ml_stats* out) ML_NOEXCEPT;
int ml_session_create(ml_handle engine, uint64_t capacity, ml_handle* out) ML_NOEXCEPT;
int ml_session_destroy(ml_handle session) ML_NOEXCEPT;
int ml_session_reset(ml_handle session) ML_NOEXCEPT;
/* Returns the input token and its zero-based position after real CUDA transport. */
int ml_session_step(ml_handle session, int32_t token, int32_t* out_token, uint64_t* out_position) ML_NOEXCEPT;
#ifdef __cplusplus
}
#endif
