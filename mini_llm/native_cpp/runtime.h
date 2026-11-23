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
enum ml_status { ML_OK=0, ML_INVALID=1, ML_CAPACITY=2, ML_CUDA=3, ML_INTERNAL=4 };
typedef struct ml_stats {
    uint64_t device_bytes, pinned_bytes, peak_bytes, sessions, budget_bytes;
} ml_stats;
/* GGUF descriptors only. No payload pointers escape; close unmaps all shards.
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
