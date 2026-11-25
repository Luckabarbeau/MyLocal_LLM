#pragma once
#include "runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
/* EXPERIMENTAL: WP7 remains OPEN; no HTTP/production enablement.
 * Owner handles are unavailable until close restores them. A sampled token is
 * unconsumed until the following successful decoder step. Queue excludes the
 * separately stored terminal event. timeout_ms is bounded to 0..120000.
 * Poll returns kind=0 on timeout/end, 1 token, 2 terminal (delivered once).
 * close cancels, drains running work, discards queued events, joins, restores
 * owners, and returns terminal metadata even if already delivered by poll. */
enum ml_generation_reason {
    ML_GEN_NONE=0, ML_GEN_MAX_TOKENS=1, ML_GEN_EOS=2,
    ML_GEN_CAPACITY=3, ML_GEN_CANCELLED=4, ML_GEN_ERROR=5
};
typedef struct ml_generation_event {
    int32_t kind, reason, status, pending_token;
    uint64_t position, sampled;
    int32_t pending, failed;
        int32_t in_flight, reserved;
    ml_sample_result sample;
    char error[512];
    /* Cumulative native step+sample wall time, excluding queue/runtime-lock waits. */
    double decode_seconds;
} ml_generation_event;
int ml_generation_start(ml_handle decoder, ml_handle sampler, int32_t input_token,
    uint64_t max_tokens, const int32_t* eos_ids, uint64_t eos_count,
    double temperature, double top_p, uint64_t top_k, uint64_t queue_capacity,
    ml_handle* out) ML_NOEXCEPT;
int ml_generation_poll(ml_handle generation, uint64_t timeout_ms,
    ml_generation_event* out) ML_NOEXCEPT;
/* Last published step/terminal only; not a live view of an in-flight step. */
int ml_generation_state(ml_handle generation, ml_generation_event* out,
    uint64_t* queued, int32_t* done) ML_NOEXCEPT;
int ml_generation_cancel(ml_handle generation) ML_NOEXCEPT;
int ml_generation_close(ml_handle generation, ml_generation_event* out) ML_NOEXCEPT;
#ifdef __cplusplus
}
#endif
