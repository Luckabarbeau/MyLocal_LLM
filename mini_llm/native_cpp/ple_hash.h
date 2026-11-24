#pragma once
#include <stdint.h>

// Host-only, single sequence. Matches PLEConfig, no ownership/allocation.
struct MlPleHashConfig {
    uint64_t multipliers[3];
    uint64_t offsets[16];
    uint64_t vocabs[16];
    int32_t eos;
    uint64_t table_rows;
};

// predecessors[2] oldest first; -1 means missing. token in [0,INT32_MAX].
// rows[16] and next_predecessors[2] are caller-owned, mutually disjoint and
// disjoint from all inputs. Invalid arguments return false without writes.
// EOS pads future hash context only; caller MUST NOT reset convolution history.
bool ml_ple_hash_rows(const MlPleHashConfig& config, int32_t token,
    const int32_t* predecessors, uint64_t* rows, int32_t* next_predecessors);
