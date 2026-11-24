// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>

// Declaration-only ABI shared by the original bridge and native callers.
using Quantize = void (*)(const float *, void *, int64_t);
using Dot = void (*)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
struct Projection {
    const uint8_t *weight;
    size_t stride;
    int qtype; // Activation type: GGML_TYPE_Q8_0=8 or GGML_TYPE_Q8_K=15.
    Quantize quantize;
    Dot dot;
};
struct PoolExpert { Projection gate, up, down; };
extern "C" {
void *mylocal_ggml_pool_create(int d, int f, int capacity, int threads) noexcept;
int mylocal_ggml_pool_set_spin(void *, int microseconds) noexcept;
int mylocal_ggml_pool_configure(void *, int host_help, const int *cpus, int n) noexcept;
int mylocal_ggml_pool_set_task_factor(void *, int factor) noexcept;
int mylocal_ggml_pool_set_task_mode(void *, int enabled) noexcept;
int mylocal_ggml_pool_set_fused_hidden(void *, int enabled) noexcept;
int mylocal_ggml_pool_clear_registered(void *) noexcept;
int mylocal_ggml_pool_register(void *, const PoolExpert *, int *index) noexcept;
int mylocal_ggml_pool_submit(void *, const PoolExpert *, int count, const float *, float *) noexcept;
int mylocal_ggml_pool_submit_registered(void *, const int *, int count, const float *, float *) noexcept;
void mylocal_ggml_pool_wait(void *) noexcept;
void mylocal_ggml_pool_destroy(void *) noexcept;
}
