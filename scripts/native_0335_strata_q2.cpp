// Diagnostic only: unchanged Strata 712fec6 Q2_0 AVX2 kernels behind a bounded C ABI.
// Compile this TU at baseline x86-64, NOT with the kernel's AVX2 flags.
#include "strata/kernels/cpu/expert.hpp"
#include <cpuid.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <new>

namespace sc = strata::kernels::cpu;
namespace {
struct Caps { bool avx2, vnni; };
Caps detect() noexcept {
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return {};
    constexpr unsigned need = (1u<<27) | (1u<<28) | (1u<<12) | (1u<<29);
    if ((c & need) != need) return {};
    unsigned lo, hi;
    // XGETBV is legal only after the OSXSAVE check above. Require OS XMM/YMM state.
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    if ((lo & 6) != 6 || !__get_cpuid_count(7, 0, &a, &b, &c, &d) || !(b & (1u<<5))) return {};
    const unsigned max_subleaf = a;
    bool vnni = max_subleaf >= 1 && __get_cpuid_count(7, 1, &a, &b, &c, &d) && (a & (1u<<4));
    return {true, vnni};
}
const Caps& caps() noexcept { static const Caps c = detect(); return c; }
struct Handle { sc::ActQ act{}; int n = 0; };
static_assert(alignof(Handle) >= 64);
constexpr int max_rows = 65536;
}

// The only nonlocal function needed by q2_avx2.cpp. Never claim VNNI from AVX2 alone.
namespace strata::kernels::cpu {
bool cpu_avxvnni_ok() {
    return caps().avx2 && caps().vnni && std::getenv("STRATA_NO_AVXVNNI") == nullptr;
}
}

#define API extern "C" __attribute__((visibility("default")))
// Status: 0 success, 1 invalid argument/state, 2 unsupported CPU, 3 unsafe numeric input.
// Opaque handles must be live handles from create; callers own buffer lifetimes and synchronization.
API int strata0335_q2_available() noexcept { return caps().avx2 ? 1 : 0; }
API int strata0335_q2_vnni() noexcept { return sc::cpu_avxvnni_ok() ? 1 : 0; }
API int strata0335_q2_bitplane() noexcept { return caps().avx2 ? sc::q2_bitplane_enabled() : 0; }
API int strata0335_q2_max_n() noexcept { return sc::H; }
API int strata0335_q2_max_rows() noexcept { return max_rows; }
API size_t strata0335_q2_act_alignment() noexcept { return alignof(Handle); }
API void* strata0335_q2_create() noexcept {
    if (!caps().avx2) return nullptr;
    return new (std::nothrow) Handle;
}
API void strata0335_q2_destroy(void* handle) noexcept { delete static_cast<Handle*>(handle); }

API int strata0335_q2_quantize(void* handle, const float* x, int n) noexcept {
    if (!caps().avx2) return 2;
    if (!handle) return 1;
    auto& h = *static_cast<Handle*>(handle);
    h.n = 0; // A rejected re-quantization must not leave an old activation usable.
    if (!x || n <= 0 || n > sc::H || n % 32) return 1;
    for (int c = 0; c < n; c += 32) {
        float maximum = 0.f;
        for (int j = 0; j < 32; ++j) {
            if (!std::isfinite(x[c+j])) return 3;
            maximum = std::fmax(maximum, std::fabs(x[c+j]));
        }
        const float scale = maximum / 127.f;
        if (scale > 0.f && !std::isfinite(1.f / scale)) return 3;
    }
    sc::act_quant_q8_1_avx2(x, n, h.act);
    h.n = n;
    return 0;
}

// Import little-endian GGML Q8_0 blocks without re-quantization. No bit-plane image
// is constructed: reject that mode rather than silently changing the selected rows.
API int strata0335_q2_import_q8_0(void* handle, const uint8_t* blocks, size_t bytes, int n) noexcept {
    if (!caps().avx2) return 2;
    if (!handle) return 1;
    auto& h = *static_cast<Handle*>(handle);
    h.n = 0;
    h.act.nchunks = 0;
    h.act.bp_pairs = 0;
    if (!blocks || n <= 0 || n > sc::H || n % 32 ||
        bytes != size_t(n/32)*34 || sc::q2_bitplane_enabled()) return 1;
    for (int c = 0; c < n/32; ++c) {
        const uint8_t* block = blocks + size_t(c)*34;
        const uint32_t half = uint32_t(block[0]) | (uint32_t(block[1]) << 8);
        const uint32_t exponent = (half >> 10) & 31u;
        uint32_t mantissa = half & 1023u;
        if (exponent == 31u || ((half & 0x8000u) && (half & 0x7fffu))) return 3;
        // IEEE binary16 -> binary32 by bits, including exact subnormals and signed
        // zero. Baseline wrapper needs no F16C and cannot flush these via FP math.
        uint32_t bits = (half & 0x8000u) << 16;
        if (exponent) {
            bits |= ((exponent + 112u) << 23) | (mantissa << 13);
        } else if (mantissa) {
            uint32_t e = 113u;
            while (!(mantissa & 1024u)) { mantissa <<= 1; --e; }
            bits |= (e << 23) | ((mantissa & 1023u) << 13);
        }
        float scale;
        std::memcpy(&scale, &bits, sizeof scale);
        int32_t sum = 0;
        for (int j = 0; j < 32; ++j) {
            const int v = block[2+j];
            if (v == 128) return 3; // The supported GGML quantizer contract is [-127,127].
            sum += v < 128 ? v : v-256;
        }
        std::memcpy(h.act.q + c*32, block+2, 32);
        h.act.scale[c] = scale;
        h.act.sum[c] = sum;
        h.act.hx[c] = scale * float(sum);
    }
    h.act.nchunks = n/32;
    h.n = n;
    return 0;
}

// Copies only the initialized n values and n/32 scales, never exposes C++ layout.
API int strata0335_q2_export(const void* handle, int8_t* q, size_t q_capacity,
                             float* scales, size_t scale_capacity) noexcept {
    if (!caps().avx2) return 2;
    if (!handle || !q || !scales) return 1;
    const auto& h = *static_cast<const Handle*>(handle);
    if (!h.n || q_capacity < size_t(h.n) || scale_capacity < size_t(h.n/32)) return 1;
    std::memcpy(q, h.act.q, size_t(h.n));
    std::memcpy(scales, h.act.scale, size_t(h.n/32)*sizeof(float));
    return 0;
}

// Raw GGUF Q2_0 rows: 18 bytes per 64 values, fp16 scale then 16 code bytes.
// Absolute output indexing: out[r0:r1], other elements untouched; nt is fixed at 1.
API int strata0335_q2_rows(const void* handle, const uint8_t* weights, size_t weight_bytes,
                           size_t stride, int k, int r0, int r1,
                           float* out, size_t out_count) noexcept {
    if (!caps().avx2) return 2;
    if (!handle || !weights || !out || k <= 0 || k > sc::H || k % 64 ||
        r0 < 0 || r1 < r0 || r1 > max_rows || out_count < size_t(r1)) return 1;
    const auto& h = *static_cast<const Handle*>(handle);
    if (h.n != k) return 1;
    const size_t row_bytes = size_t(k/64)*18;
    if (stride < row_bytes) return 1;
    if (r0 == r1) return 0;
    if (size_t(r1-1) > (std::numeric_limits<size_t>::max()-row_bytes)/stride ||
        size_t(r1-1)*stride + row_bytes > weight_bytes) return 1;
    const sc::ActQ* acts[1] = {&h.act};
    float* outputs[1] = {out};
    sc::q2_0_gguf_rows_multi_avx2(weights, stride, k/64, acts, 1, outputs, r0, r1);
    return 0;
}
