// 0106E: small dependency-free GGUF CPU decode/GEMV bridge for the shared backend.
// These are independently implemented packed block readers, not vendored GGML.
// No expanded FP16/FP32 weight matrix is ever allocated. One FP32 intermediate
// vector is allocated per expert forward. Numerical oracle: quantized_weight.py.
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

namespace {
constexpr int Q8_0 = 0, Q4_0 = 1, Q4_K = 2, Q3_K = 3, Q5_K = 4;

inline float fp16(const uint8_t* p) noexcept {
    const uint16_t h = uint16_t(p[0]) | (uint16_t(p[1]) << 8);
    uint32_t bits = uint32_t(h & 0x8000) << 16;
    const uint32_t exponent = (h >> 10) & 31;
    const uint32_t mantissa = h & 1023;
    if (exponent == 0) {
        if (mantissa != 0) {
            // Subnormals are rare for real weight scales; retain correct FP16 semantics.
            float value = std::ldexp(float(mantissa), -24);
            return (h & 0x8000) ? -value : value;
        }
        return (h & 0x8000) ? -0.0f : 0.0f;
    }
    bits |= (exponent == 31 ? 255 : exponent - 15 + 127) << 23;
    bits |= mantissa << 13;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline float packed_dot(const uint8_t* w, int fmt, int k, const float* x) noexcept {
    const int qk = (fmt == Q8_0 || fmt == Q4_0) ? 32 : 256;
    const int qb = fmt == Q8_0 ? 34 : fmt == Q4_0 ? 18 : fmt == Q4_K ? 144 : fmt == Q5_K ? 176 : 110;
    float sum = 0.0f;
    for (int b = 0; b < k / qk; ++b, w += qb, x += qk) {
        if (fmt == Q8_0) {
            const float d = fp16(w);
            float acc = 0.0f;
            for (int i = 0; i < 32; ++i) {
                int q = int(w[2 + i]);
                if (q > 127) q -= 256;
                acc += float(q) * x[i];
            }
            sum += d * acc;
        } else if (fmt == Q4_0) {
            const float d = fp16(w);
            float acc = 0.0f;
            for (int i = 0; i < 16; ++i) {
                const unsigned q = w[2 + i];
                acc += float(int(q & 15) - 8) * x[i];
                acc += float(int(q >> 4) - 8) * x[16 + i];
            }
            sum += d * acc;
        } else if (fmt == Q4_K || fmt == Q5_K) {
            const float d = fp16(w);
            const float dmin = fp16(w + 2);
            for (int group = 0; group < 8; ++group) {
                int sc, mn;
                if (group < 4) {
                    sc = w[4 + group] & 63;
                    mn = w[8 + group] & 63;
                } else {
                    sc = (w[8 + group] & 15) | ((w[group] >> 2) & 48);
                    mn = (w[8 + group] >> 4) | ((w[4 + group] >> 2) & 48);
                }
                const float scale = d * float(sc);
                const float minimum = dmin * float(mn);
                for (int j = 0; j < 32; ++j) {
                    const int i = group * 32 + j;
                    const unsigned q = w[(fmt == Q5_K ? 48 : 16) + (i / 64) * 32 + (i % 32)];
                    unsigned v = (q >> ((i % 64) / 32 * 4)) & 15;
                    if (fmt == Q5_K) {
                        const unsigned mask = w[16 + i % 32];
                        v |= ((mask >> (i / 32)) & 1) << 4;
                    }
                    sum += (scale * float(v) - minimum) * x[i];
                }
            }
        } else { // Q3_K
            const float d = fp16(w + 108);
            for (int group = 0; group < 16; ++group) {
                const unsigned sb = w[96 + group % 8];
                const unsigned hb = w[104 + group % 4];
                const int sc = int(((sb >> (4 * (group / 8))) & 15)
                                  | (((hb >> (2 * (group / 4))) & 3) << 4)) - 32;
                const float scale = d * float(sc);
                for (int j = 0; j < 16; ++j) {
                    const int i = group * 16 + j;
                    const unsigned q = w[32 + (i / 128) * 32 + i % 32];
                    const int v = int((q >> (2 * ((i % 128) / 32))) & 3);
                    const int negative = 1 - ((w[i % 32] >> (i / 32)) & 1);
                    sum += (scale * float(v - 4 * negative)) * x[i];
                }
            }
        }
    }
    return sum;
}
inline bool valid(int fmt, int n, int k) noexcept {
    return fmt >= 0 && fmt <= 4 && n > 0 && k > 0 &&
        (k % ((fmt == Q8_0 || fmt == Q4_0) ? 32 : 256) == 0);
}
inline int block_bytes(int fmt) noexcept {
    return fmt == Q8_0 ? 34 : fmt == Q4_0 ? 18 : fmt == Q4_K ? 144 : fmt == Q5_K ? 176 : 110;
}
inline int row_bytes(int fmt, int k) noexcept {
    return k / ((fmt == Q8_0 || fmt == Q4_0) ? 32 : 256) * block_bytes(fmt);
}
}  // namespace

extern "C" {
// Returns nonzero on invalid argument rather than accessing malformed buffers.
int mylocal_packed_gemv(const uint8_t* w, int fmt, int n, int k,
                        const float* x, int tokens, float* out) noexcept {
    if (!w || !x || !out || !valid(fmt, n, k) || tokens < 1) return 1;
    const int rb = row_bytes(fmt, k);
    for (int t = 0; t < tokens; ++t) {
        for (int j = 0; j < n; ++j)
            out[t * n + j] = packed_dot(w + size_t(j) * rb, fmt, k, x + size_t(t) * k);
    }
    return 0;
}

int mylocal_packed_expert(const uint8_t* gate, int gf,
                          const uint8_t* up, int uf,
                          const uint8_t* down, int df,
                          int d_model, int d_ff, const float* x,
                          int tokens, float* out) noexcept {
    if (!gate || !up || !down || !x || !out || tokens < 1 ||
        !valid(gf, d_ff, d_model) || !valid(uf, d_ff, d_model) ||
        !valid(df, d_model, d_ff)) return 1;
    const int grb = row_bytes(gf, d_model), urb = row_bytes(uf, d_model);
    const int drb = row_bytes(df, d_ff);
    std::vector<float> hidden(static_cast<size_t>(d_ff), 0.0f);
    for (int t = 0; t < tokens; ++t) {
        const float* act = x + size_t(t) * d_model;
        for (int j = 0; j < d_ff; ++j) {
            const float g = packed_dot(gate + size_t(j) * grb, gf, d_model, act);
            const float u = packed_dot(up + size_t(j) * urb, uf, d_model, act);
            // Stable sigmoid avoids overflow on unusually large activations.
            const float sigmoid = g >= 0.0f ? 1.0f / (1.0f + std::exp(-g)) :
                std::exp(g) / (1.0f + std::exp(g));
            hidden[j] = (g * sigmoid) * u;
        }
        for (int j = 0; j < d_model; ++j)
            out[size_t(t) * d_model + j] = packed_dot(
                down + size_t(j) * drb, df, d_ff, hidden.data());
    }
    return 0;
}
}
