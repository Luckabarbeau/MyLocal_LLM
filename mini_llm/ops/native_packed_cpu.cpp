// 0106E: small dependency-free GGUF CPU decode/GEMV bridge for the shared backend.
// These are independently implemented packed block readers, not vendored GGML.
// No expanded FP16/FP32 weight matrix is ever allocated. One FP32 intermediate
// vector is allocated per expert forward. Numerical oracle: quantized_weight.py.
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include "gguf_codebooks.h"  // Pinned GGML tables and MIT notice from Patch0202.

// Identical value-only heap selection to the QSA reference. In particular,
// equal scores must not acquire an index tie-breaker.
extern "C" int mylocal_qsa_select(const float* scores, int count, int width, int64_t* ids) {
    if (!scores || !ids || width < 1 || width > count) return 1;
    for (int i = 0; i < width; ++i) ids[i] = i;
    auto adjust = [&](int hole, int64_t value) {
        const int top = hole;
        int child = hole;
        while (child < (width - 1) / 2) {
            child = 2 * (child + 1);
            if (scores[ids[child]] > scores[ids[child - 1]]) --child;
            ids[hole] = ids[child];
            hole = child;
        }
        if (width % 2 == 0 && child == (width - 2) / 2) {
            child = 2 * (child + 1);
            ids[hole] = ids[child - 1];
            hole = child - 1;
        }
        int parent = (hole - 1) / 2;
        while (hole > top && scores[ids[parent]] > scores[value]) {
            ids[hole] = ids[parent];
            hole = parent;
            parent = (hole - 1) / 2;
        }
        ids[hole] = value;
    };
    if (width > 1) {
        for (int parent = (width - 2) / 2; parent >= 0; --parent) adjust(parent, ids[parent]);
    }
    for (int i = width; i < count; ++i) {
        if (scores[i] > scores[ids[0]]) adjust(0, i);
    }
    return 0;
}

namespace {
constexpr int Q8_0 = 0, Q4_0 = 1, Q4_K = 2, Q3_K = 3, Q5_K = 4, Q6_K = 5;
constexpr int IQ2_S = 6, IQ3_S = 7, IQ3_XXS = 8, IQ4_NL = 9, IQ4_XS = 10, Q2_0 = 11;

inline int block_elements(int fmt) noexcept {
    return (fmt == Q8_0 || fmt == Q4_0 || fmt == IQ4_NL) ? 32 : fmt == Q2_0 ? 64 : 256;
}
inline int block_bytes(int fmt) noexcept {
    constexpr int bytes[] = {34, 18, 144, 110, 176, 210, 82, 110, 98, 18, 136, 18};
    return bytes[fmt];
}

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
    const int qk = block_elements(fmt);
    const int qb = block_bytes(fmt);
    float sum = 0.0f;
    for (int b = 0; b < k / qk; ++b, w += qb, x += qk) {
        // Decode only the current dot-product lane, never a weight matrix/block.
        // Layouts/order follow pinned GGML dequantize_row_iq* (see 0203 docs).
        if (fmt == IQ4_NL || fmt == IQ4_XS) {
            const float d = fp16(w);
            for (int group = 0; group < qk / 32; ++group) {
                float dl = d;
                const uint8_t* qs = w + 2;
                if (fmt == IQ4_XS) {
                    const unsigned high = unsigned(w[2]) | (unsigned(w[3]) << 8);
                    const int sc = ((w[4 + group / 2] >> (4 * (group % 2))) & 15)
                        | (((high >> (2 * group)) & 3) << 4);
                    dl = d * float(sc - 32);
                    qs = w + 8 + 16 * group;
                }
                for (int j = 0; j < 32; ++j) {
                    const unsigned q = (qs[j % 16] >> (4 * (j / 16))) & 15;
                    const float value = dl * float(kvalues_iq4nl[q]);
                    sum += value * x[32 * group + j];
                }
            }
        } else if (fmt == Q2_0) {
            const float d = fp16(w);
            for (int j = 0; j < 64; ++j) {
                const int q = (w[2 + j / 4] >> (2 * (j % 4))) & 3;
                sum += (d * float(q - 1)) * x[j];
            }
        } else if (fmt == IQ2_S || fmt == IQ3_S || fmt == IQ3_XXS) {
            const float d = fp16(w);
            for (int group = 0; group < 8; ++group) {
                uint32_t aux = 0;
                float dl = 0.0f;
                if (fmt == IQ3_XXS) {
                    const uint8_t* a = w + 66 + 4 * group;
                    aux = uint32_t(a[0]) | (uint32_t(a[1]) << 8)
                        | (uint32_t(a[2]) << 16) | (uint32_t(a[3]) << 24);
                    dl = (d * (0.5f + float(aux >> 28))) * 0.5f;
                } else if (fmt == IQ3_S) {
                    const int sc = (w[106 + group / 2] >> (4 * (group % 2))) & 15;
                    dl = d * float(1 + 2 * sc);
                }
                for (int lane = 0; lane < 4; ++lane) {
                    unsigned signs;
                    const uint8_t *grid1, *grid2 = nullptr;
                    if (fmt == IQ2_S) {
                        const unsigned index = w[2 + 4 * group + lane]
                            | (((w[66 + group] >> (2 * lane)) & 3) << 8);
                        grid1 = iq2s_grid[index];
                        signs = w[34 + 4 * group + lane];
                        const int sc = (w[74 + group] >> (4 * (lane / 2))) & 15;
                        dl = (d * (0.5f + float(sc))) * 0.25f;
                    } else {
                        const int pos = 2 + 8 * group + 2 * lane;
                        if (fmt == IQ3_S) {
                            const unsigned high = w[66 + group];
                            grid1 = iq3s_grid[w[pos] | (((high >> (2 * lane)) & 1) << 8)];
                            grid2 = iq3s_grid[w[pos + 1] | (((high >> (2 * lane + 1)) & 1) << 8)];
                            signs = w[74 + 4 * group + lane];
                        } else {
                            grid1 = iq3xxs_grid[w[pos]];
                            grid2 = iq3xxs_grid[w[pos + 1]];
                            signs = ksigns_iq2xs[(aux >> (7 * lane)) & 127];
                        }
                    }
                    for (int j = 0; j < 8; ++j) {
                        const unsigned q = fmt == IQ2_S ? grid1[j] :
                            (j < 4 ? grid1[j] : grid2[j - 4]);
                        const float value = (dl * float(q)) * ((signs & (1u << j)) ? -1.0f : 1.0f);
                        sum += value * x[32 * group + 8 * lane + j];
                    }
                }
            }
        } else if (fmt == Q8_0) {
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
        } else if (fmt == Q6_K) {
            const float d = fp16(w + 208);
            for (int group = 0; group < 16; ++group) {
                int sc = int(w[192 + group]);
                if (sc >= 128) sc -= 256;
                const float scale = d * float(sc);
                for (int j = 0; j < 16; ++j) {
                    const int i = group * 16 + j;
                    const unsigned low = (w[(i / 128) * 64 + i % 64]
                                          >> ((i % 128) / 64 * 4)) & 15;
                    const unsigned high = (w[128 + (i / 128) * 32 + i % 32]
                                           >> ((i % 128) / 32 * 2)) & 3;
                    sum += (scale * float(int(low | (high << 4)) - 32)) * x[i];
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
    return fmt >= 0 && fmt <= Q2_0 && n > 0 && k > 0 &&
        (k % block_elements(fmt) == 0);
}
inline int row_bytes(int fmt, int k) noexcept {
    return k / block_elements(fmt) * block_bytes(fmt);
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
