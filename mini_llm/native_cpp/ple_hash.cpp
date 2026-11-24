#include "ple_hash.h"

bool ml_ple_hash_rows(const MlPleHashConfig& c, int32_t token,
    const int32_t* previous, uint64_t* rows, int32_t* next) {
    if (!previous || !rows || !next || token < 0 || c.eos < 0 ||
        previous[0] < -1 || previous[1] < -1 ||
        c.table_rows == 0 || c.table_rows > (uint64_t(1) << 31)) return false;
    for (int h = 0; h < 16; ++h)
        if (!c.vocabs[h] || c.offsets[h] > c.table_rows ||
            c.vocabs[h] > c.table_rows - c.offsets[h]) return false;
    uint64_t products[3];
    products[0] = uint64_t(token) * c.multipliers[0];
    bool cut = false;
    for (int i = 1; i < 3; ++i) {
        int32_t t = previous[2 - i];
        cut = cut || t < 0 || t == c.eos;
        products[i] = uint64_t(cut ? c.eos : t) * c.multipliers[i];
    }
    // Unsigned multiplication wraps mod 2^64 BEFORE XOR and per-head modulo.
    uint64_t bigram = products[0] ^ products[1];
    uint64_t trigram = bigram ^ products[2];
    for (int h = 0; h < 16; ++h)
        rows[h] = (h < 8 ? bigram : trigram) % c.vocabs[h] + c.offsets[h];
    next[0] = previous[1];
    next[1] = token;
    return true;
}
