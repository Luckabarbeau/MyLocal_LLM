// Narrow ABI for Strata's MIT-licensed fused prompt kernels, not its engine.
// Source checkout/revision/compiler provenance is recorded by the build script.
#include "strata/prefill/moe_fused_iq.hpp"
#include "ggml.h"
#include <cstdint>
#include <cstring>

namespace fused = strata::prefill::fused;
extern "C" {
int mylocal_strata_prefill_width() { return fused::kMoeEmbd; }
int mylocal_strata_prefill_ff() { return fused::kMoeFF; }
int mylocal_strata_prefill_type(const char* name) {
    if (!name) return -1;
#define FORMAT(N) if (!std::strcmp(name, #N)) return GGML_TYPE_##N
    FORMAT(IQ2_XXS); FORMAT(IQ2_XS); FORMAT(IQ2_S); FORMAT(IQ3_XXS);
    FORMAT(IQ3_S); FORMAT(IQ4_XS); FORMAT(IQ4_NL); FORMAT(Q2_0);
    FORMAT(Q4_K); FORMAT(Q5_K); FORMAT(Q6_K); FORMAT(Q8_0);
#undef FORMAT
    return -1;
}
int mylocal_strata_prefill_supported(int gate, int down) {
    return fused::native_supported(gate, down) ? 1 : 0;
}
size_t mylocal_strata_prefill_act_bytes(int64_t rows, int64_t width) {
    return fused::act_bytes(rows, width);
}
size_t mylocal_strata_prefill_group_bytes(int64_t rows, int experts) {
    return fused::group_bytes(rows, experts);
}
void mylocal_strata_prefill_quantize(const float* x, int64_t rows, int64_t width,
        void* packed, void* stream) {
    fused::quantize_act_native(x, rows, width, packed, stream);
}
void mylocal_strata_prefill_group(const int32_t* ids, int64_t rows, int k, int experts,
        void* scratch, int32_t* slot, int32_t* src, void* stream) {
    fused::group(ids, rows, k, experts, scratch, slot, src, stream);
}
int mylocal_strata_prefill_experts(int first, int last, const uint64_t* pointers,
        int gate_type, int down_type, size_t gate_row, size_t down_row,
        size_t up_offset, size_t down_offset, int experts, int64_t rows,
        const void* scratch, const void* input, const int32_t* src,
        void* hidden, float* output, void* stream) {
    if (first < 0 || last <= first || last > experts || last-first > fused::kMaxBatch) return 1;
    fused::Batch batch;
    batch.e0 = first; batch.e1 = last;
    for (int i = first; i < last; ++i)
        batch.blob[i-first] = reinterpret_cast<const uint8_t*>(pointers[i-first]);
    fused::NativeGeom geometry;
    geometry.gu_type = gate_type; geometry.d_type = down_type;
    geometry.gu_row = gate_row; geometry.d_row = down_row;
    geometry.up_off = up_offset; geometry.down_off = down_offset;
    fused::experts_native(batch, geometry, experts, rows, scratch, input, src, hidden, output, stream);
    return 0;
}
}
