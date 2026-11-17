// Diagnostic codec bridge: all quantization arithmetic is linked upstream.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <cstring>
#include <vector>
extern "C" int activation_roundtrip(int weight_type, const float * input, float * output, int n) {
    if (!input || !output || n <= 0 || n > 10240) return 1;
    auto weight = ggml_get_type_traits_cpu((ggml_type)weight_type);
    if (!weight) return 2;
    auto type = weight->vec_dot_type;
    if (type == GGML_TYPE_F32) { std::memcpy(output,input,n*sizeof(float)); return 0; }
    if (n % ggml_blck_size(type)) return 3;
    auto cpu = ggml_get_type_traits_cpu(type);
    auto traits = ggml_get_type_traits(type);
    if (!cpu->from_float || (!traits->to_float && type != GGML_TYPE_Q8_K)) return 4;
    std::vector<unsigned char> bytes(ggml_row_size(type,n));
    cpu->from_float(input,bytes.data(),n);
    if (type == GGML_TYPE_Q8_K) dequantize_row_q8_K((const block_q8_K *)bytes.data(),output,n);
    else traits->to_float(bytes.data(),output,n);
    return 0;
}
