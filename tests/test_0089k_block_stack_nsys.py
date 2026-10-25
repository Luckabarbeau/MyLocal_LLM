from benchmarks.analyze_0089k_block_stack_nsys import classify_operation


def test_classifies_verified_cupy_kernel_families():
    assert classify_operation("kernel", "_dkdv_kernel", None) == "Attention"
    assert classify_operation("kernel", "bf16_swiglu_fwd", None) == "SwiGLU forward"
    assert classify_operation("kernel", "bf16_swiglu_bwd", None) == "SwiGLU backward"
    assert classify_operation("kernel", "rmsnorm_bwd_dx_bf16dy", None) == "RMSNorm"
    assert classify_operation("kernel", "inverse_rope_pack_qkv_bf16", None) == "QKV/RoPE processing"
    assert classify_operation("kernel", "nvjet_sm120_tst_mma", None) == "GEMM"


def test_classifies_verified_torch_kernel_families():
    assert (
        classify_operation(
            "kernel", "triton_tem_fused__to_copy_flex_attention_backward_permute_view_5", None
        )
        == "Attention"
    )
    assert (
        classify_operation("kernel", "triton_poi_fused_mul_silu_backward_1", None)
        == "SwiGLU backward"
    )
    assert (
        classify_operation("kernel", "triton_per_fused__to_copy_add_mean_mul_rsqrt_view_8", None)
        == "RMSNorm"
    )


def test_preserves_ambiguous_symbols_as_unknown():
    assert classify_operation("kernel", "opaque_generated_kernel", None) == "Unknown"


def test_memory_activities_are_not_counted_as_kernels():
    assert classify_operation("memcpy", "memcpy", None) == "Memory/data movement"
    assert classify_operation("memset", "memset", None) == "Memory/data movement"


def test_source_attribution_distinguishes_accumulation_and_residual():
    assert (
        classify_operation(
            "kernel", "cupy_add__float32_float32_float32", "moe.expert.gate_wgrad"
        )
        == "Parameter-gradient accumulation"
    )
    assert (
        classify_operation(
            "kernel", "cupy_add__bfloat16_bfloat16_bfloat16", "moe.expert.dx_add"
        )
        == "Residual"
    )
