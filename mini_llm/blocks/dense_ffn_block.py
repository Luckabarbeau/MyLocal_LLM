"""Router-free dense SwiGLU using the same optimized ExpertFFN math as MoE."""
from mini_llm.ops.experts import ExpertFFN


class DenseFFN:
    """One dense expert applied to all tokens, without routing/dispatch kernels."""

    def __init__(self, d_model, d_ff, input_std, output_std, rng,
                 name="dense_ffn", dtype="float32"):
        self.expert = ExpertFFN(
            d_model=d_model, d_ff=d_ff, input_std=input_std,
            output_std=output_std, rng=rng, name=f"{name}.expert", dtype=dtype,
        )

    def parameters(self):
        return self.expert.parameters()

    def zero_grad(self):
        self.expert.zero_grad()

    def forward(self, x, return_cache=True):
        # CuPy's batched (rank-3) matmul does not support the ml_dtypes BF16
        # dispatch path used by this backend (dtype 'E').  The sparse MoE
        # already gathers its selected tokens into rank-2 matrices, so give
        # the dense FFN the same efficient GEMM shape: [B*T, D] @ [D, F].
        # Reshape is a view for contiguous inputs, not a BF16->FP32 cast.
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])
        result = self.expert.forward(
            x_2d, return_cache=return_cache, cache_input=True
        )
        if not return_cache:
            return result.reshape(original_shape)
        y_2d, cache = result
        return y_2d.reshape(original_shape), cache

    def backward(self, dy, cache, *, output_scale=None,
                 return_output_scale_grad=False,
                 accumulate_parameter_grads=True):
        if output_scale is not None or return_output_scale_grad:
            raise NotImplementedError("DenseFFN does not implement progressive gates")
        # The forward tape contains the flattened input.  Keep backward's
        # expert GEMMs rank-2 as well, and restore the Transformer shape.
        original_shape = dy.shape
        dy_2d = dy.reshape(-1, original_shape[-1])
        dx_2d = self.expert.backward(
            dy_2d, cache,
            accumulate_parameter_grads=accumulate_parameter_grads,
        )
        return dx_2d.reshape(original_shape)
