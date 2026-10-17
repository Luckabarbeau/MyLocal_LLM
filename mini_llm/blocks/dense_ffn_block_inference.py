"""Router-free dense feed-forward for cached autoregressive inference."""
from mini_llm.ops.experts_inference import ExpertFFNInference


class DenseFFNInference:
    def __init__(self, d_model, d_ff, dtype="float32"):
        self.expert = ExpertFFNInference(d_model=d_model, d_ff=d_ff, dtype=dtype)

    def set_weights(self, dense_ffn):
        self.expert.set_weights(dense_ffn.expert)

    def _forward(self, x):
        # Match training's rank-2 BF16 GEMM path.  CuPy's rank-3 @ dispatch
        # can fail for bfloat16 ("data type 'E' not understood").
        original_shape = x.shape
        y_2d = self.expert.forward(x.reshape(-1, original_shape[-1]))
        return y_2d.reshape(original_shape)

    def prefill(self, x):
        return self._forward(x)

    def decode_one(self, x):
        return self._forward(x)
