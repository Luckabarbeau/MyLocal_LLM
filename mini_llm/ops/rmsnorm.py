from ..backend import xp, is_low_precision_dtype
from ..init import ones_parameter


class RMSNorm:
    """
    Root Mean Square Layer Normalization.
    
    y = gamma * x / sqrt(mean(x^2) + eps)
    
    For low-precision inputs, performs numerically sensitive operations in float32
    to avoid underflow/overflow issues while maintaining memory efficiency.
    """

    def __init__(self, d_model, eps=1e-6, name="rmsnorm", dtype="float32"):
        self.eps = float(eps)
        self.gamma = ones_parameter((d_model,), f"{name}.gamma", dtype=dtype, decay=False)

    def parameters(self):
        return [self.gamma]

    def forward(self, x):
        """
        Forward pass with mixed precision support.
        
        For low-precision inputs:
        - Perform numerically sensitive reductions in float32
        - Keep element-wise operations in float16 for speed
        - Return output in original dtype
        """
        input_dtype = x.dtype
        
        if is_low_precision_dtype(input_dtype):
            # Convert to float32 only for the numerically sensitive computation
            x_f32 = x.astype("float32", copy=False)
            
            # Compute mean of squares in FP32 (reduction is numerically sensitive)
            mean_sq = xp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
            
            # Compute inverse RMS in FP32
            inv_rms = 1.0 / xp.sqrt(mean_sq + self.eps)
            
            # Normalize: convert gamma to FP32, do element-wise op
            x_hat_f32 = x_f32 * inv_rms
            
            # Scale by gamma - keep this in FP32 to avoid extra conversion
            y_f32 = x_hat_f32 * self.gamma.data.astype("float32", copy=False)
            
            # Cache FP32 quantities for backward
            cache = {
                "x_f32": x_f32,
                "x_hat_f32": x_hat_f32,
                "inv_rms": inv_rms,
                "original_dtype": input_dtype,
            }
            
            # Convert output back to the original compute dtype
            return y_f32.astype(input_dtype), cache
        else:
            # Float32 path - standard computation
            mean_sq = xp.mean(x * x, axis=-1, keepdims=True)
            inv_rms = 1.0 / xp.sqrt(mean_sq + self.eps)
            x_hat = x * inv_rms
            y = x_hat * self.gamma.data
            
            cache = {
                "x": x,
                "x_hat": x_hat,
                "inv_rms": inv_rms,
                "original_dtype": input_dtype,
            }
            
            return y, cache

    def backward(self, dy, cache):
        """
        Backward pass with mixed precision support.
        
        For low-precision inputs:
        - Perform reductions in float32
        - Return gradient in original dtype
        """
        # The backward path must be selected from the dtype used in forward,
        # not from dy.dtype.  In mixed precision a FP32 RMSNorm input can
        # legitimately receive a BF16 gradient from the following projection.
        # Inferring from dy would then select the low-precision cache layout
        # even though forward cached {x, x_hat, inv_rms}.
        input_dtype = cache["original_dtype"]
        
        if is_low_precision_dtype(input_dtype):
            # Get cached FP32 values
            x_f32 = cache["x_f32"]
            x_hat_f32 = cache["x_hat_f32"]
            inv_rms = cache["inv_rms"]  # Already FP32
            
            # Convert dy to float32 for computation
            dy_f32 = dy.astype("float32", copy=False)
            
            d = x_f32.shape[-1]
            
            # Compute gamma gradient in FP32
            reduce_axes = tuple(range(dy.ndim - 1))
            self.gamma.grad += xp.sum(dy_f32 * x_hat_f32, axis=reduce_axes)
            
            # Compute dx_hat in FP32
            dx_hat_f32 = dy_f32 * self.gamma.data.astype("float32", copy=False)
            
            # Compute projection in FP32
            projection = xp.sum(dx_hat_f32 * x_f32, axis=-1, keepdims=True)
            
            # Compute final gradient in FP32
            dx_f32 = dx_hat_f32 * inv_rms - x_f32 * (inv_rms ** 3) * projection / float(d)
            
            return dx_f32.astype(input_dtype)
        else:
            # Float32 path - standard computation
            x = cache["x"]
            x_hat = cache["x_hat"]
            inv_rms = cache["inv_rms"]
            d = x.shape[-1]

            reduce_axes = tuple(range(dy.ndim - 1))
            self.gamma.grad += xp.sum(dy * x_hat, axis=reduce_axes)

            dx_hat = dy * self.gamma.data
            projection = xp.sum(dx_hat * x, axis=-1, keepdims=True)
            return dx_hat * inv_rms - x * (inv_rms ** 3) * projection / float(d)
