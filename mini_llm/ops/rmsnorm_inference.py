"""Inference-only RMSNorm without backward caches."""


from mini_llm.backend import xp, BACKEND_NAME


class RMSNormInference:
    """
    RMSNorm for inference - no cache needed for backward pass.
    
    Computes: y = gamma * x / sqrt(mean(x^2) + eps)
    
    For inference, we don't need to store intermediates for backprop.
    This is a simplified version of the training RMSNorm.
    """

    def __init__(self, d_model, eps=1e-6, dtype="float32"):
        """
        Initialize RMSNorm for inference.
        
        Args:
            d_model: Model dimension
            eps: Numerical stability constant
            dtype: Data type
        """
        self.d_model = d_model
        self.eps = float(eps)
        self.dtype = dtype
        
        # Will be set externally
        self.gamma = None

    def set_weights(self, gamma):
        """Set normalization weights (shared with training model)."""
        # Ensure gamma is on the correct backend
        if BACKEND_NAME == "cupy" and not hasattr(gamma, '__cuda_array_interface__'):
            self.gamma = xp.asarray(gamma)
        else:
            self.gamma = gamma

    def forward(self, x):
        """
        Forward pass through RMSNorm.
        
        Args:
            x: Input tensor [B, T, D]
            
        Returns:
            y: Output tensor [B, T, D]
        """
        input_dtype = x.dtype
        
        if input_dtype == "float16":
            # Convert to float32 for numerically sensitive computation
            x_f32 = x.astype("float32", copy=False)
            
            mean_sq = xp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
            inv_rms = 1.0 / xp.sqrt(mean_sq + self.eps)
            
            x_hat_f32 = x_f32 * inv_rms
            # Ensure gamma is on the correct backend
            if BACKEND_NAME == "cupy" and not hasattr(self.gamma, '__cuda_array_interface__'):
                gamma_f32 = xp.asarray(self.gamma).astype("float32", copy=False)
            else:
                gamma_f32 = self.gamma.astype("float32", copy=False)
            y_f32 = x_hat_f32 * gamma_f32
            
            return y_f32.astype(input_dtype)
        else:
            mean_sq = xp.mean(x * x, axis=-1, keepdims=True)
            inv_rms = 1.0 / xp.sqrt(mean_sq + self.eps)
            x_hat = x * inv_rms
            # Ensure gamma is on the correct backend
            if BACKEND_NAME == "cupy" and not hasattr(self.gamma, '__cuda_array_interface__'):
                y = x_hat * xp.asarray(self.gamma)
            else:
                y = x_hat * self.gamma
            
            return y
