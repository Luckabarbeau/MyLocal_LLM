from ..backend import xp


class AdamW:
    """
    AdamW optimizer with explicit mixed precision support.
    
    When model parameters are float16, this implementation:
    - Maintains FP32 first and second moment estimates (m, v)
    - Maintains FP32 master weight copies for each trainable parameter
    - Converts gradients to FP32 before computing products
    - Applies updates in FP32, then casts back to model dtype
    
    This avoids underflow issues in float16 while keeping the model
    weights in float16 for memory efficiency.
    """
    
    def __init__(
        self, parameters, lr=3e-4, beta1=0.9, beta2=0.95,
        eps=1e-8, weight_decay=0.1, loss_scale: float = 1.0
    ):
        """
        Initialize AdamW optimizer.
        
        Args:
            parameters: List of Parameter objects
            lr: Learning rate
            beta1: First moment decay rate
            beta2: Second moment decay rate  
            eps: Small constant for numerical stability
            weight_decay: Weight decay coefficient
            loss_scale: Static loss scale factor (default 1.0)
        """
        self.parameters = list(parameters)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.loss_scale = float(loss_scale)  # Static loss scale
        self.step_index = 0
        
        # FP32 master weights and moments (even for float16 parameters)
        self.master_weights = []
        self.m = []
        self.v = []
        
        for p in self.parameters:
            # Master weights are always float32
            self.master_weights.append(
                p.data.astype("float32", copy=True)
            )
            
            # Moments are always float32
            self.m.append(xp.zeros_like(p.data, dtype="float32"))
            self.v.append(xp.zeros_like(p.data, dtype="float32"))

    def step(self, lr=None):
        """
        Perform one optimizer step with mixed precision.
        
        Args:
            lr: Learning rate override (defaults to configured lr)
        """
        self.step_index += 1
        eta = self.lr if lr is None else float(lr)
        
        # Bias correction terms
        c1 = 1.0 - self.beta1 ** self.step_index
        c2 = 1.0 - self.beta2 ** self.step_index
        
        for i, p in enumerate(self.parameters):
            # Get gradient and convert to float32 for computation
            g = p.grad.astype("float32", copy=False)
            
            # Scale gradient by loss_scale (for mixed precision)
            g_scaled = g / self.loss_scale
            
            # Update FP32 moments
            self.m[i] *= self.beta1
            self.m[i] += (1.0 - self.beta1) * g_scaled
            
            self.v[i] *= self.beta2
            self.v[i] += (1.0 - self.beta2) * (g_scaled * g_scaled)
            
            # Bias-corrected moments (in FP32)
            m_hat = self.m[i] / c1
            v_hat = self.v[i] / c2
            
            # Compute update in FP32
            # Weight decay is applied to master weights
            if p.decay and self.weight_decay != 0.0:
                self.master_weights[i] *= (1.0 - eta * self.weight_decay)
            
            # Adam update on master weights
            update = eta * m_hat / (xp.sqrt(v_hat) + self.eps)
            self.master_weights[i] -= update
            
            # Copy updated master weights back to model parameter
            p.data[...] = self.master_weights[i].astype(p.data.dtype)

    def zero_grad(self):
        """Zero out all gradients."""
        for p in self.parameters:
            p.zero_grad()
