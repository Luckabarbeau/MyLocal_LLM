try:
    import numpy as np
except ImportError:
    np = None

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
        eps=1e-8, weight_decay=0.1, numerical_debug=False
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
            numerical_debug: Enable expensive per-parameter finite checks.
                This copies arrays to host and must remain disabled during
                normal performance-sensitive training.
        """
        self.parameters = list(parameters)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.numerical_debug = bool(numerical_debug)
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
            
            # Optional deep numerical debugging.  These checks copy full
            # parameter arrays to the host and synchronize the GPU, so they
            # must never run in the normal training hot path.
            if self.numerical_debug and np is not None:
                g_cpu = g.get() if hasattr(g, "get") else g
                if not np.all(np.isfinite(g_cpu)):
                    raise FloatingPointError(
                        f"Nonfinite gradient in {p.name or f'parameter_{i}'} "
                        f"at step {self.step_index}"
                    )
            
            # Update FP32 moments (gradient is already scaled by trainer)
            self.m[i] *= self.beta1
            self.m[i] += (1.0 - self.beta1) * g
            
            self.v[i] *= self.beta2
            self.v[i] += (1.0 - self.beta2) * (g * g)
            
            # Bias-corrected moments (in FP32)
            m_hat = self.m[i] / c1
            v_hat = self.v[i] / c2
            
            # Optional deep numerical debugging (expensive host sync/copy).
            if self.numerical_debug and np is not None:
                m_cpu = m_hat.get() if hasattr(m_hat, "get") else m_hat
                v_cpu = v_hat.get() if hasattr(v_hat, "get") else v_hat
                if not (np.all(np.isfinite(m_cpu)) and np.all(np.isfinite(v_cpu))):
                    raise FloatingPointError(
                        f"Nonfinite moment in {p.name or f'parameter_{i}'} "
                        f"at step {self.step_index}"
                    )
            
            # Compute update in FP32
            # Weight decay is applied to master weights
            if p.decay and self.weight_decay != 0.0:
                self.master_weights[i] *= (1.0 - eta * self.weight_decay)
            
            # Adam update on master weights
            update = eta * m_hat / (xp.sqrt(v_hat) + self.eps)
            self.master_weights[i] -= update
            
            # Optional deep numerical debugging (expensive host sync/copy).
            if self.numerical_debug and np is not None:
                w_cpu = self.master_weights[i].get() if hasattr(self.master_weights[i], "get") else self.master_weights[i]
                if not np.all(np.isfinite(w_cpu)):
                    raise FloatingPointError(
                        f"Nonfinite master weight in {p.name or f'parameter_{i}'} "
                        f"at step {self.step_index}"
                    )
            
            # Copy updated master weights back to model parameter
            p.data[...] = self.master_weights[i].astype(p.data.dtype)

    def zero_grad(self):
        """Zero out all gradients."""
        for p in self.parameters:
            p.zero_grad()
