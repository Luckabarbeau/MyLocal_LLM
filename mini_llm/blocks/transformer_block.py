"""Dense Transformer block with residual streams."""

from mini_llm.backend import xp, resolve_dtype, is_low_precision_dtype
from mini_llm.ops.attention import GQAAttention
from mini_llm.ops.rmsnorm import RMSNorm
from mini_llm.ops.router import Router
from mini_llm.ops.experts import Experts
from mini_llm.blocks.moe_block import MoE


class TransformerBlock:
    """
    Sparse Transformer block for decoder-only language models using Mixture of Experts.
    
    Structure:
        X → RMSNorm → GQAAttention → +X → RMSNorm → MoE → +Y
        
    Where Y is the final output and MoE is a Mixture of Experts with
    a router that selects top-k experts per position.
    
    All operations preserve shape B×T×d_model.
    """
    
    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_head: int,
        d_ff: int,
        n_experts: int,
        top_k: int,
        input_std: float,
        output_std: float,
        rope_base: float,
        rng,
        eps: float = 1e-6,
        name: str = "block",
        dtype: str = "float32"
    ):
        """Initialize the Transformer block."""
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.top_k = top_k
        self.compute_dtype = resolve_dtype(dtype)
        self.use_fp32_residual = is_low_precision_dtype(self.compute_dtype)
        
        # First RMSNorm (input to attention)
        self.norm1 = RMSNorm(d_model, eps=eps, name=f"{name}.norm1", dtype=dtype)
        
        # Attention sublayer
        self.attention = GQAAttention(
            d_model=d_model,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            input_std=input_std,
            output_std=output_std,
            rope_base=rope_base,
            rng=rng,
            name=f"{name}.attention",
            dtype=dtype
        )
        
        # Second RMSNorm (input to MoE)
        self.norm2 = RMSNorm(d_model, eps=eps, name=f"{name}.norm2", dtype=dtype)
        
        # Feed-forward network (MoE)
        self.moe = MoE(
            d_model=d_model,
            d_ff=d_ff,
            n_experts=n_experts,
            k=top_k,
            input_std=input_std,
            output_std=output_std,
            rng=rng,
            name=f"{name}.moe",
            dtype=dtype
        )
    
    def parameters(self):
        """Return all trainable parameters."""
        params = []
        params.extend(self.norm1.parameters())
        params.extend(self.attention.parameters())
        params.extend(self.norm2.parameters())
        params.extend(self.moe.parameters())
        return params
    
    def zero_grad(self):
        """Zero out all gradients."""
        for p in self.parameters():
            p.zero_grad()
    
    def forward(self, x, finite_trace=None, layer_idx=None):
        """
        Forward pass through the Transformer block.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            
        Returns:
            y: Output tensor of shape (B, T, d_model)
            cache: Dictionary containing intermediate values for backward pass
        """
        # Keep the residual stream in FP32 for low-precision training.  The residual
        # additions are cheap element-wise operations, but they can overflow
        # FP16 even when both branch operands are individually finite.  The
        # expensive attention / MoE kernels still receive low-precision inputs below,
        # so tensor-core GEMM throughput is preserved.
        if self.use_fp32_residual:
            x = x.astype("float32", copy=False)

        # First residual branch: attention
        residual1 = x
        
        # Norm → Attention → residual. RMSNorm follows the residual dtype; cast
        # only its normalized output back to the compute dtype before GEMMs.
        norm1_out, norm1_cache = self.norm1.forward(x)
        if finite_trace is not None:
            finite_trace.append((f"block{layer_idx}.norm1", xp.all(xp.isfinite(norm1_out))))

        norm1_compute = (
            norm1_out.astype(self.compute_dtype, copy=False)
            if self.use_fp32_residual else norm1_out
        )
        attn_out, attn_cache = self.attention.forward(norm1_compute)
        if finite_trace is not None:
            finite_trace.append((f"block{layer_idx}.attention", xp.all(xp.isfinite(attn_out))))

        attn_residual = (
            attn_out.astype("float32", copy=False)
            if self.use_fp32_residual else attn_out
        )
        x = residual1 + attn_residual
        if finite_trace is not None:
            finite_trace.append((f"block{layer_idx}.post_attention_residual", xp.all(xp.isfinite(x))))
        
        # Second residual branch: MoE
        residual2 = x
        
        # Norm → MoE → residual. As above, only the branch compute is FP16.
        norm2_out, norm2_cache = self.norm2.forward(x)
        if finite_trace is not None:
            finite_trace.append((f"block{layer_idx}.norm2", xp.all(xp.isfinite(norm2_out))))

        norm2_compute = (
            norm2_out.astype(self.compute_dtype, copy=False)
            if self.use_fp32_residual else norm2_out
        )
        moe_out, moe_cache = self.moe.forward(norm2_compute)
        if finite_trace is not None:
            finite_trace.append((f"block{layer_idx}.moe", xp.all(xp.isfinite(moe_out))))

        moe_residual = (
            moe_out.astype("float32", copy=False)
            if self.use_fp32_residual else moe_out
        )
        y = residual2 + moe_residual
        if finite_trace is not None:
            finite_trace.append((f"block{layer_idx}.output", xp.all(xp.isfinite(y))))
        
        cache = {
            "residual1": residual1,
            "norm1_cache": norm1_cache,
            "attn_cache": attn_cache,
            "residual2": residual2,
            "norm2_cache": norm2_cache,
            "moe_cache": moe_cache,
        }
        
        return y, cache
    
    def backward(self, dy, cache):
        """
        Backward pass through the Transformer block.
        
        Args:
            dy: Gradient of loss w.r.t. output y, shape (B, T, d_model)
            cache: Dictionary from forward pass containing intermediates
            
        Returns:
            dx: Gradient of loss w.r.t. input x, shape (B, T, d_model)
        """
        residual2 = cache["residual2"]
        norm2_cache = cache["norm2_cache"]
        moe_cache = cache["moe_cache"]
        
        # Low-precision models keep residual-gradient accumulation in FP32 as well.
        # Cast only branch gradients to the compute dtype before the expensive
        # attention / MoE backward kernels, then promote their outputs before
        # adding them back to the residual gradient.
        if self.use_fp32_residual:
            dy = dy.astype("float32", copy=False)

        # Backward through second residual: y = residual2 + moe_out
        dresidual2 = dy
        dmoe_out = (
            dy.astype(self.compute_dtype, copy=False)
            if self.use_fp32_residual else dy
        )
        
        # Backward through MoE
        dnorm2_out = self.moe.backward(dmoe_out, moe_cache)
        if self.use_fp32_residual:
            dnorm2_out = dnorm2_out.astype("float32", copy=False)
        
        # Backward through second RMSNorm - this gives gradient through FFN path
        dx_norm2_through_ffn = self.norm2.backward(dnorm2_out, norm2_cache)
        if self.use_fp32_residual:
            dx_norm2_through_ffn = dx_norm2_through_ffn.astype("float32", copy=False)
        
        # Total gradient through first residual is sum of direct and FFN paths
        dx_norm2 = dresidual2 + dx_norm2_through_ffn
        
        # Backward through first residual: x = residual1 + attn_out
        dresidual1 = dx_norm2
        dattn_out = (
            dx_norm2.astype(self.compute_dtype, copy=False)
            if self.use_fp32_residual else dx_norm2
        )
        
        # Backward through attention
        dnorm1_out = self.attention.backward(dattn_out, cache["attn_cache"])
        if self.use_fp32_residual:
            dnorm1_out = dnorm1_out.astype("float32", copy=False)
        
        # Backward through first RMSNorm
        dx_norm1 = self.norm1.backward(dnorm1_out, cache["norm1_cache"])
        if self.use_fp32_residual:
            dx_norm1 = dx_norm1.astype("float32", copy=False)
        
        # Combine gradients for residual1 and norm1 in the residual dtype.
        dx = dresidual1 + dx_norm1
        
        return dx
