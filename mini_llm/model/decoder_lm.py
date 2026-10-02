"""Decoder-only language model built from Transformer blocks."""

from mini_llm.backend import xp, resolve_dtype, is_low_precision_dtype, is_bfloat16_dtype
from mini_llm.config import ModelConfig
from mini_llm.blocks.transformer_block import TransformerBlock
from mini_llm.ops.embedding import Embedding
from mini_llm.ops.linear import Linear
from mini_llm.ops.loss import cross_entropy_forward, cross_entropy_backward
from mini_llm.ops.rmsnorm import RMSNorm


class DecoderLanguageModel:
    """
    Decoder-only language model.
    
    Architecture:
        Token IDs → Embedding → Transformer blocks → RMSNorm → Output Proj → Logits → Loss
        
    Uses separate output projection layer (untied embeddings by default).
    The embedding matrix can be optionally tied to the output projection via weight sharing.
    """
    
    def __init__(self, config: ModelConfig, rng_seed: int = 0, dtype: str = None):
        """
        Initialize the decoder language model.
        
        Args:
            config: ModelConfig with all hyperparameters
            rng_seed: Seed for random number generation
            dtype: Data type (overrides config if provided)
        """
        self.config = config
        self.dtype = resolve_dtype(dtype if dtype is not None else config.dtype)
        
        # Random stream for initialization
        from mini_llm.backend import RandomStream
        rng = RandomStream(rng_seed)
        
        # Embedding layer
        self.embedding = Embedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            std=config.init_std,
            rng=rng,
            name="embedding",
            dtype=self.dtype
        )
        
        # Create transformer blocks
        self.blocks = []
        for layer_idx in range(config.n_layers):
            block = TransformerBlock(
                d_model=config.d_model,
                n_q_heads=config.n_q_heads,
                n_kv_heads=config.n_kv_heads,
                d_head=config.d_head,
                d_ff=config.d_ff,
                n_experts=config.n_experts,
                top_k=config.top_k,
                input_std=config.init_std,
                output_std=config.residual_init_std,
                rope_base=config.rope_base,
                rng=rng,
                eps=config.rms_eps,
                name=f"blocks.{layer_idx}",
                dtype=self.dtype
            )
            self.blocks.append(block)
        
        # Final normalization
        self.final_norm = RMSNorm(
            config.d_model,
            eps=config.rms_eps,
            name="final_norm",
            dtype=self.dtype
        )
        
        # Output projection layer (maps d_model -> vocab_size)
        self.output_proj = Linear(
            config.d_model,
            config.vocab_size,
            config.init_std,
            rng,
            name="output_proj",
            dtype=self.dtype
        )
    
    def parameters(self):
        """Return all trainable parameters."""
        params = []
        params.extend(self.embedding.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_norm.parameters())
        params.extend(self.output_proj.parameters())
        return params
    
    def zero_grad(self):
        """Zero out all gradients."""
        for p in self.parameters():
            p.zero_grad()
    
    def forward(self, token_ids, finite_trace=None):
        """
        Forward pass through the model.
        
        Args:
            token_ids: Input token IDs, shape (B, T)
            
        Returns:
            logits: Output logits before softmax, shape (B, T, vocab_size)
            cache: Dictionary for backward pass
        """
        B, T = token_ids.shape
        
        # Embed tokens - unpack output and cache
        x, embed_cache = self.embedding.forward(token_ids)
        if finite_trace is not None:
            finite_trace.append(("embedding", xp.all(xp.isfinite(x))))
        
        # Pass through transformer blocks - collect caches
        block_caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block.forward(x, finite_trace=finite_trace, layer_idx=i)
            block_caches.append(cache)
        
        # Final normalization - unpack output and cache.  FP16 models keep the
        # transformer residual stream in FP32, but the large output projection
        # remains on the fast FP16 GEMM path.
        x, final_norm_cache = self.final_norm.forward(x)
        if finite_trace is not None:
            finite_trace.append(("final_norm", xp.all(xp.isfinite(x))))

        output_proj_input = (
            x.astype(self.dtype, copy=False)
            if is_low_precision_dtype(self.dtype) else x
        )
        
        # Output projection - maps d_model -> vocab_size
        logits, output_proj_cache = self.output_proj.forward(output_proj_input)
        if finite_trace is not None:
            finite_trace.append(("logits", xp.all(xp.isfinite(logits))))
        
        cache = {
            "token_ids": token_ids,
            "embedding_input": output_proj_input,  # Input to logits projection
            "embed_cache": embed_cache,
            "final_norm_cache": final_norm_cache,
            "output_proj_cache": output_proj_cache,
            "block_caches": block_caches,  # List of caches from each transformer block
        }
        
        return logits, cache
    
    def compute_loss(self, logits, targets, loss_mask=None):
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Logits from forward pass, shape (B, T, vocab_size)
            targets: Target token IDs, shape (B, T)
            loss_mask: Optional mask, shape (B, T). Non-zero entries select
                target positions that contribute to the loss. This is used for
                assistant-only supervised fine-tuning.

        Returns:
            loss: Scalar loss value
            cache: Dictionary for backward pass
        """
        return cross_entropy_forward(logits, targets, loss_mask=loss_mask)
    
    def backward_loss(self, loss_cache):
        """
        Backward pass through the loss layer.
        
        Args:
            loss_cache: Cache (second element) from compute_loss
            
        Returns:
            d_logits: Gradient w.r.t. logits
        """
        # loss_cache is the cache dict returned by compute_loss (second element)
        return cross_entropy_backward(loss_cache)
    
    def backward(self, d_logits, cache):
        """
        Full backward pass through the model.
        
        Args:
            d_logits: Gradient from loss, shape (B, T, vocab_size)
            cache: Cache from forward pass
            
        Returns:
            None (gradients stored in parameter.grad)
        """
        # Get intermediate values from cache
        x = cache["embedding_input"]  # Final normalized state (input to logits)
        token_ids = cache["token_ids"]
        block_caches = cache["block_caches"]
        
        # BF16 has FP32-like exponent range, so the FP32 cross-entropy gradient
        # can safely be cast back to BF16 for the large vocabulary GEMMs. FP16
        # keeps the FP32 loss gradient because its tiny non-target components
        # can underflow before accumulation.
        d_logits_compute = (
            d_logits.astype(self.dtype, copy=False)
            if is_bfloat16_dtype(self.dtype) else d_logits
        )
        dx = self.output_proj.backward(d_logits_compute, cache["output_proj_cache"])
        
        # Backward through final RMSNorm
        dx = self.final_norm.backward(dx, cache["final_norm_cache"])
        
        # Backward through transformer blocks (in reverse order)
        for block, block_cache in zip(reversed(self.blocks), reversed(block_caches)):
            dx = block.backward(dx, block_cache)
        
        # Backward through embedding - this computes the gradient for W_E
        self.embedding.backward(dx, cache["embed_cache"])
        

