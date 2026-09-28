"""Decoder-only language model built from Transformer blocks."""

from mini_llm.backend import xp
from mini_llm.config import ModelConfig
from mini_llm.blocks.transformer_block import TransformerBlock
from mini_llm.ops.embedding import Embedding
from mini_llm.ops.loss import cross_entropy_forward, cross_entropy_backward
from mini_llm.ops.rmsnorm import RMSNorm


class DecoderLanguageModel:
    """
    Decoder-only language model.
    
    Architecture:
        Token IDs → Embedding → Transformer blocks → RMSNorm → Logits → Loss
        
    The embedding matrix is tied to the output logits (shared weights).
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
        self.dtype = dtype if dtype is not None else config.dtype
        
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
                input_std=config.init_std,
                output_std=config.residual_init_std,
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
    
    def parameters(self):
        """Return all trainable parameters."""
        params = []
        params.extend(self.embedding.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_norm.parameters())
        return params
    
    def zero_grad(self):
        """Zero out all gradients."""
        for p in self.parameters():
            p.zero_grad()
    
    def forward(self, token_ids):
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
        
        # Pass through transformer blocks - collect caches
        block_caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block.forward(x)
            block_caches.append(cache)
        
        # Final normalization - unpack output and cache
        x, final_norm_cache = self.final_norm.forward(x)
        
        # Compute logits using tied embedding matrix
        # logits = x @ W_E^T
        # x: (B, T, d_model), W_E: (vocab_size, d_model)
        # So logits: (B, T, vocab_size)
        logits = x @ self.embedding.W.data.T
        
        cache = {
            "token_ids": token_ids,
            "embedding_input": x,  # Input to logits (after final norm)
            "embed_cache": embed_cache,
            "final_norm_cache": final_norm_cache,
            "block_caches": block_caches,  # List of caches from each transformer block
        }
        
        return logits, cache
    
    def compute_loss(self, logits, targets):
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Logits from forward pass, shape (B, T, vocab_size)
            targets: Target token IDs, shape (B, T)
            
        Returns:
            loss: Scalar loss value
            cache: Dictionary for backward pass
        """
        return cross_entropy_forward(logits, targets)
    
    def backward_loss(self, loss_cache):
        """
        Backward pass through the loss layer.
        
        Args:
            loss_cache: Cache from cross_entropy_forward
            
        Returns:
            d_logits: Gradient w.r.t. logits
        """
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
        
        # Gradient of logits w.r.t. embedding output
        # logits = x @ W_E^T
        # d_logits: (B, T, vocab_size)
        # x: (B, T, d_model)
        # W_E: (vocab_size, d_model)
        
        BT = x.shape[0] * x.shape[1]
        d_logits_flat = d_logits.reshape(BT, -1)  # (B*T, vocab_size)
        x_flat = x.reshape(BT, -1)  # (B*T, d_model)
        
        # Gradient for embedding weight matrix
        # dW_E = d_logits^T @ x
        self.embedding.W.grad = d_logits_flat.T @ x_flat
        
        # Gradient w.r.t. x (embedding output)
        # dx = d_logits @ W_E
        dx = d_logits_flat @ self.embedding.W.data
        
        # Reshape dx back to (B, T, d_model)
        dx = dx.reshape(x.shape)
        
        # Backward through final RMSNorm
        dx = self.final_norm.backward(dx, cache["final_norm_cache"])
        
        # Backward through transformer blocks (in reverse order)
        for block, block_cache in zip(reversed(self.blocks), reversed(block_caches)):
            dx = block.backward(dx, block_cache)
        
        # Backward through embedding (if needed - currently not used since we compute gradient directly)
        # This would require modifying Embedding.backward to handle the case where we pass dx
        # For now, the gradient is computed directly in this method
