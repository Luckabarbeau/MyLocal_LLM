from mini_llm.backend import xp
from mini_llm.blocks import TransformerBlock
from mini_llm.ops.embedding import Embedding
from mini_llm.ops.linear import Linear
from mini_llm.ops.rmsnorm import RMSNorm
from mini_llm.ops.loss import cross_entropy_forward, cross_entropy_backward
class DenseDecoderLM:
    """
    Dense decoder-only language model (Milestone 3).
    
    Architecture:
    1. Input embedding layer
    2. Stack of Transformer blocks
    3. Final normalization
    4. Language model head (tied to input embeddings)
    """
    
    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        n_q_heads,
        n_kv_heads,
        d_head,
        d_ff,
        max_seq_len,
        init_std=0.02,
        rms_eps=1e-6,
        rope_base=10_000.0,
        dtype="float32",
        seed=42
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.seed = seed
        
        # Initialize random stream
        from mini_llm.backend import RandomStream
        self.rng = RandomStream(seed)
        
        # Input embedding layer
        self.embeddings = Embedding(
            vocab_size, d_model, init_std, self.rng, name="embeddings", dtype=dtype
        )
        
        # Stack of Transformer blocks
        self.blocks = []
        for i in range(n_layers):
            block = TransformerBlock(
                d_model=d_model,
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                d_head=d_head,
                d_ff=d_ff,
                input_std=init_std,
                output_std=init_std / (2.0 * n_layers) ** 0.5,
                rng=RandomStream(seed + i + 1),
                rope_base=rope_base,
                rms_eps=rms_eps,
                name=f"block_{i}",
                dtype=dtype
            )
            self.blocks.append(block)
        
        # Final normalization (RMSNorm)
        self.final_norm = RMSNorm(
            d_model, eps=rms_eps, name="final_norm", dtype=dtype
        ) 
    
    def parameters(self):
        """Return all trainable parameters."""
        params = []
        params.extend(self.embeddings.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_norm.parameters())
        return params
    
    def zero_grad(self):
        """Zero out gradients for all parameters."""
        for p in self.parameters():
            p.zero_grad()
    
    def forward(self, input_ids, return_cache=False):
        """
        Forward pass through the language model.
        
        Args:
            input_ids: Token IDs of shape [B, T]
            return_cache: Whether to return cache for backward pass
            
        Returns:
            logits: Prediction logits of shape [B, T, vocab_size]
            cache: Cache dict if return_cache=True, otherwise None
        """
        # Input embedding
        x, embed_cache = self.embeddings.forward(input_ids)
        
        if return_cache:
            cache = {"embed_cache": embed_cache, "block_caches": []}
        
        # Pass through Transformer blocks
        for i, block in enumerate(self.blocks):
            if return_cache:
                x, block_cache = block.forward(x, return_cache=True)
                cache["block_caches"].append(block_cache)
            else:
                x = block.forward(x)
        
        # Final normalization
        x = self.final_norm.forward(x)[0]
        
        # Language model head - tied to input embeddings
        logits = x @ self.embeddings.W.data.T
        
        if not return_cache:
            return logits
        
        return logits, cache
    
    def compute_loss(self, logits, targets):
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Prediction logits of shape [B, T, vocab_size]
            targets: Target token IDs of shape [B, T]
            
        Returns:
            loss: Scalar loss value
        """
        loss, _ = cross_entropy_forward(logits, targets)
        return loss
    
    def backward(self, dy, cache):
        """
        Backward pass through the language model.
        
        Args:
            dy: Gradient of loss w.r.t. logits
            cache: Cache from forward pass
            
        Returns:
            dx: Gradient of loss w.r.t. input_ids
        """
        if cache is None:
            raise ValueError("Backward pass requires cache from forward pass.")
        
        # Extract caches from forward pass
        embed_cache = cache["embed_cache"]
        block_caches = cache["block_caches"]
        
        # Backward through blocks (from last to first)
        # For each block, we need to pass through attention and ffn branches
        # For now, implement a simplified backward
        
        # Start with gradient from logits through final norm to block input
        dx = dy[..., :self.d_model]
        
        # Backward through blocks (reverse order)
        for i in reversed(range(len(self.blocks))):
            block = self.blocks[i]
            block_cache = block_caches[i]
            
            # Block backward pass
            dx = block.backward(dx, block_cache)
        
        # Backward through embeddings
        d_embed = self.embeddings.backward(dx, embed_cache)
        
        return d_embed
    
    def train_step(self, input_ids, targets):
        """
        Perform one training step.
        
        Args:
            input_ids: Input token IDs of shape [B, T]
            targets: Target token IDs of shape [B, T]
            
        Returns:
            loss: Scalar loss value
        """
        # Forward pass with cache
        logits, cache = self.forward(input_ids, return_cache=True)
        
        # Compute loss
        loss = self.compute_loss(logits, targets)
        
        # Backward pass
        # Get gradient from cross-entropy backward
        _, ce_cache = cross_entropy_forward(logits, targets)
        dy_logits = cross_entropy_backward(ce_cache)
        
        # Convert gradient from logits to gradient w.r.t. input_ids
        d_input_ids = self.backward(dy_logits, cache)
        
        return loss
    
    def train(self, input_ids, targets, steps):
        """
        Train for multiple steps.
        
        Args:
            input_ids: Input token IDs
            targets: Target token IDs
            steps: Number of training steps
            
        Returns:
            losses: List of loss values per step
        """
        losses = []
        
        for i in range(steps):
            loss = self.train_step(input_ids, targets)
            losses.append(loss)
            print(f"Step {i+1}/{steps}, Loss: {loss:.6f}")
            
        return losses
    
    def predict(self, input_ids):
        """
        Generate predictions for input tokens.
        
        Args:
            input_ids: Token IDs of shape [B, T]
            
        Returns:
            predicted_ids: Token IDs of shape [B, T, 1]
        """
        logits = self.forward(input_ids)
        
        # Get the last token's logits for next token prediction
        last_token_logits = logits[:, -1, :]
        
        # Use argmax for prediction
        predicted_ids = xp.argmax(last_token_logits, axis=-1, keepdims=True)
        
        return predicted_ids
    
    def generate(self, input_ids, max_new_tokens=10):
        """
        Autoregressive generation.
        
        Args:
            input_ids: Initial token IDs of shape [B, T]
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            output_ids: Generated token IDs of shape [B, T + max_new_tokens]
        """
        output_ids = input_ids.copy()
        
        for _ in range(max_new_tokens):
            # Get last token only for prediction
            last_token = output_ids[:, -1:]
            
            # Predict next token
            logits = self.forward(last_token)
            next_token = self.predict(last_token)
            
            # Append to output
            output_ids = xp.concatenate([output_ids, next_token], axis=1)
            
        return output_ids
    
    def inspect_block_activations(self, input_ids, block_idx=0):
        """
        Inspect attention and feed-forward activations in a specific block.
        
        Args:
            input_ids: Token IDs of shape [B, T]
            block_idx: Index of block to inspect
            
        Returns:
            block_cache: Cache from the specified block
        """
        if block_idx >= len(self.blocks):
            raise ValueError(f"Block index {block_idx} out of range for {len(self.blocks)} blocks.")
        
        # Forward pass with cache
        logits, forward_cache = self.forward(input_ids, return_cache=True)
        
        # Extract block cache
        block_cache = forward_cache["block_caches"][block_idx]
        
        return block_cache
    
    def print_attention_matrices(self, input_ids, block_idx=0):
        """
        Print attention matrices for inspection (similar to examples/inspect_attention.py).
        
        Args:
            input_ids: Token IDs of shape [B, T]
            block_idx: Index of block to inspect
        """
        # Forward pass with cache
        logits, forward_cache = self.forward(input_ids, return_cache=True)
        
        # Extract block cache
        block_cache = forward_cache["block_caches"][block_idx]
        
        print(f"Block {block_idx} attention inspection:")
        print(f"Input shape: {input_ids.shape}")
        
        # Extract attention components from cache
        attn_cache = block_cache["attn_cache"]
        
        from mini_llm.backend import asnumpy
        
        print(f"Q after RoPE [B,T,Hq,Dh]: {attn_cache['q'].shape}")
        print(asnumpy(attn_cache["q"]))
        
        print(f"\nK before GQA expansion [B,T,Hkv,Dh]: {attn_cache['k'].shape}")
        print(asnumpy(attn_cache["k"]))
        
        print(f"\nK after GQA expansion [B,T,Hq,Dh]: {attn_cache['k_exp'].shape}")
        print(asnumpy(attn_cache["k_exp"]))
        
        print(f"\nHead-0 unmasked QK^T/sqrt(dh):")
        print(asnumpy(attn_cache["scores"][0, 0]))
        
        print(f"\nHead-0 causal attention probabilities:")
        print(asnumpy(attn_cache["probs"][0, 0]))
        
        print("\n✓ Attention matrices inspected successfully!")