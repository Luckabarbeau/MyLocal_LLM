"""HF Llama-style inference through the existing MyLocal_LLM NumPy/CuPy engine.

This deliberately creates no trainable DecoderLanguageModel or optimizer state.
Incoming weights are *canonical HF orientation* and are validated before any
matrix is copied or transposed for the native inference engine.
"""

from types import SimpleNamespace

from mini_llm.backend import xp
from mini_llm.config import ModelConfig
from mini_llm.inference_model import InferenceModel

from .config import ExternalLlamaConfig


class ExternalLlamaModel:
    def __init__(self, config: ExternalLlamaConfig, *, dtype="float32"):
        if not isinstance(config, ExternalLlamaConfig):
            raise TypeError("config must be ExternalLlamaConfig")
        config.validate()
        self.config = config
        self.dtype = dtype
        native_config = ModelConfig(
            tokenizer_vocab_size=config.vocab_size,
            context_length=config.max_position_embeddings,
            n_layers=config.num_hidden_layers,
            d_model=config.hidden_size,
            n_q_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            d_head=config.head_dim,
            n_experts=1,
            top_k=1,
            d_ff=config.intermediate_size,
            ffn_mode="dense",
            tie_embeddings=config.tie_word_embeddings,
            rope_base=config.rope_theta,
            rms_eps=config.rms_norm_eps,
            dtype=dtype,
            attention_layers=None,
        )
        self.engine = InferenceModel(native_config, dtype=dtype)
        # The native default is adjacent-pair RoPE. Llama checkpoints use
        # rotate_half; only opt these external attention instances into it.
        for block in self.engine.blocks:
            block.attention.rope_layout = "half"
        self._loaded = False

    def expected_shapes(self):
        cfg = self.config
        d, f, v = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
        q = cfg.num_attention_heads * cfg.head_dim
        kv = cfg.num_key_value_heads * cfg.head_dim
        result = {
            "model.embed_tokens.weight": (v, d),
            "model.norm.weight": (d,),
        }
        for i in range(cfg.num_hidden_layers):
            stem = f"model.layers.{i}."
            result.update({
                stem + "input_layernorm.weight": (d,),
                stem + "post_attention_layernorm.weight": (d,),
                stem + "self_attn.q_proj.weight": (q, d),
                stem + "self_attn.k_proj.weight": (kv, d),
                stem + "self_attn.v_proj.weight": (kv, d),
                stem + "self_attn.o_proj.weight": (d, q),
                stem + "mlp.gate_proj.weight": (f, d),
                stem + "mlp.up_proj.weight": (f, d),
                stem + "mlp.down_proj.weight": (d, f),
            })
        if not cfg.tie_word_embeddings:
            result["lm_head.weight"] = (v, d)
        return result

    def set_weights(self, weights, *, strict=True):
        """Bind canonical HF-named arrays to native inference-only operations.

        Reject incomplete imports before modifying any live inference weights.
        Shape checks do not copy tensors to the GPU. A future Safetensors
        reader can pass a name -> ndarray mapping without involving PyTorch.
        """
        expected = self.expected_shapes()
        missing = sorted(set(expected) - set(weights))
        optional = {"lm_head.weight"} if self.config.tie_word_embeddings else set()
        extra = sorted(set(weights) - set(expected) - optional)
        if missing or (strict and extra):
            raise ValueError(f"Invalid Llama weight names: missing={missing[:5]}, extra={extra[:5]}")
        shape_of = getattr(weights, "shape_of", None)
        for name, shape in expected.items():
            actual_shape = shape_of(name) if shape_of is not None else weights[name].shape
            if tuple(actual_shape) != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {tuple(actual_shape)}")
        if self.config.tie_word_embeddings and "lm_head.weight" in weights:
            head_shape = (self.config.vocab_size, self.config.hidden_size)
            serialized_head_shape = (shape_of("lm_head.weight") if shape_of is not None
                                     else weights["lm_head.weight"].shape)
            if tuple(serialized_head_shape) != head_shape:
                raise ValueError("lm_head.weight has incorrect tied-head shape")
            # The two serialized names may be physically distinct but must
            # contain the same values for a tied checkpoint.
            if not bool(xp.all(
                xp.asarray(weights["lm_head.weight"]) == xp.asarray(weights["model.embed_tokens.weight"])
            )):
                raise ValueError("Serialized lm_head.weight differs from tied embeddings")
        convert = lambda key: xp.asarray(weights[key], dtype=self.dtype)
        embed = convert("model.embed_tokens.weight")
        # InferenceModel._embedding_lookup uses only embedding.W.data; these
        # minimal read-only views avoid allocating gradient buffers.
        self.engine.embedding = SimpleNamespace(W=SimpleNamespace(data=embed))
        if self.config.tie_word_embeddings:
            self.engine.lm_head = embed.T
        else:
            self.engine.lm_head = convert("lm_head.weight").T
        self.engine.final_norm.set_weights(convert("model.norm.weight"))
        for i, block in enumerate(self.engine.blocks):
            prefix = f"model.layers.{i}."
            block.norm1.set_weights(convert(prefix + "input_layernorm.weight"))
            block.norm2.set_weights(convert(prefix + "post_attention_layernorm.weight"))
            block.attention.set_weights(
                convert(prefix + "self_attn.q_proj.weight").T,
                convert(prefix + "self_attn.k_proj.weight").T,
                convert(prefix + "self_attn.v_proj.weight").T,
                convert(prefix + "self_attn.o_proj.weight").T,
            )
            expert = block.moe.expert
            expert.W_gate = convert(prefix + "mlp.gate_proj.weight").T
            expert.W_up = convert(prefix + "mlp.up_proj.weight").T
            expert.W_down = convert(prefix + "mlp.down_proj.weight").T
        self._loaded = True

    def _require_weights(self):
        if not self._loaded:
            raise RuntimeError("External weights have not been loaded")

    def create_generation_state(self, batch_size=1, max_length=None):
        self._require_weights()
        if max_length is None:
            max_length = self.config.max_position_embeddings
        if not 1 <= int(max_length) <= self.config.max_position_embeddings:
            raise ValueError("Requested KV horizon exceeds configured positions")
        return self.engine.create_generation_state(int(batch_size), int(max_length))

    def prefill(self, token_ids, state):
        self._require_weights()
        return self.engine.prefill(token_ids, state)

    def decode_one(self, token_ids, state):
        self._require_weights()
        return self.engine.decode_one(token_ids, state)

    def forward(self, token_ids):
        """Return *all* causal next-token logits `[batch, seq, vocab]`.

        This parity/benchmark path intentionally uses the native full-causal
        prefill implementation; large T can consume O(T^2) memory. No training
        cache or autograd is instantiated.
        """
        self._require_weights()
        tokens = xp.asarray(token_ids, dtype=xp.int32)
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.config.max_position_embeddings:
            raise ValueError("token_ids must be [B,T] within configured context")
        state = self.create_generation_state(tokens.shape[0], tokens.shape[1])
        x = self.engine._embedding_lookup(tokens)
        for i, block in enumerate(self.engine.blocks):
            x = block.prefill(x, state.k_cache[i], state.v_cache[i], 0)
        x = self.engine.final_norm.forward(x)
        flat = x.astype(self.dtype, copy=False).reshape(-1, self.config.hidden_size)
        logits = self.engine._lm_head_forward(flat)
        return logits.reshape(tokens.shape[0], tokens.shape[1], self.config.vocab_size)
