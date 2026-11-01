"""Native TextGenerator compatibility adapter (0103E).

KV-cache inference delegates sampling and token lifecycle to generation.py.
The full-prefix numerical reference remains explicit for debugging.
"""
import time
import numpy as np
from mini_llm.backend import xp, BACKEND_NAME, synchronize
from mini_llm.ops.attention import GQAAttention

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply softmax with temperature scaling in FP32.

    CuPy's sorting/partitioning stack does not support BF16, and probability
    calculations are numerically safer in FP32 anyway.  Sampling operates on
    only one vocabulary vector per generated token, so this promotion is
    negligible compared with the model forward pass.
    """
    logits = logits.astype(xp.float32, copy=False)
    # Subtract max for numerical stability
    logits = logits / temperature
    logits = logits - xp.max(logits, axis=-1, keepdims=True)
    exp_logits = xp.exp(logits)
    return exp_logits / xp.sum(exp_logits, axis=-1, keepdims=True)


# Preserve historical helper names for CLI and third-party imports; actual
# sampling now has exactly one implementation in mini_llm.generation.
from .generation import sample_token as _sample_token


def greedy_decode(logits) -> int:
    return _sample_token(logits, temperature=0.0)


def temperature_sample(logits, temperature: float = 1.0) -> int:
    return _sample_token(logits, temperature=temperature)


def top_k_sample(logits, k: int, temperature: float = 1.0) -> int:
    if k is None:
        raise ValueError("top_k must be provided for top-k sampling")
    return _sample_token(logits, temperature=temperature, top_k=k)


def top_p_sample(logits, p: float, temperature: float = 1.0) -> int:
    return _sample_token(logits, temperature=temperature, top_p=p)


class TextGenerator:
    """Text generation wrapper for the language model."""

    def __init__(self, model, tokenizer, max_context: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_context = max_context

    def _get_eos_id(self) -> int:
        """Get EOS token ID from tokenizer or use default."""
        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
            return self.tokenizer.encode(self.tokenizer.eos_token)[0]
        # Default: 2 is commonly used as EOS
        return 2

    def generate(
        self, prompt: str, max_new_tokens: int = 50,
        strategy: str = "greedy", temperature: float = 1.0,
        top_k: int | None = None, top_p: float | None = None,
        seed: int | None = None, log_speed: bool = False,
        use_kv_cache: bool = True, return_token_ids: bool = False,
    ) -> tuple[str, dict]:
        """Generate using the common session or an explicit full-prefix reference.

        Production KV-cache decoding and HTTP serving share exactly the same
        sampler, streaming stop logic, and token-state lifecycle in 0103E.
        """
        if strategy not in {"greedy", "temperature", "top-k", "top-p"}:
            raise ValueError(f"Unknown strategy: {strategy}")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if seed is not None:
            np.random.seed(seed)
            xp.random.seed(seed)
        self.last_execution_report = None
        if use_kv_cache:
            from .generation import GenerationSession, NativeGeneratorAdapter
            # Retain the adapter/session for multiple requests; the underlying
            # captured graph can be reused by _ensure_inference_model(reset=True).
            if not hasattr(self, "_generation_session"):
                self._generation_session = GenerationSession(NativeGeneratorAdapter(self))
            sess = self._generation_session
            params = {
                "greedy": (0.0, None, 1.0),
                "temperature": (temperature, None, 1.0),
                "top-k": (temperature, top_k, 1.0),
                "top-p": (temperature, None, top_p),
            }
            temp, k, p = params[strategy]
            if strategy == "top-k" and top_k is None:
                raise ValueError("top_k must be provided for top-k sampling")
            if strategy == "top-p" and top_p is None:
                raise ValueError("top_p must be provided for top-p sampling")
            if log_speed:
                synchronize()
                started = time.perf_counter()
            outcome = sess.generate(
                prompt, max_tokens=max_new_tokens, temperature=temp,
                top_k=k, top_p=p, seed=seed, log_speed=log_speed,
                include_stop_token=True)
            self._state = sess.state
            self.last_execution_report = sess.model.execution_report(sess.state)
            timing = {}
            if log_speed:
                synchronize()
                duration = time.perf_counter() - started
                n = outcome.completion_tokens
                forward = sum(sess.forward_times)
                sampling = sum(sess.sample_times)
                prefill = sess.prefill_time_s
                decode = max(duration - prefill, 0.0)
                timing = {
                    "tokens_per_sec": n / duration if duration else 0.0,
                    "decode_tokens_per_sec": n / decode if decode else 0.0,
                    "prefill_time_s": prefill,
                    "total_tokens": n,
                    "total_time_s": duration,
                    "forward_times": sess.forward_times,
                    "other_times": sess.sample_times,
                    "total_forward_time_s": forward,
                    "total_other_time_s": sampling,
                    "avg_forward_time_s": forward / len(sess.forward_times) if sess.forward_times else 0.0,
                    "avg_other_time_s": sampling / len(sess.sample_times) if sess.sample_times else 0.0,
                    "breakdown_is_async_approx": BACKEND_NAME == "cupy",
                }
            if return_token_ids:
                timing["generated_token_ids"] = list(outcome.token_ids)
                timing["eos_token_id"] = self._get_eos_id()
            return outcome.text, timing

        # Numerical reference: full-prefix forward, never a production fast path.
        input_ids = list(self.tokenizer.encode(prompt))[-self.max_context:]
        if not input_ids:
            raise ValueError("Prompt tokenized to an empty sequence")
        generated_ids = list(input_ids)
        eos_id = self._get_eos_id()
        forward_times = []
        if log_speed:
            synchronize()
            started = time.perf_counter()
        from .generation import sample_token
        for _ in range(max_new_tokens):
            if log_speed:
                synchronize()
                start = time.perf_counter()
            context = generated_ids[-self.max_context:]
            tokens = xp.asarray([context], dtype=xp.int32)
            logits = self.model.forward(tokens, return_cache=False)
            if log_speed:
                synchronize()
                forward_times.append(time.perf_counter() - start)
            params = {"greedy": (0.0, None, 1.0),
                      "temperature": (temperature, None, 1.0),
                      "top-k": (temperature, top_k, 1.0),
                      "top-p": (temperature, None, top_p)}
            temp, k, p = params[strategy]
            if (strategy == "top-k" and k is None) or (strategy == "top-p" and p is None):
                raise ValueError(f"{strategy} requires its sampling parameter")
            next_id = sample_token(logits[0, -1], temperature=temp, top_k=k, top_p=p)
            generated_ids.append(next_id)
            del logits, tokens
            GQAAttention.clear_caches()
            if BACKEND_NAME == "cupy":
                xp.get_default_memory_pool().free_all_blocks()
                xp.get_default_pinned_memory_pool().free_all_blocks()
            if next_id == eos_id:
                break
        result_ids = generated_ids[len(input_ids):]
        timing = {}
        if log_speed:
            synchronize()
            duration = time.perf_counter() - started
            forward = sum(forward_times)
            timing = {
                "tokens_per_sec": len(result_ids) / duration if duration else 0.0,
                "decode_tokens_per_sec": 0.0,
                "prefill_time_s": 0.0,
                "total_tokens": len(result_ids),
                "total_time_s": duration,
                "forward_times": forward_times,
                "other_times": [],
                "total_forward_time_s": forward,
                "total_other_time_s": 0.0,
                "avg_forward_time_s": forward / len(forward_times) if forward_times else 0.0,
                "avg_other_time_s": 0.0,
                "breakdown_is_async_approx": BACKEND_NAME == "cupy",
            }
        if return_token_ids:
            timing["generated_token_ids"] = result_ids
            timing["eos_token_id"] = eos_id
        return self.tokenizer.decode(result_ids), timing

    def _ensure_inference_model(self, reset: bool = False):
        """Ensure inference model and state are initialized for KV cache.

        Args:
            reset: If True, create a fresh state (used for new prompts)
        """
        if not hasattr(self, '_inference_model'):
            from mini_llm.inference_model import InferenceModel

            self._inference_model = InferenceModel(
                self.model.config,
                dtype=self.model.dtype
            )
            self._inference_model.set_weights(self.model)

            self._state = self._inference_model.create_generation_state(
                batch_size=1,
                max_length=self.max_context
            )
        elif reset:
            # Captured graphs are bound to the existing cache addresses. Reuse
            # that state so repeated requests do not recapture the graph.
            if getattr(self._state, '_native_cuda_graph', None) is not None:
                self._state.reset()
            else:
                self._state = self._inference_model.create_generation_state(
                    batch_size=1,
                    max_length=self.max_context
                )
