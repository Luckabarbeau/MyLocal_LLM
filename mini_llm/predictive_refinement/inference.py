"""0079D progressive commitment with real committed-context KV caching.

Every refinement runs the *same* inference Transformer at full depth, solely
on active K-token queries. The only permanent KV modifications occur in the
normal causal append after actual tokens are selected. Sparse/routed history
is explicitly unsupported until a correct block-specific layout is available.
"""

from dataclasses import dataclass

from mini_llm.backend import xp, RandomStream
from .soft_embeddings import FuturePositionEmbeddings, SoftTokenExpectation, stable_softmax


@dataclass(frozen=True)
class RefinementGenerationConfig:
    block_size: int = 4
    feedback_temperature: float = 1.0
    sampling_temperature: float = 1.0
    commitment_policy: str = "greedy"
    eos_token_id: int | None = None

    def __post_init__(self):
        if not 1 <= int(self.block_size) <= 16:
            raise ValueError("block_size must be in [1,16]")
        if self.feedback_temperature <= 0 or self.feedback_temperature == float("inf"):
            raise ValueError("feedback_temperature must be finite and positive")
        if self.sampling_temperature <= 0 or self.sampling_temperature == float("inf"):
            raise ValueError("sampling_temperature must be finite and positive")
        if self.commitment_policy not in {"greedy", "sample"}:
            raise ValueError("commitment_policy must be greedy or sample")


class CachedPredictiveRefinementGenerator:
    """Opt-in generator for an already-trained *refinement* checkpoint.

    Pass ``states=training_reference.states`` to share unknown/soft/fixed
    learned vectors. This class never initializes new state embeddings when
    explicitly given trained ones, and does not silently adapt an AR checkpoint.
    """

    def __init__(self, inference_model, states: FuturePositionEmbeddings, config=None, *, seed=0):
        self.model = inference_model
        self.states = states
        self.config = config or RefinementGenerationConfig()
        if inference_model.sparse_attention or inference_model.terminal_landmark_memory or inference_model.routed_prefix_memory:
            raise NotImplementedError("0079D supports dense KV cache only")
        if inference_model.embedding is None:
            raise RuntimeError("inference weights must be loaded before generation")
        self.expectation = SoftTokenExpectation(inference_model.embedding.W)
        self.rng = RandomStream(seed).rng

    def _choose(self, logits):
        if self.config.commitment_policy == "greedy":
            return int(xp.argmax(logits))
        q = stable_softmax(logits[None, :], self.config.sampling_temperature)[0]
        return int(self.rng.choice(int(q.shape[0]), p=q))

    def generate_block(self, state, *, max_new_tokens=None, return_trace=False):
        """Return committed token IDs and pass count; advance causal KV exactly once.

        Restricted to batch size one in this first reference generator because
        EOS-aware termination can produce per-example variable block lengths.
        The active-block inference kernel itself supports arbitrary batch size.
        """
        if state.batch_size != 1:
            raise NotImplementedError("EOS-aware progressive decoding is currently B=1")
        remaining = int(state.remaining_capacity())
        if max_new_tokens is not None:
            remaining = min(remaining, int(max_new_tokens))
        k = min(int(self.config.block_size), remaining)
        if k <= 0:
            return {"tokens": [], "passes": 0, "eos": False}
        active = self.states.unknown_block(1, k)
        prev_logits = self.model.forward_provisional_block(active, state)
        trace = [prev_logits.copy()] if return_trace else None
        committed = []
        eos = False
        for j in range(k):
            soft, _ = self.expectation.forward(
                prev_logits[:, j:, :], temperature=self.config.feedback_temperature
            )
            soft_active = self.states.soft_block(soft)
            if j:
                prefix_ids = xp.asarray([committed], dtype=xp.int32)
                fixed = self.states.fixed_block(prefix_ids, self.model.embedding.W)
                active = xp.concatenate((fixed, soft_active), axis=1)
            else:
                active = soft_active
            prev_logits = self.model.forward_provisional_block(active, state)
            if return_trace:
                trace.append(prev_logits.copy())
            token = self._choose(prev_logits[0, j, :])
            committed.append(token)
            if token == self.config.eos_token_id:
                eos = True
                break
        # Bidirectional provisional K/V are never part of GenerationState.
        # Finalization is one standard causal prefill for the committed subset.
        self.model.append_causal_block(xp.asarray([committed], dtype=xp.int32), state)
        result = {"tokens": committed, "passes": len(committed) + 1, "eos": eos}
        if return_trace:
            result["trace"] = trace
        return result

    def generate(self, state, max_new_tokens):
        if int(max_new_tokens) < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        emitted = []
        passes = 0
        while len(emitted) < int(max_new_tokens):
            result = self.generate_block(
                state, max_new_tokens=int(max_new_tokens) - len(emitted)
            )
            if not result["tokens"]:
                break
            emitted.extend(result["tokens"])
            passes += result["passes"]
            if result["eos"]:
                break
        return {"tokens": emitted, "passes": passes, "eos": bool(emitted and emitted[-1] == self.config.eos_token_id)}
