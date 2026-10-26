"""Phase 1: checkpoint-exact, forward-only shared foundation interface.

This is *not* the online probability-fusion teacher loader: that loader
explicitly disables historical context, while specialists must retain it.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from mini_llm.backend import xp
from mini_llm.checkpoint import load_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.parameter import no_parameter_grad_allocation


@dataclass(frozen=True)
class FrozenFoundation:
    """Read-only foundation and its exact frozen output-head weights.

    `hidden()` returns the representation *after* the foundation's final
    normalization. The residual specialist will eventually add ΔH there,
    prior to the single shared vocabulary projection. This class never
    allocates a backward tape or optimizer state for the foundation.

    Each specialist can reuse this same object with its own future memory
    policy; independent specialist memory is NOT implemented in Phase 1.
    """

    model: DecoderLanguageModel
    checkpoint_path: Path | None = None

    @classmethod
    def from_checkpoint(cls, checkpoint_dir):
        path = Path(checkpoint_dir).expanduser().resolve()
        with (path / "config.json").open(encoding="utf-8") as source:
            config = ModelConfig(**json.load(source))
        # Critically, do not replace or disable config.memory_context.
        with no_parameter_grad_allocation():
            model = DecoderLanguageModel(config, rng_seed=42, dtype=config.dtype)
        params = model.parameters()
        names = [p.name for p in params]
        if len(names) != len(set(names)):
            raise ValueError("foundation contains duplicate parameter names")
        state, _, _ = load_checkpoint(
            path, param_names=names, skip_optimizer=True, skip_training=True
        )
        expected = set(names)
        missing = expected - set(state)
        if missing:
            raise ValueError(f"foundation checkpoint missing: {sorted(missing)[:8]}")
        for param in params:
            value = state[param.name]
            if tuple(value.shape) != tuple(param.data.shape):
                raise ValueError(
                    f"foundation shape mismatch {param.name}: "
                    f"{value.shape} != {param.data.shape}"
                )
            param.data[...] = value
        del state
        if any(p.grad is not None for p in params):
            raise AssertionError("frozen foundation unexpectedly allocated gradients")
        return cls(model=model, checkpoint_path=path)

    @property
    def config(self):
        return self.model.config

    def hidden(self, token_ids, *, memory_metadata=None, position_ids=None):
        """Return `(normalized_hidden, target_slice)` without forward caches.

        Historical foundation inputs must use the checkpoint's complete source
        horizon and the original causal metadata convention. `target_slice`
        identifies supervised active positions, if the model has memory.
        """
        return self.model.forward_body(
            token_ids, return_cache=False, return_target_slice=True,
            memory_metadata=memory_metadata, position_ids=position_ids,
        )

    def project(self, normalized_hidden, target_slice=None):
        """Project one shared-head hidden representation (no weight gradients).

        For a zero correction this is exactly the original decoder logits path.
        Production training should use the chunked loss path instead to avoid
        a persistent complete-vocabulary tensor.
        """
        head = normalized_hidden
        if target_slice is not None:
            start, stop = map(int, target_slice)
            if not 0 <= start < stop <= head.shape[1]:
                raise ValueError("invalid foundation target slice")
            head = xp.array(head[:, start:stop, :], copy=True, order="C")
        logits, _ = self.model.output_proj.forward(head)
        return logits

    def predict(self, token_ids, *, memory_metadata=None, position_ids=None):
        hidden, target_slice = self.hidden(
            token_ids, memory_metadata=memory_metadata, position_ids=position_ids
        )
        return self.project(hidden, target_slice)

    def head_loss(self, corrected_hidden, targets, *, target_slice=None,
                  loss_mask=None, chunk_tokens=512, return_device_loss=False):
        """Compute chunked CE with the frozen vocabulary head.

        Return its usual `(loss, cache)` to be immediately consumed by
        `head_backward()`. Only the specialist will consume resulting dH;
        never call `model.backward_body()` on the frozen foundation.
        """
        return self.model.chunked_lm_head_loss_forward(
            corrected_hidden, targets, target_slice=target_slice,
            loss_mask=loss_mask, chunk_tokens=chunk_tokens,
            return_device_loss=return_device_loss,
        )

    def head_backward(self, cache, *, grad_scale=1.0):
        """Return dH without accumulating a single foundation parameter grad."""
        return self.model.chunked_lm_head_backward(
            cache, grad_scale=grad_scale, frozen_output_projection=True
        )
