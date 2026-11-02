"""0104D: model-centered training lifecycle over the validated native trainer.

A TrainingRun owns optimizer/data/step mutable state. A Model retains only
architecture and parameters. This adapter intentionally does not implement
an additional optimizer, sampler, forward or backward path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingRun:
    """Thin, explicit lifecycle for one existing ExtendedTrainer instance."""
    model: Any
    trainer: Any

    @classmethod
    def from_native(cls, model, *, train_shard_paths, val_shard_paths,
                    **training_options):
        from .train_extended import ExtendedTrainer
        if not train_shard_paths and not training_options.get("train_source_shards"):
            raise ValueError("Training requires train_shard_paths or train_source_shards")
        if not val_shard_paths and not training_options.get("val_source_shards"):
            raise ValueError("Training requires val_shard_paths or val_source_shards")
        trainer = ExtendedTrainer(
            model=model._native_model,
            train_shard_paths=list(train_shard_paths or ()),
            val_shard_paths=list(val_shard_paths or ()),
            **training_options,
        )
        if trainer.model is not model._native_model:
            raise AssertionError("TrainingRun must reuse the original model instance")
        return cls(model, trainer)

    @classmethod
    def wrap_native(cls, model, trainer):
        """Attach existing production trainers without reallocating state."""
        if trainer.model is not model._native_model:
            raise ValueError("Trainer and Model do not share the same native model")
        return cls(model, trainer)

    @property
    def step(self) -> int:
        return int(self.trainer.step)

    @property
    def supervised_tokens(self) -> float:
        return float(self.trainer.supervised_tokens_total)

    @property
    def grad_accum_steps(self) -> int:
        return int(self.trainer.grad_accum_steps)

    def train_steps(self, steps: int, *, log_interval: int = 10):
        """Run the *existing* optimizer/data/validation/checkpoint pipeline."""
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        return self.trainer.train(num_steps=steps, log_interval=log_interval)

    def save_checkpoint(self):
        """Save full trainer state (including optimizer and RNG)."""
        if self.trainer.checkpoint_dir is None:
            raise ValueError("TrainingRun has no checkpoint_dir")
        return self.trainer.save()

    def inspect(self):
        return {
            "model_id": self.model.model_id,
            "step": self.step,
            "supervised_tokens": self.supervised_tokens,
            "grad_accum_steps": self.grad_accum_steps,
            "checkpoint_dir": (str(self.trainer.checkpoint_dir)
                               if self.trainer.checkpoint_dir is not None else None),
        }
