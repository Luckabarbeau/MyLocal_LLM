"""Framework-independent model configuration registry (0103-0).

Registrations provide architectural configuration, not numerical kernels.
Adding a new backend must not require modifying architecture presets.
"""

from __future__ import annotations

from typing import Callable


class ModelRegistry:
    def __init__(self):
        self._factories: dict[str, Callable] = {}

    def register(self, model_id: str, config_factory: Callable) -> None:
        if not model_id or not callable(config_factory):
            raise ValueError("A model ID and a callable config factory are required")
        if model_id in self._factories:
            raise ValueError(f"Model ID already registered: {model_id}")
        self._factories[model_id] = config_factory

    def config(self, model_id: str):
        try:
            factory = self._factories[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model ID: {model_id}") from exc
        return factory()

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


registry = ModelRegistry()


def _model_b_config():
    from .config import ModelConfig
    return ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()


registry.register("model_b_348m_sparse16", _model_b_config)
