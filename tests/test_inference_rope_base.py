import dataclasses

from mini_llm.config import ModelConfig
from mini_llm.inference_model import InferenceModel


def test_inference_model_propagates_configured_rope_base():
    config = dataclasses.replace(
        ModelConfig.tiny_inspection(),
        rope_base=123_456.0,
    )

    model = InferenceModel(config, dtype=config.dtype)

    assert model.blocks
    for block in model.blocks:
        assert block.attention.rope_base == config.rope_base
