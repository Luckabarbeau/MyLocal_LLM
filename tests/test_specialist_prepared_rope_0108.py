"""Specialist layers share one prepared RoPE table, including replay."""

import numpy as np

from mini_llm.ops.rope import PreparedRoPEPositions
from mini_llm.specialists import residual_specialist as specialist_module
from mini_llm.specialists.residual_specialist import ResidualSpecialist, SpecialistConfig


def test_specialist_prepares_rope_once_for_layers_and_checkpoint_replay(monkeypatch):
    monkeypatch.delenv("MINI_LLM_ROPE_PREPARED_POSITIONS", raising=False)
    specialist = ResidualSpecialist(SpecialistConfig(
        specialist_id="rope", n_layers=2, d_model=16, d_ff=32,
        n_q_heads=2, n_kv_heads=1, d_head=8,
        full_window=16, local_window=8, dtype="float32",
    ), rng_seed=7)

    prepare_calls = []
    original_prepare = specialist_module.prepare_rope_positions

    def counted_prepare(*args, **kwargs):
        prepared = original_prepare(*args, **kwargs)
        prepare_calls.append(prepared)
        return prepared

    monkeypatch.setattr(specialist_module, "prepare_rope_positions", counted_prepare)
    forwarded_positions = []
    for block in specialist.blocks:
        original_forward = block.forward

        def capture_forward(*args, _forward=original_forward, **kwargs):
            forwarded_positions.append(kwargs.get("position_ids"))
            return _forward(*args, **kwargs)

        monkeypatch.setattr(block, "forward", capture_forward)

    hidden = np.random.default_rng(11).normal(size=(1, 12, 16)).astype(np.float32)
    positions = np.arange(31, 43, dtype=np.int64)[None, :]
    delta, cache = specialist.forward_delta(
        hidden, position_ids=positions, return_cache=True,
        activation_checkpoint=True,
    )
    specialist.backward_delta(np.ones_like(delta), cache)

    assert len(prepare_calls) == 1
    prepared = prepare_calls[0]
    assert isinstance(prepared, PreparedRoPEPositions)
    assert len(forwarded_positions) == 2 * len(specialist.blocks)
    assert all(value is prepared for value in forwarded_positions)
