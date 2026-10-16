"""0080AB: trained decoder packed QKV must match live parameter weights.

Regression for a production-only failure mode: checkpoint reload rebuilt packed
weights correctly while continuous training kept stale decoder attention packs.
"""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy
from test_block_proposal_0080n_production import _close, _docs, _trainer


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="deterministic CPU regression")
def test_production_optimizer_refreshes_decoder_packed_qkv(monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_LLM_PACKED_QKV", "1")
    trainer = _trainer(_docs(tmp_path))
    try:
        attention_layers = [layer.attn for layer in trainer.training_objective.net.decoder.layers]
        initial = []
        for attn in attention_layers:
            attn.refresh_compute_buffers()
            initial.append(asnumpy(attn._packed_qkv_weight).copy())

        loss, norm = trainer.train_step()
        assert np.isfinite(loss) and np.isfinite(norm)
        assert all(attn._packed_qkv_valid for attn in attention_layers)
        for attn in attention_layers:
            current = np.concatenate(
                [asnumpy(attn.Wq.data), asnumpy(attn.Wk.data), asnumpy(attn.Wv.data)],
                axis=1,
            )
            np.testing.assert_array_equal(asnumpy(attn._packed_qkv_weight), current)
        assert any(
            not np.array_equal(previous, asnumpy(attn._packed_qkv_weight))
            for previous, attn in zip(initial, attention_layers)
        ), "at least one decoder attention matrix should have been updated"
    finally:
        _close(trainer)

@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="deterministic CPU regression")
def test_decoder_packed_qkv_matches_after_checkpoint_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_LLM_PACKED_QKV", "1")
    from mini_llm.block_proposal.checkpoint import load_two_stage_checkpoint

    shard = _docs(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    live = _trainer(shard, checkpoint_dir=str(checkpoint))
    restored = _trainer(shard)
    try:
        live.train_step()
        live.save()
        load_two_stage_checkpoint(checkpoint, restored.training_objective.net,
                                  optimizer=restored.optimizer)
        for original_layer, restored_layer in zip(
            live.training_objective.net.decoder.layers,
            restored.training_objective.net.decoder.layers,
        ):
            original, recovered = original_layer.attn, restored_layer.attn
            np.testing.assert_array_equal(
                asnumpy(original._get_packed_qkv_weight()),
                asnumpy(recovered._get_packed_qkv_weight()),
            )
    finally:
        _close(live)
        _close(restored)
