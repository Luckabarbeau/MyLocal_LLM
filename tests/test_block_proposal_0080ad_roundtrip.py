"""0080AD: real production optimizer update -> checkpoint -> FRESH model parity."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy
from mini_llm.block_proposal.checkpoint import load_two_stage_checkpoint
from mini_llm.diagnostics.block_proposal_roundtrip import (
    CAPTURE_KEYS, compare_fingerprints, decoder_pack_integrity, fixed_batch_fingerprint,
)
from test_block_proposal_0080n_production import _close, _docs, _trainer
from test_block_proposal_0080l_baselines import _window

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='small CPU production-regression model')


def test_real_optimizer_step_fresh_model_checkpoint_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv('MINI_LLM_PACKED_QKV', '1')
    monkeypatch.setenv('MINI_LLM_BLOCK_PROPOSAL_COMPACT_TARGET', '1')
    shard = _docs(tmp_path)
    checkpoint = tmp_path / 'checkpoint'
    live = _trainer(shard, checkpoint_dir=str(checkpoint))
    fresh = _trainer(shard)
    try:
        src, targets, mask, metadata = _window()
        # Real production step: two actual gradient-accumulation microbatches
        # and an optimizer update, NOT grad_scale=0 or a forward-only surrogate.
        loss, grad_norm = live.train_step()
        assert np.isfinite(loss) and grad_norm > 0
        assert live.optimizer.step_index == 1
        live_packs = decoder_pack_integrity(live.training_objective.net)
        before = fixed_batch_fingerprint(live.training_objective.net,
                                         src, targets, mask, metadata, group_size=2)
        live.save()  # ExtendedTrainer production serializer (not test-only save).
        saved_state = load_two_stage_checkpoint(checkpoint, fresh.training_objective.net,
                                                optimizer=fresh.optimizer)
        assert saved_state['step'] == live.step
        assert fresh.optimizer.step_index == live.optimizer.step_index
        fresh_packs = decoder_pack_integrity(fresh.training_objective.net)
        assert live_packs == fresh_packs
        after = fixed_batch_fingerprint(fresh.training_objective.net,
                                        src, targets, mask, metadata, group_size=2)
        parity = compare_fingerprints(before, after, ce_atol=1e-11, logits_atol=1e-11)
        assert parity['max_ce_error'] <= 1e-11
        assert set(parity['stage_max_abs_errors']) == set(CAPTURE_KEYS)
        for p, q in zip(live.training_objective.parameters(),
                        fresh.training_objective.parameters()):
            assert p.name == q.name
            np.testing.assert_array_equal(asnumpy(p.data), asnumpy(q.data), err_msg=p.name)
    finally:
        _close(live)
        _close(fresh)


def test_guard_catches_stale_decoder_cache_before_save(monkeypatch, tmp_path):
    monkeypatch.setenv('MINI_LLM_PACKED_QKV', '1')
    live = _trainer(_docs(tmp_path))
    try:
        live.train_step()
        decoder_pack_integrity(live.training_objective.net)
        attn = live.training_objective.net.decoder.layers[0].attn
        # Simulates original 0080AB failure: canonical QKV updates but the
        # optimized forward still uses the old packed matrix.
        attn.Wq.data[...] += 0.05
        with pytest.raises(AssertionError, match='stale packed QKV'):
            decoder_pack_integrity(live.training_objective.net)
    finally:
        _close(live)


def test_fingerprint_comparison_is_sensitive_to_decoder_and_proposal_changes(monkeypatch, tmp_path):
    monkeypatch.setenv('MINI_LLM_PACKED_QKV', '1')
    live = _trainer(_docs(tmp_path))
    try:
        src, targets, mask, metadata = _window()
        base = fixed_batch_fingerprint(live.training_objective.net,
                                       src, targets, mask, metadata, group_size=2)
        different = {**base, 'stages': dict(base['stages'])}
        different['stages']['soft_plan_embeddings'] = (
            base['stages']['soft_plan_embeddings'].copy() + 0.1)
        with pytest.raises(AssertionError, match='soft_plan_embeddings'):
            compare_fingerprints(base, different, ce_atol=1e-10, logits_atol=1e-10)
    finally:
        _close(live)


def test_cold_model_roundtrip_cli_core_on_cpu(monkeypatch, tmp_path):
    """Exercise sequential-save/new-model CLI path (500M uses same orchestration)."""
    from types import SimpleNamespace
    from benchmarks import audit_block_proposal_checkpoint_roundtrip as audit

    monkeypatch.setenv('MINI_LLM_PACKED_QKV', '1')
    shard = _docs(tmp_path)
    root = tmp_path / 'production_checkpoint'
    live = _trainer(shard, checkpoint_dir=str(root))
    created = []
    try:
        live.train_step()
        live.save()

        def load_small(_checkpoint, mixture_root=None):
            newly = _trainer(shard)
            created.append(newly)
            load_two_stage_checkpoint(root, newly.training_objective.net)
            return (newly.training_objective.net, object(), 100,
                    {'step': live.step},
                    [SimpleNamespace(name='fineweb', val_paths=[shard])], shard)

        monkeypatch.setattr(audit, 'load_production', load_small)
        monkeypatch.setattr(audit, 'sample_heldout',
                            lambda paths, **kwargs: iter([_window()]))
        output = tmp_path / 'roundtrip.json'
        args = SimpleNamespace(
            checkpoint=str(root), mixed_shard_root=str(tmp_path), source='fineweb',
            batch_size=2, group_size=2, seed=78078, expect_step=1,
            ce_tolerance=1e-11, logit_tolerance=1e-11,
            temporary_root=str(tmp_path), output=str(output))
        report = audit.run(args)
        assert report['passed'] is True
        assert report['step'] == 1
        assert report['decoder_packed_qkv_verified'] == 1
        assert report['decoder_proposal_tensors_verified'] > 0
        assert output.exists()
        assert report['parity']['max_ce_error'] < 1e-11
    finally:
        _close(live)
        for trainer in created:
            _close(trainer)
