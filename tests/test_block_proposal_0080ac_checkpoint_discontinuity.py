"""0080AC: historical QKV probe must be strictly non-mutating."""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME, asnumpy
from benchmarks.audit_decoder_checkpoint_discontinuity import (
    fresh_initial_decoder, temporarily_replace_decoder_packs_with_initial,
)
from test_block_proposal_0080n_production import _trainer, _docs, _close


@pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='CPU regression')
def test_initialization_era_swap_is_reversible(monkeypatch, tmp_path):
    monkeypatch.setenv('MINI_LLM_PACKED_QKV','1')
    trainer = _trainer(_docs(tmp_path))
    try:
        dec = trainer.training_objective.net.decoder
        for layer in dec.layers:
            layer.attn.refresh_compute_buffers()
        fresh = fresh_initial_decoder(dec)
        originally = [(layer.attn._packed_qkv_weight,
                       np.array(asnumpy(layer.attn._packed_qkv_weight)),
                       layer.attn._packed_qkv_valid) for layer in dec.layers]
        # Perturb trainable attention without refreshing: the swap should not
        # touch these canonical model parameters.
        q_before = [np.asarray(asnumpy(layer.attn.Wq.data)).copy() for layer in dec.layers]
        with temporarily_replace_decoder_packs_with_initial(dec, fresh):
            for cur, initial in zip(dec.layers, fresh.layers):
                expected = np.concatenate([
                    asnumpy(initial.attn.Wq.data), asnumpy(initial.attn.Wk.data),
                    asnumpy(initial.attn.Wv.data)],axis=1)
                np.testing.assert_array_equal(cur.attn._packed_qkv_weight,expected)
        for cur, (ref, values, valid), q in zip(dec.layers, originally, q_before):
            assert cur.attn._packed_qkv_weight is ref
            assert cur.attn._packed_qkv_valid == valid
            np.testing.assert_array_equal(cur.attn._packed_qkv_weight,values)
            np.testing.assert_array_equal(asnumpy(cur.attn.Wq.data),q)
    finally:
        _close(trainer)

@pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='CPU regression')
def test_loaded_proposal_decoder_weights_equal_checkpoint_and_masters(tmp_path):
    from mini_llm.block_proposal.checkpoint import load_two_stage_checkpoint
    from benchmarks.audit_decoder_checkpoint_discontinuity import check_decoder_saved_weights
    data = _docs(tmp_path)
    ckpt = tmp_path / 'saved'
    trainer = _trainer(data, checkpoint_dir=str(ckpt))
    loaded = _trainer(data)
    try:
        trainer.train_step()
        trainer.save()
        load_two_stage_checkpoint(ckpt, loaded.training_objective.net,
                                  optimizer=loaded.optimizer)
        report = check_decoder_saved_weights(ckpt, loaded.training_objective.net)
        assert report['decoder_and_proposal_tensors'] > 0
        assert report['optimizer_master_tensors'] == report['decoder_and_proposal_tensors']
        assert not report['checkpoint_to_loaded_byte_mismatches']
        assert not report['fp32_masters_not_equal_to_saved_after_rounding']
    finally:
        _close(trainer)
        _close(loaded)
