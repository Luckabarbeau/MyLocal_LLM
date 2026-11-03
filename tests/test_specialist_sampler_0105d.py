import json
import hashlib
import numpy as np
import pytest
from mini_llm.specialists.document_sampler import SpecialistDocumentSampler


def mixture(tmp_path):
    root = tmp_path / 'mixed'
    root.mkdir()
    tokenizer = root/'tokenizer.json'
    tokenizer.write_text('{}')
    sources = {}
    for name, weight, fill in [('web', .75, 5), ('code', .25, 9)]:
        (root/name).mkdir()
        for kind in ('train','val'):
            # Consistent EOS and next-token windows.
            np.array([fill]*21 + [63] + [fill+1]*35 + [63],
                     dtype=np.uint16).tofile(root/name/f'{kind}_shard_000.bin')
        sources[name] = dict(weight=weight, relative_shard_dir=name,
                             train_shards=1, val_shards=1)
    manifest = dict(format_version='weighted_packed_mixture_v1', vocab_size=64,
                    tokenizer_sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
                    sources=sources)
    (root/'mixture_manifest.json').write_text(json.dumps(manifest))
    return root


@pytest.mark.parametrize('batch_size', [1, 2, 4])
def test_restart_preserves_exact_batches_and_metadata(tmp_path, batch_size):
    root = mixture(tmp_path)
    a = SpecialistDocumentSampler(root, context_length=32, batch_size=batch_size,
                                   eos_token_id=63, block_size=4, seed=42)
    for _ in range(5):
        a.next_batch()
    state = a.checkpoint_state()
    b = SpecialistDocumentSampler(root, context_length=32, batch_size=batch_size,
                                   eos_token_id=63, block_size=4, seed=99)
    b.restore(state)
    for _ in range(8):
        ai, at, am = a.next_batch()
        bi, bt, bm = b.next_batch()
        np.testing.assert_array_equal(ai, bi)
        np.testing.assert_array_equal(at, bt)
        np.testing.assert_array_equal(am['target_loss_mask'], bm['target_loss_mask'])
        np.testing.assert_array_equal(am['dual_resolution_metadata']['block_valid'],
                                      bm['dual_resolution_metadata']['block_valid'])
        assert am['source_name'] == bm['source_name']
        for row, valid in enumerate(am['target_valid_lengths']):
            assert len(np.unique(am['dual_resolution_metadata']['document_ids'][row, :valid])) == 1


def test_resume_rejects_changed_sampling_contract(tmp_path):
    root=mixture(tmp_path)
    a=SpecialistDocumentSampler(root, context_length=32,batch_size=1,
                                 eos_token_id=63,block_size=4)
    state=a.checkpoint_state()
    state['context_length']=16
    with pytest.raises(ValueError, match='context_length'):
        a.restore(state)


@pytest.mark.parametrize('batch_size', [1, 2, 4])
def test_specialist_sampler_matches_extended_trainer_source_and_document_sequence(tmp_path, monkeypatch, batch_size):
    """Check actual source-order/RNG parity, not only internal restart parity."""
    from mini_llm.config import ModelConfig, DualResolutionConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.train_extended import ExtendedTrainer
    from mini_llm.data.mixed_shards import (
        load_weighted_shard_sources, source_paths_by_split, source_weight_dict)
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_SHAPE_BUCKET', 'fine')
    root=mixture(tmp_path)
    sources,_=load_weighted_shard_sources(root)
    train=source_paths_by_split(sources,'train')
    val=source_paths_by_split(sources,'val')
    cfg=ModelConfig(tokenizer_vocab_size=64,context_length=32,n_layers=1,
        d_model=16,d_ff=32,n_q_heads=2,n_kv_heads=1,d_head=8,
        ffn_mode='dense',n_experts=1,top_k=1,dtype='float32',
        dual_resolution=DualResolutionConfig(enabled=True,block_size=4,
            local_window=8,compression_dim=8,compression_heads=2,
            max_blocks=16,global_query_chunk=4))
    legacy=ExtendedTrainer(model=DecoderLanguageModel(cfg,rng_seed=7),
        train_shard_paths=[p for paths in train.values() for p in paths],
        val_shard_paths=[p for paths in val.values() for p in paths],
        train_source_shards=train,val_source_shards=val,
        source_weights=source_weight_dict(sources),
        seq_length=32,batch_size=batch_size,eos_token_id=63,
        rng_seed=42,total_steps=3,warmup_steps=1,val_steps=1,
        val_interval=100,save_interval=100)
    specialist=SpecialistDocumentSampler(root,context_length=32,batch_size=batch_size,
                                        eos_token_id=63,block_size=4,seed=42)
    for _ in range(10):
        a,b,am=legacy.get_train_batch(return_metadata=True)
        c,d,cm=specialist.next_batch()
        np.testing.assert_array_equal(a,c)
        np.testing.assert_array_equal(b,d)
        np.testing.assert_array_equal(am['target_loss_mask'],cm['target_loss_mask'])
        assert legacy._last_train_batch_source==cm['source_name']
