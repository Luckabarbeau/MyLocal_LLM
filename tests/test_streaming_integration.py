"""0077B integration with ExtendedTrainer and checkpoint provenance."""
import json
import numpy as np
from test_streaming_sources import ResumableRows, TinyTokenizer, spec

def test_streaming_trainer_batch_and_checkpoint_resume(tmp_path, monkeypatch):
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.train_extended import ExtendedTrainer
    import mini_llm.data.streaming_sources as streaming

    monkeypatch.setattr(streaming, '_default_factory', lambda _: ResumableRows())
    shard = tmp_path/'local.bin'
    np.random.default_rng(77).integers(3, 40, size=4096, dtype=np.uint16).tofile(shard)

    def make():
        model = DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=12)
        trainer = ExtendedTrainer(
            model=model, train_shard_paths=[str(shard)], val_shard_paths=[str(shard)],
            train_source_shards={'local': [str(shard)]},
            val_source_shards={'local': [str(shard)]},
            source_weights={'local': 1.0},
            batch_size=1, seq_length=16, grad_accum_steps=1,
            warmup_steps=1, total_steps=5, checkpoint_dir=str(tmp_path/'checkpoint'),
            stream_specs=[spec()], stream_tokenizer=TinyTokenizer(),
            stream_segment_tokens=1024, stream_reuse_batches=3,
            stream_cache_dir=str(tmp_path),
        )
        trainer._choose_train_source = lambda _: 'fake'
        return trainer

    a = make()
    try:
        a.get_train_batch()
        a.get_train_batch()
        a.save()
        saved = json.loads((tmp_path/'checkpoint'/'training_state.json').read_text())
        assert saved['data_provenance']['streaming']['sources']['fake']['active']['microbatches'] == 2
        assert saved['train_source_batch_counts']['fake'] == 2
        assert saved['train_source_supervised_tokens']['fake'] == 32
        assert saved['data_provenance']['sample_trace_microbatches'] == 2
        assert saved['data_provenance']['sample_trace_sha256'] != '00' * 32
        b = make()
        try:
            b.streaming_cache.load_state_dict(saved['data_provenance']['streaming'])
            b.train_rng.bit_generator.state = saved['train_rng_state']
            b.sample_trace_sha256 = saved['data_provenance']['sample_trace_sha256']
            b.sample_trace_microbatches = saved['data_provenance']['sample_trace_microbatches']
            xa, ya = a.get_train_batch()
            xb, yb = b.get_train_batch()
            np.testing.assert_array_equal(xa, xb)
            np.testing.assert_array_equal(ya, yb)
            assert a.sample_trace_sha256 == b.sample_trace_sha256
        finally:
            b.streaming_cache.close()
    finally:
        a.streaming_cache.close()
