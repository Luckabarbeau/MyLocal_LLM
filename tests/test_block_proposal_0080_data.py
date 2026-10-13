"""Original document-first long-memory sampler and first-token alignment."""
import numpy as np
from mini_llm.block_proposal.real_data import PackedRoutedBlockProposalDocuments


def test_short_documents_preserved_unshifted_and_eos_not_bridged(tmp_path):
    np.asarray([10,11,12,99],dtype=np.uint16).tofile(tmp_path/'train_shard_000.bin')
    data=PackedRoutedBlockProposalDocuments(shard_dir=tmp_path,split='train',
                                             eos_token_id=99,token_dtype='uint16')
    x,y,valid,md,obs=data.sample_causal(np.random.default_rng(17),batch_size=3,
                       memory_length=32,target_length=16,min_history_tokens=10)
    assert x.shape==(3,48) and y.shape==(3,16)
    assert np.array_equal(md['target_valid_lengths'],[3,3,3])
    assert np.array_equal(md['history_valid_starts'],[32,32,32])
    assert np.array_equal(x[:,32:35],np.array([[10,11,12]]*3))
    assert np.array_equal(y[:,:3],np.array([[11,12,99]]*3))
    assert np.all(valid[:,:3]) and not valid[:,3:].any()
    assert md['causal_prefix_query'] is True
    assert all(x['sampled_document_span_tokens']==4 for x in obs)
