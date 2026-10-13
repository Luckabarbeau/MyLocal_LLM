"""0080D production-original next-token sampler for 64K routed block training.

Deliberately DOES NOT call 0079Q4's align_next_token_targets: the block
boundary is h(x0) with labels beginning at x1. EOS and document-first sampling
are delegated to the original 0078E data factory, unmodified.
"""
import numpy as np
from mini_llm.data.token_shards import create_hierarchical_memory_minibatch
from mini_llm.predictive_refinement.real_data import PackedRefinementDocuments


class PackedRoutedBlockProposalDocuments(PackedRefinementDocuments):
    def sample_causal(self,rng,*,batch_size,memory_length,target_length,
                      min_history_tokens=0):
        if min(int(batch_size),int(memory_length),int(target_length))<=0:
            raise ValueError('positive batch, history and target lengths required')
        sources=[];labels=[];masks=[];starts=[];lengths=[];observations=[]
        for _ in range(int(batch_size)):
            s=int(rng.choice(len(self.sources),p=self.source_probs))
            name,_,paths=self.sources[s]
            shard=int(rng.choice(len(paths),p=self.shard_weights[s]))
            tokens,index=self._get_shard(paths[shard])
            x,y,md=create_hierarchical_memory_minibatch(
                tokens,1,memory_length,target_length,rng,
                eos_token_id=self.eos,document_index=index,
                pad_token_id=self.eos,document_aware=True,return_metadata=True,
                min_history_tokens=int(min_history_tokens),sampling_policy='document_first')
            sources.append(x[0]);labels.append(y[0]);
            count=int(md['target_valid_lengths'][0]);lengths.append(count)
            masks.append(np.arange(target_length)<count)
            starts.append(int(md['history_valid_starts'][0]))
            observations.append(dict(source=name,
                sampled_document_span_tokens=int(md['document_ends'][0]-md['document_starts'][0]),
                supervised_tokens=count,history_tokens=memory_length-starts[-1]))
        metadata=dict(history_valid_starts=np.asarray(starts,dtype=np.int64),
            target_valid_lengths=np.asarray(lengths,dtype=np.int64),
            router_stochastic=False,causal_prefix_query=True,
            causal_masked_target_refinement=True,use_retrieval=True)
        return (np.stack(sources),np.stack(labels),np.stack(masks),metadata,observations)
