"""0080E: cached two-stage generation for direct dense main-model contexts.

The production original ``InferenceModel`` supplies the normalized h at each
committed block boundary. Main KV is updated with one causal K-token chunk.
The small prefix-LM caches its full PLAN once and appends each DEC token.

This adapter intentionally rejects sparse and long routed main caches until
those append and refresh policies pass exact chunk-vs-single-step tests.
"""
from dataclasses import dataclass
from mini_llm.backend import xp, RandomStream
from mini_llm.predictive_refinement.soft_embeddings import stable_softmax


@dataclass(frozen=True)
class BlockGenerationConfig:
    strategy: str='greedy'
    temperature: float=1.0
    eos_token_id: int | None=None

    def __post_init__(self):
        if self.strategy not in {'greedy','sample'}:
            raise ValueError('strategy must be greedy or sample')
        if not 0<float(self.temperature)<float('inf'):
            raise ValueError('temperature must be positive and finite')


class DirectCachedBlockGenerator:
    def __init__(self,inference_model,proposal_head,decoder,*,config=None,seed=42):
        if inference_model.sparse_attention or inference_model.routed_prefix_memory or inference_model.terminal_landmark_memory:
            raise NotImplementedError('0080E validated generator currently supports direct dense KV only')
        if proposal_head.model is not decoder.model or proposal_head.block_size!=decoder.block_size:
            raise ValueError('proposal/decoder must share main model and K')
        if inference_model.lm_head is not proposal_head.model.output_proj.W.data:
            raise ValueError('inference model must share the corresponding training main model weights')
        self.inference=inference_model
        self.proposal=proposal_head
        self.decoder=decoder
        self.config=config or BlockGenerationConfig()
        self.rng=RandomStream(seed).rng

    def choose(self,logits):
        if self.config.strategy=='greedy':
            return int(xp.argmax(logits))
        probs=stable_softmax(logits[None,:],self.config.temperature)[0]
        return int(self.rng.choice(int(probs.shape[0]),p=probs))

    def generate_block(self,state,*,max_new_tokens=None):
        if state.batch_size!=1:
            raise NotImplementedError('EOS-aware inference currently requires batch=1')
        n=state.remaining_capacity()
        if max_new_tokens is not None:
            n=min(n,int(max_new_tokens))
        k=min(int(self.proposal.block_size),int(n))
        if k<=0:
            return {'tokens':[], 'eos':False,'main_updates':0,'plan_prefills':0}
        if not hasattr(self.inference,'last_hidden'):
            raise RuntimeError('first call inference_model.prefill(prompt,state)')
        lp=self.proposal.forward(self.inference.last_hidden,return_cache=False)
        cache=self.decoder.prepare_plan_cache(lp)
        tokens=[]
        eos=False
        for j in range(k):
            previous=None if not tokens else xp.asarray([tokens[-1]],dtype=xp.int64)
            scores=self.decoder.cached_step(cache,previous)
            token=self.choose(scores[0])
            tokens.append(token)
            if self.config.eos_token_id is not None and token==int(self.config.eos_token_id):
                eos=True
                break
        self.inference.append_causal_block(xp.asarray([tokens],dtype=xp.int32),state)
        return {'tokens':tokens, 'eos':eos,'main_updates':1,'plan_prefills':1}

    def generate(self,state,max_new_tokens):
        if int(max_new_tokens)<0:
            raise ValueError('max_new_tokens must be nonnegative')
        result=[]
        updates=0
        while len(result)<max_new_tokens:
            part=self.generate_block(state,max_new_tokens=max_new_tokens-len(result))
            result.extend(part['tokens'])
            updates+=part['main_updates']
            if part['eos'] or not part['tokens']:
                break
        return {'tokens':result,'main_updates':updates,
                'eos':bool(self.config.eos_token_id is not None and
                           result and result[-1]==self.config.eos_token_id)}
