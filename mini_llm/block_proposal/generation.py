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
        if self.decoder.direct_latents:
            lp,latents=self.proposal.forward(self.inference.last_hidden,return_cache=False,return_latents=True)
        else:
            lp=self.proposal.forward(self.inference.last_hidden,return_cache=False)
            latents=None
        cache=self.decoder.prepare_plan_cache(lp,boundary_h=self.inference.last_hidden,proposal_latents=latents)
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


class SequentialCommitBlockGenerator:
    """Correctness-first K-proposal generation with tokenwise causal main KV updates.

    Unlike ``DirectCachedBlockGenerator``, this path supports the configured
    sparse attention and routed-prefix inference modes through their original
    ``InferenceModel.decode_one`` implementation.  Crucially it NEVER calls the
    original AR sampler to choose a token: choices are from the prefix decoder.

    Not a speed benchmark for batched K-token KV updates.  Such updates remain
    unsupported for sparse/routed caches and need separate correctness testing.
    """

    def __init__(self, inference_model, proposal_head, decoder, *, chooser,
                 eos_token_id=None, commit_stride=None):
        if proposal_head.model is not decoder.model or proposal_head.block_size != decoder.block_size:
            raise ValueError('proposal and decoder must share the same model and block size')
        if inference_model.lm_head is not proposal_head.model.output_proj.W.data:
            raise ValueError('inference main model does not share checkpoint model weights')
        if not callable(chooser):
            raise TypeError('chooser must take one corrected-vocabulary logits vector')
        self.inference = inference_model
        self.proposal = proposal_head
        self.decoder = decoder
        self.choose = chooser
        self.eos_token_id = eos_token_id
        horizon = int(proposal_head.block_size)
        stride = horizon if commit_stride is None else int(commit_stride)
        if stride < 1 or stride > horizon:
            raise ValueError(f'commit_stride must be between 1 and K={horizon}')
        self.commit_stride = stride

    def generate(self, state, max_new_tokens):
        if state.batch_size != 1:
            raise NotImplementedError('only batch=1 generation is supported')
        if not hasattr(self.inference, 'last_hidden'):
            raise RuntimeError('call inference_model.prefill(prompt,state) first')
        remaining = int(max_new_tokens)
        if remaining < 0:
            raise ValueError('max_new_tokens must be nonnegative')
        tokens = []
        proposals = 0
        causal_updates = 0
        eos = False
        while remaining > 0 and state.remaining_capacity() > 0:
            n_commit = min(self.commit_stride, remaining, state.remaining_capacity())
            # Always compute the FULL K-token plan even for stride=1/2.
            # All K proposals are conditioned on the SAME causal boundary,
            # with no ground-truth future tokens available.
            if self.decoder.direct_latents:
                logits,latents = self.proposal.forward(self.inference.last_hidden,return_cache=False,return_latents=True)
            else:
                logits = self.proposal.forward(self.inference.last_hidden,return_cache=False)
                latents = None
            plan = self.decoder.prepare_plan_cache(logits,boundary_h=self.inference.last_hidden,proposal_latents=latents)
            proposals += 1
            committed = []
            for j in range(n_commit):
                previous = None if j == 0 else xp.asarray([committed[-1]], dtype=xp.int64)
                corrected = self.decoder.cached_step(plan, previous)
                selected = int(self.choose(corrected[0]))
                if not 0 <= selected < self.inference.vocab_size:
                    raise ValueError('decoder sampler returned an invalid token ID')
                committed.append(selected)
                tokens.append(selected)
                remaining -= 1
                # We must make the REAL token part of the main causal state.
                # Never cache provisional states as if they were committed.
                self.inference.decode_one(xp.asarray([[selected]], dtype=xp.int32), state)
                causal_updates += 1
                if self.eos_token_id is not None and selected == self.eos_token_id:
                    eos = True
                    break
            if eos:
                break
        return dict(tokens=tokens, eos=eos, proposal_blocks=proposals,
                    causal_main_updates=causal_updates, commit_stride=self.commit_stride,
                    proposal_horizon=int(self.proposal.block_size),
                    implementation='sequential_main_kv')

    def score_reference_tokens(self, state, reference_token_ids):
        """Causal teacher-forced NLL under THIS stride's true inference procedure.

        Every plan is computed before the corresponding reference token is read.
        Subsequent plans see only reference tokens already committed to main KV.
        This is distinct from the cheaper training-time, fixed-block CE path.
        """
        if state.batch_size != 1:
            raise NotImplementedError('reference scoring supports batch=1 only')
        if not hasattr(self.inference, 'last_hidden'):
            raise RuntimeError('prefill the genuine prompt before scoring')
        references = [int(token) for token in reference_token_ids]
        if any(not 0 <= token < self.inference.vocab_size for token in references):
            raise ValueError('invalid reference token ID')
        if len(references) > state.remaining_capacity():
            raise ValueError('reference sequence exceeds remaining causal context capacity')
        position = 0
        losses = []
        proposal_blocks = 0
        while position < len(references):
            if self.decoder.direct_latents:
                logits,latents = self.proposal.forward(self.inference.last_hidden,return_cache=False,return_latents=True)
            else:
                logits = self.proposal.forward(self.inference.last_hidden,return_cache=False)
                latents = None
            cache = self.decoder.prepare_plan_cache(logits,boundary_h=self.inference.last_hidden,proposal_latents=latents)
            proposal_blocks += 1
            previous = None
            for _ in range(min(self.commit_stride, len(references) - position)):
                scores = self.decoder.cached_step(cache, previous)[0]
                target = references[position]
                # Stable log-sum-exp with no extra [K,V] allocation.
                floating = scores.astype(xp.float64, copy=False)
                maximum = float(xp.max(floating))
                log_sum = maximum + float(xp.log(xp.sum(xp.exp(floating - maximum))))
                losses.append(log_sum - float(floating[target]))
                self.inference.decode_one(xp.asarray([[target]], dtype=xp.int32), state)
                position += 1
                previous = xp.asarray([target], dtype=xp.int64)
        return dict(nll_sum=float(sum(losses)), n_tokens=len(losses),
                    mean_ce=float(sum(losses)/len(losses)) if losses else None,
                    per_token_nll=losses, proposal_blocks=proposal_blocks,
                    causal_main_updates=len(references), commit_stride=self.commit_stride)
