"""0080C: K-proposal-conditioned small *dense* prefix-LM Transformer.

This is a fully differentiable CPU / CuPy research implementation with manual
backward. The decoder never receives main Transformer hidden vectors: its only
conditioning is P@E and the shifted committed token stream. All K outputs are
computed by one teacher-forced forward per block group, not K model passes.

Production 0080D must chunk these groups to bound peak full-vocabulary memory;
0080E will add the actual decoder inference KV-cache path.
"""

from dataclasses import dataclass
import os

from mini_llm.backend import xp, RandomStream, is_low_precision_dtype
from mini_llm.init import matrix_parameter
from mini_llm.parameter import Parameter
from mini_llm.ops.attention import GQAAttention
from mini_llm.ops.linear import Linear
from mini_llm.ops.rmsnorm import RMSNorm
from mini_llm.ops.swiglu import SwiGLU
from mini_llm.ops.loss import cross_entropy_forward, cross_entropy_backward
from mini_llm.predictive_refinement.soft_embeddings import SoftTokenExpectation
from .contracts import prefix_lm_allow_mask, shifted_decoder_tokens


class DenseDecoderLayer:
    """RMSNorm / dense GQA / RMSNorm / conventional SwiGLU (no MoE)."""

    def __init__(self, width, heads, ff_width, rng, name, dtype):
        if width % heads:
            raise ValueError('small decoder width must be divisible by n_heads')
        self.width = width
        self.dtype = dtype
        self.low_precision = is_low_precision_dtype(dtype)
        self.norm1 = RMSNorm(width, name=f'{name}.norm1', dtype=dtype)
        self.attn = GQAAttention(
            d_model=width,n_q_heads=heads,n_kv_heads=heads,d_head=width//heads,
            input_std=0.02,output_std=0.02,rope_base=10000.0,
            rng=rng,name=f'{name}.attn',dtype=dtype,
        )
        self.norm2 = RMSNorm(width, name=f'{name}.norm2', dtype=dtype)
        self.ffn = SwiGLU(width,ff_width,0.02,0.02,rng,name=f'{name}.ffn',dtype=dtype)

    def parameters(self):
        return (self.norm1.parameters()+self.attn.parameters()+
                self.norm2.parameters()+self.ffn.parameters())

    def forward(self,x,allow):
        if self.low_precision:
            x = x.astype(xp.float32,copy=False)
        n1,c1 = self.norm1.forward(x)
        n1 = n1.astype(self.dtype,copy=False) if self.low_precision else n1
        a,ca = self.attn.forward(n1,attention_allow_mask=allow)
        mid = x + a.astype(xp.float32,copy=False) if self.low_precision else x+a
        n2,c2 = self.norm2.forward(mid)
        n2 = n2.astype(self.dtype,copy=False) if self.low_precision else n2
        f,cf = self.ffn.forward(n2.reshape(-1,self.width))
        f = f.reshape(mid.shape)
        y = mid+f.astype(xp.float32,copy=False) if self.low_precision else mid+f
        return y,(c1,ca,c2,cf)

    def backward(self,dy,caches):
        c1,ca,c2,cf = caches
        if self.low_precision:
            dy = dy.astype(xp.float32,copy=False)
        d_ff = dy.astype(self.dtype,copy=False) if self.low_precision else dy
        df = self.ffn.backward(d_ff.reshape(-1,self.width),cf).reshape(dy.shape)
        dn2 = self.norm2.backward(df.astype(xp.float32,copy=False) if self.low_precision else df,c2)
        dmid = dy+dn2
        da = dmid.astype(self.dtype,copy=False) if self.low_precision else dmid
        dn1 = self.attn.backward(da,ca)
        dnorm1 = self.norm1.backward(dn1.astype(xp.float32,copy=False) if self.low_precision else dn1,c1)
        return dmid+dnorm1


@dataclass(frozen=True)
class DecoderCache:
    shifted: object
    expectation_cache: object
    plan_linear_cache: dict
    token_linear_cache: dict
    layer_caches: tuple
    norm_cache: dict
    up_cache: dict
    output_cache: dict
    block_size: int


class BlockPrefixDecoder:
    """Single-pass teacher-forced decoder operating on **full** proposals.

    W_vocab and E are borrowed from the main model, not copied or tied.
    ``forward`` takes main *proposal* logits, not main hidden states.
    """

    def __init__(self,model,block_size=8,*,width=384,n_layers=3,n_heads=6,
                 ff_multiplier=4,rng_seed=81):
        k=int(block_size); width=int(width); layers=int(n_layers)
        if k<=0 or width<=0 or layers<=0 or int(n_heads)<=0 or width%int(n_heads):
            raise ValueError('positive K,width,layers and matching head dimensions required')
        self.model=model
        self.block_size=k
        self.width=width
        self.low_precision=is_low_precision_dtype(model.dtype)
        rng=RandomStream(rng_seed)
        dtype=model.dtype
        d=int(model.config.d_model)
        self.plan_projection=Linear(d,width,0.02,rng,name='block_decoder.plan',dtype=dtype)
        self.token_projection=Linear(d,width,0.02,rng,name='block_decoder.tokens',dtype=dtype)
        self.plan_positions=Parameter(rng.normal((k,width),std=.02,dtype=dtype),name='block_decoder.plan_positions',decay=False)
        self.dec_positions=Parameter(rng.normal((k,width),std=.02,dtype=dtype),name='block_decoder.dec_positions',decay=False)
        self.bos=Parameter(rng.normal((width,),std=.02,dtype=dtype),name='block_decoder.bos',decay=False)
        self.layers=[DenseDecoderLayer(width,int(n_heads),width*int(ff_multiplier),rng,f'block_decoder.layer{i}',dtype)
                     for i in range(layers)]
        self.final_norm=RMSNorm(width,name='block_decoder.final_norm',dtype=dtype)
        self.up=Linear(width,d,0.001,rng,name='block_decoder.correction',dtype=dtype)
        self.expectation=SoftTokenExpectation(model.embedding.W)
        self.allow=prefix_lm_allow_mask(k)

    def parameters(self):
        params=self.plan_projection.parameters()+self.token_projection.parameters()
        params+=[self.plan_positions,self.dec_positions,self.bos]
        for layer in self.layers:
            params.extend(layer.parameters())
        params+=self.final_norm.parameters()+self.up.parameters()
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def forward(self,proposal_logits,targets,valid,*,plan_condition='normal',
                prefix_condition='normal',plan_override=None,
                trusted_block_schedule=False):
        """Return final prior-plus-correction logits and differentiable tape.

        ``targets`` are consumed only *shifted*, so DEC_j sees y_<j. They
        are NEVER used when building the bidirectional PLAN prefix.
        """
        if proposal_logits.ndim != 3 or proposal_logits.shape[1] != self.block_size:
            raise ValueError('proposal logits expected [N,K,V]')
        n,k,v=proposal_logits.shape
        if targets.shape != (n,k) or valid.shape != (n,k):
            raise ValueError('target/mask must be [N,K]')
        if not trusted_block_schedule and not bool(xp.any(valid)):
            raise ValueError('no valid proposal tokens')
        if plan_condition not in ('normal','neutral','shuffled','external'):
            raise ValueError('unknown plan conditioning ablation')
        if prefix_condition not in ('normal','masked'):
            raise ValueError('unknown committed-prefix ablation')
        if (plan_condition in ('shuffled','external')) != (plan_override is not None):
            raise ValueError('shuffled/external plan conditioning requires a plan bank')
        shifted=shifted_decoder_tokens(
            targets,valid,trusted_prefix_contiguous=trusted_block_schedule)
        if prefix_condition=='masked':
            # Mask *all* committed-token content, retaining DEC positions,
            # PLAN attention, BOS embedding, and all original prior logits.
            shifted=xp.full_like(shifted,-1)
        if not trusted_block_schedule and bool(xp.any((shifted>=v) | (shifted < -1))):
            raise ValueError('committed decoder token ID outside vocabulary')
        z,expectation_cache=self.expectation.forward(proposal_logits)
        if plan_condition=='neutral':
            # Mean of all vocabulary embeddings = uniform-proposal expectation.
            # No additional full [N,K,V] logits/probability tensor is needed.
            neutral=xp.mean(self.model.embedding.W.data.astype(xp.float32),axis=0)
            z=xp.broadcast_to(neutral.astype(z.dtype),z.shape)
        elif plan_condition in ('shuffled','external'):
            if plan_override.shape!=z.shape:
                raise ValueError('external plan embedding shape differs from proposal expectation')
            z=plan_override
        plan,plan_cache=self.plan_projection.forward(z)
        # Previous token IDs are read only, never labels of the same/later position.
        safe_ids=xp.maximum(shifted,0)
        embedding=self.model.embedding.W.data[safe_ids]
        real,token_cache=self.token_projection.forward(embedding)
        real=xp.where((shifted>=0)[:,:,None],real,self.bos.data[None,None,:])
        plan=plan + self.plan_positions.data[None,:,:]
        real=real + self.dec_positions.data[None,:,:]
        x=xp.concatenate((plan,real),axis=1)
        cache_layers=[]
        for layer in self.layers:
            x,c=layer.forward(x,self.allow)
            cache_layers.append(c)
        decoded=x[:,k:,:]
        normalized,norm_cache=self.final_norm.forward(decoded)
        normalized=normalized.astype(self.model.dtype,copy=False) if self.low_precision else normalized
        up,up_cache=self.up.forward(normalized)
        delta,output_cache=self.model.output_proj.forward(up)
        # Main distribution remains a soft and correctable additive prior.
        final_logits=proposal_logits+delta
        return final_logits,DecoderCache(
            shifted=shifted, expectation_cache=expectation_cache,
            plan_linear_cache=plan_cache,token_linear_cache=token_cache,
            layer_caches=tuple(cache_layers),norm_cache=norm_cache,
            up_cache=up_cache,output_cache=output_cache,block_size=k)

    def backward(self,d_final_logits,cache):
        """Return **all** gradients with respect to proposal logits.

        The residual-prior identity gradient plus the full differentiable
        probability embedding feedback gradient are both included.
        """
        k=cache.block_size
        if d_final_logits.ndim!=3 or d_final_logits.shape[1]!=k:
            raise ValueError('expected final logits gradient [N,K,V]')
        df=d_final_logits.astype(self.model.dtype,copy=False) if self.low_precision else d_final_logits
        d_up=self.model.output_proj.backward(df,cache.output_cache)
        d_norm=self.up.backward(d_up,cache.up_cache)
        d_decode=self.final_norm.backward(d_norm.astype(xp.float32,copy=False)
                                               if self.low_precision else d_norm,
                                            cache.norm_cache)
        d_hidden=xp.zeros((d_decode.shape[0],2*k,self.width),dtype=xp.float32 if self.low_precision else d_decode.dtype)
        d_hidden[:,k:,:]=d_decode
        for layer,c in reversed(list(zip(self.layers,cache.layer_caches))):
            d_hidden=layer.backward(d_hidden,c)
        d_plan=d_hidden[:,:k,:]
        d_real=d_hidden[:,k:,:]
        self.plan_positions.grad+=xp.sum(d_plan.astype(self.plan_positions.grad.dtype,copy=False),axis=0)
        self.dec_positions.grad+=xp.sum(d_real.astype(self.dec_positions.grad.dtype,copy=False),axis=0)
        d_z=self.plan_projection.backward(d_plan.astype(self.model.dtype,copy=False)
                                                if self.low_precision else d_plan,
                                          cache.plan_linear_cache)
        valid_previous=cache.shifted>=0
        self.bos.grad+=xp.sum(xp.where(valid_previous[:,:,None],0,d_real),axis=(0,1))
        d_tokens=self.token_projection.backward(
            xp.where(valid_previous[:,:,None],d_real,0).astype(self.model.dtype,copy=False)
            if self.low_precision else xp.where(valid_previous[:,:,None],d_real,0),
            cache.token_linear_cache)
        # Use only previous genuine tokens. Never scatter padding or BOS into E.
        # Empty indexed scatter is a no-op, avoiding an implicit GPU
        # synchronization for every decoder group.
        xp.add.at(self.model.embedding.W.grad,
                  cache.shifted[valid_previous],
                  d_tokens[valid_previous])
        d_prop_from_plan=self.expectation.backward(d_z,cache.expectation_cache)
        return d_final_logits + d_prop_from_plan




def _cached_per_token_nll(ce_cache, targets, valid):
    """Per-position diagnostics from CE's existing probability cache.

    No second [N,K,V] softmax or FP32 full-vocabulary workspace is allocated.
    BF16 probabilities give approximate log diagnostics; the loss itself
    continues to use stable loss-layer FP32 log-sum-exp arithmetic.
    """
    row_losses = ce_cache.get('row_losses')
    if row_losses is not None:
        # Exactly the CE layer's stable FP32 per-target log-sum-exp, including
        # its validity mask, rather than -log(rounded BF16 probabilities).
        return row_losses.reshape(targets.shape)
    probs = ce_cache.get('probs_bf16')
    if probs is None:
        probs = ce_cache.get('probs_f32')
    if probs is None:
        probs = ce_cache['probs']
    n,k=map(int,targets.shape)
    probs=probs.reshape(n,k,-1)
    y=xp.where(valid,targets,0).astype(xp.int64)
    target_p=xp.take_along_axis(probs,y[:,:,None],axis=2)[:,:,0]
    nll=-xp.log(xp.maximum(target_p.astype(xp.float32,copy=False),1e-30))
    return xp.where(valid,nll,0.)


@dataclass(frozen=True)
class JointCache:
    head_cache: object
    decoder_cache: DecoderCache
    final_ce_cache: dict
    proposal_ce_cache: dict
    n_unique: int


class TwoStageBlockModel:
    """0080C composable objective for one bounded group of independent blocks.

    Caller owns the single shared main causal forward/backward and chunked
    group scheduling (0080D). No direct raw main h enters the small decoder.
    """

    def __init__(self,proposal_head,decoder,*,lambda_proposal=0.5):
        if decoder.model is not proposal_head.model or decoder.block_size!=proposal_head.block_size:
            raise ValueError('proposal and decoder must share model and K')
        if float(lambda_proposal)<=0:
            raise ValueError('lambda_proposal must be positive')
        self.proposal_head=proposal_head
        self.decoder=decoder
        self.lambda_proposal=float(lambda_proposal)

    def parameters(self):
        return (self.proposal_head.model.parameters()+
                self.proposal_head.parameters()+self.decoder.parameters())

    def zero_grad(self):
        self.proposal_head.model.zero_grad()
        self.proposal_head.zero_grad()
        self.decoder.zero_grad()

    def forward(self,main_boundaries,targets,valid,*,plan_condition='normal',
                prefix_condition='normal',plan_override=None,
                defer_metrics=False,known_valid_count=None,
                trusted_block_schedule=False):
        lp,cp=self.proposal_head.forward(main_boundaries)
        lf,cd=self.decoder.forward(lp,targets,valid,
            plan_condition=plan_condition,prefix_condition=prefix_condition,
            plan_override=plan_override,
            trusted_block_schedule=trusted_block_schedule)
        # In-place BF16 CE may overwrite the logits buffer; preserve raw
        # proposal logits for the 0080C backward and diagnostics.
        copy_ce=self.decoder.low_precision and os.environ.get('MINI_LLM_INPLACE_BF16_CE','0') not in ('0','false','off','')
        # Production 0080 supplies the verified group count once per group
        # from its compact block schedule. Legacy research calls retain the
        # old independently verified count and host-scalar metrics.
        n_unique=(int(known_valid_count) if known_valid_count is not None
                  else int(xp.sum(valid)))
        if n_unique < 1 or n_unique > int(valid.size):
            raise ValueError('invalid group supervised-token count')
        safe_targets=xp.where(valid,targets,0)
        prop_loss,prop_cache=cross_entropy_forward(lp.copy() if copy_ce else lp,
                        safe_targets,loss_mask=valid,
                        return_device_loss=defer_metrics,
                        normalizer_count=n_unique, return_row_losses=True)
        final_loss,final_cache=cross_entropy_forward(lf,safe_targets,loss_mask=valid,
                        return_device_loss=defer_metrics,
                        normalizer_count=n_unique, return_row_losses=True)
        prop_nll=_cached_per_token_nll(prop_cache,targets,valid)
        final_nll=_cached_per_token_nll(final_cache,targets,valid)
        metrics={'proposal_position_ce_sums':xp.sum(prop_nll,axis=0),
                 'final_position_ce_sums':xp.sum(final_nll,axis=0),
                 'position_counts':xp.sum(valid.astype(xp.int64),axis=0),
                 'block_row_final_ce_sums':xp.sum(final_nll,axis=1),
                 'proposal_ce':prop_loss if defer_metrics else float(prop_loss),
                 'final_ce':final_loss if defer_metrics else float(final_loss),
                 'total_loss':(final_loss+self.lambda_proposal*prop_loss
                               if defer_metrics else
                               float(final_loss)+self.lambda_proposal*float(prop_loss)),
                 'unique_supervised_tokens':n_unique}
        return metrics,JointCache(cp,cd,final_cache,prop_cache,n_unique)

    def backward(self,cache,*,scale=1.0):
        d_final=cross_entropy_backward(cache.final_ce_cache)*float(scale)
        d_prop=self.decoder.backward(d_final,cache.decoder_cache)
        d_prop+=cross_entropy_backward(cache.proposal_ce_cache)*(self.lambda_proposal*float(scale))
        return self.proposal_head.backward(d_prop,cache.head_cache)

@dataclass
class PrefixDecoderGenerationState:
    """One immutable PLAN prefix with mutable per-layer DEC KV, one block."""
    proposal_logits: object
    keys: list
    values: list
    next_position: int = 0


def _cached_attend_one(attn, x, prefix_keys, prefix_values, position):
    """Exact one-row GQA after a bidirectional PLAN prefill.

    This is deliberately a small-K correctness path; it never materializes a
    long main-model 64K mask and has no dependency on future target labels.
    """
    from mini_llm.ops.rope import rope_forward
    n= int(x.shape[0]); h=int(attn.n_q_heads); hkv=int(attn.n_kv_heads)
    d=int(attn.d_head)
    q=(x @ attn.Wq.data).reshape(n,1,h,d)
    key=(x @ attn.Wk.data).reshape(n,1,hkv,d)
    value=(x @ attn.Wv.data).reshape(n,1,hkv,d)
    pos=xp.asarray([int(position)],dtype=xp.int64)
    q,_=rope_forward(q,attn.rope_base,position_ids=pos)
    key,_=rope_forward(key,attn.rope_base,position_ids=pos)
    keys=xp.concatenate((prefix_keys,key),axis=1)
    values=xp.concatenate((prefix_values,value),axis=1)
    # Hq and Hkv may differ; map each query head to its original GQA KV head.
    kv_map=xp.arange(h,dtype=xp.int64)//(h//hkv)
    acc_dtype = xp.float32 if is_low_precision_dtype(x.dtype) else x.dtype
    k_select=keys[:,:,kv_map,:].astype(acc_dtype,copy=False)
    v_select=values[:,:,kv_map,:].astype(acc_dtype,copy=False)
    query=q[:,0,:,:].astype(acc_dtype,copy=False)
    # [B,S,H,D] dot [B,H,D] -> [B,H,S]
    scores=xp.sum(k_select*query[:,None,:,:],axis=-1).transpose(0,2,1)*float(attn.scale)
    scores=scores-xp.max(scores,axis=-1,keepdims=True)
    p=xp.exp(scores)
    p/=xp.sum(p,axis=-1,keepdims=True)
    result=xp.sum(p.transpose(0,2,1)[:,:,:,None]*v_select,axis=1)
    merged=result.reshape(n,h*d).astype(x.dtype,copy=False)
    y=merged @ attn.Wo.data
    return y,keys,values


def _decoder_prepare_plan(self,proposal_logits):
    """Run all K PLAN tokens once, caching their learned K/V at every layer."""
    if proposal_logits.ndim!=3 or proposal_logits.shape[1]!=self.block_size:
        raise ValueError('proposal logits must be [B,K,V]')
    z,_=self.expectation.forward(proposal_logits)
    plan,_=self.plan_projection.forward(z)
    plan=plan+self.plan_positions.data[None,:,:]
    keys=[];values=[]
    allow=xp.ones((self.block_size,self.block_size),dtype=xp.bool_)
    for layer in self.layers:
        plan,cache=layer.forward(plan,allow)
        keys.append(cache[1]['k'].copy())
        values.append(cache[1]['v'].copy())
    return PrefixDecoderGenerationState(proposal_logits=proposal_logits,keys=keys,values=values)


def _decoder_cached_step(self,state,previous_token_ids=None):
    """Append one decoder position, preserving the unmodified PLAN proposal."""
    j=int(state.next_position)
    if j>=self.block_size:
        raise ValueError('all block positions have been decoded')
    n=int(state.proposal_logits.shape[0])
    if j==0:
        if previous_token_ids is not None:
            raise ValueError('first DEC position uses learned BOS, not a real token')
        x=xp.broadcast_to(self.bos.data[None,:],(n,self.width)).copy()
    else:
        if previous_token_ids is None:
            raise ValueError('subsequent DEC positions require last committed token ID')
        ids=xp.asarray(previous_token_ids,dtype=xp.int64).reshape(-1)
        if ids.shape!=(n,) or bool(xp.any((ids<0)|(ids>=self.model.config.vocab_size))):
            raise ValueError('previous committed tokens must be valid [B] token IDs')
        embedded=self.model.embedding.W.data[ids]
        x,_=self.token_projection.forward(embedded)
    x=x+self.dec_positions.data[None,j,:]
    for i,layer in enumerate(self.layers):
        if layer.low_precision:
            x=x.astype(xp.float32,copy=False)
        normalized,_=layer.norm1.forward(x)
        normalized=normalized.astype(self.model.dtype,copy=False) if self.low_precision else normalized
        a,k,v=_cached_attend_one(layer.attn,normalized,state.keys[i],state.values[i],self.block_size+j)
        state.keys[i]=k;state.values[i]=v
        mid=x+a.astype(xp.float32,copy=False) if self.low_precision else x+a
        n2,_=layer.norm2.forward(mid)
        n2=n2.astype(self.model.dtype,copy=False) if self.low_precision else n2
        ff,_=layer.ffn.forward(n2)
        x=mid+ff.astype(xp.float32,copy=False) if self.low_precision else mid+ff
    norm,_=self.final_norm.forward(x)
    norm=norm.astype(self.model.dtype,copy=False) if self.low_precision else norm
    up,_=self.up.forward(norm)
    delta,_=self.model.output_proj.forward(up)
    state.next_position+=1
    return state.proposal_logits[:,j,:]+delta


# These cached correctness operations use the exact trainable parameters in
# BlockPrefixDecoder, without an independent shadow inference model.
BlockPrefixDecoder.prepare_plan_cache = _decoder_prepare_plan
BlockPrefixDecoder.cached_step = _decoder_cached_step
