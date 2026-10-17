"""0080C: K-proposal-conditioned small *dense* prefix-LM Transformer.

This is a fully differentiable CPU / CuPy research implementation with manual
backward. Historical decoder modes condition on P@E; 0080BD conditions directly on
K proposal latents and the shifted committed token stream. All K outputs are
computed by one teacher-forced forward per block group, not K model passes.

Production 0080D must chunk these groups to bound peak full-vocabulary memory;
0080E will add the actual decoder inference KV-cache path.
"""

from dataclasses import dataclass
import os

from mini_llm.backend import xp, RandomStream, is_low_precision_dtype, BACKEND_NAME
from mini_llm.init import matrix_parameter
from mini_llm.parameter import Parameter
from mini_llm.ops.attention import GQAAttention
from mini_llm.ops.linear import Linear
from mini_llm.ops.rmsnorm import RMSNorm
from mini_llm.ops.swiglu import SwiGLU
from mini_llm.ops.loss import cross_entropy_forward, cross_entropy_backward
from mini_llm.predictive_refinement.soft_embeddings import SoftTokenExpectation
from mini_llm.performance_profiler import block_proposal_detail_scope
from .contracts import prefix_lm_allow_mask, shifted_decoder_tokens
from .context_attention import context_attention_forward, context_attention_backward
from .soft_interface_config import SoftInterfaceConfig
from .soft_vocab_read import SoftVocabularyRead


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
    vocab_read_cache: object = None
    boundary_cache: object = None
    probability_adapter_cache: object = None
    probability_plan_norm_cache: object = None
    context_attention_cache: object = None


class BlockPrefixDecoder:
    """Single-pass teacher-forced decoder operating on **full** proposals.

    W_vocab and E are borrowed from the main model, not copied or tied.
    The 0080BD mode also accepts pre-unembedding proposal latents; it never
    consumes future ground-truth tokens to produce PLAN embeddings.
    """

    def __init__(self,model,block_size=8,*,width=384,n_layers=3,n_heads=6,
                 ff_multiplier=4,rng_seed=81,soft_interface_config=None,
                 experimental_cpu_full_vocab=False, experimental_gpu_full_vocab=False,
                 vocab_read_algorithm='source_mixture',vocab_read_tile=2048,
                 vocab_read_block_tile=8):
        k=int(block_size); width=int(width); layers=int(n_layers)
        if k<=0 or width<=0 or layers<=0 or int(n_heads)<=0 or width%int(n_heads):
            raise ValueError('positive K,width,layers and matching head dimensions required')
        # 0080AE: validate before allocating parameters or consuming the old
        # decoder's dedicated RandomStream. v1 defaults do not add tensors,
        # change initialization ordering, or alter forward/backward/cached KV.
        interface = (SoftInterfaceConfig() if soft_interface_config is None
                     else soft_interface_config)
        if not isinstance(interface, SoftInterfaceConfig):
            raise TypeError('soft_interface_config must be SoftInterfaceConfig')
        interface.validate(model_width=model.config.d_model,
                           decoder_layers=layers, block_size=k)
        if experimental_cpu_full_vocab and experimental_gpu_full_vocab:
            raise ValueError('choose one experimental backend opt-in')
        if experimental_cpu_full_vocab or experimental_gpu_full_vocab:
            # Experimental training only; generation and v1 checkpoints
            # remain disabled until their own parity and migration patches.
            if interface.is_v1_baseline:
                raise ValueError('experimental variant must change the v1 interface')
            if experimental_cpu_full_vocab and (
                BACKEND_NAME != 'numpy' or int(model.config.vocab_size) > 512 or
                    is_low_precision_dtype(model.dtype)):
                raise NotImplementedError('CPU read requires NumPy float32/64 and V<=512')
            if experimental_gpu_full_vocab and BACKEND_NAME != 'cupy':
                raise NotImplementedError('GPU read requires CuPy')
        else:
            interface.require_implemented()
        self.soft_interface_config=interface
        self.model=model
        self.block_size=k
        self.width=width
        self.direct_probability=(interface.mode == 'probability_embedding_direct')
        self.direct_latents=(interface.mode == 'proposal_latents_direct')
        self.direct_width=(self.direct_probability or self.direct_latents)
        if self.direct_width and width != int(model.config.d_model):
            raise ValueError('direct probability decoder must keep model hidden width')
        # The output projection remains shared with the original model.
        # A small fixed scale replaces the removed learned 512->768 up-projector.
        self.direct_correction_scale=0.1 if self.direct_width else 1.0
        self.direct_plan_scale=0.02 if self.direct_width else 1.0
        self.low_precision=is_low_precision_dtype(model.dtype)
        rng=RandomStream(rng_seed)
        dtype=model.dtype
        d=int(model.config.d_model)
        self.plan_projection=(None if self.direct_width else
                              Linear(d,width,0.02,rng,name='block_decoder.plan',dtype=dtype))
        self.token_projection=(None if self.direct_width else
                               Linear(d,width,0.02,rng,name='block_decoder.tokens',dtype=dtype))
        self.plan_positions=Parameter(rng.normal((k,width),std=.02,dtype=dtype),name='block_decoder.plan_positions',decay=False)
        self.dec_positions=Parameter(rng.normal((k,width),std=.02,dtype=dtype),name='block_decoder.dec_positions',decay=False)
        self.bos=Parameter(rng.normal((width,),std=.02,dtype=dtype),name='block_decoder.bos',decay=False)
        self.layers=[DenseDecoderLayer(width,int(n_heads),width*int(ff_multiplier),rng,f'block_decoder.layer{i}',dtype)
                     for i in range(layers)]
        self.final_norm=RMSNorm(width,name='block_decoder.final_norm',dtype=dtype)
        self.up=(None if self.direct_width else
                 Linear(width,d,0.001,rng,name='block_decoder.correction',dtype=dtype))
        self.expectation=SoftTokenExpectation(model.embedding.W)
        # 0080AM: a dedicated trainable map from full proposal probabilities
        # to a compact distribution representation. Separate RNG and unique
        # names preserve every existing v1/v2 parameter and its initialization.
        self.probability_embedding=None
        self.probability_adapter=None
        self.probability_expectation=None
        self.probability_plan_norm=(RMSNorm(width,eps=1e-10,
            name='block_decoder.probability_plan_norm',dtype=dtype)
            if self.direct_probability else None)
        if interface.mode in ('probability_embedding','probability_embedding_direct'):
            dim=int(interface.probability_embedding_dim)
            prob_rng=RandomStream(8093)
            self.probability_embedding=Parameter(
                prob_rng.normal((int(model.config.vocab_size),dim),std=.02,dtype=dtype),
                name='block_decoder.probability_embedding')
            if not self.direct_probability:
                self.probability_adapter=Linear(
                    dim,d,0.02,RandomStream(8094),
                    name='block_decoder.probability_adapter',dtype=dtype)
            self.probability_expectation=SoftTokenExpectation(self.probability_embedding)
        self.allow=prefix_lm_allow_mask(k)
        # Allocate only after *all* v1 layers and projections are initialized.
        # The independent read RNG never changes their previous random draws.
        self.vocab_read = None
        if interface.mode == 'full_vocab':
            self.vocab_read = SoftVocabularyRead(
                model.embedding.W, width, n_heads=int(interface.read_heads),
                temperature=float(interface.temperature),
                read_scale=float(interface.read_scale),
                rng_seed=8091, source_scope=interface.source_scope, block_size=k,
                execution='tiled' if experimental_gpu_full_vocab else 'reference',
                all_source_algorithm=vocab_read_algorithm,
                vocab_tile=int(vocab_read_tile),block_tile=int(vocab_read_block_tile))
        self.boundary_projection = None
        if interface.boundary_condition == 'dec_residual':
            # Independent RNG: legacy decoder and read parameters unchanged.
            self.boundary_projection = Linear(d, width, 0.0, RandomStream(8092),
                                              name='block_decoder.boundary', dtype=dtype)

    def parameters(self):
        params=[]
        if self.plan_projection is not None:
            params+=self.plan_projection.parameters()+self.token_projection.parameters()
        params+=[self.plan_positions,self.dec_positions,self.bos]
        for layer in self.layers:
            params.extend(layer.parameters())
        params+=self.final_norm.parameters()
        if self.up is not None:
            params+=self.up.parameters()
        if self.vocab_read is not None:
            params+=self.vocab_read.parameters()
        if self.boundary_projection is not None:
            params+=self.boundary_projection.parameters()
        if self.probability_plan_norm is not None:
            params+=self.probability_plan_norm.parameters()
        if self.probability_embedding is not None:
            params.append(self.probability_embedding)
            if self.probability_adapter is not None:
                params+=self.probability_adapter.parameters()
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def _distribution_plan(self, proposal_logits):
        """Full softmax over V; one learned expectation per proposed horizon.

        The adapter maps a configurable probability width into the *existing*
        plan projection's d_model width, retaining old parameters and allowing
        exact cached/teacher-forced parity without extra decoder layers.
        """
        if self.probability_expectation is None:
            z,expectation_cache=self.expectation.forward(proposal_logits)
            return z,expectation_cache,None
        raw,expectation_cache=self.probability_expectation.forward(proposal_logits)
        if self.probability_adapter is None:
            return raw,expectation_cache,None
        z,adapter_cache=self.probability_adapter.forward(raw)
        return z,expectation_cache,adapter_cache

    def forward(self,proposal_logits,targets,valid,*,plan_condition='normal',
                prefix_condition='normal',plan_override=None,
                trusted_block_schedule=False,boundary_h=None,proposal_latents=None,
                context_memory=None,context_valid=None):
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
        if self.direct_latents:
            if proposal_latents is None or proposal_latents.shape != (n,k,self.width):
                raise ValueError('0080BD requires proposal_latents [N,K,d_model]')
            if plan_condition != 'normal' or plan_override is not None:
                raise NotImplementedError('0080BD PLAN ablations need a latent-space definition')
        elif proposal_latents is not None:
            raise ValueError('proposal_latents are only accepted in 0080BD mode')
        if self.probability_expectation is not None and plan_condition != 'normal':
            # Other modes need a separate exact VJP for neutral/external PLAN.
            # Never misattribute their gradients to the live proposals.
            raise NotImplementedError('v3 PLAN ablations require a dedicated probability-embedding backward')
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
        if self.direct_latents:
            # Retain all information in the original proposal states. A fixed
            # nonzero scale matches the small decoder's initialized PLAN range
            # without a vocabulary projection or trainable probability table.
            z=proposal_latents*self.direct_plan_scale
            expectation_cache=adapter_cache=None
        else:
            with block_proposal_detail_scope('block_proposal.expectation.forward'):
                z,expectation_cache,adapter_cache=self._distribution_plan(proposal_logits)
        probability_norm_cache=None
        if self.probability_plan_norm is not None:
            z,probability_norm_cache=self.probability_plan_norm.forward(z)
            z=z*self.direct_plan_scale
        if plan_condition=='neutral':
            # Mean of all vocabulary embeddings = uniform-proposal expectation.
            # No additional full [N,K,V] logits/probability tensor is needed.
            neutral=xp.mean(self.model.embedding.W.data.astype(xp.float32),axis=0)
            z=xp.broadcast_to(neutral.astype(z.dtype),z.shape)
        elif plan_condition in ('shuffled','external'):
            if plan_override.shape!=z.shape:
                raise ValueError('external plan embedding shape differs from proposal expectation')
            z=plan_override
        if self.direct_width:
            plan=z
            plan_cache=None
        else:
            plan,plan_cache=self.plan_projection.forward(z)
        # Previous token IDs are read only, never labels of the same/later position.
        safe_ids=xp.maximum(shifted,0)
        embedding=self.model.embedding.W.data[safe_ids]
        if self.direct_width:
            real=embedding
            token_cache=None
        else:
            real,token_cache=self.token_projection.forward(embedding)
        real=xp.where((shifted>=0)[:,:,None],real,self.bos.data[None,None,:])
        plan=plan + self.plan_positions.data[None,:,:]
        real=real + self.dec_positions.data[None,:,:]
        boundary_cache=None
        if self.boundary_projection is not None:
            if boundary_h is None or boundary_h.shape != (n,self.model.config.d_model):
                raise ValueError('boundary conditioning requires pre-block h [G,model_width]')
            projected,boundary_cache=self.boundary_projection.forward(
                boundary_h.astype(self.model.dtype,copy=False)
                if self.low_precision else boundary_h)
            real=real+projected[:,None,:]
        x=xp.concatenate((plan,real),axis=1)
        context_cache=None
        if context_memory is not None:
            # 0080BJ: M <= 128 causal main-trunk states are K/V only.
            with block_proposal_detail_scope('block_proposal.context_attention.forward'):
                x,context_cache=context_attention_forward(x,context_memory,context_valid)
        elif context_valid is not None:
            raise ValueError('context validity supplied without memory')
        cache_layers=[]
        read_cache=None
        with block_proposal_detail_scope('block_proposal.small_decoder.layers.forward'):
            for i, layer in enumerate(self.layers):
                x,c=layer.forward(x,self.allow)
                cache_layers.append(c)
                if self.vocab_read is not None and i == self.soft_interface_config.read_layer:
                    # Query after a causally masked layer: DEC_j sees y_<j,
                    # PLAN sees only proposals from the pre-block boundary.
                    with block_proposal_detail_scope('block_proposal.vocab_read.forward'):
                        residual,read_cache=self.vocab_read.forward(
                            x[:,k:,:],proposal_logits)
                    # Build an independent tensor: cached query inputs must
                    # not be mutated by the residual update.
                    x=xp.concatenate((x[:,:k,:],x[:,k:,:]+residual),axis=1)
        decoded=x[:,k:,:]
        normalized,norm_cache=self.final_norm.forward(decoded)
        normalized=normalized.astype(self.model.dtype,copy=False) if self.low_precision else normalized
        if self.direct_width:
            up=normalized
            up_cache=None
        else:
            up,up_cache=self.up.forward(normalized)
        with block_proposal_detail_scope('block_proposal.correction_vocab.forward'):
            delta,output_cache=self.model.output_proj.forward(up)
            if self.direct_width:
                delta=delta*self.direct_correction_scale
        # Main distribution remains a soft and correctable additive prior.
        final_logits=proposal_logits+delta
        return final_logits,DecoderCache(
            shifted=shifted, expectation_cache=expectation_cache,
            plan_linear_cache=plan_cache,token_linear_cache=token_cache,
            layer_caches=tuple(cache_layers),norm_cache=norm_cache,
            up_cache=up_cache,output_cache=output_cache,block_size=k,
            vocab_read_cache=read_cache,boundary_cache=boundary_cache,
            probability_adapter_cache=adapter_cache,
            probability_plan_norm_cache=probability_norm_cache,
            context_attention_cache=context_cache)

    def backward(self,d_final_logits,cache):
        """Return prior-logit gradient and, for 0080BD, direct latent gradient.

        Older interfaces return the residual-prior identity gradient plus
        the differentiable probability-embedding feedback gradient.
        """
        k=cache.block_size
        if d_final_logits.ndim!=3 or d_final_logits.shape[1]!=k:
            raise ValueError('expected final logits gradient [N,K,V]')
        df=d_final_logits.astype(self.model.dtype,copy=False) if self.low_precision else d_final_logits
        with block_proposal_detail_scope('block_proposal.correction_vocab.backward'):
            d_up=self.model.output_proj.backward(
                df*self.direct_correction_scale if self.direct_width else df,
                cache.output_cache)
        d_norm=(d_up if self.direct_width else
                self.up.backward(d_up,cache.up_cache))
        d_decode=self.final_norm.backward(d_norm.astype(xp.float32,copy=False)
                                               if self.low_precision else d_norm,
                                            cache.norm_cache)
        d_hidden=xp.zeros((d_decode.shape[0],2*k,self.width),dtype=xp.float32 if self.low_precision else d_decode.dtype)
        d_hidden[:,k:,:]=d_decode
        d_prop_from_read=None
        with block_proposal_detail_scope('block_proposal.small_decoder.layers.backward'):
            for i in reversed(range(len(self.layers))):
                if self.vocab_read is not None and i == self.soft_interface_config.read_layer:
                    if cache.vocab_read_cache is None:
                        raise ValueError('missing full-vocabulary read tape')
                    with block_proposal_detail_scope('block_proposal.vocab_read.backward'):
                        d_read_state,d_prop_from_read=self.vocab_read.backward(
                            d_hidden[:,k:,:],cache.vocab_read_cache)
                    # Derivative of residual update is the identity PLUS
                    # the read-query branch; leave all PLAN rows unchanged.
                    d_hidden[:,k:,:]+=d_read_state
                d_hidden=self.layers[i].backward(d_hidden,cache.layer_caches[i])
        d_context=None
        if cache.context_attention_cache is not None:
            with block_proposal_detail_scope('block_proposal.context_attention.backward'):
                d_hidden,d_context=context_attention_backward(d_hidden,cache.context_attention_cache)
        d_plan=d_hidden[:,:k,:]
        d_real=d_hidden[:,k:,:]
        d_boundary=None
        if self.boundary_projection is not None:
            if cache.boundary_cache is None:
                raise ValueError('missing boundary projection tape')
            d_boundary=self.boundary_projection.backward(
                xp.sum(d_real,axis=1).astype(self.model.dtype,copy=False)
                if self.low_precision else xp.sum(d_real,axis=1),
                cache.boundary_cache)
        self.plan_positions.grad+=xp.sum(d_plan.astype(self.plan_positions.grad.dtype,copy=False),axis=0)
        self.dec_positions.grad+=xp.sum(d_real.astype(self.dec_positions.grad.dtype,copy=False),axis=0)
        d_z=(d_plan if self.direct_width else
             self.plan_projection.backward(d_plan.astype(self.model.dtype,copy=False)
                                                  if self.low_precision else d_plan,
                                            cache.plan_linear_cache))
        valid_previous=cache.shifted>=0
        self.bos.grad+=xp.sum(xp.where(valid_previous[:,:,None],0,d_real),axis=(0,1))
        valid_d_real=xp.where(valid_previous[:,:,None],d_real,0)
        d_tokens=(valid_d_real if self.direct_width else
                  self.token_projection.backward(
                      valid_d_real.astype(self.model.dtype,copy=False)
                      if self.low_precision else valid_d_real,
                      cache.token_linear_cache))
        # Use only previous genuine tokens; padding and BOS must contribute
        # exactly zero to the shared vocabulary embedding gradient.
        with block_proposal_detail_scope('block_proposal.token_embedding_scatter'):
            # 0080O10 opt-in: CuPy Boolean-indexing a GPU array creates a
            # variable-size result and may synchronize to determine its length.
            # Keep N*K indices instead, routing BOS/padding to ID 0 with an
            # *exactly zero* gradient. There is no CPU-dependent output shape,
            # and valid-token indices retain their original order.
            fixed_scatter=(os.environ.get('MINI_LLM_BLOCK_DECODER_FIXED_SCATTER', '0')
                           .strip().lower() not in ('0', 'false', 'off', 'no', ''))
            if fixed_scatter:
                ids=xp.maximum(cache.shifted, 0).reshape(-1)
                values=xp.where(valid_previous[:,:,None], d_tokens, 0)
                xp.add.at(self.model.embedding.W.grad,
                          ids, values.reshape(-1, d_tokens.shape[-1]))
            else:
                xp.add.at(self.model.embedding.W.grad,
                          cache.shifted[valid_previous],
                          d_tokens[valid_previous])
        if self.probability_plan_norm is not None:
            if cache.probability_plan_norm_cache is None:
                raise ValueError('missing direct probability norm tape')
            d_z=self.probability_plan_norm.backward(
                d_z*self.direct_plan_scale,cache.probability_plan_norm_cache)
        if self.direct_latents:
            # No vocabulary-size softmax VJP! The PLAN gradient travels
            # directly to HorizonProposalHead.backward and its latent states.
            d_latents=d_z*self.direct_plan_scale
            return ((d_final_logits,d_boundary,d_latents,d_context)
                    if d_context is not None else (d_final_logits,d_boundary,d_latents))
        with block_proposal_detail_scope('block_proposal.expectation.backward'):
            if self.probability_expectation is None:
                d_prop_from_plan=self.expectation.backward(d_z,cache.expectation_cache)
            else:
                if self.direct_probability:
                    d_raw=d_z
                else:
                    if cache.probability_adapter_cache is None:
                        raise ValueError('missing probability adapter tape')
                    d_raw=self.probability_adapter.backward(
                        d_z.astype(self.model.dtype,copy=False) if self.low_precision else d_z,
                        cache.probability_adapter_cache)
                d_prop_from_plan=self.probability_expectation.backward(
                    d_raw,cache.expectation_cache)
        # 0080O5 opt-in: the expectation's *owned* FP32 probability buffer
        # becomes the VJP output and is no longer needed as P. Accumulate the
        # identity-gradient in that same buffer, avoiding one more [N,K,V]
        # allocation. Only the explicitly destructive CUDA mode takes this
        # path; the reference and other ablations retain the old semantics.
        if (os.environ.get('MINI_LLM_INPLACE_EXPECTATION_VJP', '0').strip().lower()
                not in ('0', 'false', 'off', 'no', '')
                and getattr(d_prop_from_plan, 'dtype', None) == xp.float32
                and d_prop_from_plan is cache.expectation_cache.probs):
            xp.add(d_prop_from_plan, d_final_logits, out=d_prop_from_plan)
            if d_prop_from_read is not None:
                xp.add(d_prop_from_plan,d_prop_from_read,out=d_prop_from_plan)
            return (d_prop_from_plan,d_boundary) if d_boundary is not None else d_prop_from_plan
        result=d_final_logits + d_prop_from_plan
        if d_prop_from_read is not None:
            result+=d_prop_from_read
        return (result,d_boundary) if d_boundary is not None else result




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
    group scheduling (0080D). In 0080BD the decoder receives the proposal
    head's K latent predictions, not raw future target states.
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
                trusted_block_schedule=False,context_memory=None,context_valid=None):
        with block_proposal_detail_scope('block_proposal.proposal_head.forward'):
            if self.decoder.direct_latents:
                lp,cp,latents=self.proposal_head.forward(main_boundaries,return_latents=True)
            else:
                lp,cp=self.proposal_head.forward(main_boundaries)
                latents=None
        lf,cd=self.decoder.forward(lp,targets,valid,proposal_latents=latents,
            plan_condition=plan_condition,prefix_condition=prefix_condition,
            plan_override=plan_override,boundary_h=main_boundaries,
            trusted_block_schedule=trusted_block_schedule,
            context_memory=context_memory,context_valid=context_valid)
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
        with block_proposal_detail_scope('block_proposal.proposal_ce.forward'):
            prop_loss,prop_cache=cross_entropy_forward(lp.copy() if copy_ce else lp,
                            safe_targets,loss_mask=valid,
                            return_device_loss=defer_metrics,
                            normalizer_count=n_unique, return_row_losses=True)
        # 0080O6: final_logits is a new, exclusively owned residual-addition
        # tensor. Unlike lp it is not cached for any later gradient pathway;
        # opt-in CE may safely overwrite it with FP32 probabilities/gradients.
        # The loss op still checks dtype, ownership and contiguity itself.
        reuse_final_fp32=(os.environ.get('MINI_LLM_INPLACE_FP32_FINAL_CE','0')
                          .strip().lower() not in ('0','false','off','no',''))
        with block_proposal_detail_scope('block_proposal.final_ce.forward'):
            final_loss,final_cache=cross_entropy_forward(lf,safe_targets,loss_mask=valid,
                            return_device_loss=defer_metrics,
                            normalizer_count=n_unique, return_row_losses=True,
                            reuse_fp32_logits=reuse_final_fp32)
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
        with block_proposal_detail_scope('block_proposal.final_ce.backward'):
            d_final=cross_entropy_backward(cache.final_ce_cache)
        if cache.final_ce_cache.get('consume_fp32_probs',False):
            # 0080O6: avoid one more full-vocab gradient allocation when
            # applying the accumulation weight to the exclusively owned CE
            # output. The proposal/other loss paths retain their semantics.
            xp.multiply(d_final,float(scale),out=d_final)
        else:
            d_final=d_final*float(scale)
        with block_proposal_detail_scope('block_proposal.small_decoder.backward'):
            decoder_grad=self.decoder.backward(d_final,cache.decoder_cache)
            if self.decoder.direct_latents:
                if len(decoder_grad)==4:
                    d_prop,d_boundary,d_latents,d_context=decoder_grad
                else:
                    d_prop,d_boundary,d_latents=decoder_grad
                    d_context=None
            else:
                d_prop,d_boundary=(decoder_grad if isinstance(decoder_grad,tuple)
                                   else (decoder_grad,None))
                d_latents=None
                d_context=None
        with block_proposal_detail_scope('block_proposal.proposal_ce.backward'):
            d_prop+=cross_entropy_backward(cache.proposal_ce_cache)*(self.lambda_proposal*float(scale))
        with block_proposal_detail_scope('block_proposal.proposal_head.backward'):
            d_h=self.proposal_head.backward(d_prop,cache.head_cache,d_latents=d_latents)
            if d_boundary is not None:
                # Accumulate in FP32 before the existing single main-body
                # scatter. No second Transformer backward pass is introduced.
                d_h=(d_h.astype(xp.float32,copy=False)+
                     d_boundary.astype(xp.float32,copy=False)).astype(d_h.dtype)
            return (d_h,d_context) if d_context is not None else d_h

@dataclass
class PrefixDecoderGenerationState:
    """One immutable PLAN prefix with mutable per-layer DEC KV, one block."""
    proposal_logits: object
    keys: list
    values: list
    next_position: int = 0
    boundary_context: object = None


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


def _decoder_prepare_plan(self,proposal_logits,*,boundary_h=None,proposal_latents=None):
    if int(os.environ.get('MINI_LLM_BLOCK_DECODER_CONTEXT_TOKENS','0')):
        raise NotImplementedError('0080BJ contextual KV inference cache is not implemented; do not generate without the same context')
    """Run all K PLAN tokens once, caching their learned K/V at every layer."""
    if proposal_logits.ndim!=3 or proposal_logits.shape[1]!=self.block_size:
        raise ValueError('proposal logits must be [B,K,V]')
    if self.direct_latents:
        if proposal_latents is None or proposal_latents.shape != (proposal_logits.shape[0],self.block_size,self.width):
            raise ValueError('0080BD cached PLAN requires proposal_latents [B,K,d_model]')
        z=proposal_latents*self.direct_plan_scale
    else:
        if proposal_latents is not None:
            raise ValueError('proposal_latents are only accepted in 0080BD mode')
        z,_,_=self._distribution_plan(proposal_logits)
    if self.probability_plan_norm is not None:
        z,_=self.probability_plan_norm.forward(z)
        z=z*self.direct_plan_scale
    if self.direct_width:
        plan=z
    else:
        plan,_=self.plan_projection.forward(z)
    plan=plan+self.plan_positions.data[None,:,:]
    keys=[];values=[]
    allow=xp.ones((self.block_size,self.block_size),dtype=xp.bool_)
    for layer in self.layers:
        plan,cache=layer.forward(plan,allow)
        keys.append(cache[1]['k'].copy())
        values.append(cache[1]['v'].copy())
    context=None
    if self.boundary_projection is not None:
        if boundary_h is None or boundary_h.shape != (proposal_logits.shape[0],self.model.config.d_model):
            raise ValueError('cached decoder requires original causal boundary h')
        context,_=self.boundary_projection.forward(
            boundary_h.astype(self.model.dtype,copy=False) if self.low_precision else boundary_h)
    return PrefixDecoderGenerationState(proposal_logits=proposal_logits,keys=keys,
                                        values=values,boundary_context=context)


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
        x=embedded if self.direct_width else self.token_projection.forward(embedded)[0]
    x=x+self.dec_positions.data[None,j,:]
    if self.boundary_projection is not None:
        if state.boundary_context is None:
            raise ValueError('missing cached causal boundary state')
        x=x+state.boundary_context
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
        if self.vocab_read is not None and i == self.soft_interface_config.read_layer:
            # Correctness-first cached inference: retain the complete K-source
            # bank and original target-slot bias.  Noncurrent DEC slots are
            # inert placeholders: PLAN and prior computations never consume
            # them, and only the read for slot j is applied here.  This may
            # perform K query reads per generated token; a selected-query CUDA
            # kernel is a separate performance optimization.
            dec_states=xp.zeros((n,self.block_size,self.width),dtype=x.dtype)
            dec_states[:,j,:]=x
            residual,_=self.vocab_read.forward(dec_states,state.proposal_logits)
            x=x+residual[:,j,:]
    norm,_=self.final_norm.forward(x)
    norm=norm.astype(self.model.dtype,copy=False) if self.low_precision else norm
    up=norm if self.direct_width else self.up.forward(norm)[0]
    delta,_=self.model.output_proj.forward(up)
    if self.direct_width:
        delta=delta*self.direct_correction_scale
    state.next_position+=1
    return state.proposal_logits[:,j,:]+delta


# These cached correctness operations use the exact trainable parameters in
# BlockPrefixDecoder, without an independent shadow inference model.
BlockPrefixDecoder.prepare_plan_cache = _decoder_prepare_plan
BlockPrefixDecoder.cached_step = _decoder_cached_step
