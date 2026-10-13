"""0080F baseline strict checkpoint/resume for the two-stage architecture.

This serializer reuses the original parameter/AdamW checkpoint format while
requiring new architecture metadata, all new trainable tensors, tokenizer and
data provenance. Legacy AR and 0079 nine-pass checkpoints are never silently
accepted as 0080 models.
"""

from pathlib import Path
import json

from mini_llm.backend import xp
from mini_llm.checkpoint import load_checkpoint,save_checkpoint

ARCHITECTURE='probabilistic_block_proposal_prefix_decoder_v1'


def architecture_metadata(net):
    head,dec=net.proposal_head,net.decoder
    mc=head.model.config.memory_context
    c=head.model.config
    return dict(type=ARCHITECTURE,
        block_size=int(head.block_size),lambda_proposal=float(net.lambda_proposal),
        proposal_hidden_width=int(head.W_back.shape[0]),
        proposal_residual_scale=float(head.residual_scale),
        decoder_layers=len(dec.layers),decoder_width=int(dec.width),
        decoder_heads=int(dec.layers[0].attn.n_q_heads),
        decoder_ff_width=int(dec.layers[0].ffn.W_gate.shape[1]),
        model_config=dict(vocab_size=int(c.vocab_size),d_model=int(c.d_model),
            n_layers=int(c.n_layers),n_q_heads=int(c.n_q_heads),
            n_kv_heads=int(c.n_kv_heads),d_head=int(c.d_head),
            d_ff=int(c.d_ff),n_experts=int(c.n_experts),top_k=int(c.top_k),
            context_length=int(c.context_length),dtype=str(c.dtype),
            active_layers=int(head.model.active_layers),
            attention_layout=repr(c.attention_layers),
            memory_enabled=bool(mc.enabled),memory_integration=mc.integration_mode,
            memory_length=int(mc.memory_length),recent_length=int(mc.recent_length),
            target_length=int(mc.target_length),retrieved_length=int(mc.retrieved_length)))


def save_two_stage_checkpoint(path,net,*,optimizer=None,training_step=0,
                              supervised_tokens=0,tokenizer_hash=None,
                              data_provenance=None,rng_state=None):
    params=net.parameters()
    if len(params)!=len({id(p) for p in params}) or len(params)!=len({p.name for p in params}):
        raise ValueError('duplicate parameter in 0080 model')
    opt_state=None
    if optimizer is not None:
        if [id(p) for p in optimizer.parameters]!=[id(p) for p in params]:
            raise ValueError('optimizer parameters/order must match two-stage model')
        opt_state={'step':int(optimizer.step_index),
                   'm':{p.name:v for p,v in zip(params,optimizer.m)},
                   'v':{p.name:v for p,v in zip(params,optimizer.v)},
                   'master_weights':{p.name:v for p,v in zip(params,optimizer.master_weights)}}
    training_state=dict(architecture=architecture_metadata(net),
        step=int(training_step), supervised_tokens=int(supervised_tokens),
        tokenizer_hash=tokenizer_hash,data_provenance=data_provenance,rng_state=rng_state,
        optimizer_step=None if optimizer is None else int(optimizer.step_index),
        optimizer_options=None if optimizer is None else {
            x:float(getattr(optimizer,x)) for x in ('lr','beta1','beta2','eps','weight_decay')})
    save_checkpoint(path,{p.name:p.data for p in params},optimizer_state=opt_state,
                    training_state=training_state)


def load_two_stage_checkpoint(path,net,*,optimizer=None,
                              expected_tokenizer_hash=None,
                              expected_data_provenance=None):
    path=Path(path)
    manifest=path/'training_state.json'
    if not manifest.is_file():
        raise ValueError('missing 0080 training metadata')
    with manifest.open(encoding='utf8') as f:
        training_state=json.load(f)
    if training_state.get('architecture')!=architecture_metadata(net):
        raise ValueError('incompatible 0080 architecture or not an 0080 checkpoint')
    if expected_tokenizer_hash is not None and training_state.get('tokenizer_hash')!=expected_tokenizer_hash:
        raise ValueError('tokenizer hash mismatch')
    if expected_data_provenance is not None and training_state.get('data_provenance')!=expected_data_provenance:
        raise ValueError('data provenance mismatch')
    params=net.parameters()
    saved,saved_opt,_=load_checkpoint(path,skip_optimizer=(optimizer is None),skip_training=True)
    if set(saved)!={p.name for p in params}:
        raise ValueError('incomplete/mismatched 0080 parameter list')
    for p in params:
        if saved[p.name].shape!=p.shape:
            raise ValueError(f'parameter shape mismatch: {p.name}')
    if optimizer is not None:
        if saved_opt is None:
            raise ValueError('no AdamW optimizer state to resume')
        if [id(p) for p in optimizer.parameters]!=[id(p) for p in params]:
            raise ValueError('optimizer parameter order differs')
        for name in ('m','v','master_weights'):
            if set(saved_opt.get(name,{}))!={p.name for p in params}:
                raise ValueError(f'incomplete optimizer component: {name}')
        for key in ('lr','beta1','beta2','eps','weight_decay'):
            options=training_state.get('optimizer_options') or {}
            if key not in options or float(options[key])!=float(getattr(optimizer,key)):
                raise ValueError(f'incompatible optimizer configuration: {key}')
    for p in params:
        p.data[...]=saved[p.name]
    for block in net.proposal_head.model.blocks:
        if hasattr(block,'refresh_compute_buffers'):
            block.refresh_compute_buffers()
    for layer in net.decoder.layers:
        if hasattr(layer.attn,'refresh_compute_buffers'):
            layer.attn.refresh_compute_buffers()
    if optimizer is not None:
        for i,p in enumerate(params):
            if not optimizer.restore_master_weight(i,saved_opt['master_weights'][p.name]) or not optimizer.restore_moments(
                    i,saved_opt['m'][p.name],saved_opt['v'][p.name]):
                raise ValueError(f'incompatible AdamW tensors: {p.name}')
        optimizer.step_index=int(saved_opt['step'])
    return training_state
