"""0080F baseline strict checkpoint/resume for the two-stage architecture.

This serializer reuses the original parameter/AdamW checkpoint format while
requiring new architecture metadata, all new trainable tensors, tokenizer and
data provenance. Legacy AR and 0079 nine-pass checkpoints are never silently
accepted as 0080 models.
"""

from pathlib import Path
import json
from dataclasses import asdict

from mini_llm.backend import xp
from mini_llm.checkpoint import load_checkpoint,save_checkpoint
from mini_llm.optim.hybrid_muon import HybridMuonAdamW

ARCHITECTURE='probabilistic_block_proposal_prefix_decoder_v1'
ARCHITECTURE_V2='probabilistic_block_proposal_prefix_decoder_full_soft_v2'
ARCHITECTURE_V3='probabilistic_block_proposal_prefix_decoder_learned_probability_v3'
ARCHITECTURE_V4='probabilistic_block_proposal_prefix_decoder_direct_width_v4'
ARCHITECTURE_V5='probabilistic_block_proposal_prefix_decoder_direct_latents_v5'


def architecture_metadata(net):
    head,dec=net.proposal_head,net.decoder
    # Legacy v1 metadata is byte-for-byte compatible and has no new keys.
    interface=getattr(dec,'soft_interface_config',None)
    is_v2=interface is not None and not interface.is_v1_baseline
    is_v3=is_v2 and interface.mode == 'probability_embedding'
    is_v4=is_v2 and interface.mode == 'probability_embedding_direct'
    is_v5=is_v2 and interface.mode == 'proposal_latents_direct'
    mc=head.model.config.memory_context
    c=head.model.config
    architecture=dict(type=(ARCHITECTURE_V5 if is_v5 else ARCHITECTURE_V4 if is_v4 else ARCHITECTURE_V3 if is_v3 else ARCHITECTURE_V2) if is_v2 else ARCHITECTURE,
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
    if is_v2:
        config=asdict(interface)
        if not (is_v3 or is_v4 or is_v5):
            # The new config field must NOT alter historical v2 metadata.
            config.pop('probability_embedding_dim')
        architecture['soft_interface'] = config
        if is_v3 or is_v4:
            architecture['probability_embedding'] = dict(
                dimension=int(interface.probability_embedding_dim),
                table='independent_trainable',
                adapter='none' if is_v4 else 'probability_to_model_width',
                full_vocabulary=True)
        if is_v5:
            architecture['direct_latent_decoder'] = dict(
                hidden_width=int(dec.width),input_projections=False,
                output_projection=False,proposal_latent_dim=int(c.d_model),
                plan_normalization='none',plan_scale=float(dec.direct_plan_scale),
                correction_scale=float(dec.direct_correction_scale),
                full_vocabulary_probability_embedding=False)
        if is_v4:
            architecture['direct_decoder'] = dict(
                hidden_width=int(dec.width),input_projections=False,
                output_projection=False,probability_norm=True,
                plan_scale=float(dec.direct_plan_scale),
                correction_scale=float(dec.direct_correction_scale))
        if dec.vocab_read is not None:
            architecture['vocab_read_execution'] = dict(
                algorithm=dec.vocab_read.all_source_algorithm,
                vocab_tile=dec.vocab_read.vocab_tile,
                block_tile=dec.vocab_read.block_tile)
    return architecture


def warmstart_v2_from_v1(path,net,*,expected_tokenizer_hash):
    """Strict weights-only transfer of *all* matching v1 tensors, including decoder.

    The v2 modules retain their zero-output initialization. No old optimizer
    state is loaded. Reject mismatched architecture, missing tensors and a
    tokenizer fingerprint mismatch before modifying ANY target parameters.
    """
    from mini_llm.checkpoint import initialize_matching_parameters
    path=Path(path)
    with (path/'training_state.json').open(encoding='utf8') as handle:
        state=json.load(handle)
    source=state.get('architecture') or {}
    expected=architecture_metadata(net)
    if expected['type'] not in (ARCHITECTURE_V2,ARCHITECTURE_V3) or source.get('type')!=ARCHITECTURE:
        raise ValueError('weights-only migration requires v1 source and v2/v3 destination')
    base={k:v for k,v in expected.items() if k not in
          ('soft_interface','vocab_read_execution','probability_embedding')}
    base['type']=ARCHITECTURE
    if base != source:
        raise ValueError('v1/v2 trunk, proposal or decoder architecture mismatch')
    if (not expected_tokenizer_hash or
            state.get('tokenizer_hash')!=expected_tokenizer_hash):
        raise ValueError('v1 warm start requires identical, verified tokenizer SHA256')
    extra=set()
    if net.decoder.vocab_read is not None:
        extra.update(p.name for p in net.decoder.vocab_read.parameters())
    if net.decoder.boundary_projection is not None:
        extra.update(p.name for p in net.decoder.boundary_projection.parameters())
    if net.decoder.probability_embedding is not None:
        extra.add(net.decoder.probability_embedding.name)
        if net.decoder.probability_adapter is not None:
            extra.update(p.name for p in net.decoder.probability_adapter.parameters())
    old_params=[p for p in net.parameters() if p.name not in extra]
    expected_names={p.name for p in old_params}
    saved_names={p.stem for p in (path/'model_params').glob('*.npy')}
    if saved_names!=expected_names:
        raise ValueError('v1 source incomplete/unexpected parameter names: '
                         +repr(sorted(expected_names ^ saved_names)[:8]))
    # Reject dimension mismatches before mutating parameters.
    import numpy as np
    for parameter in old_params:
        filename=path/'model_params'/f'{parameter.name}.npy'
        with filename.open('rb') as stream:
            version=np.lib.format.read_magic(stream)
            if version==(1,0):
                shape,_,_=np.lib.format.read_array_header_1_0(stream)
            elif version in ((2,0),(3,0)):
                shape,_,_=np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError('unsupported source checkpoint numpy header')
        if tuple(shape)!=tuple(parameter.shape):
            raise ValueError('v1 parameter shape mismatch: '+parameter.name)
    loaded,missing,total=initialize_matching_parameters(path,old_params)
    if missing or set(loaded)!=expected_names:
        raise ValueError('incomplete v1 migration')
    for block in net.proposal_head.model.blocks:
        if hasattr(block,'refresh_compute_buffers'):
            block.refresh_compute_buffers()
    for layer in net.decoder.layers:
        if hasattr(layer.attn,'refresh_compute_buffers'):
            layer.attn.refresh_compute_buffers()
    return dict(loaded=len(loaded),elements=total,initialized_v2=sorted(extra),
                source=str(path),tokenizer_hash=expected_tokenizer_hash)


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
        opt_state=(optimizer.checkpoint_state()
                   if isinstance(optimizer, HybridMuonAdamW) else {
            'step':int(optimizer.step_index),
            'm':{p.name:v for p,v in zip(params,optimizer.m)},
            'v':{p.name:v for p,v in zip(params,optimizer.v)},
            'master_weights':{p.name:v for p,v in zip(params,optimizer.master_weights)}})
    training_state=dict(architecture=architecture_metadata(net),
        step=int(training_step), supervised_tokens=int(supervised_tokens),
        tokenizer_hash=tokenizer_hash,data_provenance=data_provenance,rng_state=rng_state,
        optimizer_step=None if optimizer is None else int(optimizer.step_index),
        optimizer_options=None if optimizer is None else {
            x:float(getattr(optimizer,x)) for x in ('lr','beta1','beta2','eps','weight_decay')})
    training_state['optimizer_kind'] = ('hybrid-muon' if isinstance(optimizer, HybridMuonAdamW)
                                         else 'adamw')
    if isinstance(optimizer, HybridMuonAdamW):
        training_state['muon_hyperparameters'] = {
            'momentum':optimizer.momentum, 'ns_steps':optimizer.ns_steps,
            'matmul_precision':optimizer.matmul_precision,
            'group_names':sorted(optimizer.muon_names)}
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
            raise ValueError('no optimizer state to resume')
        if [id(p) for p in optimizer.parameters]!=[id(p) for p in params]:
            raise ValueError('optimizer parameter order differs')
        saved_kind=training_state.get('optimizer_kind', 'adamw')
        hybrid=isinstance(optimizer,HybridMuonAdamW)
        if saved_kind!=('hybrid-muon' if hybrid else 'adamw'):
            raise ValueError('0080 optimizer kind mismatch')
        expected={p.name for p in params}
        if hybrid:
            muon=set(optimizer.muon_names)
            adam=expected-muon
            groups={'m':adam,'v':adam,'muon_m':muon,'master_weights':expected}
            settings=training_state.get('muon_hyperparameters') or {}
            if settings!={'momentum':optimizer.momentum,
                          'ns_steps':optimizer.ns_steps,
                          'matmul_precision':optimizer.matmul_precision,
                          'group_names':sorted(muon)}:
                raise ValueError('incompatible Muon optimizer hyperparameters or groups')
        else:
            groups={name:expected for name in ('m','v','master_weights')}
        for name, names in groups.items():
            if set(saved_opt.get(name,{}))!=names:
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
        if isinstance(optimizer,HybridMuonAdamW):
            optimizer.restore_checkpoint(saved_opt,from_hybrid=True)
        else:
            for i,p in enumerate(params):
                if not optimizer.restore_master_weight(i,saved_opt['master_weights'][p.name]) or not optimizer.restore_moments(
                        i,saved_opt['m'][p.name],saved_opt['v'][p.name]):
                    raise ValueError(f'incompatible AdamW tensors: {p.name}')
            optimizer.step_index=int(saved_opt['step'])
    return training_state
