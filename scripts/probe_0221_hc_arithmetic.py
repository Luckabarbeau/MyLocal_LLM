"""Replay token-0/layer-0 HC using canonical weights; production is unchanged."""
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from probe_0218_hc_identity import MODEL, provenance, associate, validate_layout
from probe_0219_activation_arithmetic import compile_reference, run_tile
from probe_0220_layer_evolution import metrics


def bf16_roundtrip(x):
    """Round finite FP32 to BF16 nearest-even, retaining decoded FP32 values."""
    x = np.asarray(x,dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('finite inputs required')
    bits = x.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def sigmoid(x):
    return np.exp(-np.logaddexp(np.float32(0),-x))


def replay(streams, gamma, down, up, epsilon, rounded=False):
    norm = streams / np.sqrt(np.mean(streams*streams,axis=-1,keepdims=True,dtype=np.float32)+np.float32(epsilon)) * gamma
    flat = norm.reshape(-1)
    lo = down @ (bf16_roundtrip(flat) if rounded else flat)
    lo *= np.float32(.25)
    lo *= sigmoid(lo)
    gate = up @ (bf16_roundtrip(lo) if rounded else lo)
    mixed = np.mean(norm*sigmoid(gate.reshape(streams.shape)),axis=0,dtype=np.float32)
    return mixed, flat, lo


def main():
    from mini_llm.external.gguf_store import MultiShardTensorStore
    from mini_llm.ops.quantized_weight import PackedGGUFWeight
    docs = ROOT/'docs/qwen4exp'
    output = docs/'0221_hc_arithmetic_v1.json'
    identity_output = docs/'0218_hc_identity_v2.json'
    if output.exists() or identity_output.exists():
        raise FileExistsError('preserve versioned results')
    provenance_result = provenance()
    retained = docs/'diagnostics_0218_v1'
    events = [json.loads(line) for line in (retained/'events.jsonl').read_text().splitlines()]
    ids = associate(events)
    identity = dict(scope='reanalysis of retained A captures; no model rerun', mapping_trusted=True,
                    prior_failed_reporting_artifact_preserved='0218_hc_identity_v1.json',comparisons=[])
    reference = None
    for e in events:
        stage = ids[e['tensor']['ptr']]
        a = np.load(retained/f'{stage}.npy').reshape(-1)
        validate_layout(e,a.size)
        r = np.fromfile(retained/f'eval_{e["eval_occurrence"]}.f32',np.float32)
        identity['comparisons'].append(dict(stage=stage,tensor=e['tensor'],**metrics(a,r)))
        if stage=='hc_attention_read':
            reference = r
    with MultiShardTensorStore.open(MODEL) as store:
        def dense(name):
            d = store.tensors[name]
            raw = store.tensor_bytes(name)
            if d.quant_type=='F32':
                return raw.view('<f4').reshape(d.shape).copy()
            if d.quant_type!='BF16':
                raise ValueError('HC weight is not F32/BF16')
            return (raw.view('<u2').astype(np.uint32)<<16).view(np.float32).reshape(d.shape)
        raw = store.rows('token_embd.weight',1,2)
        embedding = PackedGGUFWeight(raw,(1,2560),'IQ4_XS').dequantize_reference()[0]
        streams = np.tile(embedding,(4,1))
        gamma = dense('blk.0.hc_attn_norm.weight').reshape(4,2560)
        down, up = dense('blk.0.hc_attn_down.weight'), dense('blk.0.hc_attn_up.weight')
        eps = store.metadata['qwen4exp.attention.layer_norm_rms_epsilon']
        full, norm, _ = replay(streams,gamma,down,up,eps,False)
        rounded, _, low = replay(streams,gamma,down,up,eps,True)
        local = np.load(retained/'hc_attention_read.npy')
        with tempfile.TemporaryDirectory(prefix='hc-rounding-') as tmp:
            binary = compile_reference(tmp)
            # Verify BOTH composed replay cast inputs against upstream conversion.
            casts=[]
            for name,x in [('blk.0.hc_attn_down.weight',norm),('blk.0.hc_attn_up.weight',low)]:
                desc=store.tensors[name]
                run_tile(binary,tmp,desc.format.type_id,store.rows(name,0,32),x)
                actual=np.fromfile(str(Path(tmp)/'result')+'.xq',np.float32)
                np.testing.assert_array_equal(actual,bf16_roundtrip(x))
                casts.append(dict(tensor=name,exact_upstream_BF16_roundtrip=True))
        report=dict(scope='composed first HC read only; same original embedding and weights',
            provenance=provenance_result,production_math_changed=False,position=0,token_id=1,layer=0,
            actual_local_vs_replay=metrics(local,full),unrounded_vs_reference=metrics(full,reference),
            BF16_activation_replay_vs_reference=metrics(rounded,reference),
            rounding_effect=metrics(rounded,full),cast_checks=casts,
            explanation='Only the two HC projection inputs are rounded to BF16. Gamma, normalization, nonlinearities and stream combination remain FP32. NumPy and GGML reduction orders still differ.',
            full_graph_equivalence_claim=False)
    for p,data in [(identity_output,identity),(output,report)]:
        with p.open('x') as f: json.dump(data,f,indent=2,allow_nan=False)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
