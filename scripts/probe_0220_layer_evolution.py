"""C: bounded identity-based layer evolution; never modify production or weights."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import time
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_0218_hc_identity import ROOT, COPY, MODEL, provenance, bounded
from probe_0210_qwen4exp_reference import child_model, logits_metrics

STAGES = ('hc_attention_read', 'post_mixer', 'hc_ffn_read', 'post_moe')


def nearest_weights(event, root):
    nodes = {n['ptr']: n for n in event['nodes']}
    todo, seen, found = [root], set(), set()
    while todo:
        ptr = todo.pop()
        if not ptr or ptr in seen:
            continue
        seen.add(ptr)
        if len(seen) > 512:
            raise ValueError('ancestry bound exceeded')
        if ptr not in nodes:
            raise ValueError('incomplete bounded ancestry')
        n = nodes[ptr]
        if re.fullmatch(r'blk\.\d+\.hc_(attn|ffn)_norm\.weight', n['name']) or n['name'] == 'output_hc_norm.weight':
            found.add(n['name'])
        if n['op'] in ('RMS_NORM', 'DSV4_HC_POST') or (ptr != root and n['name'].startswith('hc_mixed-')):
            continue
        todo.extend(n['src'])
    if len(found) != 1:
        raise ValueError(f'ambiguous nearest norm ancestry: {sorted(found)}')
    return found.pop()


def classify(event):
    t = event['tensor']
    name = t['name']
    if name == 'result_norm':
        if nearest_weights(event, t['ptr']) != 'output_hc_norm.weight':
            raise ValueError('wrong output norm')
        return -1, 'hc_output_read'
    match = re.fullmatch(r'(hc_mixed|hc_combine|l_last)-(\d+)', name)
    if not match or not 0 <= int(match[2]) < 48:
        raise ValueError('invalid layer boundary name')
    layer = int(match[2])
    if match[1] == 'hc_mixed':
        weight = nearest_weights(event, t['ptr'])
        for kind, stage in [('attn', 'hc_attention_read'), ('ffn', 'hc_ffn_read')]:
            if weight == f'blk.{layer}.hc_{kind}_norm.weight':
                return layer, stage
        raise ValueError('read norm layer mismatch')
    if t['op'] != 'DSV4_HC_POST' or len(t['src']) < 3 or not all(t['src'][:3]):
        raise ValueError('writer must have block/residual/gains inputs')
    nodes = {n['ptr']: n for n in event['nodes']}
    block, residual, gains = (nodes[p] for p in t['src'][:3])
    if block['ne'] != [2560,1,1,1] or residual['ne'] != [2560,4,1,1] or gains['ne'] != [4,1,1,1]:
        raise ValueError('writer input shape mismatch')
    kind, stage = ('attn', 'post_mixer') if match[1] == 'hc_combine' else ('ffn', 'post_moe')
    if nearest_weights(event, gains['ptr']) != f'blk.{layer}.hc_{kind}_norm.weight':
        raise ValueError('writer injection norm mismatch')
    return layer, stage


def validate_layout(event, actual, stage):
    t = event['tensor']
    shape = (4,2560) if stage in ('post_mixer','post_moe') else (2560,)
    ne = [2560,4,1,1] if len(shape)==2 else [2560,1,1,1]
    nb = [4,10240,10240*ne[1],10240*ne[1]]
    if t['type'] != 0 or t['ne'] != ne or t['nb'] != nb or actual.shape != shape:
        raise ValueError(f'layout mismatch: {t}, Python {actual.shape}, expected {shape}')
    return actual.reshape(-1, order='C')


def associate(events, complete=True):
    identities, keys = {}, {}
    for event in events:
        layer, stage = classify(event)
        key = (event['position'], layer, stage)
        identity = (event['position'], event['tensor']['ptr'])
        if identity in identities and identities[identity] != key:
            raise ValueError('object identity changed boundary')
        if key in keys and keys[key] != identity:
            raise ValueError('multiple distinct objects for boundary')
        identities[identity], keys[key] = key, identity
    if complete:
        expected = {(p,l,s) for p in range(2) for l in range(48) for s in STAGES}
        expected |= {(p,-1,'hc_output_read') for p in range(2)}
        if set(keys) != expected or len(events) != 386:
            raise ValueError('missing/repeated boundaries; require exactly 386')
    return identities


def metrics(actual, reference):
    a, r = np.asarray(actual,dtype=np.float64), np.asarray(reference,dtype=np.float64)
    if a.shape != r.shape or not a.size:
        raise ValueError('metric shape mismatch')
    nonfinite = int((~np.isfinite(a)).sum() + (~np.isfinite(r)).sum())
    if nonfinite:
        raise ValueError(f'nonfinite elements: {nonfinite}')
    d = a-r
    an, rn = float(np.linalg.norm(a)), float(np.linalg.norm(r))
    largest = np.argsort(-np.abs(r),kind='stable')[:10]
    return dict(max_absolute_error=float(np.abs(d).max()), relative_l2=float(np.linalg.norm(d)/max(rn,1e-30)),
                rms_difference=float(np.sqrt(np.mean(d*d))), reference_rms=float(np.sqrt(np.mean(r*r))),
                actual_norm=an, reference_norm=rn, cosine=float(np.dot(a,r)/max(an*rn,1e-30)), nonfinite=nonfinite,
                largest_reference_elements=[dict(index=int(i), reference=float(r[i]), actual=float(a[i]),
                    relative_error=float(abs(d[i])/max(abs(r[i]),1e-30))) for i in largest])


def reserve(paths):
    if any(p.exists() for p in paths):
        raise FileExistsError('versioned output exists; refusing overwrite')


def plot_png(rows, path):
    """Dependency-free raster: 4 panels, token 0 blue/token 1 red, QSA shaded."""
    w,h = 1000,640
    image = np.full((h,w,3),255,dtype=np.uint8)
    colors = [(25,80,210),(210,45,35)]
    def line(x0,y0,x1,y1,color):
        n = max(abs(x1-x0),abs(y1-y0))+1
        xs = np.rint(np.linspace(x0,x1,n)).astype(int)
        ys = np.rint(np.linspace(y0,y1,n)).astype(int)
        image[np.clip(ys,0,h-1),np.clip(xs,0,w-1)] = color
    for panel,stage in enumerate(STAGES):
        x,y = 55+(panel%2)*490,35+(panel//2)*310
        for l in range(48):
            if (l+1)%4==0:
                xx=x+int(l*430/47)
                image[y:y+240,xx:xx+4] = (225,225,225)
        line(x,y+240,x+440,y+240,(0,0,0)); line(x,y,x,y+240,(0,0,0))
        for p in range(2):
            data=sorted((r for r in rows if r['stage']==stage and r['position']==p),key=lambda r:r['layer'])
            points=[(x+int(r['layer']*430/47),y+240-int(np.clip((np.log10(max(r['relative_l2'],1e-12))+12)/13,0,1)*240)) for r in data]
            for a,b in zip(points,points[1:]): line(*a,*b,colors[p])
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    raw=b''.join(b'\0'+row.tobytes() for row in image)
    legend=b'Description\0Panels: attention HC read, post mixer, FFN HC read, post MoE. X: layers 0..47. Y: relative L2 log10 -12..1. Blue token id 1; red token id 2. Gray QSA layers (3 mod 4).'
    with path.open('xb') as f:
        f.write(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'tEXt',legend)+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--child-model',type=Path)
    parser.add_argument('--version', choices=('v1','v2'), default='v1')
    args=parser.parse_args()
    if args.child_model:
        child_model(MODEL,args.child_model,0)
        return
    docs=ROOT/'docs/qwen4exp'
    report_path=docs/f'0220_layer_evolution_{args.version}.json'
    vectors=docs/f'diagnostics_0220_{args.version}'
    csv_path=docs/'layer_error_evolution.csv'
    png_path=docs/'layer_error_evolution.png'
    reserve([report_path,vectors,csv_path,png_path])
    report=dict(scope='C only',mapping_trusted=False,production_files_modified=False,model_complete=False,
                tokens=[1,2],threads=1,context=32,kv_dtype='F32',expected_boundaries=386,
                omitted_captures='Input streams and branch outputs omitted: buffer lifetime not independently verified.',
                interpretation='Descriptive differences only; jumps do not prove causal quantization attribution. Old occurrence-based HC read 183% is untrusted and not reused.')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MINI_LLM_BACKEND='numpy',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    try:
        report['provenance']=provenance()
        report['shards_before']=[dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(MODEL.parent.glob(MODEL.name.replace('00001','*')))]
        if len(report['shards_before'])!=2:
            raise ValueError('expected original two shards')
        vectors.mkdir()
        reference=vectors/'reference'; actual=vectors/'production'
        reference.mkdir(); actual.mkdir()
        with tempfile.TemporaryDirectory(prefix='layer-evolution-') as tmp:
            tmp=Path(tmp); binary=tmp/'layers'; lib=COPY/'build-cpu/bin'
            report['compile']=bounded(['c++','-std=c++17','-O2',str(ROOT/'tests/fixtures/qwen4exp_layers_reference.cpp'),'-I'+str(COPY/'include'),'-I'+str(COPY/'ggml/include'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lllama','-lggml','-lggml-base','-lggml-cpu','-o',str(binary)],time.monotonic()+120,tmp,'compile',env)
            report['reference_run']=bounded([str(binary),str(MODEL),str(reference)],time.monotonic()+120,tmp,'reference',env)
            report['production_run']=bounded([sys.executable,str(Path(__file__)),'--child-model',str(actual)],time.monotonic()+120,tmp,'production',env)
        events=[json.loads(line) for line in (reference/'events.jsonl').read_text().splitlines()]
        identities=associate(events)
        rows=[]
        for e in events:
            p,l,s=identities[(e['position'],e['tensor']['ptr'])]
            a=np.load(actual/f'p{p}_l{l}_{s}.npy')
            a=validate_layout(e,a,s)
            r=np.fromfile(reference/f'eval_{e["eval_id"]}.f32',dtype=np.float32)
            rows.append(dict(position=p,token_id=p+1,layer=l,stage=s,qsa=l>=0 and (l+1)%4==0,
                             eval_id=e['eval_id'],tensor=e['tensor'],sources=[n for n in e['nodes'] if n['ptr'] in e['tensor']['src']],**metrics(a,r)))
        report['comparisons']=rows
        report['logits']=[logits_metrics(np.load(actual/f'logits_{p}.npy'),np.fromfile(reference/f'logits_{p}.f32',dtype=np.float32)) for p in range(2)]
        old=json.loads((docs/'0210_reference_validation.json').read_text())
        report['cached_old_logits_comparison']=dict(note='Old callback/fusion schedule may differ; raw cached logits unavailable.',old_report_keys=list(old),metrics_changed_not_established=True)
        report['earliest_nonzero']=[next((dict(layer=r['layer'],stage=r['stage'],max_absolute_error=r['max_absolute_error']) for r in rows if r['position']==p and r['max_absolute_error']>0),None) for p in range(2)]
        jumps=[]
        for p in range(2):
            seq=[r for r in rows if r['position']==p]
            jumps.extend(dict(position=p,layer=b['layer'],stage=b['stage'],relative_l2_increase=b['relative_l2']-a['relative_l2']) for a,b in zip(seq,seq[1:]))
        report['largest_jumps']=sorted(jumps,key=lambda r:r['relative_l2_increase'],reverse=True)[:10]
        fields=['position','token_id','layer','stage','qsa','eval_id','max_absolute_error','relative_l2','rms_difference','reference_rms','actual_norm','reference_norm','cosine','nonfinite']
        with csv_path.open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
        plot_png(rows,png_path)
        report['plot']=dict(panel_order=STAGES,token_colors={'1':'blue','2':'red'},y_axis='relative L2 log10 [-12,1]',x_axis='layer 0..47',qsa='gray bands, layers 3 mod 4')
        report['shards_unchanged']=all(Path(s['path']).stat().st_size==s['size'] and Path(s['path']).stat().st_mtime_ns==s['mtime_ns'] for s in report['shards_before'])
        if not report['shards_unchanged']: raise ValueError('shard metadata changed')
        report['mapping_trusted']=True
    except Exception as error:
        report['error']=str(error)
        if hasattr(error,'result'): report['failure']=error.result
    with report_path.open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('comparisons','compile','reference_run','production_run')},indent=2))
    if not report['mapping_trusted']: raise SystemExit(1)


if __name__=='__main__':
    main()
