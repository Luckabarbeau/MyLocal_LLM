"""Read-only identity investigation A; exclusive versioned output, no B/C work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_0210_qwen4exp_reference import ROOT, PIN, bounded, metrics

SOURCE = Path('/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/vendor/llama.cpp')
COPY = Path('/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch')
MODEL = Path('/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf')


def classify(event):
    """Find the nearest HC norm weight through actual source edges, not event order.

    Stop at prior HC read boundaries so FFN ancestry cannot inherit attention's
    classification. Every traversed edge is retained in the artifact.
    """
    nodes = {n['ptr']: n for n in event['nodes']}
    root = event['tensor']['ptr']
    pending, seen, found = [root], set(), set()
    while pending:
        ptr = pending.pop()
        if not ptr or ptr in seen:
            continue
        seen.add(ptr)
        n = nodes[ptr]
        if ptr != root and n['name'] == 'hc_mixed-0':
            continue
        for kind in ('attn', 'ffn'):
            if n['name'] == f'blk.0.hc_{kind}_norm.weight':
                found.add(kind)
                # The norm input has earlier block ancestry; do not follow it.
        # Traverse mixer gate ancestry, but stop at the residual input to this
        # HC norm: MUL's other operand is RMS_NORM of the incoming streams.
        if n['op'] == 'RMS_NORM':
            continue
        pending.extend(n['src'])
    if len(found) != 1:
        raise ValueError(f'ambiguous HC weight ancestry: {found}')
    return {'attn': 'hc_attention_read', 'ffn': 'hc_ffn_read'}[found.pop()]


def validate_layout(event, width):
    t = event['tensor']
    if t['type'] != 0 or t['ne'] != [width, 1, 1, 1]:
        raise ValueError('expected HC read ne[0]=D, ne[1]=token count=1 (streams already reduced)')
    expected = [4, 4*width, 4*width, 4*width]
    if t['nb'] != expected:
        raise ValueError(f'non-contiguous F32 read: {t["nb"]}')


def associate(events):
    identities, stages = {}, {}
    for event in events:
        ptr = event['tensor']['ptr']
        stage = classify(event)
        if ptr in identities and identities[ptr] != stage:
            raise ValueError('pointer changed graph identity')
        if stage in stages and stages[stage] != ptr:
            raise ValueError('multiple distinct tensors for one boundary')
        identities[ptr], stages[stage] = stage, ptr
    if len(stages) != 2 or len(identities) != 2:
        raise ValueError('did not establish two distinct HC reads')
    return identities


def provenance():
    sha = subprocess.check_output(['git', '--no-pager', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '--no-optional-locks', '-C', str(SOURCE), 'status', '--porcelain'], text=True)
    if sha != PIN or status.strip():
        raise ValueError('original source revision/cleanliness mismatch')
    def inventory(root):
        return {str(p.relative_to(root)): p for p in root.rglob('*') if p.is_file()
                and not any(x == '.git' or x.startswith('build') or x == '__pycache__'
                            for x in p.relative_to(root).parts)}
    original, copied = inventory(SOURCE), inventory(COPY)
    if original.keys() != copied.keys():
        raise ValueError('source inventories differ')
    differences = [k for k in original if original[k].read_bytes() != copied[k].read_bytes()]
    old = 'ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);'
    base = original['src/llama-model.cpp'].read_text()
    if differences != ['src/llama-model.cpp'] or base.count(old) != 1 or copied['src/llama-model.cpp'].read_text() != base.replace(old, old.replace('(true,', '(false,')):
        raise ValueError('not exactly loader prefetch=false')
    loader = (COPY/'src/llama-model-loader.cpp').read_text()
    mmap = (COPY/'src/llama-mmap.cpp').read_text()
    if 'prefetch && use_mmap ? -1 : 0' not in loader or 'if (prefetch && lazy_ranges.empty()) { flags |= MAP_POPULATE; }' not in mmap or 'if (prefetch > 0)' not in mmap:
        raise ValueError('no-prefetch flow not verified')
    cache = (COPY/'build-cpu/CMakeCache.txt').read_text().splitlines()
    for key in ('GGML_CUDA','GGML_HIP','GGML_VULKAN','GGML_METAL','GGML_SYCL','GGML_OPENCL','GGML_RPC','GGML_BLAS'):
        if f'{key}:BOOL=OFF' not in cache:
            raise ValueError(f'non CPU build: {key}')
    if (COPY/'build-cpu/bin/libllama.so').stat().st_mtime_ns < copied['src/llama-model.cpp'].stat().st_mtime_ns:
        raise ValueError('library predates patched source')
    return dict(base_sha=sha, changed_source_files=differences, compared_files=len(original),
                zero_prefetch_control_flow_verified=True, cpu_only=True)


def child_model(output):
    sys.path.insert(0, str(ROOT))
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    class LayerZeroCaptured(Exception):
        pass
    def diagnostic(layer, stage, value):
        if layer == 0 and stage in ('hc_attention_read', 'hc_ffn_read'):
            np.save(output/f'{stage}.npy', np.array(value, dtype=np.float32, copy=True))
            if stage == 'hc_ffn_read':
                raise LayerZeroCaptured()
    with Qwen4ExpGGUFModel.open(MODEL, max_length=32, cpu_workers=1, diagnostic_callback=diagnostic) as model:
        state = model.create_generation_state(1, 32)
        try:
            model.step(1, state)
        except LayerZeroCaptured:
            return
    raise RuntimeError('layer zero callbacks missing')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child-model', type=Path)
    args = parser.parse_args()
    if args.child_model:
        child_model(args.child_model)
        return
    report_path = ROOT/'docs/qwen4exp/0218_hc_identity_v1.json'
    vectors = ROOT/'docs/qwen4exp/diagnostics_0218_v1'
    if report_path.exists() or vectors.exists():
        raise FileExistsError('versioned artifact already exists; refusing overwrite')
    report = dict(scope='A only: token 0 (id 1), layer 0', mapping_trusted=False,
                  production_files_modified=False, boundaries_beyond_layer0_validated=False,
                  context=32, threads=1, kv_dtype='F32', mmap=True, lazy_PLE=True, repack=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MINI_LLM_BACKEND='numpy',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    try:
        report['provenance'] = provenance()
        vectors.mkdir()
        with tempfile.TemporaryDirectory(prefix='hc-identity-') as tmp:
            tmp = Path(tmp)
            lib = COPY/'build-cpu/bin'
            binary = tmp/'identity'
            report['compile'] = bounded(['c++','-std=c++17','-O2',str(ROOT/'tests/fixtures/qwen4exp_identity_reference.cpp'),
                '-I'+str(COPY/'include'),'-I'+str(COPY/'ggml/include'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),
                '-lllama','-lggml','-lggml-base','-lggml-cpu','-o',str(binary)],time.monotonic()+120,tmp,'compile',env)
            report['reference_run'] = bounded([str(binary),str(MODEL),str(vectors)],time.monotonic()+120,tmp,'reference',env)
            report['model_run'] = bounded([sys.executable,str(Path(__file__)), '--child-model',str(vectors)],time.monotonic()+120,tmp,'model',env)
        events = [json.loads(line) for line in (vectors/'events.jsonl').read_text().splitlines()]
        identities = associate(events)
        comparisons, summaries = {}, []
        for event in events:
            stage = identities[event['tensor']['ptr']]
            actual = np.load(vectors/f'{stage}.npy')
            validate_layout(event, actual.size)
            ref = np.fromfile(vectors/f'eval_{event["eval_occurrence"]}.f32',dtype=np.float32)
            comparisons[str(event['eval_occurrence'])] = dict(stage=stage, **metrics(actual.reshape(-1),ref),
                actual_l2=float(np.linalg.norm(actual.astype(np.float64))))
            summaries.append(dict(eval_occurrence=event['eval_occurrence'],pointer_occurrence=event['pointer_occurrence'],
                                  tensor=event['tensor'],stage=stage))
        report.update(mapping_trusted=True, events=summaries, comparisons=comparisons,
                      identity_to_stage=identities, distinct_tensors=len(identities), eval_callbacks=len(events),
                      layout='HC read ne=[D,1,1,1]; ne[1] is token count, not HC streams C; contiguous D flatten verified')
    except Exception as error:
        report['error'] = str(error)
        if hasattr(error,'result'):
            report['failure'] = error.result
    with report_path.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))
    if not report['mapping_trusted']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
