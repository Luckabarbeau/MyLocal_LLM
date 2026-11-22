#!/usr/bin/env python3
"""Build only the locally available Strata fused prompt kernels plus a narrow ABI.

No engine/server is linked or launched. Requires the local MIT-licensed Strata
checkout, its matching GGML headers, and CUDA nvcc. The output manifest records
source hashes/revision and the full command; no system installation is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/home/lucka/Documents/my_local_LLM/qwen4exp-references/Strata'))
    parser.add_argument('--nvcc', default='/usr/local/cuda/bin/nvcc')
    parser.add_argument('--arch', default='sm_120')
    parser.add_argument('--output', type=Path, required=True, help='New build artifact directory')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    ggml = source/'third_party/strata-ggml/ggml'
    inputs = [source/'src/prefill/moe_fused.cu', source/'src/prefill/moe_fused_iq.cu',
              ROOT/'scripts/native_strata_prefill.cpp']
    library = out/'libmylocal_strata_prefill.so'
    command = [args.nvcc, '-O3', '-std=c++17', '--extended-lambda', '--use_fast_math',
               '--shared', '-Xcompiler', '-fPIC', '--cudart', 'shared', '-arch='+args.arch,
               '-I'+str(source/'include'), '-I'+str(ggml/'include'), '-I'+str(ggml/'src'),
               *map(str, inputs), '-o', str(library)]
    revision = subprocess.run(['git','--no-pager','-C',str(source),'rev-parse','HEAD'],
                              check=True,capture_output=True,text=True,timeout=10).stdout.strip()
    manifest = dict(source=str(source), source_revision=revision, command=command,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        scope='Only fused expert kernels, not Strata engine or model execution',
        license='MIT; original sources/headers retain their upstream notices')
    (out/'build.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (out/'build.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    manifest['returncode'] = result.returncode
    if result.returncode == 0:
        manifest['library_sha256'] = hashlib.sha256(library.read_bytes()).hexdigest()
    (out/'build.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
