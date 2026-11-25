"""Bounded sampler-only evidence capture; run before and after changing the library."""
import argparse
from contextlib import ExitStack
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from mini_llm.native import Runtime
from test_native_engine_hc import upload

V = 248320


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def snapshot_input():
    x = np.random.default_rng(8).normal(size=V).astype(np.float32)
    x[:1000] = 5  # Stable top-k boundary ties for k=1/50/500.
    return x


def capture(sampler, uploaded, k, p):
    result = sampler.sample(uploaded, temperature=.7, top_k=k, top_p=p,
                            uniform=.213, metadata=True)
    ids, probabilities, cdf = sampler.readback_for_test()
    return dict(k=k, top_p=p, result_hex=bytes(result).hex(),
                ids_sha256=digest(ids), probabilities_sha256=digest(probabilities),
                cdf_sha256=digest(cdf), active_ids=ids[:k].tolist(),
                active_probability_bits=probabilities[:k].view(np.uint32).tolist(),
                active_cdf_bits=cdf[:k].view(np.uint32).tolist())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--library', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error('repeats must be 1..20')
    import cupy as cp
    root = Path(__file__).resolve().parents[1]
    library = Path(args.library)
    report = dict(library=str(library), library_sha256=digest(np.fromfile(library, np.uint8)),
                  source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                 for name in ('mini_llm/native_cpp/sampling.cu',
                                              'tests/test_native_engine_sampler.py')},
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  visible_count=cp.cuda.runtime.getDeviceCount(), ordinal=cp.cuda.Device().id,
                  gpu_name=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                  free_total_bytes=list(cp.cuda.runtime.memGetInfo()), cupy=cp.__version__,
                  vocab=V, input_seed=8, warmups=2, repeats=args.repeats)
    with tempfile.TemporaryDirectory() as path, ExitStack() as stack:
        r = Runtime(library)
        e = stack.enter_context(r.engine(budget_bytes=32*2**20))
        x = np.random.default_rng(8).normal(size=V).astype(np.float32)
        uploaded = upload(stack, Path(path), r, e, [x])[0][0]
        sampler = stack.enter_context(e.sampler(V, seed=8))
        report['input_sha256'] = digest(x)
        report['workspace_bytes'] = e.stats.device_bytes-x.nbytes
        report['benchmarks'] = {}
        for name, kwargs in [('greedy', {}), ('topk50_nucleus', dict(temperature=.7, top_p=.9, top_k=50)),
                             ('full_stochastic', dict(temperature=1))]:
            sampler.reset(seed=8)
            for _ in range(report['warmups']):
                sampler.sample(uploaded, **kwargs)
            times, tokens = [], []
            for _ in range(args.repeats):
                start = time.perf_counter_ns()
                tokens.append(sampler.sample(uploaded, **kwargs))
                times.append((time.perf_counter_ns()-start)/1e6)
            report['benchmarks'][name] = dict(host_ms=times, median_host_ms=float(np.median(times)), tokens=tokens)
        tied = upload(stack, Path(path), r, e, [snapshot_input()])[0][0]
        report['snapshot_input_sha256'] = digest(snapshot_input())
        report['snapshots'] = [capture(sampler, tied, k, p) for k in (1,50,500) for p in (.9,1.)]
        sampler.reset(seed=8)
        report['seeded_results_hex'] = [bytes(sampler.sample(uploaded, temperature=.7, top_k=50,
                                                           top_p=.9, metadata=True)).hex() for _ in range(4)]
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({name: values['median_host_ms'] for name, values in report['benchmarks'].items()}))
    print('workspace_bytes', report['workspace_bytes'])


if __name__ == '__main__':
    main()
