"""Reproducible bounded WP3 benchmark, no CuPy or checkpoint dependency.

Run from repository root: MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=...
.venv/bin/python -m benchmarks.benchmark_native_qsa_selection --device 0
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import struct
import tempfile

import numpy as np
from mini_llm.native import Runtime


def fixture(path, arrays):
    """Minimal F32 GGUF; same exact stored-byte input path as runtime tests."""
    def string(s):
        raw = s.encode()
        return struct.pack('<Q', len(raw)) + raw
    header = b'GGUF' + struct.pack('<IQQ', 3, len(arrays), 0)
    offset, payloads = 0, []
    for name, a in arrays.items():
        dims = a.shape[::-1]
        header += string(name) + struct.pack('<I', len(dims))
        header += struct.pack('<'+'Q'*len(dims), *dims) + struct.pack('<IQ', 0, offset)
        payload = a.tobytes()
        padded = payload + bytes((-len(payload)) % 32)
        payloads.append(padded)
        offset += len(padded)
    path.write_bytes(header + bytes((-len(header)) % 32) + b''.join(payloads))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--samples', type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10000 or args.samples < 1:
        parser.error('repeats must be 1..10000 and samples positive')
    rng, capacity = np.random.default_rng(20261123), 16384
    arrays = dict(q=rng.normal(size=(4,128)).astype(np.float32),
                  p=rng.normal(size=(capacity//4,128)).astype(np.float32),
                  d=rng.normal(size=128).astype(np.float32))
    runtime = Runtime()
    rows = []
    with tempfile.TemporaryDirectory(prefix='native-qsa-') as directory, ExitStack() as stack:
        path = Path(directory)/'inputs.gguf'
        fixture(path, arrays)
        store = stack.enter_context(runtime.gguf(path))
        engine = stack.enter_context(runtime.engine(device=args.device, budget_bytes=2**24))
        uploads = [stack.enter_context(engine.upload(stack.enter_context(store.lease(name))))
                   for name in arrays]
        w = stack.enter_context(engine.qsa_selection(capacity))
        for n in [1, 2051, 2053, 4096, 8192, 16384]:
            w.set_n(n)
            for policy in ['ggml_cpu', 'strata']:
                for _ in range(3):
                    w.run(*uploads, policy=policy)
                samples = [w.run(*uploads, policy=policy, repeats=args.repeats)
                           for _ in range(args.samples)]
                rows.append(dict(n=n, policy=policy, milliseconds_per_iteration=samples,
                                 median_ms=float(np.median(samples))))
        result = dict(device=args.device, capacity=capacity, seed=20261123,
                      repeats=args.repeats, samples=args.samples,
                      timing='CUDA events: status clear + scoring + selection; excludes transfers/readback',
                      device_bytes=engine.stats.device_bytes, pinned_bytes=engine.stats.pinned_bytes,
                      results=rows, speedup_claim=None)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
