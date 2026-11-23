"""End-to-end orchestration test for try_triton_prefill_experts without the
real checkpoint: synthetic 64-expert layer with the production geometry,
exercising waves, the pinned ring (3 slots force reuse), worker memmoves and
all four kernels.  Output is compared to an FP32 host reference computed from
the same dequantized weights within FP16-policy tolerance.
"""
import os
os.environ.setdefault('MINI_LLM_BACKEND', 'cupy')
os.environ['MINI_LLM_TRITON_WAVE'] = '8'   # 8 waves over 64 experts -> ring reuse
os.environ['MINI_LLM_TRITON_THREADS'] = '4'
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from mini_llm.backend import xp
import mini_llm.ops.triton_prefill_experts as tpe
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
from mini_llm.ops.quantized_weight import PackedGGUFWeight

rng = np.random.default_rng(7)
T, D, NFF, NE, TOPK = 512, 2560, 640, 64, 10


def _valid_bytes(n_rows, k, qb, qk):
    data = rng.integers(0, 256, (n_rows, k // qk, qb), dtype=np.uint8)
    data[:, :, 0:2] = np.frombuffer(np.float16(0.004).tobytes(), np.uint8)
    return data.reshape(n_rows, -1)


def main():
    weights = {}
    for e in range(NE):
        gate = PackedGGUFWeight(_valid_bytes(NFF, D, 110, 256).reshape(-1),
                                (NFF, D), 'IQ3_S')
        up = PackedGGUFWeight(_valid_bytes(NFF, D, 110, 256).reshape(-1),
                              (NFF, D), 'IQ3_S')
        down = PackedGGUFWeight(_valid_bytes(D, NFF, 18, 32).reshape(-1),
                                (D, NFF), 'IQ4_NL')
        weights[e] = (gate, up, down)
    ref = {e: tuple(w.dequantize_reference().astype(np.float32) for w in trio)
           for e, trio in weights.items()}

    ids = np.stack([rng.choice(NE, size=TOPK, replace=False)
                    for _ in range(T)]).astype(np.int64)
    routes = (rng.random((T, TOPK)) * 0.1).astype(np.float32)
    x = (rng.standard_normal((1, T, D)) * 0.002).astype(np.float32)

    def _ffn(t, e, j):
        g, u, d = ref[e]
        row = x[0, t]
        gate_v = row @ g.T
        up_v = row @ u.T
        act = gate_v / (1.0 + np.exp(-gate_v)) * up_v
        return routes[t, j] * (act @ d.T)

    href = np.zeros((T, D), np.float32)
    for t in range(T):
        for j in range(TOPK):
            href[t] += _ffn(t, int(ids[t, j]), j)

    executor = HeterogeneousExpertExecutor.__new__(HeterogeneousExpertExecutor)
    executor.model_id = 'synthetic'
    executor.execution_mode = 'mixed'
    executor.routing_profiler = None
    executor.cache = SimpleNamespace(_admission_limit=0,
                                     idle_eviction_requests=False)
    from collections import Counter
    executor.counters = Counter()
    executor.dispatch_scope = lambda layer_id, active: nullcontext()
    executor.cpu_source_weights = lambda key: weights[key.expert_id]
    experts = SimpleNamespace(expert_executor=executor, n_experts=NE,
                              layer_id=0)

    dev_x = xp.asarray(x)
    dev_routes = xp.asarray(routes[None])
    dev_ids = xp.asarray(ids[None])
    out = tpe.try_triton_prefill_experts(experts, dev_x, dev_routes, dev_ids)
    if out is None:
        print('ORCHESTRATION: returned None (ineligible?) - FAIL')
        return 1
    got = xp.asnumpy(out[0])
    rel = float(np.max(np.abs(got - href) / np.maximum(np.abs(href), 1.0)))
    print('orchestration max rel vs fp32 host reference:', f'{rel:.3e}')
    print('counters:', executor.counters)
    ok = rel < 5e-2
    print('TRITON ORCHESTRATION', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
