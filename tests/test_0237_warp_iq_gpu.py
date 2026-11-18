"""Opt-in warp scheduling versus retained block reduction; no Q8 arithmetic."""
import json
import os
from pathlib import Path
import resource

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops import iq_packed_gpu as iq
from mini_llm.ops import grouped_quantized_experts as grouped
from mini_llm.ops.quantized_linear import PackedExpertFFNInference, packed_to_device
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from test_0231_grouped_iq_gpu import ACTUAL, IQ, handles
from test_0234_iq_codebook_storage import measured_call

ENV = 'MINI_LLM_IQ_GPU_GROUPED_KERNEL'
GPU = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='actual CuPy required')


def metrics(actual, reference):
    a, b = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    result = dict(max_absolute_error=float(np.max(np.abs(a-b))),
                  relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30)))
    assert result['relative_l2'] < 1e-5, result
    assert result['max_absolute_error'] < 1e-7 + 2e-5*float(np.max(np.abs(b))), result
    return result


def test_option_rejects_invalid_mode_without_cuda(monkeypatch):
    for mode in ('', 'WARP', 'warp ', '4'):
        monkeypatch.setenv(ENV, mode)
        with pytest.raises(ValueError, match='block or warp'):
            iq._grouped_kernel()
    monkeypatch.delenv(ENV)
    assert iq._grouped_kernel() == 'block'


@GPU
@pytest.mark.parametrize('formats', [(f, f, f) for f in IQ] + ACTUAL)
@pytest.mark.parametrize('storage', ['constant', 'global'])
def test_warp_against_block_formats_routes_and_stream(formats, storage, monkeypatch):
    import cupy as cp
    monkeypatch.setenv('MINI_LLM_IQ_GPU_CODEBOOK_STORAGE', storage)
    _, experts = handles(formats, count=3, d=768, f=256)
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        x = cp.asarray(np.random.default_rng(237).normal(0, .1, (1, 768)).astype(np.float32))
        for routes in (cp.asarray([.125, -.375, .5], dtype=cp.float32),
                       cp.asarray([.125, -.375, .5], dtype=cp.float32)[::-1],
                       cp.broadcast_to(cp.asarray([.125], dtype=cp.float32), (3,))):
            for slots in (None, (2, 0, 1)):
                monkeypatch.setenv(ENV, 'block')
                expected = iq.grouped(x, experts, routes, slots=slots)
                monkeypatch.setenv(ENV, 'warp')
                actual = iq.grouped(x, experts, routes, slots=slots)
                dependent = actual * cp.float32(2)
                stream.synchronize()
                metrics(cp.asnumpy(dependent), cp.asnumpy(expected)*2)


@GPU
@pytest.mark.skipif(os.environ.get('MINI_LLM_IQ_WARP_REAL_BENCHMARK') != '1',
                    reason='opt-in bounded real expert benchmark')
def test_real_seven_format_warp_benchmark(monkeypatch):
    import cupy as cp
    from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
    from mini_llm.ops.expert_residency import ExpertKey
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    model = '/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S'
    index = Qwen4ExpGGUFIndex.open(model, model_id='0237')
    monkeypatch.setenv('MINI_LLM_IQ_GPU_CODEBOOK_STORAGE', 'global')
    reads = []
    original = index.store.rows
    def bounded(name, first, last, *, outer=()):
        size = (last-first)*index.store.tensors[name].row_bytes
        assert len(outer) == 1 and size < 4*1024**2
        reads.append(size)
        return original(name, first, last, outer=outer)
    monkeypatch.setattr(index.store, 'rows', bounded)
    def forbidden(*args, **kwargs):
        pytest.fail('full expert decode forbidden')
    monkeypatch.setattr(PackedGGUFWeight, 'dequantize_reference', forbidden)
    monkeypatch.setattr(index.store, 'packed_weight', forbidden)
    stamps = [(p.stat().st_size, p.stat().st_mtime_ns) for p in index.store.paths]
    seen, records = set(), []
    try:
        x = cp.asarray(np.random.default_rng(237).normal(0, .1, (1, 2560)).astype(np.float32))
        routes = cp.asarray(np.arange(1, 11, dtype=np.float32)/np.float32(55))
        for layer in range(index.n_layers):
            signature = tuple(index.descriptors[f'blk.{layer}.ffn_{p}_exps.weight'].quant_type
                              for p in ('gate', 'up', 'down'))
            if signature in seen:
                continue
            seen.add(signature)
            experts = []
            for e in np.linspace(0, 511, 10, dtype=int):
                ws = index.provider(ExpertKey(index.model_id, layer, int(e)))
                experts.append(PackedExpertFFNInference(*(packed_to_device(w) for w in ws)))
            values = {}
            samples = {'block': [], 'warp': []}
            for mode in samples:
                monkeypatch.setenv(ENV, mode)
                values[mode] = cp.asnumpy(grouped.grouped_quantized_experts_forward(x, experts, routes))
            errors = metrics(values['warp'], values['block'])
            for i in range(10):
                order = ('block', 'warp', 'warp', 'block') if i % 2 == 0 else ('warp', 'block', 'block', 'warp')
                for mode in order:
                    monkeypatch.setenv(ENV, mode)
                    samples[mode].append(measured_call(cp, lambda: grouped.grouped_quantized_experts_forward(x, experts, routes)))
            times = {mode: dict(event_ms=float(np.median(np.asarray(v)[:, 0])),
                               wall_ms=float(np.median(np.asarray(v)[:, 1]))) for mode, v in samples.items()}
            records.append(dict(layer=layer, formats=signature, errors=errors, timings=times,
                                speedup=times['block']['event_ms']/times['warp']['event_ms']))
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-baseline < 2*1024**3
        assert len(records) == 7
        assert stamps == [(p.stat().st_size, p.stat().st_mtime_ns) for p in index.store.paths]
        report = dict(status='passed', scope='resident real top10 operator only; FP32, global codebooks',
                      records=records, logical_read_bytes=sum(reads), max_single_read_bytes=max(reads),
                      additional_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-baseline,
                      median_event_speedup=float(np.median([r['speedup'] for r in records])),
                      timings='20 warm samples per mode, balanced order, excludes JIT/upload; includes pointer metadata upload',
                      checkpoint_size_mtime_unchanged=True)
        output = os.environ.get('MINI_LLM_IQ_WARP_REPORT')
        if output:
            with Path(output).open('x') as f:
                json.dump(report, f, indent=2, allow_nan=False)
        print(json.dumps(report, indent=2))
    finally:
        index.store.close()
