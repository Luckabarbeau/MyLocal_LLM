"""Native checkpoint architecture probe; metadata/descriptors only, no payload IO."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from mini_llm.native import Runtime
from mini_llm.external.gguf_store import _discover
from docs.native_engine.probe_gguf_inventory import identity, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    paths, _ = _discover(args.checkpoint)
    before = [identity(p) for p in paths]
    runtime = Runtime()
    with runtime.gguf(args.checkpoint) as store:
        h = store.validate_qwen4exp()
        report = dict(status='passed', scope='native descriptors + authoritative Python architecture; no payload reads/decodes/uploads',
                      tensor_count=len(h.shapes), width=h.width, vocab=h.vocab,
                      context=h.context, hc_rank=h.rank, layers=len(h.recurrent),
                      recurrent_layers=[i for i, rec in enumerate(h.recurrent) if rec],
                      qsa_layers=[i for i, rec in enumerate(h.recurrent) if not rec],
                      ple_layers=h.ple_layers, dense_fp32_bytes=h.dense_bytes,
                      expert_scale=h.expert_scale, shards=before,
                      library_sha256=sha256(runtime._lib._name))
        report['schema_sha256'] = hashlib.sha256(json.dumps(h.shapes, sort_keys=True).encode()).hexdigest()
    assert before == [identity(p) for p in paths], 'checkpoint stat identity changed'
    sources = ['mini_llm/external/qwen4exp_weights.py', 'mini_llm/external/qwen4exp_gguf.py',
               'docs/qwen4exp/reference_contract.md', 'mini_llm/native/__init__.py',
               'mini_llm/native_cpp/runtime.cpp', 'mini_llm/native_cpp/runtime.h',
               'tests/test_native_engine_qwen_contract.py', __file__]
    report.update(source_sha256={p: sha256(p) for p in sources},
                  elapsed_seconds=time.monotonic()-start,
                  limitations='Not payload integrity or full-model numerical validation; no standalone C++ architecture metadata parser')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
