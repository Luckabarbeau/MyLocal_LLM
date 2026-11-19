# Patch0201 — additive multi-shard GGUF store

## Scope and invariants

`mini_llm/external/gguf_store.py` adds `MultiShardTensorStore` without changing
`Qwen3NextGGUFIndex`, model loading, packed kernels, or package enums. The optional
installed `gguf` reader parses the header; a local subclass overrides
`_build_tensors` to retain only tensor-info fields, leaving `reader.tensors=[]`.
It never asks the reader for tensor payload views. Header reads are bounds checked.

Descriptors use installed `GGML_QUANT_SIZES`, independently of the native packed
backend's `BLOCK_FORMATS`. The explicit verified extension is raw type **42,
Q2_0, 64 elements / 18 bytes** (fp16 scale plus 16 packed bytes). IQ3_S, IQ2_S,
Q2_0 and other formats can therefore be indexed before their packed decoders exist.
No monkeypatch of the package enum is used.

## API

```python
from mini_llm.external.gguf_store import MultiShardTensorStore

with MultiShardTensorStore.open("/models/model-00002-of-00002.gguf") as store:
    descriptor = store.tensors["blk.0.ffn_down_exps.weight"]
    raw = store.tensor_bytes(descriptor.name)  # zero-copy read-only uint8
    # For formats already supported by PackedGGUFWeight only:
    expert = store.packed_expert(descriptor.name, 0)
    # Equivalently: store.packed_weight(descriptor.name).select_outer(0)
```

`open` accepts a standalone file, any numbered shard, or an unambiguous directory.
Numbered files must form one complete `name-00001-of-000NN.gguf` set; unrelated
model families in a directory are rejected rather than guessed. Each numbered
shard must supply matching `split.count` and `split.tensors.count`; `split.no` is
zero-based and must match filename order. Later shards may contain **only these
three split fields**. Standalone split metadata, when present, must describe one
shard and the actual tensor count.

`metadata` is a first-occurrence union of detached `field.contents()` values,
including complete tokenizer/numeric arrays, not `contents(0)` truncations.
`shard_metadata` preserves each file's values. Overlapping metadata must agree in
both type and value, except `split.no` and per-shard `GGUF.*` bookkeeping. Metadata
missing from later shards is allowed; values unique to later shards are merged.

Each immutable tensor descriptor records:

- canonical row-major NumPy `shape` and original `ggml_shape` (their reverse);
- raw format ID, name, block elements/bytes;
- owning `shard_index`, Python-int relative and absolute byte offsets;
- `nbytes`, `row_bytes`, `byte_shape`, and `byte_strides`.

For packed weights, `byte_shape` replaces the logical innermost dimension by its
**encoded byte width**. Thus `byte_strides` describe a uint8 view, not a fictional
per-element byte stride for sub-byte quantization. A block's byte stride is the
format's `block_bytes`. File offsets and dimension arithmetic use Python integers
before addition/product, avoiding uint64 overflow or float promotion.

`rows(name, first, last, outer=())` returns a zero-copy uint8 matrix of exactly
`[last-first, row_bytes]`. It fixes every axis preceding `[rows,width]` through
`outer`, validates the bounded contiguous row interval, and works for PLE tables
and formats whose decoder is not yet implemented. It does not gather arbitrary
IDs or decode any row. Callers explicitly decode selected supported blocks later.
`packed_weight(name)` rejects unsupported formats with `NotImplementedError`
(Patch0202); it never silently expands them. `packed_expert` delegates to the
existing `PackedGGUFWeight.select_outer` implementation.

## Validation and memory ownership

Before exposing tensor bytes, indexing rejects duplicate names (within/across
shards), zero/invalid dimensions or rank outside GGML's 1..4 layout, unknown raw
types, block-unaligned innermost rows, misaligned tensor offsets, truncated header
or payload ranges, and intersecting payload intervals within a shard. It supports
only little-endian GGUF on little-endian hosts and the standard contiguous GGML
layout; no custom strides, transposed payload interpretation, or big-endian
conversion is provided. GGUF descriptors do not encode custom strides, so an
unmarked nonstandard layout cannot be detected from a header alone.

There is one independent read-only `np.memmap` owner per shard. No tensor payload
concatenation, full-table decode, or eager tensor-byte view occurs during indexing.
Reader fields are discarded after detaching metadata and descriptors, avoiding
retaining the reader's many small header views. Parsing large tokenizer arrays
can nevertheless have a substantial transient Python-object cost; virtual mapped
file size is not resident RAM. Header-only means no requested payload access, not
an assurance that the OS never readaheads a page adjacent to a header.

`close()` clears store ownership and disables future byte access. It does **not**
call private `mmap.close()` while exported NumPy or packed views may still exist:
those views keep their owner/base chain alive and remain valid until their final
reference disappears. Consequently close does not promise immediate unmapping.
Descriptors and detached metadata remain inspectable after close. The store is
not a synchronized concurrent-close API, checksum verifier, or protection against
files externally modified/truncated after indexing.

## Verification procedure

The new tests use a minimal `struct` GGUF binary writer, no checkpoints and no
GGUFWriter dependency. `pytest.importorskip("gguf")` covers optional installation.
Fixtures cover numbering, split-only later headers, metadata overlap and arrays,
duplicates, malformed descriptors/headers, sparse >4 GiB offsets, extreme uint64
truncation, endian rejection, unsupported indexed formats, bounded rows,
independent mmap owners, exported-view lifetime, and instrumented header-only
reader calls with forbidden payload builder/dequantizer.

Focused test command (no persistent cache/bytecode writes):

```sh
/usr/bin/time -v env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_0201_gguf_store.py tests/test_0106p_gguf_rows.py
```

The local target check opens the IQ3_S two-shard model identified by the existing
inventory, requires exactly **1224** descriptors, reports format counts and PLE
namespace examples, and instruments reader `_get` ranges against each header's
data offset. It does not call `tensor_bytes`, packed access, rows, or dequantizers.
Resident/peak RAM and actual command results are recorded below.

## Observed results

Both commands passed on the initial execution; no corrective implementation edit
or test rerun was required. Environment: repository `.venv`, Python 3.12,
installed gguf 0.19.0, CPU-only.

Focused test output:

```text
.............................................. [100%]
46 passed in 0.19s
Elapsed wall clock: 0:00.37
Maximum resident set size (kbytes): 58824
Exit status: 0
```

Exact real-file probe command (instrumentation is process-local; no package or
repository source is monkeypatched on disk):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B - <<'PY'
from collections import Counter
import resource
import time
import numpy as np
import mini_llm.external.gguf_store as module

path = '/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'
def ram():
    with open('/proc/self/status') as stream:
        return {line.split(':')[0]: line.split(':')[1].strip() for line in stream if line.startswith(('VmRSS:', 'VmHWM:', 'VmSize:'))}
print('RAM baseline:', ram(), flush=True)
reader_type, formats = module._reader_api()
original_get = reader_type._get
original_build = reader_type._build_tensors
requests = {}
checks = []
def checked_get(self, offset, dtype, count=1, override_order=None):
    key = id(self)
    calls, end = requests.get(key, (0, 0))
    requests[key] = (calls + 1, max(end, int(offset) + np.dtype(dtype).itemsize * int(count)))
    return original_get(self, offset, dtype, count, override_order)
def checked_build(self, start, fields):
    calls, end = requests.pop(id(self))
    assert end <= int(start), (end, start)
    original_build(self, start, fields)
    assert self.tensors == []
    checks.append({'header_requests': calls, 'largest_requested_end': end, 'data_offset': int(start), 'descriptors': len(fields)})
reader_type._get = checked_get
reader_type._build_tensors = checked_build
module._reader_api = lambda: (reader_type, formats)
def forbidden(*args, **kwargs):
    raise AssertionError('Payload access is forbidden in this index check')
module.MultiShardTensorStore.tensor_bytes = forbidden
module.MultiShardTensorStore.packed_weight = forbidden
module.MultiShardTensorStore.packed_expert = forbidden
module.MultiShardTensorStore.rows = forbidden
start = time.monotonic()
with module.MultiShardTensorStore.open(path) as store:
    assert len(store) == 1224, len(store)
    assert len(store.paths) == 2
    assert store.metadata['general.architecture'] == 'qwen4exp'
    assert len(store.metadata['tokenizer.ggml.tokens']) == 248320
    print('TARGET tensor count:', len(store), 'shards:', len(store.paths))
    print('FORMAT counts:', dict(Counter(d.quant_type for d in store.tensors.values())))
    print('NAMESPACE samples:', [(d.name, d.shape, d.quant_type, d.shard_index) for d in store.tensors.values() if 'ple' in d.name or 'ngram' in d.name][:12])
    print('LAST namespace entry:', list(store.tensors)[-1])
    print('HEADER checks:', checks)
    print('RAM indexed:', ram())
print('RAM after ownership release:', ram())
print('PEAK ru_maxrss KiB:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print('ELAPSED seconds:', round(time.monotonic() - start, 3))
print('PASS: real 1224-tensor namespace, headers only; no payload views or decodes')
PY
```

Real-file results:

```text
TARGET tensor count: 1224 shards: 2
FORMAT counts: {'Q6_K': 129, 'BF16': 484, 'F32': 292, 'IQ4_XS': 45,
 'Q4_K': 47, 'IQ4_NL': 87, 'IQ3_XXS': 34, 'Q5_K': 35, 'Q2_0': 9,
 'IQ2_S': 40, 'F16': 1, 'IQ3_S': 20, 'Q8_0': 1}
LAST namespace entry: per_layer_token_embd.weight
HEADER checks:
 shard 0: 1247867 requests, largest end 11024355, data offset 11024384,
          1223 descriptors
 shard 1: 21 requests, largest end 173, data offset 192, 1 descriptor
PEAK ru_maxrss KiB: 904896
ELAPSED seconds: 5.447
PASS: real 1224-tensor namespace, headers only; no payload views or decodes
```

PLE namespace examples included `blk.1.ple_conv1d.weight` `(10240, 4)` F16,
`blk.1.ple_key.weight` `(10240, 2560)` BF16, and
`blk.1.ple_value.weight` `(2560, 2560)` BF16, all on shard 0. The second shard's
`per_layer_token_embd.weight` was accessible through the descriptor namespace;
its payload was not requested.

| Memory point | VmRSS (KiB) | VmHWM (KiB) | VmSize (KiB) |
| --- | ---: | ---: | ---: |
| Baseline | 38600 | 38600 | 1062048 |
| Indexed | 249628 | 904896 | 82922444 |
| Store owners released | 238684 | 904896 | 1264568 |

Indexing therefore used about **244 MiB resident RAM**, with a transient
**884 MiB peak**. Approximately 79 GiB of indexed virtual address space is mostly
file mappings, not resident payload allocation. Python allocator retention and
detached tokenizer metadata explain why releasing mappings does not restore
baseline RSS immediately. These observations are not a hard RAM budget guarantee.
No model inference, all-payload decode, payload checksum, or new IQ/Q2 decoder
validation was performed.
