"""0080G: small, reproducible, EOS-bounded three-source CPU research corpus.

Run from repository root with an installed editable package. Does not alter
or synthesize the original pretraining shards/tokenizer.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from mini_llm.tokenizer.tokenizer import FastBPETokenizer

EXTENSIONS = {'.parquet', '.jsonl', '.ndjson', '.json', '.txt', '.md', '.py', '.arrow'}
IGNORE_NAMES = {'readme.md', 'license.txt', 'license.md', 'dataset_info.json', 'dataset_state.json', 'state.json'}
TEXT_FIELDS = ('text', 'content', 'code', 'document', 'body', 'markdown')
DEFAULT_SOURCES = (
    ('fine_edu', '../pretraining_dataset/fine_edu', .4),
    ('cosmopedia_v2', '../pretraining_dataset/cosmopedia_v2', .3),
    ('python_edu', '../pretraining_dataset/python_edu', .3),
)


def _iter_texts(folder: Path, *, text_field: str | None = None):
    if not folder.is_dir():
        raise FileNotFoundError(f'dataset directory missing: {folder}')
    paths = sorted(p for p in folder.rglob('*') if p.is_file()
        and p.name.lower() not in IGNORE_NAMES
        and (p.suffix.lower() in EXTENSIONS or p.name.lower().endswith(('.jsonl.gz','.ndjson.gz'))))
    if not paths:
        raise FileNotFoundError(f'no supported data files in {folder}; expected Parquet, JSONL, JSON or text')
    for path in paths:
        ext = path.suffix.lower()
        if ext == '.gz':
            ext = Path(path.stem).suffix.lower()
        if ext in ('.txt', '.md', '.py'):
            text = path.read_text(encoding='utf8', errors='replace')
            if text.strip():
                yield text
            continue
        if ext == '.arrow':
            try:
                import pyarrow as pa
            except ImportError as exc:
                raise RuntimeError('reading Arrow requires pyarrow: pip install pyarrow') from exc
            with pa.memory_map(str(path), 'r') as fd:
                try:
                    reader = pa.ipc.open_stream(fd)
                    batches = reader
                except pa.ArrowInvalid:
                    fd.seek(0)
                    reader = pa.ipc.open_file(fd)
                    batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
                for batch in batches:
                    available = set(batch.schema.names)
                    field = text_field or next((x for x in TEXT_FIELDS if x in available), None)
                    if field not in available:
                        raise ValueError(f'{path}: no text column among {sorted(available)}; use --text-field source=column')
                    for value in batch.column(batch.schema.get_field_index(field)).to_pylist():
                        if isinstance(value, str) and value.strip():
                            yield value
            continue
        if ext == '.parquet':
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError('reading Parquet requires pyarrow: pip install pyarrow') from exc
            parquet = pq.ParquetFile(path)
            available = set(parquet.schema.names)
            field = text_field or next((x for x in TEXT_FIELDS if x in available), None)
            if field not in available:
                raise ValueError(f'{path}: no text column among {sorted(available)}; use --text-field source=column')
            for batch in parquet.iter_batches(columns=[field], batch_size=128):
                for value in batch.column(0).to_pylist():
                    if isinstance(value, str) and value.strip():
                        yield value
            continue
        if ext in ('.jsonl', '.ndjson'):
            open_text = gzip.open if path.suffix.lower() == '.gz' else open
            with open_text(path, 'rt', encoding='utf8') as inp:
                values = (json.loads(line) for line in inp if line.strip())
                yield from _extract_records(values, path, text_field)
            continue
        if ext == '.json':
            payload = json.loads(path.read_text(encoding='utf8'))
            if isinstance(payload, list):
                yield from _extract_records(payload, path, text_field)
            elif isinstance(payload, dict):
                if isinstance(payload.get('data'), list):
                    yield from _extract_records(payload['data'], path, text_field)
                elif any(k in payload for k in TEXT_FIELDS) or (text_field and text_field in payload):
                    yield from _extract_records([payload], path, text_field)
                # Dataset metadata JSON is not a training document.


def _extract_records(records, path, text_field):
    for record in records:
        if isinstance(record, str):
            value = record
        elif isinstance(record, dict):
            field = text_field or next((k for k in TEXT_FIELDS if isinstance(record.get(k), str)), None)
            if field is None:
                raise ValueError(f'{path}: no supported text field in {sorted(record)}; specify --text-field')
            value = record.get(field)
        else:
            continue
        if isinstance(value, str) and value.strip():
            yield value


def _is_validation_doc(text: str, *, seed: int, fraction: float) -> bool:
    # The same document content always lands in the same split, even when a
    # duplicate occurs in two sources. Do not train the tokenizer on validation.
    digest = hashlib.sha256(str(seed).encode('ascii') + b'\0' + text.encode('utf8')).digest()
    return int.from_bytes(digest[:8], 'little') / (1 << 64) < fraction


def _parse_pair(raw, *, label):
    if '=' not in raw:
        raise ValueError(f'{label} must be NAME=VALUE, got {raw!r}')
    k, v = raw.split('=', 1)
    if not k or not v:
        raise ValueError(f'invalid {label}: {raw!r}')
    return k, v


def prepare(args):
    names = tuple(DEFAULT_SOURCES) if not args.source else tuple(
        (name, path, 1.0) for name, path in (_parse_pair(x, label='--source') for x in args.source))
    fields = dict(_parse_pair(x, label='--text-field') for x in args.text_field)
    source_paths = [(name, Path(path).expanduser().resolve(), float(weight)) for name,path,weight in names]
    if len(set(name for name,_,_ in source_paths)) != len(source_paths):
        raise ValueError('duplicate source name')
    for name, path, _ in source_paths:
        if not path.is_dir():
            raise FileNotFoundError(f'{name}: {path}')
    if args.vocab_size < 512 or args.vocab_size > 65536:
        raise ValueError('vocab size must be within [512,65536]')
    if args.max_documents_per_source < 2:
        raise ValueError('at least two documents per source are required')
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError('val fraction must be in (0,1)')
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'output not empty, refusing to overwrite: {output}')
    output.mkdir(parents=True,exist_ok=True)
    if args.tokenizer:
        tok_path = Path(args.tokenizer).expanduser().resolve()
        shutil.copyfile(tok_path, output/'tokenizer.json')
        tok = FastBPETokenizer.load(str(output/'tokenizer.json'))
    else:
        tok = FastBPETokenizer(vocab_size=args.vocab_size)
        # Training BPE on a bounded set of documents. Iterate without loading all text.
        def train_texts():
            for name, folder, _ in source_paths:
                for i, doc in enumerate(_iter_texts(folder, text_field=fields.get(name))):
                    if i >= args.tokenizer_documents_per_source:
                        break
                    if not _is_validation_doc(doc, seed=args.seed, fraction=args.val_fraction):
                        yield doc[:args.tokenizer_char_limit]
        print('Training small CPU-smoke-test tokenizer...', flush=True)
        tok.train(train_texts(), vocab_size=args.vocab_size)
        tok.save(str(output/'tokenizer.json'))
    vocab_size = len(tok)
    if vocab_size > 65536:
        raise ValueError('uint16 format only supports vocab <=65536')
    eos = tok._tokenizer.token_to_id(tok.eos_token)
    if eos is None:
        raise ValueError('tokenizer missing EOS')
    h = hashlib.sha256((output/'tokenizer.json').read_bytes()).hexdigest()
    result = {}
    for name, folder, weight in source_paths:
        dest = output/name
        dest.mkdir()
        n_train = n_val = train_tokens = val_tokens = 0
        samples = 0
        # Keep documents intact and EOS-terminated. Split by document content,
        # not by token windows, to avoid cross-split leakage.
        with (dest/'train_shard_00000.bin').open('wb') as train_out, \
             (dest/'val_shard_00000.bin').open('wb') as val_out:
            for i, doc in enumerate(_iter_texts(folder, text_field=fields.get(name))):
                if i >= args.max_documents_per_source:
                    break
                ids = tok.encode(doc)
                if not ids:
                    continue
                # Stable document holdout across sources; labels never cross EOS.
                held_out = _is_validation_doc(doc, seed=args.seed, fraction=args.val_fraction)
                ids = list(ids)
                if ids[-1] != eos:
                    ids.append(eos)
                arr = np.asarray(ids, dtype=np.uint16)
                (val_out if held_out else train_out).write(arr.tobytes())
                n_val += int(held_out)
                n_train += int(not held_out)
                val_tokens += len(arr) if held_out else 0
                train_tokens += len(arr) if not held_out else 0
                samples += 1
        if not n_train or not n_val:
            raise RuntimeError(f'{name}: need nonempty train and val splits, got {n_train}/{n_val}; '
                               'increase --max-documents-per-source or --val-fraction')
        result[name] = dict(weight=weight, relative_shard_dir=name,
            train_shards=1,val_shards=1,train_tokens=train_tokens,val_tokens=val_tokens,
            documents=samples,train_documents=n_train,val_documents=n_val,
            input_directory=str(folder))
        print(f'{name}: {samples} docs, train={train_tokens:,} tokens, val={val_tokens:,} tokens',flush=True)
    manifest = dict(format_version='weighted_packed_mixture_v1', complete=True,
        tokenizer_sha256=h,token_dtype='uint16',vocab_size=vocab_size,
        eos_token_id=int(eos), split='sha256(seed,document_text)',
        val_fraction=args.val_fraction,preprocessing_seed=args.seed,sources=result)
    (output/'mixture_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(f'Prepared mixture: {output}; vocab={vocab_size}, tokenizer_sha256={h}',flush=True)
    return manifest


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',action='append', default=[], metavar='NAME=DIR',
        help='override the default three dataset folders')
    p.add_argument('--text-field',action='append',default=[],metavar='NAME=COLUMN')
    p.add_argument('--output',type=Path,default=Path('data/block_proposal_cpu'))
    p.add_argument('--tokenizer',type=Path,help='reuse existing tokenizer.json instead of training a small one')
    p.add_argument('--vocab-size',type=int,default=2048)
    p.add_argument('--max-documents-per-source',type=int,default=128)
    p.add_argument('--tokenizer-documents-per-source',type=int,default=96)
    p.add_argument('--tokenizer-char-limit',type=int,default=4000)
    p.add_argument('--val-fraction',type=float,default=.15)
    p.add_argument('--seed',type=int,default=42)
    return p.parse_args(argv)


if __name__ == '__main__':
    prepare(parse_args())
