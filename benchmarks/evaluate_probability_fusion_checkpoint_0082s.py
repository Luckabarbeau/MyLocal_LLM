#!/usr/bin/env python3
"""0082S: streaming, independent test-corpus evaluation of frozen-member fusion.

Each member is loaded *sequentially* per bounded segment. This avoids requiring
all frozen LLMs in 16 GiB VRAM together, and never materializes full-corpus
full-vocabulary logits. Shared token windows and labels are identical for all
methods. Context-disabled fusion checkpoints only (the tested 0082F setup).

Run from repository root, e.g.:
 MINI_LLM_BACKEND=cupy python benchmarks/evaluate_probability_fusion_checkpoint_0082s.py \\
   --fusion-checkpoint ./checkpoints/fusion_0082f_deepwide_streaming_weighted_context_accum3.npz \\
   --member 0:./checkpoints/0080BX_deep512_fullmix_100k \\
   --member 1:./checkpoints/0080BX_wide768_fullmix_100k \\
   --data ../benchmarks/wikitext-2-raw/wiki.test.raw --format text \\
   --context-length 512 --stride 512 --max-tokens 4096

LAMBADA is evaluated using only the final target word's tokenizer tokens;
word accuracy requires every teacher-forced greedy prediction to match.
"""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

# The existing benchmark modules are scripts, rather than package-relative
# imports. This makes their proven corpus/tokenizer helpers importable both
# when this file is executed and when it is exercised under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_corpus import tokenize_documents  # noqa: E402
from evaluate_lambada import iter_lambada, split_lambada_target  # noqa: E402
from evaluate_wikitext import load_model, load_tokenizer  # noqa: E402
from mini_llm.backend import BACKEND_NAME, xp, asnumpy, synchronize  # noqa: E402
from mini_llm.collective import ProbabilityResidualCombiner  # noqa: E402


def parse_members(specs):
    if not specs:
        raise ValueError('provide at least one --member ID:CHECKPOINT')
    sources = []
    for spec in specs:
        label, sep, location = spec.partition(':')
        if not sep or not location:
            raise ValueError(f'expected MODEL_ID:CHECKPOINT, got {spec!r}')
        mid = int(label)
        if mid < 0 or any(mid == prev for prev, _ in sources):
            raise ValueError(f'model ID must be unique and nonnegative: {mid}')
        sources.append((mid, Path(location)))
    return sources


def tokenizer_path(checkpoint):
    for candidate in (checkpoint / 'tokenizer.json', checkpoint.parent / 'tokenizer.json'):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'tokenizer.json not found for {checkpoint}')


def validate_member_tokenizers(sources):
    reference = None
    for mid, checkpoint in sources:
        if not (checkpoint / 'config.json').is_file():
            raise FileNotFoundError(checkpoint / 'config.json')
        path = tokenizer_path(checkpoint)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if reference is not None and digest != reference:
            raise ValueError(f'tokenizer SHA256 mismatch for member {mid}: {path}')
        reference = digest
    return reference


def iter_corpus_windows(streams, *, context, stride):
    """Match evaluate_wikitext.run_windowed_evaluation's exact scored positions.

    Yield (input_ids, selected_output_positions, target_ids, example_id).
    Stride determines scoring frequency; each token is scored exactly once.
    """
    if context < 1 or not 1 <= stride <= context:
        raise ValueError('context must be positive and 1 <= stride <= context')
    for doc_idx, stream in enumerate(streams):
        for target_start in range(1, len(stream), stride):
            end = min(target_start + stride, len(stream))
            begin = max(0, end - (context + 1))
            seq = np.asarray(stream[begin:end], dtype=np.int64)
            offset = target_start - begin - 1
            positions = np.arange(offset, len(seq) - 1, dtype=np.int64)
            targets = seq[positions + 1]
            if not len(positions) or int(positions[0]) < 0:
                raise RuntimeError('invalid corpus scoring window')
            yield seq[:-1], positions, targets, doc_idx


def iter_lambada_windows(data, tokenizer, *, context, text_field, max_examples):
    """One full causal input per target word, score only its target-token suffix.

    Long contexts are truncated to the most recent `context` input tokens;
    very long target words that do not fit are skipped (with counts printed).
    """
    valid = skipped = 0
    for example in iter_lambada(data, text_field):
        if max_examples is not None and valid >= max_examples:
            break
        try:
            history, target, _ = split_lambada_target(tokenizer, example)
        except ValueError:
            skipped += 1
            continue
        full = np.asarray(history + target, dtype=np.int64)
        begin = max(0, len(full) - (context + 1))
        position = len(history) - 1 - begin
        if position < 0 or len(target) > context:
            skipped += 1
            continue
        positions = np.arange(position, len(full) - begin - 1, dtype=np.int64)
        if len(positions) != len(target):
            raise RuntimeError('LAMBADA scoring mismatch')
        yield full[begin:-1], positions, full[begin + positions + 1], valid
        valid += 1
    # Skips are also tracked in the returned benchmark summary via the caller.


def segment_windows(windows, max_windows):
    if max_windows < 1:
        raise ValueError('segment-windows must be positive')
    pending = []
    for window in windows:
        pending.append(window)
        if len(pending) == max_windows:
            yield pending
            pending = []
    if pending:
        yield pending


def export_one_member(model, windows, *, output_file, vocab, lm_head_rows, batch_windows):
    """Bounded LM-head projection using the same fast path as 0082A export.

    Groups only adjacent windows of identical input length and selected positions
    so that batching never inserts padding tokens or changes positional meaning.
    """
    row_count = sum(len(item[2]) for item in windows)
    storage = np.lib.format.open_memmap(output_file, mode='w+', dtype=np.float16,
                                        shape=(row_count, vocab))
    offsets = np.cumsum([0] + [len(item[2]) for item in windows])
    i = 0
    try:
        while i < len(windows):
            inputs, positions, _, _ = windows[i]
            end = i + 1
            while (end < len(windows) and end - i < batch_windows
                   and len(windows[end][0]) == len(inputs)
                   and np.array_equal(windows[end][1], positions)):
                end += 1
            batch = xp.asarray(np.stack([w[0] for w in windows[i:end]], axis=0))
            body = model.forward_body(batch, return_cache=False)
            selected = body[:, positions, :].reshape(-1, body.shape[-1])
            base = int(offsets[i])
            for start in range(0, selected.shape[0], lm_head_rows):
                stop = min(start + lm_head_rows, selected.shape[0])
                tile = xp.array(selected[start:stop, None, :], copy=True, order='C')
                logits, _ = model.output_proj.forward(tile)
                storage[base + start:base + stop] = asnumpy(logits[:, 0, :]).astype(np.float16)
                del tile, logits
            del batch, body, selected
            i = end
    finally:
        storage.flush()
        del storage


def score_rows(combiner, ids, arrays, targets, *, batch_rows, example_ids=None):
    """All comparisons evaluated on the same aligned logits and target tokens.

    Returns NLL sums for each method; for LAMBADA, also word-exact counts.
    """
    vocab = int(combiner.config.vocab_size)
    n = len(targets)
    results = {'fusion': 0., 'weighted': 0., 'arithmetic': 0., 'geometric': 0.}
    results.update({f'model_{int(mid)}': 0. for mid in ids})
    exact = {key: 0 for key in results} if example_ids is not None else None
    exact_states = {key: {} for key in results} if example_ids is not None else None
    weights = xp.asarray(combiner.member_weights(ids), dtype=xp.float32)
    for start in range(0, n, batch_rows):
        end = min(start + batch_rows, n)
        block = np.stack([np.asarray(arr[start:end], dtype=np.float32) for arr in arrays], axis=1)
        if not np.isfinite(block).all():
            raise FloatingPointError('nonfinite parent logits in export')
        logits = xp.asarray(block)
        y = xp.asarray(targets[start:end], dtype=xp.int64)
        row = xp.arange(end - start)
        z = logits - xp.max(logits, axis=2, keepdims=True)
        logp = z - xp.log(xp.sum(xp.exp(z), axis=2, keepdims=True))
        p = xp.exp(logp)
        member_true = logp[row[:, None], xp.arange(len(ids))[None, :], y[:, None]]
        for j, mid in enumerate(ids):
            key = f'model_{int(mid)}'
            results[key] += -float(asnumpy(xp.sum(member_true[:, j])))
        arithmetic = xp.mean(p, axis=1)
        weighted = xp.sum(p * weights[None, :, None], axis=1)
        geom = xp.mean(logp, axis=1)
        geom -= xp.log(xp.sum(xp.exp(geom - xp.max(geom, axis=1, keepdims=True)), axis=1,
                                   keepdims=True)) + xp.max(geom, axis=1, keepdims=True)
        fused = combiner.forward(logits, ids)
        pred = {'arithmetic': arithmetic, 'weighted': weighted,
                'fusion': fused}
        for name, probs in pred.items():
            results[name] += -float(asnumpy(xp.sum(xp.log(xp.maximum(
                probs[row, y], xp.float32(1e-37))))))
        results['geometric'] += -float(asnumpy(xp.sum(geom[row, y])))
        if example_ids is not None:
            preds = {key: np.asarray(asnumpy(xp.argmax(val, axis=-1)), dtype=np.int64)
                     for key, val in pred.items()}
            preds['geometric'] = np.asarray(asnumpy(xp.argmax(geom, axis=-1)), dtype=np.int64)
            for j, mid in enumerate(ids):
                preds[f'model_{int(mid)}'] = np.asarray(
                    asnumpy(xp.argmax(logits[:, j, :], axis=-1)), dtype=np.int64)
            for k in range(start, end):
                eid = int(example_ids[k])
                for name, guessed in preds.items():
                    state = exact_states[name]
                    state[eid] = state.get(eid, True) and (int(guessed[k - start]) == int(targets[k]))
        del logits, z, logp, p, arithmetic, weighted, geom, fused
    if exact_states is not None:
        for key, state in exact_states.items():
            exact[key] = sum(bool(value) for value in state.values())
    return results, exact


def _evaluate_segment(combiner, sources, windows, *, temp_path, context, lm_head_rows,
                      batch_windows, batch_rows, lambada=False):
    ids = np.asarray([mid for mid, _ in sources], dtype=np.int64)
    vocab = int(combiner.config.vocab_size)
    targets = np.concatenate([np.asarray(w[2], dtype=np.int64) for w in windows])
    example_ids = np.concatenate([np.full(len(w[2]), w[3], dtype=np.int64) for w in windows])
    paths = []
    teacher_times = {}
    for mid, checkpoint in sources:
        path = temp_path / f'model_{mid}.npy'
        start = time.perf_counter()
        model, cfg = load_model(checkpoint, context_length=context, memory_mode='plain')
        if int(cfg.vocab_size) != vocab:
            raise ValueError(f'member {mid} vocab_size {cfg.vocab_size} != fusion {vocab}')
        try:
            export_one_member(model, windows, output_file=path, vocab=vocab,
                              lm_head_rows=lm_head_rows, batch_windows=batch_windows)
        finally:
            del model
            gc.collect()
            if BACKEND_NAME == 'cupy':
                xp.get_default_memory_pool().free_all_blocks()
        teacher_times[f'model_{mid}'] = time.perf_counter() - start
        paths.append(path)
    arrays = [np.load(path, mmap_mode='r', allow_pickle=False) for path in paths]
    try:
        sums, exact = score_rows(combiner, ids, arrays, targets, batch_rows=batch_rows,
                                 example_ids=example_ids if lambada else None)
    finally:
        del arrays
    return sums, exact, teacher_times, len(targets)


def run(args):
    sources = parse_members(args.member)
    validate_member_tokenizers(sources)
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.context_length < 2 or args.batch_rows < 1 or args.batch_windows < 1 or args.lm_head_rows < 1:
        raise ValueError('context-length >=2, batch-rows/batch-windows/lm-head-rows >0 required')
    if args.stride is None:
        args.stride = args.context_length
    if not 1 <= args.stride <= args.context_length:
        raise ValueError('stride must be in [1, context-length]')
    combiner = ProbabilityResidualCombiner.load(args.fusion_checkpoint)
    if combiner.config.context_heads:
        raise ValueError('0082S evaluates context_heads=0 only; contextual fusion requires aligned context metadata')
    ids = [mid for mid, _ in sources]
    if max(ids) >= combiner.config.max_models:
        raise ValueError('member ID outside the trained combiner capacity')
    tokenizer, tpath = load_tokenizer(sources[0][1])
    if args.data_format == 'lambada':
        if args.max_tokens is not None or args.max_documents is not None:
            raise ValueError('--max-tokens and --max-documents apply to corpus benchmarks only')
        windows = iter_lambada_windows(args.data, tokenizer,
                                       context=args.context_length, text_field=args.text_field,
                                       max_examples=args.max_examples)
        token_count = source_documents = source_bytes = None
    else:
        if args.max_examples is not None:
            raise ValueError('--max-examples applies to LAMBADA only')
        streams, token_count, source_documents, _, source_bytes = tokenize_documents(
            args.data, tokenizer, data_format=args.data_format, text_field=args.text_field,
            max_documents=args.max_documents, max_tokens=args.max_tokens)
        windows = iter_corpus_windows(streams, context=args.context_length, stride=args.stride)
    totals = {key: 0. for key in ['fusion', 'weighted', 'arithmetic', 'geometric']}
    totals.update({f'model_{mid}': 0. for mid in ids})
    exact_totals = {key: 0 for key in totals}
    count = 0
    examples = 0
    teacher_time = {f'model_{mid}': 0. for mid in ids}
    segment_number = 0
    started = time.perf_counter()
    print(f'0082S fusion benchmark: {args.dataset_name or args.data.name}', flush=True)
    print(f'  members={sources} fusion={args.fusion_checkpoint} vocab={combiner.config.vocab_size}', flush=True)
    print(f'  tokenizer={tpath} context={args.context_length} stride={args.stride} memory=plain', flush=True)
    print(f'  segment_windows={args.segment_windows} batch_windows={args.batch_windows} '
          f'lm_head_rows={args.lm_head_rows} fusion_batch_rows={args.batch_rows}', flush=True)
    synchronize()
    with tempfile.TemporaryDirectory(prefix='0082s_fusion_', dir=args.temp_root) as folder:
        workdir = Path(folder)
        for segment in segment_windows(windows, args.segment_windows):
            # Remove previous segment's full-vocabulary arrays before next export.
            for old in workdir.glob('model_*.npy'):
                old.unlink()
            segment_number += 1
            sums, exact, times, rows = _evaluate_segment(
                combiner, sources, segment, temp_path=workdir,
                context=args.context_length, lm_head_rows=args.lm_head_rows,
                batch_windows=args.batch_windows, batch_rows=args.batch_rows,
                lambada=args.data_format == 'lambada')
            if args.data_format == 'lambada':
                examples += len(segment)
                for name in exact_totals:
                    exact_totals[name] += exact[name]
            for key, value in sums.items():
                totals[key] += value
            count += rows
            for key, value in times.items():
                teacher_time[key] += value
            print(f'  segment={segment_number} scored={count:,} '
                  f'fusion_CE={totals["fusion"]/count:.6f} '
                  f'best_member_CE={min(totals[f"model_{mid}"] for mid in ids)/count:.6f}', flush=True)
    synchronize()
    elapsed = time.perf_counter() - started
    if count == 0:
        raise RuntimeError('no scoreable token positions')
    print('\n' + '=' * 64)
    print(f'{args.dataset_name or args.data.name} — fusion benchmark results')
    print('=' * 64)
    print(f'Scored tokens:       {count:,}')
    print(f'Context length:      {args.context_length:,} (plain)')
    print(f'Stride:              {args.stride:,}')
    print(f'Segments:            {segment_number:,}')
    if examples:
        print(f'LAMBADA words:       {examples:,}')
    for key, nll in totals.items():
        ce = nll / count
        print(f'{key:20s} CE={ce:.6f} PPL={math.exp(ce):.4f}' +
              (f' exact_words={exact_totals[key]}/{examples} '
               f'accuracy={exact_totals[key]/examples:.2%}' if examples else ''))
    if source_bytes and args.max_tokens is None and args.max_documents is None:
        print(f'Fusion bits/byte:    {totals["fusion"]/(source_bytes * math.log(2.0)):.6f}')
        print(f'Tokens per byte:     {token_count/source_bytes:.6f}')
    print(f'Gain over weighted:  {(totals["weighted"]-totals["fusion"])/count:+.6f} CE')
    print(f'Gain over best:      {(min(totals[f"model_{mid}"] for mid in ids)-totals["fusion"])/count:+.6f} CE')
    print(f'Elapsed:             {elapsed:.2f} s')
    print(f'Scored tok/s:        {count/elapsed:.1f}')
    print('Teacher time:        ' + ', '.join(f'{k}={v:.1f}s' for k, v in teacher_time.items()))
    if args.json_output:
        output = {'benchmark': args.dataset_name or args.data.name,
                  'fusion_checkpoint': str(args.fusion_checkpoint),
                  'member_checkpoints': {str(mid): str(path) for mid, path in sources},
                  'context_length': args.context_length, 'stride': args.stride,
                  'memory_mode': 'plain', 'scored_tokens': count, 'segments': segment_number,
                  'cross_entropy': {key: val / count for key, val in totals.items()},
                  'perplexity': {key: math.exp(val / count) for key, val in totals.items()},
                  'exact_words': exact_totals if examples else None,
                  'lambada_examples': examples if examples else None,
                  'elapsed_seconds': elapsed, 'scored_tokens_per_second': count / elapsed}
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--fusion-checkpoint', required=True, type=Path)
    parser.add_argument('--member', required=True, action='append', help='MODEL_ID:CHECKPOINT_DIR')
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--format', required=True, choices=['wikitext','text','jsonl','lambada'],
                        dest='data_format')
    parser.add_argument('--dataset-name')
    parser.add_argument('--text-field', default='text')
    parser.add_argument('--context-length', type=int, default=512)
    parser.add_argument('--stride', type=int)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--max-documents', type=int)
    parser.add_argument('--max-examples', type=int)
    parser.add_argument('--segment-windows', type=int, default=128,
                        help='128x512x2 members ~8 GiB temporary disk cache')
    parser.add_argument('--lm-head-rows', type=int, default=64)
    parser.add_argument('--batch-windows', type=int, default=1)
    parser.add_argument('--batch-rows', type=int, default=8)
    parser.add_argument('--temp-root', type=Path, default=None)
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args(argv)
    if any(x is not None and x < 1 for x in (args.max_tokens, args.max_documents, args.max_examples)):
        parser.error('max-tokens, max-documents and max-examples must be positive')
    if args.temp_root is not None:
        args.temp_root.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == '__main__':
    main()
