"""Export aligned teacher-forced logits from frozen MyLocal_LLM checkpoints.

This first exporter uses independent short-context windows (plain memory mode)
so positions are precisely aligned across different LLM architectures.  Both
models must have identical tokenizer.json bytes and equal vocabulary size.
Long-history routed-prefix evaluation is a separate follow-on integration.

Use --score-last-k / --score-stride to retain multiple predictions per
window without materializing the complete LM head at all context positions.
All exported rows contain *complete* vocabulary logits, never top-k truncation.

Input windows.npy: [num_windows, window_length], containing token IDs including
the next-token target.  With --last-only, score only the final position of each
window; otherwise score all positions in each window except the last.

Example (from repository root):
  MINI_LLM_BACKEND=cupy python -m examples.export_probability_fusion_logits_0082a \\
      --member 0:./checkpoints/deep \\
      --member 1:./checkpoints/wide \\
      --windows ./runtime/fusion_snapshots/train_windows.npy \\
      --output-dir ./runtime/fusion_snapshots/train --last-only
"""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp, asnumpy
from benchmarks.evaluate_wikitext import load_model
from mini_llm.collective.context_windows import checkpoint_fingerprint


def _scored_positions(context_rows, *, last_only=False, score_last_k=None, score_stride=1):
    """Return ascending, causal prediction positions; always include the last.

    position j's logits predict input-window token j+1. This position
    array is shared by all members, labels, and metadata.
    """
    if context_rows < 1 or score_stride < 1:
        raise ValueError('context_rows and score_stride must be positive')
    if last_only and (score_last_k is not None or score_stride != 1):
        raise ValueError('--last-only cannot be combined with score-last-k or score-stride')
    count = 1 if last_only else (context_rows if score_last_k is None else score_last_k)
    if count < 1 or count > context_rows:
        raise ValueError('--score-last-k must be between 1 and the input context length')
    # Working backwards guarantees the last prediction is retained even
    # if count is not a multiple of score_stride.
    return np.arange(context_rows - 1, context_rows - count - 1, -score_stride,
                     dtype=np.int64)[::-1].copy()


def _resolve_tokenizer(checkpoint):
    for p in (checkpoint / 'tokenizer.json', checkpoint.parent / 'tokenizer.json'):
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing tokenizer for {checkpoint}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--member', action='append', required=True, help='Stable model ID:checkpoint_directory')
    parser.add_argument('--windows', required=True, help='NumPy [windows, seq_len] token IDs, including targets')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--last-only', action='store_true', help='Only score the last token in each window')
    parser.add_argument('--context-length', type=int, default=512)
    parser.add_argument('--score-last-k', type=int, default=None,
                        help='Use last K context prediction positions (default: all; --last-only: 1)')
    parser.add_argument('--score-stride', type=int, default=1,
                        help='Keep every S-th prediction position, counting backwards from last')
    parser.add_argument('--lm-head-rows', type=int, default=64,
                        help='Maximum selected hidden-state rows per LM-head projection (VRAM bound)')
    parser.add_argument('--batch-windows', type=int, default=1,
                        help='Number of independent windows per forward (try 4 or 8 on GPU)')
    args = parser.parse_args(argv)
    if args.context_length < 2:
        parser.error('--context-length must be at least 2')
    if args.batch_windows < 1:
        parser.error('--batch-windows must be positive')
    if args.lm_head_rows < 1:
        parser.error('--lm-head-rows must be positive')
    sources = []
    for spec in args.member:
        mid, sep, ckpt = spec.partition(':')
        if not sep:
            parser.error(f'expected MODEL_ID:CHECKPOINT, got {spec!r}')
        try:
            index = int(mid)
        except ValueError:
            parser.error(f'invalid model ID {mid!r}')
        if index < 0 or index in (s[0] for s in sources):
            parser.error('model IDs must be distinct and nonnegative')
        sources.append((index, Path(ckpt)))
    windows = np.load(args.windows, mmap_mode='r', allow_pickle=False)
    if windows.ndim != 2 or windows.shape[0] < 1 or not 2 <= windows.shape[1] <= args.context_length + 1:
        parser.error('windows must be [nonzero_windows, 2..context_length+1]')
    if not np.issubdtype(windows.dtype, np.integer):
        parser.error('window tokens must be integer IDs')
    nwin, width = windows.shape
    try:
        positions = _scored_positions(width - 1, last_only=args.last_only,
                                      score_last_k=args.score_last_k,
                                      score_stride=args.score_stride)
    except ValueError as err:
        parser.error(str(err))
    tokens = np.asarray(windows)
    if np.any(tokens < 0):
        parser.error('negative token IDs are invalid')
    digest_windows = hashlib.sha256()
    with Path(args.windows).open('rb') as window_file:
        for data in iter(lambda: window_file.read(1 << 20), b''):
            digest_windows.update(data)
    token_hash = digest_windows.hexdigest()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scored = int(len(positions))
    labels = np.lib.format.open_memmap(out / 'targets.npy', mode='w+', dtype=np.int32,
                                       shape=(nwin * scored,))
    for row in range(nwin):
        labels[row*scored:(row+1)*scored] = np.asarray(windows[row, positions + 1], dtype=np.int32)
    del labels
    provenance = {'token_windows_sha256': token_hash, 'window_count': int(nwin),
                  'window_length': int(width), 'last_only': args.last_only,
                  'memory_mode': 'plain', 'batch_windows': args.batch_windows,
                  'prediction_positions': positions.tolist(),
                  'scored_positions_per_window': scored,
                  'score_last_k': args.score_last_k, 'score_stride': args.score_stride,
                  'lm_head_rows': args.lm_head_rows,
                  'models': []}
    # Fail on tokenizer mismatch before running expensive model forwards.
    digest_list = [(mid, checkpoint, hashlib.sha256(_resolve_tokenizer(checkpoint).read_bytes()).hexdigest())
                   for mid, checkpoint in sources]
    tokenizer_digest = digest_list[0][2]
    for mid, checkpoint, digest in digest_list:
        if digest != tokenizer_digest:
            raise ValueError(f'tokenizer SHA256 mismatch for model {mid}: {_resolve_tokenizer(checkpoint)}')
    vocab = None
    for mid, checkpoint, digest in digest_list:
        print(f'Loading model {mid}: {checkpoint}', flush=True)
        model, config = load_model(checkpoint, context_length=args.context_length, memory_mode='plain')
        model_vocab = int(config.vocab_size)
        if vocab is None:
            vocab = model_vocab
            if np.any(tokens >= vocab):
                raise ValueError('token ID outside model vocabulary')
        elif vocab != model_vocab:
            raise ValueError('model vocab sizes differ')
        filename = f'model_{mid}_logits.npy'
        estimated_gib = nwin * scored * model_vocab * 2 / (1024 ** 3)
        print(f'model={mid} selected_positions/window={scored} rows={nwin*scored} '
              f'full_vocab_fp16_disk={estimated_gib:.2f} GiB', flush=True)
        exported = np.lib.format.open_memmap(out / filename, mode='w+', dtype=np.float16,
                                               shape=(nwin * scored, vocab))
        model_start = time.perf_counter()
        for start in range(0, nwin, args.batch_windows):
            stop = min(start + args.batch_windows, nwin)
            # Separate windows share no attention context, but can share a GPU
            # forward call.  This preserves exactly the original causal masking.
            batch = np.asarray(windows[start:stop, :-1], dtype=np.int64)
            inputs = xp.asarray(batch)
            # Compute the Transformer hidden states once. Score only the
            # selected causal positions using bounded LM-head tiles.
            # No probability vector is truncated: every exported row has V
            # original vocabulary logits. Output rows are window-major.
            head = model.forward_body(inputs, return_cache=False)
            selected = head[:, positions, :]
            flat = selected.reshape((stop - start) * scored, -1)
            export_start = start * scored
            for tile_start in range(0, len(flat), args.lm_head_rows):
                tile_stop = min(tile_start + args.lm_head_rows, len(flat))
                # BF16 GEMM fast paths require a contiguous tensor.
                tile = xp.array(flat[tile_start:tile_stop, None, :],
                                copy=True, order='C')
                logits, _ = model.output_proj.forward(tile)
                exported[export_start + tile_start: export_start + tile_stop] = (
                    asnumpy(logits[:, 0, :]).astype(np.float16))
                del tile, logits
            del head, selected, flat, inputs
            if stop // 100 != start // 100 or stop == nwin:
                exported.flush()
                rate = stop / max(1e-9, time.perf_counter() - model_start)
                print(f'model={mid} windows={stop}/{nwin} rate={rate:.2f} windows/s', flush=True)
        exported.flush()
        del exported, model
        gc.collect()
        if BACKEND_NAME == 'cupy':
            xp.get_default_memory_pool().free_all_blocks()
        provenance['models'].append({'id': mid, 'checkpoint': str(checkpoint),
                                     'checkpoint_fingerprint': checkpoint_fingerprint(checkpoint),
                                     'logits': filename, 'tokenizer_sha256': digest})
    provenance.update({'tokenizer_sha256': tokenizer_digest, 'vocab_size': vocab,
                       'rows': int(nwin * scored)})
    (out / 'metadata.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(f'Exported {nwin*scored} aligned supervised positions to {out}', flush=True)


if __name__ == '__main__':
    main()
