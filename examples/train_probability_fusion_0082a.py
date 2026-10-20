"""Train 0082A on aligned, frozen-model logit snapshots (.npy arrays).

Each file is [supervised_positions, vocab], float16 or float32 logits.
All member files and target tokens MUST refer to exactly the same token
positions in the same order and use the exact same tokenizer mapping.

Example:
  MINI_LLM_BACKEND=cupy python -m examples.train_probability_fusion_0082a \\
      --member 0:./runtime/fusion_snapshots/deep_train.npy \\
      --member 1:./runtime/fusion_snapshots/wide_train.npy \\
      --targets ./runtime/fusion_snapshots/targets_train.npy \\
      --val-member 0:./runtime/fusion_snapshots/deep_val.npy \\
      --val-member 1:./runtime/fusion_snapshots/wide_val.npy \\
      --val-targets ./runtime/fusion_snapshots/targets_val.npy \\
      --steps 1000 --batch-rows 2 --output ./checkpoints/fusion_0082a.npz

Use data held out from BOTH base LLMs.  Do not use their memorized train
predictions to claim improved generalization.
"""

import argparse
import math
from pathlib import Path
import numpy as np
from mini_llm.backend import xp, scalar, asnumpy
from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner
from mini_llm.optim.adamw import AdamW


def _open_members(specs, targets_path):
    if not specs:
        raise ValueError("provide at least one --member ID:FILE")
    ids, arrays = [], []
    for spec in specs:
        label, sep, path = spec.partition(":")
        if not sep:
            raise ValueError(f"expected MODEL_ID:LOGITS_NPY, got {spec!r}")
        member_id = int(label)
        if member_id in ids:
            raise ValueError(f"duplicate model ID {member_id}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2:
            raise ValueError(f"member logits must be [rows, vocab]: {path}")
        ids.append(member_id)
        arrays.append(array)
    targets = np.load(targets_path, mmap_mode="r", allow_pickle=False)
    if targets.ndim != 1 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("all model logits must have matching [rows, vocab] shapes")
    if targets.shape[0] != arrays[0].shape[0] or not len(targets):
        raise ValueError("targets and member logits must have matching nonzero row counts")
    if np.any(np.asarray(targets) < 0) or np.any(np.asarray(targets) >= arrays[0].shape[1]):
        raise ValueError("invalid targets for vocabulary")
    return np.asarray(ids, dtype=np.int64), arrays, targets


def _batch(arrays, row_idx, member_idx, *, validate_finite=False):
    # The exporter stores host-side FP16 logits. Checking this buffer before
    # CUDA upload avoids a synchronization in the combiner's GPU forward pass.
    host = np.stack([np.asarray(arrays[i][row_idx], dtype=np.float32)
                     for i in member_idx], axis=1)
    if validate_finite and not np.all(np.isfinite(host)):
        raise ValueError('LM logits must be finite')
    return xp.asarray(host)


def _evaluate(model, source, batch_rows, max_rows=None, *, return_weighted=False, context_source=None):
    ids, arrays, targets = source
    weighted_loss, weighted_baseline, weighted_geom, weighted_ref, seen = 0.0, 0.0, 0.0, 0.0, 0
    model_weights = model.member_weights(ids) if return_weighted and not model.config.context_heads else None
    if model.config.context_heads and context_source is None:
        raise ValueError("context-enabled combiner requires aligned validation context")
    member_nll = np.zeros(len(ids), dtype=np.float64)
    if max_rows is not None and max_rows < 1:
        raise ValueError('max_rows must be positive if specified')
    # Deterministic and identical subsets across evaluations/checkpoints.
    # This avoids scanning millions of held-out token rows every 100 steps.
    eval_rows = (np.arange(len(targets)) if max_rows is None or max_rows >= len(targets)
                 else np.linspace(0, len(targets) - 1, num=max_rows, dtype=np.int64))
    for start in range(0, len(eval_rows), batch_rows):
        end = min(start + batch_rows, len(eval_rows))
        row_idx = eval_rows[start:end]
        context_kwargs = {}
        if context_source is not None:
            tokens, lengths = context_source.batch(row_idx)
            context_kwargs = {'context_tokens': tokens, 'context_lengths': lengths}
        prob = model.forward(_batch(arrays, row_idx, range(len(ids))), ids,
                             **context_kwargs)
        gathered = scalar(xp.sum(xp.log(xp.maximum(prob[xp.arange(len(row_idx)),
                                                xp.asarray(targets[row_idx])], 1e-37))))
        loss = -gathered / len(row_idx)
        # Use numerically stable log-probabilities for arithmetic baseline.
        logits = _batch(arrays, row_idx, range(len(ids)))
        z = logits - xp.max(logits, axis=-1, keepdims=True)
        logp = z - xp.log(xp.sum(xp.exp(z), axis=-1, keepdims=True))
        pa = xp.mean(xp.exp(logp), axis=1)
        baseline = -scalar(xp.mean(xp.log(xp.maximum(
            pa[xp.arange(len(row_idx)), xp.asarray(targets[row_idx])], 1e-37))))
        # Independent and geometric references use log-softmax, not floored
        # probabilities, so even very low-probability labels remain measurable.
        selected_members = xp.take_along_axis(
            logp, xp.asarray(targets[row_idx])[:, None, None], axis=2
        )[:, :, 0]
        member_nll += -np.asarray(
            selected_members.get() if hasattr(selected_members, 'get') else selected_members
        ).sum(axis=0)
        if return_weighted:
            if model.config.context_heads:
                model_weights = model.member_weights(ids, **context_kwargs)
            # Only target-token probabilities are needed for baseline CE.
            target_prob = xp.sum(xp.exp(selected_members) * (model_weights if model.config.context_heads else model_weights[None, :]), axis=1)
            weighted_ref += -scalar(xp.sum(xp.log(xp.maximum(target_prob, xp.float32(1e-37)))))
        gm = xp.mean(logp, axis=1)
        gm_shift = gm - xp.max(gm, axis=1, keepdims=True)
        geom_log = gm_shift - xp.log(xp.sum(xp.exp(gm_shift), axis=1, keepdims=True))
        weighted_geom += -scalar(xp.sum(
            geom_log[xp.arange(len(row_idx)), xp.asarray(targets[row_idx])]
        ))
        weighted_loss += loss * len(row_idx)
        weighted_baseline += baseline * len(row_idx)
        seen += len(row_idx)
    result = (weighted_loss / seen, weighted_baseline / seen,
              weighted_geom / seen, member_nll / seen)
    return result + (weighted_ref / seen,) if return_weighted else result


def _sample_epoch_rows(rng, n_rows, n_select, state):
    """Sample every available row once before reshuffling for another epoch.

    state is [permutation, offset] and persists across optimization steps.
    """
    portions = []
    left = n_select
    while left:
        order, offset = state
        if offset == len(order):
            order = rng.permutation(n_rows)
            offset = 0
        take = min(left, len(order) - offset)
        portions.append(order[offset:offset + take])
        state[0], state[1] = order, offset + take
        left -= take
    return np.concatenate(portions)


def _clip_grad_norm_(parameters, max_norm, *, device_only=False):
    """Globally clip fusion gradients without constraining residual logits.

    Returns the pre-clipping Euclidean norm. `max_norm=0` disables clipping.
    Model output corrections remain entirely unbounded during inference.
    """
    if max_norm == 0:
        return None
    params = list(parameters)
    squared = xp.asarray(0.0, dtype=xp.float32)
    for param in params:
        squared = squared + xp.sum(xp.asarray(param.grad, dtype=xp.float32) ** 2)
    if device_only:
        # No branch on a GPU scalar: device-side scaling prevents a host
        # synchronization per step. This is equivalent to the legacy clipped
        # norm to FP32 tolerance for finite gradients.
        norm = xp.sqrt(squared)
        factor = xp.minimum(xp.float32(1.0),
                            xp.float32(max_norm) / (norm + xp.float32(1e-12)))
        for param in params:
            param.grad *= factor
        return norm
    norm = math.sqrt(float(scalar(squared)))
    if not math.isfinite(norm):
        raise FloatingPointError(f"fusion gradient norm is nonfinite: {norm}")
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for param in params:
            param.grad *= scale
    return norm


def _should_stop_early(stale_evals, patience):
    return patience > 0 and stale_evals >= patience


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--member", action="append", required=True, help="Model ID:train_logits.npy")
    parser.add_argument("--targets", required=True, help="Aligned train targets.npy")
    parser.add_argument("--val-member", action="append", help="Model ID:val_logits.npy")
    parser.add_argument("--val-targets", help="Aligned val targets.npy")
    parser.add_argument("--max-models", type=int, default=10)
    parser.add_argument("--fusion-embed-dim", "--embed-dim", dest="embed_dim", type=int, default=16)
    parser.add_argument("--fusion-member-width", "--member-width", dest="member_width", type=int, default=8)
    parser.add_argument("--fusion-decoder-width", "--decoder-width", dest="decoder_width", type=int, default=24)
    parser.add_argument("--fusion-decoder-layers", "--decoder-layers", dest="decoder_layers", type=int, default=1)
    parser.add_argument("--batch-rows", type=int, default=2)
    parser.add_argument("--val-max-rows", type=int, default=0,
                        help="Fixed evenly-spaced validation subset (0: full set)")
    parser.add_argument("--sample-all-rows", action="store_true",
                        help="Shuffled epochs without replacement instead of random row replacement")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Optional AdamW weight decay (not a residual-output constraint)")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Optional global gradient norm clip (0: disabled)")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop after this many validations without improvement (0: disabled)")
    parser.add_argument("--min-members", type=int, default=2,
                        help="Random member subset size minimum (for eventual 2..10 generalization)")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=82)
    parser.add_argument("--output", required=True, help="Best model .npz output")
    args = parser.parse_args(argv)
    if min(args.embed_dim, args.member_width, args.decoder_width, args.decoder_layers) < 1:
        parser.error("fusion size parameters must be >=1")
    if args.batch_rows < 1 or args.steps < 1 or args.min_members < 1 or args.val_max_rows < 0:
        parser.error("batch-rows, steps and min-members must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be positive and finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be nonnegative and finite")
    if not math.isfinite(args.grad_clip) or args.grad_clip < 0:
        parser.error("--grad-clip must be nonnegative and finite")
    if args.early_stop_patience < 0:
        parser.error("--early-stop-patience must be nonnegative")
    if args.early_stop_patience and not args.val_member:
        parser.error("--early-stop-patience requires validation members")
    if args.val_member is None and args.val_targets is not None:
        parser.error("--val-targets requires --val-member")
    if args.val_member is not None and args.val_targets is None:
        parser.error("--val-member requires --val-targets")
    ids, arrays, targets = _open_members(args.member, args.targets)
    if args.min_members > len(ids):
        parser.error("--min-members cannot exceed provided model count")
    if np.any(ids < 0) or np.any(ids >= args.max_models):
        parser.error("model ID outside --max-models range")
    val = None
    if args.val_member:
        val = _open_members(args.val_member, args.val_targets)
        if sorted(val[0].tolist()) != sorted(ids.tolist()):
            parser.error("train and validation must have the same model identities")
        if val[1][0].shape[1] != arrays[0].shape[1]:
            parser.error("train/validation vocabulary sizes disagree")
    config = FusionConfig(vocab_size=arrays[0].shape[1], max_models=args.max_models,
                          embed_dim=args.embed_dim, member_width=args.member_width,
                          decoder_width=args.decoder_width, decoder_layers=args.decoder_layers, seed=args.seed)
    model = ProbabilityResidualCombiner(config)
    print(f'fusion architecture: weighted_reference=trainable_by_model_id (equal_initially) '
          f'embed_dim={args.embed_dim} member_width={args.member_width} '
          f'decoder_width={args.decoder_width} decoder_layers={args.decoder_layers} '
          f'parameters={sum(int(p.data.size) for p in model.parameters()):,}; '
          f'input=[prediction_rows,models,{arrays[0].shape[1]}] (no context axis)', flush=True)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    best_score = float("inf")
    best_step = None
    stale_evals = 0
    running_loss = running_baseline = running_weighted = 0.0
    running_batches = 0
    epoch_state = [rng.permutation(len(targets)), 0] if args.sample_all_rows else None
    for step in range(1, args.steps + 1):
        row_idx = (_sample_epoch_rows(rng, len(targets), args.batch_rows, epoch_state)
                   if args.sample_all_rows else
                   rng.integers(0, len(targets), size=args.batch_rows))
        count = int(rng.integers(args.min_members, len(ids) + 1))
        member_idx = rng.choice(len(ids), size=count, replace=False)
        opt.zero_grad()
        loss, baseline, weighted_ce = model.loss_and_backward(
            _batch(arrays, row_idx, member_idx), ids[member_idx], targets[row_idx],
            return_weighted=True
        )
        grad_norm = _clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        running_loss += loss
        running_baseline += baseline
        running_weighted += weighted_ce
        running_batches += 1
        if step == 1 or step % args.log_interval == 0:
            log_suffix = (f" grad_norm={grad_norm:.5f}" if grad_norm is not None else "")
            print(f"step={step} members={count} train_ce={loss:.5f} "
                  f"arithmetic_ce={baseline:.5f} weighted_ce={weighted_ce:.5f} "
                  f"train_ce_mean={running_loss/running_batches:.5f} "
                  f"arithmetic_ce_mean={running_baseline/running_batches:.5f} "
                  f"weighted_ce_mean={running_weighted/running_batches:.5f} "
                  f"train_gain_mean={(running_baseline-running_loss)/running_batches:+.5f} "
                  f"sampled_rows={step*args.batch_rows}{log_suffix}", flush=True)
            running_loss = running_baseline = running_weighted = 0.0
            running_batches = 0
        if val and (step == 1 or step % args.eval_interval == 0 or step == args.steps):
            metrics = _evaluate(model, val, args.batch_rows,
                                max_rows=args.val_max_rows or None,
                                return_weighted=True)
            if len(metrics) == 4:  # Support existing test/evaluator adapters.
                val_loss, val_baseline, geom_ce, member_ce = metrics
                weighted_val = val_baseline
            else:
                val_loss, val_baseline, geom_ce, member_ce, weighted_val = metrics
            scores = ' '.join(f'model_{mid}_ce={ce:.5f}'
                              for mid, ce in zip(val[0], member_ce))
            mixture = ' '.join(f'weight_model_{int(mid)}={float(w):.4f}'
                               for mid, w in zip(val[0], asnumpy(model.member_weights(val[0]))))
            print(f"step={step} val_ce={val_loss:.5f} "
                  f"arithmetic_ce={val_baseline:.5f} geometric_ce={geom_ce:.5f} "
                  f"weighted_ce={weighted_val:.5f} "
                  f"gain={val_baseline-val_loss:+.5f} "
                  f"gain_weighted={weighted_val-val_loss:+.5f} "
                  f"{mixture} {scores}", flush=True)
            if val_loss < best_score:
                best_score = val_loss
                best_step = step
                stale_evals = 0
                model.save(args.output)
                print(f"fusion_best step={step} val_ce={best_score:.5f} "
                      f"saved={args.output}", flush=True)
            else:
                stale_evals += 1
            if _should_stop_early(stale_evals, args.early_stop_patience):
                print(f"fusion_early_stop step={step} best_step={best_step} "
                      f"best_val_ce={best_score:.5f} "
                      f"stale_validations={stale_evals}", flush=True)
                break
    if val is None:
        model.save(args.output)
    else:
        print(f"fusion_best_final step={best_step} val_ce={best_score:.5f}", flush=True)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
