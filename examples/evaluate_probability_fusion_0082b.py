"""0082B: evaluate a saved 0082A combiner on a separate held-out split.

Reports identical-position individual, arithmetic, geometric and learned-fusion
cross-entropies.  Inputs must be aligned frozen-member logit arrays exported
by 0082A; this tool does not load or update base LLM checkpoints.
"""

import argparse

from mini_llm.collective import ProbabilityResidualCombiner
from mini_llm.collective.context_windows import AlignedContextWindows
from examples.train_probability_fusion_0082a import _open_members, _evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Trained fusion .npz checkpoint')
    parser.add_argument('--member', action='append', required=True, help='Stable ID:logits.npy')
    parser.add_argument('--targets', required=True, help='Aligned ground-truth targets.npy')
    parser.add_argument('--context-windows', help='Token windows used for prediction export (0082J checkpoints)')
    parser.add_argument('--context-metadata', help='Exporter metadata.json with exact row positions and window SHA256')
    parser.add_argument('--batch-rows', type=int, default=2)
    parser.add_argument('--max-rows', type=int, default=0,
                        help='Evenly spaced subset size (0 evaluates all rows)')
    args = parser.parse_args(argv)
    if args.batch_rows < 1 or args.max_rows < 0:
        parser.error('--batch-rows must be positive and --max-rows nonnegative')
    model = ProbabilityResidualCombiner.load(args.checkpoint)
    source = _open_members(args.member, args.targets)
    ids, arrays, _ = source
    if arrays[0].shape[1] != model.config.vocab_size:
        parser.error('fusion checkpoint and member logits vocabularies differ')
    if min(ids) < 0 or max(ids) >= model.config.max_models:
        parser.error('model ID is outside trained combiner capacity')
    context_source = None
    if model.config.context_heads:
        if not args.context_windows or not args.context_metadata:
            parser.error('0082J checkpoints require --context-windows and --context-metadata')
        context_source = AlignedContextWindows(args.context_windows, args.context_metadata,
                                                source[2])
    fusion_ce, arithmetic_ce, geometric_ce, member_ce, weighted_ce = _evaluate(
        model, source, args.batch_rows, max_rows=args.max_rows or None,
        return_weighted=True, context_source=context_source)
    print(f'evaluated_rows={min(args.max_rows, len(source[2])) if args.max_rows else len(source[2])}')
    for mid, ce in zip(ids, member_ce):
        print(f'model_{mid}_ce={ce:.6f}')
    print(f'arithmetic_ce={arithmetic_ce:.6f}')
    print(f'geometric_ce={geometric_ce:.6f}')
    print(f'weighted_ce={weighted_ce:.6f}')
    if model.config.context_heads:
        from mini_llm.backend import asnumpy
        import numpy as np
        from mini_llm.backend import xp
        base_scores = np.asarray(asnumpy(model.reference_logits.data[ids]), dtype=np.float32)
        normalized = np.exp(base_scores - np.max(base_scores))
        normalized /= normalized.sum()
        for mid, weight in zip(ids, normalized):
            print(f'global_weight_model_{int(mid)}={float(weight):.6f}')
    else:
        for mid, weight in zip(ids, model.member_weights(ids)):
            print(f'weight_model_{int(mid)}={float(weight):.6f}')
    print(f'fusion_ce={fusion_ce:.6f}')
    print(f'fusion_minus_arithmetic={fusion_ce-arithmetic_ce:+.6f}')
    print(f'gain_over_arithmetic={arithmetic_ce-fusion_ce:+.6f}')
    print(f'gain_over_weighted={weighted_ce-fusion_ce:+.6f}')
    print(f'gain_over_best_member={float(min(member_ce))-fusion_ce:+.6f}')


if __name__ == '__main__':
    main()
