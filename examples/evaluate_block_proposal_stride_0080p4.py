"""0080P4: held-out causal-inference CE for K=4 commitment strides S=1,2,4.

The fixed 0080I validation samples and checkpoint/corpus provenance are verified
exactly as in 0080K.  This test is NOT the training forward: it uses the same
incremental causal main KV and decoder cache as actual generation, but commits
true previous tokens so all strides are evaluated on identical continuations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.block_proposal.benchmark_adapter import load_0080i_block_checkpoint
from mini_llm.block_proposal.generation import SequentialCommitBlockGenerator
from mini_llm.block_proposal.real_data import PackedRoutedBlockProposalDocuments
from mini_llm.inference_model import InferenceModel
from mini_llm.predictive_refinement.real_data import sha256
from examples.evaluate_block_proposal_ablations_0080j import _checkpoint, _reconstruct
from examples.train_predictive_refinement_text import _load_tokenizer
from examples.train_block_proposal_cpu_learning_0080i import _signature


def _parse_strides(value):
    strides = [int(s.strip()) for s in value.split(',')]
    if not strides or len(set(strides)) != len(strides) or any(s < 1 for s in strides):
        raise argparse.ArgumentTypeError('strides must be unique positive comma-separated integers')
    return strides


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mixed-shard-root', type=Path, required=True)
    p.add_argument('--training-output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--strides', type=_parse_strides, default=[4,2,1])
    p.add_argument('--validation-rows', type=int, default=8,
                   help='Number of fixed validation examples to score (default: 8)')
    p.add_argument('--target-tokens', type=int, default=32,
                   help='Maximum tokens per document continuation (default: 32)')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.validation_rows <= 0 or args.target_tokens <= 0:
        p.error('--validation-rows and --target-tokens must be positive')
    return args


def score_fixed_rows(net, rows, *, strides, max_context, target_tokens):
    """Rows: (genuine causal prefix IDs, held-out continuation IDs)."""
    totals = {s: {'nll_sum':0.,'tokens':0,'plans':0,'rows':0} for s in strides}
    for prompt, continuation in rows:
        prompt = [int(x) for x in prompt]
        continuation = [int(x) for x in continuation[:target_tokens]]
        if not prompt or not continuation:
            continue
        if len(prompt) + len(continuation) > max_context:
            raise ValueError('prompt + targets exceed checkpoint memory horizon; reduce --target-tokens')
        for stride in strides:
            inf = InferenceModel(net.proposal_head.model.config,
                                 dtype=net.proposal_head.model.dtype)
            inf.set_weights(net.proposal_head.model)
            state = inf.create_generation_state(batch_size=1, max_length=max_context)
            inf.prefill(xp.asarray([prompt], dtype=xp.int32), state)
            gen = SequentialCommitBlockGenerator(inf,net.proposal_head,net.decoder,
                chooser=lambda v: int(xp.argmax(v)),commit_stride=stride)
            score = gen.score_reference_tokens(state, continuation)
            total = totals[stride]
            total['nll_sum'] += score['nll_sum']
            total['tokens'] += score['n_tokens']
            total['plans'] += score['proposal_blocks']
            total['rows'] += 1
            if state.length != len(prompt) + len(continuation):
                raise RuntimeError('teacher-forced causal state failed to advance exactly')
    if any(value['tokens'] == 0 for value in totals.values()):
        raise ValueError('no valid held-out targets were scored')
    return {str(s): dict(**totals[s],
                final_ce=totals[s]['nll_sum']/totals[s]['tokens'],
                perplexity=float(np.exp(totals[s]['nll_sum']/totals[s]['tokens'])),
                plans_per_token=totals[s]['plans']/totals[s]['tokens'])
            for s in strides}


def main(argv=None):
    if BACKEND_NAME != 'numpy':
        raise RuntimeError('0080P4 frozen CPU scoring requires MINI_LLM_BACKEND=numpy')
    a = parse_args(argv)
    out = a.output.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f'refusing to overwrite existing stride evaluation: {out}')
    checkpoint = _checkpoint(a)
    signature = json.loads((checkpoint/'training_state.json').read_text())['data_provenance']
    root = a.mixed_shard_root.expanduser().resolve()
    tok = root/'tokenizer.json'
    tokenizer, eos = _load_tokenizer(tok)
    opts, cfg = _reconstruct(signature, len(tokenizer))
    k = opts.block_size
    if any(s > k for s in a.strides):
        raise ValueError(f'commit strides must be in [1, K={k}]')
    train = PackedRoutedBlockProposalDocuments(mixed_root=root,eos_token_id=eos,token_dtype='uint16',split='train')
    val = PackedRoutedBlockProposalDocuments(mixed_root=root,eos_token_id=eos,token_dtype='uint16',split='val')
    rng = np.random.default_rng(opts.seed+4)
    fixed = [val.sample_causal(rng,batch_size=opts.batch_size,
             memory_length=cfg.memory_context.distant_memory_length,
             target_length=cfg.memory_context.target_length,
             min_history_tokens=(cfg.memory_context.top_k_blocks+1)*cfg.memory_context.block_size-1)
             for _ in signature['validation_digests']]
    expected = {key:value for key,value in signature.items() if key != 'warm_start'}
    if _signature(opts,cfg,root,tok,train,val,fixed) != expected:
        raise ValueError('fixed validation samples, tokenizer or mixture differ from checkpoint')
    net, _, info = load_0080i_block_checkpoint(checkpoint, tok)
    max_context = int(cfg.memory_context.memory_length)
    rows = []
    prefix_boundary = int(cfg.memory_context.distant_memory_length)
    for source, targets, valid, md, _ in fixed:
        for i in range(len(source)):
            count = min(int(np.sum(valid[i])), a.target_tokens)
            if not count:
                continue
            prompt = source[i,int(md['history_valid_starts'][i]):prefix_boundary+1]
            # Trim old history ONLY when needed to stay inside the trained
            # inference horizon; never lose the immediately preceding token.
            capacity = max_context - count
            prompt = prompt[-capacity:]
            rows.append((prompt,targets[i,:count]))
            if len(rows) >= a.validation_rows:
                break
        if len(rows) >= a.validation_rows:
            break
    print(f'0080P4 checkpoint step {info["step"]}, {len(rows)} fixed held-out rows, '
          f'max {a.target_tokens} targets each, K={k}',flush=True)
    results = score_fixed_rows(net,rows,strides=a.strides,
                               max_context=max_context,target_tokens=a.target_tokens)
    for stride in a.strides:
        r = results[str(stride)]
        print(f'  S={stride}: CE={r["final_ce"]:.5f}, tokens={r["tokens"]}, '
              f'plans={r["plans"]}, plans/token={r["plans_per_token"]:.3f}',flush=True)
    report = dict(experiment='0080P4_commit_stride_teacher_forced_inference',
        checkpoint=str(checkpoint),checkpoint_step=info['step'],
        tokenizer_sha256=sha256(tok),validation_digests=signature['validation_digests'],
        selected_rows=len(rows),max_target_tokens_per_row=a.target_tokens,
        proposal_horizon=k,strides=results,
        comparison_note=('Frozen model, same held-out target sequence, real causal KV updates. '
            'This is inference-procedure teacher-forcing CE, not training-window CE. '
            'Replanning each token changes boundary alignment; no checkpoint retraining.'))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(f'Wrote {out}',flush=True)
    return report


if __name__ == '__main__':
    main()
