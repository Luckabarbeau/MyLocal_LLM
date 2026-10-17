"""0080AH+AI standalone CUDA module benchmark (NOT full model throughput).

Run from repository root:
  MINI_LLM_BACKEND=cupy python -m benchmarks.benchmark_block_proposal_vocab_streaming \
      --source-scope all --block-groups 4 8 16 32 64 --repeat 3

Reports synchronized forward/backward times, full-vocab positions/s,
current and retained CuPy pool bytes.  Full model VRAM/speed requires a
separate production-window benchmark after cache and checkpoint support.
"""
import argparse
import json
import statistics
import time

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.parameter import Parameter
from mini_llm.block_proposal.soft_vocab_read import SoftVocabularyRead


def _milliseconds(f):
    xp.cuda.Stream.null.synchronize()
    a=time.perf_counter()
    val=f()
    xp.cuda.Stream.null.synchronize()
    return 1000*(time.perf_counter()-a),val


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-scope',choices=['own','all'],default='all')
    parser.add_argument('--all-source-algorithm', choices=['source_mixture','source_loop','batched_mixture','fused_mixture','fused_pair','both','batched_pair','all_three'],
                        default='source_mixture',
                        help='0080AJ2 batched heads/queries, existing mixture, legacy loop, or controlled pairs')
    parser.add_argument('--block-groups',type=int,nargs='+',default=[4,8,16,32,64])
    parser.add_argument('--repeat',type=int,default=3)
    parser.add_argument('--vocab',type=int,default=65280)
    parser.add_argument('--width',type=int,default=768)
    parser.add_argument('--decoder-width',type=int,default=512)
    parser.add_argument('--heads',type=int,default=4)
    parser.add_argument('--k',type=int,default=4)
    parser.add_argument('--vocab-tile',type=int,default=2048)
    parser.add_argument('--block-tile',type=int,default=8)
    args=parser.parse_args()
    if BACKEND_NAME!='cupy':
        parser.error('CUDA CuPy backend required')
    if args.repeat<1 or min(args.block_groups)<1:
        parser.error('positive groups and repeat required')
    import ml_dtypes
    dtype=ml_dtypes.bfloat16
    xp.random.seed(81)
    emb=Parameter((xp.random.standard_normal((args.vocab,args.width),dtype=xp.float32)*.02).astype(dtype),name='E')
    obj=SoftVocabularyRead(emb,args.decoder_width,n_heads=args.heads,
        block_size=args.k,source_scope=args.source_scope,execution='tiled',
        vocab_tile=args.vocab_tile,block_tile=args.block_tile)
    obj.output_projection.W.data[:]=xp.asarray(.015,dtype=dtype)
    pool=xp.get_default_memory_pool()
    print('backend=cupy, full-vocabulary, scratch-only benchmark; not 500M training',flush=True)
    if args.source_scope == 'own' and args.all_source_algorithm in ('both', 'batched_pair', 'all_three', 'batched_mixture', 'fused_mixture', 'fused_pair'):
        parser.error('algorithm comparison applies only to all-source')
    for g in args.block_groups:
        try:
            states=xp.random.standard_normal((g,args.k,args.decoder_width),dtype=xp.float32)
            logits=xp.random.standard_normal((g,args.k,args.vocab),dtype=xp.float32).astype(dtype)
            gradient=xp.random.standard_normal(states.shape,dtype=xp.float32)
            algorithms = ({'both': ('source_loop', 'source_mixture'),
                           'batched_pair': ('source_mixture', 'batched_mixture'),
                           'fused_pair': ('batched_mixture', 'fused_mixture'),
                           'all_three': ('source_loop', 'source_mixture', 'batched_mixture')}
                          .get(args.all_source_algorithm, (args.all_source_algorithm,)))
            comparisons = {}
            for algorithm in algorithms:
                obj.all_source_algorithm=algorithm
                fw=[];bw=[]
                for i in range(args.repeat+1):
                    obj.zero_grad(); emb.zero_grad()
                    f,(out,cache)=_milliseconds(lambda: obj.forward(states,logits))
                    b,(dx,dl)=_milliseconds(lambda: obj.backward(gradient,cache))
                    if i:
                        fw.append(f);bw.append(b)
                total=statistics.mean(fw)+statistics.mean(bw)
                row={
                    'group_blocks':g,'scope':args.source_scope,
                    'algorithm': algorithm,
                    'mean_forward_ms':statistics.mean(fw),
                    'mean_backward_ms':statistics.mean(bw),
                    'p95_total_ms':sorted([a+b for a,b in zip(fw,bw)])[max(0,int(.95*(len(fw)-1)))],
                    'query_positions_per_second':round(g*args.k*1000/total,2),
                    'scored_source_vocab_entries_per_second':round(g*args.k*args.vocab*(args.k if args.source_scope=='all' else 1)*args.heads*1000/total,2),
                    'pool_used_MiB':round(pool.used_bytes()/1048576,1),
                    'pool_reserved_MiB':round(pool.total_bytes()/1048576,1)}
                baseline = ('batched_mixture' if args.all_source_algorithm == 'fused_pair' else
                            ('source_mixture' if args.all_source_algorithm == 'batched_pair' else 'source_loop'))
                if baseline in comparisons and algorithm != baseline:
                    orig=comparisons[baseline]
                    row[f'speedup_over_{baseline}']=round(
                        (orig['mean_forward_ms']+orig['mean_backward_ms'])/total,3)
                    row['max_abs_correction_diff']=float(xp.max(xp.abs(out-orig['out'])))
                    row['max_abs_dlogits_diff']=float(xp.max(xp.abs(dl-orig['dl'])))
                    row['max_abs_dstates_diff']=float(xp.max(xp.abs(dx-orig['dx'])))
                print(json.dumps({k:v for k,v in row.items() if k not in ('out','dl')},
                                 sort_keys=True),flush=True)
                # Retain the *same-input* reference outputs only in pair mode;
                # never keep both parameter-gradient tapes or scratch buffers.
                if ((args.all_source_algorithm == 'both' and algorithm == 'source_loop') or
                    (args.all_source_algorithm == 'batched_pair' and algorithm == 'source_mixture') or
                    (args.all_source_algorithm == 'fused_pair' and algorithm == 'batched_mixture') or
                    (args.all_source_algorithm == 'all_three' and algorithm == 'source_loop')):
                    comparisons[algorithm] = dict(row, out=out.copy(), dl=dl.copy(), dx=dx.copy())
                del cache,out,dx,dl
            del states,logits,gradient,comparisons
        except xp.cuda.memory.OutOfMemoryError:
            print(json.dumps({'group_blocks':g,'error':'CUDA out of memory'}),flush=True)
            pool.free_all_blocks()


if __name__=='__main__':
    main()
