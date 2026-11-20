"""Causal equal-slot heat-policy replay; estimates expert misses, not throughput.

Each window is evaluated BEFORE using its routes to propose moves for the next
window. Real transfers, changed CPU/GPU rounding, placement cost and future route
changes are not simulated. No look-ahead oracle is advertised as performance.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mini_llm.ops.expert_heat_policy import propose_heat_moves


def decode_frame(cache):
    rows=sorted(cache.get('native_handoffs',[]),key=lambda row:row['layer'])
    if not rows: return None
    if len(rows)>64 or [r['layer'] for r in rows]!=list(range(len(rows))):
        raise ValueError('Incomplete/invalid layer profile')
    values=[r['route_heat'] for r in rows]
    if any(not isinstance(v,list) or not 1<=len(v)<=512 for v in values):
        raise ValueError('Native route profiling was not enabled')
    heat=np.asarray(values,dtype=np.int64)
    if heat.ndim!=2 or (heat<0).any(): raise ValueError('Invalid route counts')
    steps=[r['calls'] for r in rows]
    if min(steps)!=max(steps) or min(steps)<0: raise ValueError('Layer steps disagree')
    resident=np.zeros(heat.shape,np.bool_)
    for row in rows:
        ids=row['profile_resident_ids']
        if len(ids)!=len(set(ids)) or any(not 0<=i<heat.shape[1] for i in ids):
            raise ValueError('Invalid resident expert IDs')
        resident[row['layer'],ids]=True
    return dict(heat=heat,resident=resident,steps=steps[0],
                sizes=np.asarray([r['expert_packed_bytes'] for r in rows],dtype=np.int64))


def simulate(frames,*,limit,decay=.9):
    if not frames: raise ValueError('No native profile frames')
    if not 0<=decay<=1: raise ValueError('Decay must be in0..1')
    original=frames[0]['resident']
    resident=original.copy()
    scores=np.zeros(original.shape,np.float64)
    previous=np.zeros(original.shape,np.int64)
    steps=0; baseline=0; adapted=0; moves=0; transfer_bytes=0; windows=[]
    for frame in frames:
        if frame['heat'].shape!=previous.shape or not np.array_equal(frame['resident'],original):
            raise ValueError('Expected unchanged original placement/geometry in diagnostic capture')
        counts=frame['heat']-previous
        if frame['steps']<steps or (counts<0).any(): raise ValueError('Profile counts reset or regressed')
        if frame['steps']==steps:
            if counts.any(): raise ValueError('Routes changed without steps')
            continue
        cold=int(counts[~original].sum())
        current=int(counts[~resident].sum())
        baseline+=cold; adapted+=current
        # Only completed-window observations inform the following placement.
        scores=decay*scores+counts
        proposals=propose_heat_moves(scores,resident,limit=limit)
        copied=0
        for hot,victim in proposals:
            resident[victim]=False; resident[hot]=True
            copied+=int(frame['sizes'][hot[0]])
        windows.append(dict(from_step=steps,to_step=frame['steps'],baseline_misses=cold,
                            proposed_placement_misses=current,next_window_moves=len(proposals),
                            next_window_transfer_bytes=copied))
        moves+=len(proposals); transfer_bytes+=copied
        previous=frame['heat']; steps=frame['steps']
    return dict(limit_per_window=limit,decay=decay,steps=steps,baseline_cpu_experts=baseline,
                causal_placement_cpu_experts=adapted,miss_reduction_fraction=1-adapted/max(1,baseline),
                proposed_moves=moves,proposed_transfer_bytes=transfer_bytes,windows=windows,
                caveat='Counts only; excludes transfer/graph rebuild cost and changed execution-site rounding. Last proposals have no measured following window.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    progress=args.artifact_dir/'progress.jsonl'
    if progress.stat().st_size>64*1024**2: raise ValueError('Profile artifact exceeds64MiB bound')
    frames=[]
    with progress.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['request_id']!='performance-short-r1': continue
            frame=decode_frame(row['cache'])
            if frame is not None: frames.append(frame)
    result=json.loads((args.artifact_dir/'result.json').read_text())
    target=next(c for c in result['cases'] if c['request_id']=='performance-short-r1')
    frames.append(decode_frame(target['cache_after']))
    reports=[simulate(frames,limit=n) for n in (0,8,16,32,64,128)]
    with args.output.open('x') as out:
        json.dump(dict(source=str(args.artifact_dir),scope='Causal route-count estimate only',
                       policies=reports),out,indent=2,allow_nan=False)
    for row in reports:
        print(row['limit_per_window'],row['baseline_cpu_experts'],row['causal_placement_cpu_experts'],
              row['miss_reduction_fraction'],row['proposed_transfer_bytes'])


if __name__=='__main__': main()
