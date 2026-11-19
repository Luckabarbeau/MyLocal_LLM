"""Summarize node-level Nsight GPU occupancy and gaps, not additive CPU latency."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def summarize(path):
    connection=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    connection.row_factory=sqlite3.Row
    strings=dict(connection.execute('select id,value from StringIds'))
    intervals=[]; kernels=defaultdict(lambda:[0,0]); copies=defaultdict(lambda:[0,0,0])
    for row in connection.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
        name=strings[row['shortName']]
        intervals.append((row['start'],row['end'],'kernel:'+name))
        kernels[name][0]+=row['end']-row['start']; kernels[name][1]+=1
    for row in connection.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY'):
        name='copy:'+str(row['copyKind'])+':'+str(row['bytes'])
        intervals.append((row['start'],row['end'],name))
        copies[name][0]+=row['end']-row['start']; copies[name][1]+=1; copies[name][2]+=row['bytes']
    for row in connection.execute('select * from CUPTI_ACTIVITY_KIND_MEMSET'):
        intervals.append((row['start'],row['end'],'memset'))
    intervals.sort()
    union=0; end=intervals[0][0]; gaps=defaultdict(lambda:[0,0,0])
    for a,b,name in intervals:
        if a>end:
            gaps[name][0]+=a-end; gaps[name][1]+=1; gaps[name][2]=max(gaps[name][2],a-end)
        union+=max(0,b-max(end,a)); end=max(end,b)
    runtime=defaultdict(lambda:[0,0])
    for row in connection.execute('select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
        name=strings[row['nameId']]
        runtime[name][0]+=row['end']-row['start']; runtime[name][1]+=1
    nvtx=defaultdict(lambda:[0,0])
    for row in connection.execute('select start,end,text,textId from NVTX_EVENTS where end is not null'):
        name=row['text'] or strings.get(row['textId'],'')
        nvtx[name][0]+=row['end']-row['start']; nvtx[name][1]+=1
    def records(values,extra=()):
        return [dict(name=k,total_ms=v[0]/1e6,count=v[1],**{n:v[i+2] for i,n in enumerate(extra)})
                for k,v in sorted(values.items(),key=lambda pair:-pair[1][0])]
    result=dict(source=str(path),gpu_operation_union_ms=union/1e6,
        gpu_span_ms=(end-intervals[0][0])/1e6,
        summed_kernel_ms=sum(v[0] for v in kernels.values())/1e6,
        kernels=records(kernels),copies=records(copies,('bytes',)),runtime_api=records(runtime),
        nvtx_wall=records(nvtx),gaps_by_next_operation=records(gaps,('maximum_gap_ns',)),
        caveat='GPU gaps include CPU dependencies, graph scheduling and host submission; these categories overlap CPU/API spans.')
    connection.close()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sqlite',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x') as out: json.dump(summarize(args.sqlite),out,indent=2)


if __name__=='__main__': main()
