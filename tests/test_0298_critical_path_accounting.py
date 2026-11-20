"""A spinning dependency wait is not useful arithmetic or an additive latency."""
import importlib.util
from pathlib import Path
import sqlite3

spec=importlib.util.spec_from_file_location('critical_path',
    Path(__file__).resolve().parents[1]/'scripts/analyze_0285_critical_path.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_wait_kernel_excluded_from_useful_union(tmp_path):
    path=tmp_path/'trace.sqlite'
    with sqlite3.connect(path) as c:
        c.executescript('''
          create table StringIds(id integer,value text);
          create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,shortName integer);
          create table CUPTI_ACTIVITY_KIND_MEMCPY(start integer,end integer,copyKind integer,bytes integer);
          create table CUPTI_ACTIVITY_KIND_MEMSET(start integer,end integer);
          create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,nameId integer);
          create table NVTX_EVENTS(start integer,end integer,text text,textId integer);
        ''')
        c.executemany('insert into StringIds values(?,?)',[(1,'await_cpu'),(2,'q8_linear')])
        c.executemany('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?)',
                      [(0,20_000_000,1),(15_000_000,25_000_000,2)])
    report=module.summarize(path)
    assert report['gpu_operation_union_ms']==25
    assert report['non_wait_gpu_operation_union_ms']==10
    assert report['summed_wait_kernel_ms']==20
    assert report['summed_kernel_ms']==30
