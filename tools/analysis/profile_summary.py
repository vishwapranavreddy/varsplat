"""Summarize synchronous nested timings without double-counting child calls."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def summarize(report):
    if report.get('status')!='completed' or report.get('profile')!='stages':
        raise ValueError('Require a completed diagnostic stages run')
    pending=[];roots=[]
    for row in report['timings']:
        d=row['nesting_depth'];seconds=row['seconds']
        if type(d) is not int or d<0 or not math.isfinite(seconds) or seconds<0 or row['status']!='completed':
            raise ValueError('Invalid timing row')
        children=[]
        while pending and pending[-1]['depth']>d:
            child=pending.pop()
            if child['depth']!=d+1:raise ValueError('Missing enclosing timing span')
            children.append(child)
        children.reverse()
        exclusive=seconds-sum(c['inclusive_seconds'] for c in children)
        if exclusive < -1e-6:raise ValueError('Children exceed inclusive parent time')
        node=dict(stage=row['stage'],depth=d,inclusive_seconds=seconds,
                  exclusive_seconds=max(0.,exclusive),children=children)
        if d==0:
            if pending:raise ValueError('Unclosed timing spans')
            roots.append(node)
        else:pending.append(node)
    if pending:raise ValueError('Missing final timing parent')
    slam=[n for n in roots if n['stage']=='slam']
    if len(slam)!=1:raise ValueError('Require exactly one SLAM span')
    root=slam[0];totals=defaultdict(lambda:dict(calls=0,inclusive_seconds=0.,exclusive_seconds=0.))
    def visit(node):
        t=totals[node['stage']];t['calls']+=1
        for k in ('inclusive_seconds','exclusive_seconds'):t[k]+=node[k]
        for child in node['children']:visit(child)
    visit(root)
    exclusive_total=sum(v['exclusive_seconds'] for v in totals.values())
    if not math.isclose(exclusive_total,root['inclusive_seconds'],abs_tol=1e-5,rel_tol=1e-9):
        raise ValueError('Timing partition does not reconcile')
    return dict(scope='Synchronized diagnostics only; never pool with wall benchmarks',
                slam_seconds=root['inclusive_seconds'],
                exclusive_partition_seconds=exclusive_total,
                stages=dict(sorted(totals.items(),key=lambda p:-p[1]['exclusive_seconds'])),
                note='slam exclusive is uninstrumented work plus profiler overhead; inclusive stage totals overlap')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=summarize(json.loads(args.report.read_bytes()))
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
