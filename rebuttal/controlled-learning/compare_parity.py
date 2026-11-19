"""Direct value comparisons of learning, online actions and expert data."""
import runtime
import argparse
from pathlib import Path
import json
import numpy as np
import torch


def compare_tree(left,right):
    checked=0;differences=[];elements=0
    def visit(a,b,path):
        nonlocal checked,elements
        if torch.is_tensor(a):
            checked+=1;elements+=a.numel()
            if not torch.is_tensor(b) or a.shape!=b.shape or a.dtype!=b.dtype:
                differences.append(dict(path=path,reason='type, shape or dtype'));return
            if not torch.equal(a,b):
                delta=(a.double()-b.double()).abs()
                differences.append(dict(path=path,reason='values',
                    differing_elements=int((a!=b).sum()),max_abs=float(delta.max())))
        elif isinstance(a,dict):
            if set(a)!=set(b):differences.append(dict(path=path,reason='dictionary keys'));return
            for k in a:visit(a[k],b[k],f'{path}/{k}')
        elif isinstance(a,(list,tuple)):
            if len(a)!=len(b):differences.append(dict(path=path,reason='length'));return
            for i,(x,y) in enumerate(zip(a,b)):visit(x,y,f'{path}/{i}')
        elif a!=b:differences.append(dict(path=path,reason='scalar',left=a,right=b))
    visit(left,right,'')
    return dict(identical=not differences,tensors=checked,elements=elements,differences=differences)


def compare_pair(cpu,jointloom,include_rounds=True):
    cpu=Path(cpu);jointloom=Path(jointloom)
    names=['last.pt']
    if include_rounds:names+=sorted(v.name for v in cpu.glob('round-*.pt'))
    checkpoints={}
    for name in names:
        checkpoints[name]=compare_tree(torch.load(cpu/name,map_location='cpu',weights_only=False),
            torch.load(jointloom/name,map_location='cpu',weights_only=False))
    # Include optimizer state, but exclude backend-specific run metadata/timings.
    states=[torch.load(p/'training-state.pt',map_location='cpu',weights_only=False) for p in (cpu,jointloom)]
    checkpoints['final-optimizer']=compare_tree(states[0]['optimizer'],states[1]['optimizer'])
    traces=[json.loads((p/'update-trace.json').read_text()) for p in (cpu,jointloom)]
    update_trace=traces[0]==traces[1]
    actions=[np.load(p/'executed-actions.npy') for p in (cpu,jointloom)]
    action_check=dict(identical=np.array_equal(*actions),shape=list(actions[0].shape),
        actions=int(actions[0].size),differences=int(np.sum(actions[0]!=actions[1])))
    corpora=[torch.load(p/'visited-labeled-states.pt',weights_only=False) for p in (cpu,jointloom)]
    queried=[]
    if len(corpora[0])==len(corpora[1]):
        for i,(a,b) in enumerate(zip(*corpora)):
            same=all(a[k]==b[k] for k in ('round','step','env'))
            same=same and all(a.get(k)==b.get(k) for k in ('episode_step','episode_bin','time_bin','arrived_agents'))
            same=same and np.array_equal(a['rows'],b['rows']) and np.array_equal(a['executed_actions'],b['executed_actions'])
            strip=lambda e:{k:v for k,v in e.items() if k not in ('pid','cpu_affinity','seconds')}
            same=same and strip(a['expert'])==strip(b['expert'])
            if not same:queried.append(i)
    else:queried=['length']
    result=dict(cpu=str(cpu),jointloom=str(jointloom),checkpoints=checkpoints,
        update_trace_identical=update_trace,updates=len(traces[0]),actions=action_check,
        queried_states=len(corpora[0]),query_differences=queried,
        identical=all(v['identical'] for v in checkpoints.values()) and update_trace
            and action_check['identical'] and not queried)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('cpu',type=Path);p.add_argument('jointloom',type=Path)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    result=compare_pair(args.cpu,args.jointloom)
    runtime.json_write(args.out,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if not result['identical']:raise RuntimeError('learning/trajectory pair differs; see comparison')


if __name__=='__main__':main()
