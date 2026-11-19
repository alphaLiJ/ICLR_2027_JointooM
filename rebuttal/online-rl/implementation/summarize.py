"""Summarize the bounded pilot without treating iterations as independent seeds."""
import bootstrap
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=bootstrap.ARTIFACTS/'pilot-v1')
    root=parser.parse_args().root;rows=[];summaries=[];initial_matches=[]
    for seed in range(3):
        weights=[]
        for backend in ('cpu','cuda'):
            out=root/f'{backend}-s{seed}'
            values=[json.loads(line) for line in (out/'iterations.jsonl').read_text().splitlines()]
            rows.extend(dict(backend=backend,seed=seed,**r) for r in values)
            summary=json.loads((out/'summary.json').read_text());summaries.append(summary)
            weights.append(torch.load(out/'initial.pt',map_location='cpu',weights_only=True))
            final=torch.load(out/'final.pt',map_location='cpu',weights_only=True)
            assert final['agent_samples']==1048576 and len(values)==16
            assert all(torch.isfinite(t).all() for t in final['model'].values())
        initial_matches.append(all(torch.equal(weights[0][k],weights[1][k]) for k in weights[0]))
    assert all(initial_matches)
    for name,data in [('iterations.csv',rows),('run-summary.csv',summaries)]:
        with (root/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    result=dict(seeds=[0,1,2],runs=6,agent_samples_per_run=1048576,total_agent_samples=6291456,
        main_configuration='16 environments x 128 agents, T32, minibatch2048, one epoch',
        steady_iterations='iterations 2 through 15 inclusive, first two excluded; final iteration includes reset',
        cpu_backend='Pogema 1.2.2 soft movement + cached CPU exact distances/vectorized CostTracer view; 8 workers/16 CPUs',
        cuda_backend='JointLoom resident simulator + exact-distance cache + CostTracer view',
        initial_weights_equal_per_seed=initial_matches,
        scope='short efficiency/learning-path pilot; no final-quality or convergence equivalence claim',
        backends={})
    for backend in ('cpu','cuda'):
        d=[s for s in summaries if s['backend']==backend];agg={}
        for key in ('mean_iteration_seconds','mean_sample_gae_seconds','mean_update_seconds',
                    'first_progress_reward','last_progress_reward','total_goals','peak_gpu_memory_bytes'):
            v=np.array([s[key] for s in d],dtype=float)
            agg[key]=dict(mean=float(v.mean()),seed_sample_std=float(v.std(ddof=1)))
        agg['agent_samples_per_second']=65536/agg['mean_iteration_seconds']['mean']
        result['backends'][backend]=agg
    cpu=result['backends']['cpu'];gpu=result['backends']['cuda']
    result['complete_iteration_speedup']=cpu['mean_iteration_seconds']['mean']/gpu['mean_iteration_seconds']['mean']
    result['sample_gae_speedup']=cpu['mean_sample_gae_seconds']['mean']/gpu['mean_sample_gae_seconds']['mean']
    result['per_seed_complete_iteration_speedups']=[
        next(d['mean_iteration_seconds'] for d in summaries if d['backend']=='cpu' and d['seed']==s)/
        next(d['mean_iteration_seconds'] for d in summaries if d['backend']=='cuda' and d['seed']==s)
        for s in range(3)]
    (root/'results.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    colors={'cpu':'#c57a38','cuda':'#207ca4'}
    ax=axes[0]
    for i,backend in enumerate(('cpu','cuda')):
        d=result['backends'][backend]
        sample=d['mean_sample_gae_seconds']['mean'];update=d['mean_update_seconds']['mean']
        ax.bar(i,sample,color='#207ca4',label='Online sampling + GAE' if i==0 else None)
        ax.bar(i,update,bottom=sample,color='#bbbbbb',label='PPO update' if i==0 else None)
        ax.text(i,sample+update+.008,f'{sample+update:.3f} s',ha='center')
    ax.set_xticks([0,1],['Pogema CPU','JointLoom CUDA']);ax.set_ylabel('Complete iteration (s)')
    ax.set_ylim(0,.56);ax.legend(fontsize=8,loc='upper right')
    axes[1].set_title('Short training pilot (3 seeds)')
    for backend in ('cpu','cuda'):
        traces=np.array([[r['mean_progress_reward'] for r in rows if r['backend']==backend and r['seed']==s]
                         for s in range(3)])
        mean=traces.mean(0);sd=traces.std(0,ddof=1);x=np.arange(1,17)*65536/1e6
        axes[1].plot(x,mean,label='Pogema CPU' if backend=='cpu' else 'JointLoom CUDA',color=colors[backend])
        axes[1].fill_between(x,mean-sd,mean+sd,alpha=.16,color=colors[backend])
    axes[1].set_xlabel('Agent samples (million)');axes[1].set_ylabel('Mean progress reward / sample')
    axes[1].legend(fontsize=8)
    fig.savefig(root/'pilot.png',dpi=180);fig.savefig(root/'pilot.pdf')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
