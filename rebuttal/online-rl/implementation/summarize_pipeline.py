"""Keep the old shared-learner control and the new reference placement distinct."""
import bootstrap
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=bootstrap.ARTIFACTS/'pipeline-v2-trials')
    root=p.parse_args().root;rows=[];summaries=[];equal=[];initial_vs_old=[]
    profiles=('reference_cpu_buffers','jointloom_optimized')
    for seed in range(3):
        weights=[]
        for profile in profiles:
            folder=root/f'{profile}-s{seed}'
            data=[json.loads(x) for x in (folder/'iterations.jsonl').read_text().splitlines()]
            rows.extend(dict(profile=profile,seed=seed,**r) for r in data)
            summary=json.loads((folder/'summary.json').read_text());summaries.append(summary)
            assert summary['adam_step_counters']==[512.] and summary['agent_samples']==1048576
            weights.append(torch.load(folder/'initial.pt',map_location='cpu',weights_only=True))
            checkpoint=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
            assert all(bool(torch.isfinite(v).all()) for v in checkpoint['model'].values())
        equal.append(all(torch.equal(weights[0][k],weights[1][k]) for k in weights[0]))
        old=torch.load(bootstrap.ARTIFACTS/f'pilot-v1/cpu-s{seed}/initial.pt',map_location='cpu',weights_only=True)
        initial_vs_old.append(all(torch.equal(old[k],weights[0][k]) for k in old))
    assert all(equal+initial_vs_old)
    with (root/'iterations.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(runs=6,seeds=[0,1,2],agent_samples_per_run=1048576,
        total_agent_samples=6291456,adam_updates_per_run=512,
        initial_weights_equal=equal,initial_weights_equal_to_pilot_v1=initial_vs_old,
        statistics='mean and sample std across three seed-level means; first two iterations excluded',
        reference_identity='SF-v1 CPU-rollout/GAE/batch-upload placement, synchronous adapter; original author trainer unavailable',
        precision='FP32, AMP off, TF32 off',profiles={})
    for profile in profiles:
        d=[s for s in summaries if s['profile']==profile];aggregated={}
        for key in ('sample_seconds','gae_prepare_seconds','update_seconds','iteration_seconds'):
            x=np.array([s['steady_mean'][key] for s in d])
            aggregated[key]=dict(mean=float(x.mean()),seed_std=float(x.std(ddof=1)))
        for key in ('setup_seconds','first_progress_reward','last_progress_reward','total_goals'):
            x=np.array([s[key] for s in d]);aggregated[key]=dict(mean=float(x.mean()),seed_std=float(x.std(ddof=1)))
        aggregated['samples_per_second']=65536/aggregated['iteration_seconds']['mean']
        result['profiles'][profile]=aggregated
    ref=result['profiles'][profiles[0]];opt=result['profiles'][profiles[1]]
    old=json.loads((bootstrap.ARTIFACTS/'pilot-v1/results.json').read_text())
    result['speedups']=dict(
        complete_reference_to_optimized=ref['iteration_seconds']['mean']/opt['iteration_seconds']['mean'],
        gae_and_prepare=ref['gae_prepare_seconds']['mean']/opt['gae_prepare_seconds']['mean'],
        own_previous_to_optimized=old['backends']['cuda']['mean_iteration_seconds']['mean']/opt['iteration_seconds']['mean'],
        historical_shared_gpu_learner_cpu_to_optimized=old['backends']['cpu']['mean_iteration_seconds']['mean']/opt['iteration_seconds']['mean'])
    result['historical_pilot_v1']=dict(complete_speedup=old['complete_iteration_speedup'],
        cpu_seconds=old['backends']['cpu']['mean_iteration_seconds']['mean'],
        jointloom_seconds=old['backends']['cuda']['mean_iteration_seconds']['mean'],
        identity='both sides use shared GPU rollout/GAE/learner; historical control, not relabeled as CPU-buffer reference')
    (root/'results.json').write_text(json.dumps(result,indent=2))
    with (root/'run-summary.csv').open('w',newline='') as f:
        flat=[dict(profile=s['profile'],seed=s['seed'],**s['steady_mean'],
                   samples=s['agent_samples'],adam_steps=s['adam_step_counters'][0]) for s in summaries]
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    labels=['CPU buffers\nreference','JointLoom\nprevious','JointLoom\noptimized']
    sample=[ref['sample_seconds']['mean']+ref['gae_prepare_seconds']['mean'],
            old['backends']['cuda']['mean_sample_gae_seconds']['mean'],
            opt['sample_seconds']['mean']+opt['gae_prepare_seconds']['mean']]
    updates=[ref['update_seconds']['mean'],old['backends']['cuda']['mean_update_seconds']['mean'],opt['update_seconds']['mean']]
    axes[0].bar(range(3),sample,color='#207ca4',label='Sampling + GAE / preparation')
    axes[0].bar(range(3),updates,bottom=sample,color='#bbbbbb',label='PPO update')
    for i in range(3):axes[0].text(i,sample[i]+updates[i]+.009,f'{sample[i]+updates[i]:.3f} s',ha='center')
    axes[0].set_xticks(range(3),labels);axes[0].set_ylabel('Complete iteration (s)')
    axes[0].set_ylim(0,.64);axes[0].legend(fontsize=8,loc='upper right')
    for profile,color,label in zip(profiles,('#c57a38','#207ca4'),('CPU buffers reference','JointLoom optimized')):
        traces=np.array([[r['mean_progress_reward'] for r in rows if r['profile']==profile and r['seed']==s]
                         for s in range(3)])
        x=np.arange(1,17)*65536/1e6;mean=traces.mean(0);sd=traces.std(0,ddof=1)
        axes[1].plot(x,mean,color=color,label=label)
        axes[1].fill_between(x,mean-sd,mean+sd,color=color,alpha=.16)
    axes[1].set_xlabel('Agent samples (million)');axes[1].set_ylabel('Mean progress reward / sample')
    axes[1].set_title('Short pilot, 3 seeds');axes[1].legend(fontsize=8)
    fig.savefig(root/'pipeline-v2.png',dpi=180);fig.savefig(root/'pipeline-v2.pdf')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
