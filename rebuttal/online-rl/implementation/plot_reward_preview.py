"""Reuse existing shared-learner CPU/CUDA training logs; NOT an SF comparison."""
import bootstrap
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    source=bootstrap.ARTIFACTS/'pilot-v1'
    runs={};raw=[]
    for backend in ('cpu','cuda'):
        runs[backend]=[]
        for seed in (0,1,2):
            folder=source/f'{backend}-s{seed}'
            rows=[json.loads(s) for s in (folder/'iterations.jsonl').read_text().splitlines()]
            samples=np.array([r['agent_samples'] for r in rows])
            reward=np.array([r['mean_progress_reward'] for r in rows])
            seconds=np.cumsum([r['iteration_seconds'] for r in rows])
            runs[backend].append(dict(samples=samples,reward=reward,seconds=seconds))
            raw.extend(dict(backend=backend,seed=seed,iteration=r['iteration'],
                agent_samples=int(n),measured_training_seconds=float(t),mean_progress_reward=float(v),
                source=str(folder/'iterations.jsonl')) for r,n,t,v in zip(rows,samples,seconds,reward))
    with (out/'raw-reward-points.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=raw[0]);w.writeheader();w.writerows(raw)
    fig,axes=plt.subplots(1,2,figsize=(10.1,3.9),layout='constrained')
    colors=dict(cpu='#c07836',cuda='#247ba0')
    labels=dict(cpu='CPU Pogema + shared GPU learner',cuda='JointLoom + shared GPU learner')
    stats={}
    for backend in ('cpu','cuda'):
        group=runs[backend];color=colors[backend]
        x=group[0]['samples']
        if not all(np.array_equal(r['samples'],x) for r in group):raise ValueError('sample axes differ')
        y=np.stack([r['reward'] for r in group]);mean=y.mean(0);sd=y.std(0,ddof=1)
        for r in group:
            axes[0].plot(r['samples']/1e6,r['reward'],color=color,alpha=.16,lw=.8)
            axes[1].plot(r['seconds'],r['reward'],color=color,alpha=.16,lw=.8)
        axes[0].plot(x/1e6,mean,color=color,lw=2,label=labels[backend])
        axes[0].fill_between(x/1e6,mean-sd,mean+sd,color=color,alpha=.15)
        # Interpolate only within the common time coverage of all three seeds.
        # These are display coordinates, not extra observations or repetitions.
        grid=np.linspace(max(r['seconds'][0] for r in group),min(r['seconds'][-1] for r in group),128)
        ys=np.stack([np.interp(grid,r['seconds'],r['reward']) for r in group])
        axes[1].plot(grid,ys.mean(0),color=color,lw=2,label=labels[backend])
        axes[1].fill_between(grid,ys.mean(0)-ys.std(0,ddof=1),ys.mean(0)+ys.std(0,ddof=1),color=color,alpha=.15)
        stats[backend]=dict(first_reward=float(mean[0]),last_reward=float(mean[-1]),
            last_reward_seed_sd=float(sd[-1]),mean_final_training_seconds=float(np.mean([r['seconds'][-1] for r in group])),
            processed_agent_samples=int(x[-1]),seeds=3)
    axes[0].set_xlabel('Processed agent samples (millions)')
    axes[1].set_xlabel('Measured training time (seconds)')
    axes[0].set_title('Learning progress per sample')
    axes[1].set_title('Learning progress per measured training time')
    for ax in axes:
        ax.set_ylabel('Mean progress reward per agent-step')
        ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    axes[0].legend(fontsize=8,loc='lower right')
    fig.suptitle('Existing short PPO pilot: shared-learner control, 3 seeds (not Sample Factory)',fontsize=10)
    fig.savefig(out/'reward-preview.png',dpi=180)
    fig.savefig(out/'reward-preview.pdf')
    metadata=dict(source=str(source),statistics=stats,
        baseline='Pogema CPU environment with shared original PPO learner; not native Sample Factory',
        settings='16 env x128 agents, rollout32, minibatch2048, epoch1, 16 macros, common 16-core budget, FP32',
        reward='0.1 only for a new historical distance record; mean over 32 rollout steps and all agents',
        bands='sample standard deviation across 3 data/training seeds, not a confidence interval',
        sample_axis='raw macro endpoints; no smoothing',
        time_axis='sum of synchronized iteration_seconds including first two iterations; excludes setup/logging/checkpoint; time-grid interpolation without extrapolation',
        scope='training-period short-run progress; not held-out policy quality, convergence, or time-to-target evidence',
        source_protocol='original common-learner final-observation bootstrap; not the later SF2-compatible profile')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    (out/'README.txt').write_text('\n'.join([
        'Existing reward-curve preview; no new training was run.',
        'Source is pilot-v1 CPU/CUDA with the same original PPO learner, not Sample Factory.',
        'Left: reward vs processed agent samples. Right: reward vs measured training iteration time.',
        'Reward is mean progress reward per agent-step, not episode return or MAPF success rate.',
        'Three seeds; shaded bands are seed sample standard deviations. No smoothing.',
        'The wall axis sums iteration timings including the first two iterations, excluding setup/log/checkpoint.',
        'Time means use interpolation inside each method\'s common three-seed time range; no extrapolation.',
        'The 16-macro runs provide a short training trace, not convergence or quality equivalence.',
        'Actual native SF runs currently have no aligned reward series in TensorBoard. A formal SF reward figure needs uniformly logged bounded training.',
        'Reward-preview.png/pdf and raw-reward-points.csv are for reviewing the proposed presentation.',
        '']))
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
