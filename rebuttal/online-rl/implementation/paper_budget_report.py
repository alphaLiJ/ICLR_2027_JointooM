"""Single-seed full-budget training traces and fixed held-out checks."""
import bootstrap
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reward_report import clock_bridge,extract


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root;training=root/'training'
    jobs=[j for j in json.loads((training/'status.json').read_text()) if j['run'].startswith('formal-')]
    if len(jobs)!=2 or not all(j['status']=='complete' for j in jobs):raise RuntimeError('incomplete pair')
    bridge=clock_bridge();budget=jobs[0]['iterations']*65536
    raw=[];episodes=[];groups={}
    for j in jobs:
        rows,ep=extract(training,j,bridge,budget);raw.extend(rows);episodes.extend(ep);groups[j['profile']]=rows
    with (root/'reward-points.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=raw[0]);w.writeheader();w.writerows(raw)
    colors={'sf-async':'#c07836','jointloom':'#207da4'}
    labels={'sf-async':'Native SF async + Pogema','jointloom':'JointLoom GPU resident'}
    fig,axes=plt.subplots(1,2,figsize=(10.3,4),layout='constrained')
    per_profile={}
    for profile,rows in groups.items():
        reward=np.array([r['mean_progress_reward'] for r in rows])
        samples=np.array([r['collected_agent_samples'] for r in rows])
        seconds=np.array([r['elapsed_job_seconds'] for r in rows])
        mean=np.convolve(reward,np.ones(16)/16,mode='valid')
        axes[0].plot(samples/1e6,reward,color=colors[profile],alpha=.08,lw=.6)
        axes[1].plot(seconds,reward,color=colors[profile],alpha=.08,lw=.6)
        axes[0].plot(samples[15:]/1e6,mean,color=colors[profile],lw=1.7,label=labels[profile])
        axes[1].plot(seconds[15:],mean,color=colors[profile],lw=1.7,label=labels[profile])
        per_profile[profile]=dict(collected_samples=int(samples[-1]),
            tail16_mean_progress_reward=float(reward[-16:].mean()),
            full_job_seconds_to_collected_budget=float(seconds[-1]),
            training_seconds_to_collected_budget=rows[-1]['training_seconds'])
        job=next(j for j in jobs if j['profile']==profile)
        per_profile[profile]['full_training_job_seconds']=job['finished']-job['started']
    axes[0].set_xlabel('Collected agent transitions (millions)')
    axes[1].set_xlabel('Elapsed job time, including startup (seconds)')
    axes[0].set_title('Reward versus interactions');axes[1].set_title('Reward versus wall time')
    for ax in axes:ax.set_ylabel('Raw progress reward / agent-step');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('CostTracer paper-budget reference: 16 env × 128 agents, seed 0',fontsize=11)
    fig.savefig(root/'reward-curves.png',dpi=180);fig.savefig(root/'reward-curves.pdf')
    epoints=[];n_episodes=jobs[0]['iterations']//16
    for job in jobs:
        profile=job['profile'];folder=training/job['run']
        start=int(job['started']*1e9)-bridge['wall_minus_perf_ns']
        if profile=='jointloom':
            rows=[json.loads(x) for x in (folder/'iterations.jsonl').read_text().splitlines()]
            for e in range(n_episodes):
                chunk=rows[e*16:(e+1)*16]
                epoints.append(dict(profile=profile,episode=e+1,per_env_steps=(e+1)*512,
                    mean_agent_episode_return=sum(r['collected_reward_sum'] for r in chunk)/2048,
                    elapsed_job_seconds=(chunk[-1]['sample_completed_perf_counter_ns']-start)/1e9))
        else:
            for e in range(1,n_episodes+1):
                chunk=[r for r in episodes if r['episode']==e]
                if len(chunk)!=16:continue
                epoints.append(dict(profile=profile,episode=e,per_env_steps=e*512,
                    mean_agent_episode_return=float(np.mean([r['mean_agent_episode_return'] for r in chunk])),
                    elapsed_job_seconds=(max(r['completed_perf_counter_ns'] for r in chunk)-start)/1e9))
    with (root/'episode-return-points.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=epoints[0]);w.writeheader();w.writerows(epoints)
    efig,eaxes=plt.subplots(1,2,figsize=(10.3,4),layout='constrained')
    for profile in groups:
        rows=[r for r in epoints if r['profile']==profile]
        val=[r['mean_agent_episode_return'] for r in rows]
        eaxes[0].plot([r['per_env_steps'] for r in rows],val,color=colors[profile],lw=1.6,label=labels[profile])
        eaxes[1].plot([r['elapsed_job_seconds'] for r in rows],val,color=colors[profile],lw=1.6,label=labels[profile])
    eaxes[0].set_xlabel('Completed interaction steps per environment')
    eaxes[1].set_xlabel('Elapsed job time, including startup (seconds)')
    eaxes[0].set_title('512-step episode return');eaxes[1].set_title('Episode return versus wall time')
    for ax in eaxes:ax.set_ylabel('Mean agent return over 512 steps');ax.grid(alpha=.2)
    eaxes[0].legend(fontsize=8)
    efig.suptitle('CostTracer paper-budget reference: seed 0, no smoothing',fontsize=11)
    efig.savefig(root/'episode-return-curves.png',dpi=180);efig.savefig(root/'episode-return-curves.pdf')
    evaluation=json.loads((root/'evaluation/results.json').read_text())
    native_envs=[json.loads(p.read_text()) for p in (training/'formal-sf-async-s0').glob('reward-env-*.json')]
    max_cpu_steps=max(r['total_joint_steps'] for r in native_envs if r['records'])
    gpu_rows=[json.loads(s) for s in (training/'formal-jointloom-s0/iterations.jsonl').read_text().splitlines()]
    max_gpu_steps=gpu_rows[-1]['env_joint_steps']
    summary=dict(reference_training_steps=75000000,step_definition='agent transitions (declared SF counting convention)',
        requested_macros=jobs[0]['iterations'],rounded_processed_budget=budget,
        training_seed=0,profiles=per_profile,heldout=evaluation,
        actual_training=[r for r in json.loads((training/'results.json').read_text()) if r['phase']=='formal'],
        task_bank_episodes=96,max_env_steps=dict(sf_async=max_cpu_steps,jointloom=max_gpu_steps),
        task_bank_cycles=dict(sf_async=max_cpu_steps//(96*512),jointloom=max_gpu_steps//(96*512)),
        caution='one paired seed, declared adaptation of published table; not full recovery of unpublished CostTracer trainer or convergence proof')
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['CostTracer paper-budget reference: complete paired training results for seed0','',
        f'Target75000000;65536 samples per macrobatch, rounded up to {budget} agent transitions / {jobs[0]["iterations"]} macrobatches.',
        '16env,128agents/env,2048 concurrent agents; RTX5080, eight physical CPU cores per method; shared initial weights.',
        'Learning parameters follow Table1 of the paper. The unpublished original trainer was not fully recovered; training steps are explicitly counted here as SF agent transitions.',
        'A new96-episode task bank, without cycling in this run, is independently sampled from the same task distribution. The previous64-episode bank is retained; old and new curves are separate.','',
        'Raw reward over the last16 training windows and collection-budget completion time including startup:']
    for profile,value in per_profile.items():
        lines.append(f"{profile}: reward={value['tail16_mean_progress_reward']:.7f}; same collection budget: {value['full_job_seconds_to_collected_budget']:.3f}s; full process (including saving/exit): {value['full_training_job_seconds']:.3f}s.")
    lines += ['', 'Fixed unseen-map evaluation (two mazes, eight environments per map,128agents/env,512 steps, shared categorical tasks/RNG):']
    for r in evaluation:
        lines.append(f"{r['model']}: mean agent episode return={r['mean_agent_episode_return']:.6f}; goals={r['completed_goals']:.0f}; goals/env-step={r['goals_per_joint_env_step']:.6f}。")
    lines += ['', 'Training reward curves use16-window moving averages with raw points retained. Episode curves show unsmoothed returns over complete512-step episodes.',
        'Curves and evaluation use one training seed and do not establish quality equivalence or statistical significance.',
        'Asynchronous SF retains policy lag and additional in-flight batches; actual processing budgets are recorded in summary.actual_training.',
        'Completed goals measure lifelong throughput, without requiring all agents to arrive simultaneously.',
        'Full raw rewards, tasks, checkpoints and fixed evaluation records are in this output directory.','']
    (root/'RESULTS.txt').write_text('\n'.join(lines));print('\n'.join(lines[:18]))


if __name__=='__main__':main()
