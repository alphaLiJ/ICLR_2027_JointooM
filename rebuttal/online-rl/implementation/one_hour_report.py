"""Equal wall-budget traces, separate in-flight tails, and final held-out checks."""
import bootstrap
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reward_report import extract

LABELS={'sf-async':'Native SF async + Pogema','jointloom':'JointLoom GPU resident'}
COLORS={'sf-async':'#c07836','jointloom':'#207da4'}


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--desktop',type=Path,required=True)
    args=p.parse_args();root=args.root;training=root/'training'
    state=json.loads((root/'state.json').read_text())
    jobs=[j for j in state['jobs'] if j['phase']=='formal']
    if len(jobs)!=2 or not all(j['status']=='complete' for j in jobs):raise RuntimeError('incomplete training pair')
    all_raw=[];all_ep=[];groups={};summaries={}
    for job in jobs:
        profile=job['profile'];folder=training/job['run'];duration=job['budget_seconds']
        start=job['start_perf_counter_ns']
        bridge={'wall_minus_perf_ns':int(job['started']*1e9)-start}
        if profile=='jointloom':
            raw=[json.loads(s) for s in (folder/'iterations.jsonl').read_text().splitlines()]
            max_samples=raw[-1]['agent_samples'];native_ep=[]
        else:
            files=[json.loads(p.read_text()) for p in folder.glob('reward-env-*-pid-*.json')]
            max_samples=sum(sum(r['kind']=='rollout' for r in f['records']) for f in files)//16*65536
        rows,native_ep=extract(training,job,bridge,max_samples)
        for r in rows:r['within_wall_budget']=r['elapsed_job_seconds']<=duration
        all_raw.extend(rows);within=[r for r in rows if r['within_wall_budget']];groups[profile]=within
        ep=[]
        if profile=='jointloom':
            for index in range(len(raw)//16):
                chunk=raw[index*16:(index+1)*16]
                ep.append(dict(profile=profile,episode=index+1,per_env_steps=(index+1)*512,
                    mean_agent_episode_return=sum(r['collected_reward_sum'] for r in chunk)/2048,
                    elapsed_job_seconds=(chunk[-1]['sample_completed_perf_counter_ns']-start)/1e9))
        else:
            by_episode={}
            for record in native_ep:by_episode.setdefault(record['episode'],[]).append(record)
            for index,chunk in sorted(by_episode.items()):
                if len(chunk)!=16 or {r['env_id'] for r in chunk}!=set(range(16)):continue
                ep.append(dict(profile=profile,episode=index,per_env_steps=index*512,
                    mean_agent_episode_return=float(np.mean([r['mean_agent_episode_return'] for r in chunk])),
                    elapsed_job_seconds=(max(r['completed_perf_counter_ns'] for r in chunk)-start)/1e9))
        for r in ep:r['within_wall_budget']=r['elapsed_job_seconds']<=duration
        all_ep.extend(ep);ep_in=[r for r in ep if r['within_wall_budget']]
        summaries[profile]=dict(budget_seconds=duration,
            processed_samples_within_deadline=job['processed_samples_within_deadline'],
            adam_steps_within_deadline=job['processed_samples_within_deadline']//2048,
            collected_complete_group_samples_within_deadline=within[-1]['collected_agent_samples'],
            checkpoint_processed_samples=job['checkpoint_processed_samples'],
            checkpoint_adam_steps=job['checkpoint_adam_steps'],
            processed_samples_after_deadline=job['processed_samples_after_deadline'],
            full_job_seconds=job['full_job_seconds'],
            process_seconds_beyond_budget=job['full_job_seconds']-duration,
            tail16_mean_progress_reward=float(np.mean([r['mean_progress_reward'] for r in within[-16:]])),
            full_episodes_within_budget=len(ep_in),
            last16_episode_mean_return=float(np.mean([r['mean_agent_episode_return'] for r in ep_in[-16:]])),
            max_env_joint_steps=job['max_env_joint_steps'],task_bank_cycles=job['task_bank_cycles'],
            checkpoint=job['checkpoint'])
    write_csv(root/'reward-points.csv',all_raw);write_csv(root/'episode-return-points.csv',all_ep)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for profile,rows in groups.items():
        y=np.array([r['mean_progress_reward'] for r in rows]);smooth=min(16,len(y))
        mean=np.convolve(y,np.ones(smooth)/smooth,mode='valid')
        x=np.array([r['collected_agent_samples'] for r in rows])/1e6
        t=np.array([r['elapsed_job_seconds'] for r in rows])/60
        axes[0].plot(x,y,color=COLORS[profile],alpha=.08,lw=.5)
        axes[1].plot(t,y,color=COLORS[profile],alpha=.08,lw=.5)
        axes[0].plot(x[smooth-1:],mean,color=COLORS[profile],lw=1.5,label=LABELS[profile])
        axes[1].plot(t[smooth-1:],mean,color=COLORS[profile],lw=1.5,label=LABELS[profile])
    axes[0].set_xlabel('Collected agent transitions (millions)')
    axes[1].set_xlabel('Elapsed job time, including startup (minutes)')
    axes[0].set_title('Reward versus interactions');axes[1].set_title('Reward versus equal wall time')
    axes[1].set_xlim(0,max(j['budget_seconds'] for j in jobs)/60)
    for ax in axes:ax.set_ylabel('Raw progress reward / agent-step');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('CostTracer: one hour per method, 16 env × 128 agents, seed 0',fontsize=11)
    fig.savefig(root/'reward-curves.png',dpi=180);fig.savefig(root/'reward-curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for profile in groups:
        rows=[r for r in all_ep if r['profile']==profile and r['within_wall_budget']]
        y=[r['mean_agent_episode_return'] for r in rows]
        axes[0].plot([r['per_env_steps']*2048/1e6 for r in rows],y,color=COLORS[profile],lw=.8,label=LABELS[profile])
        axes[1].plot([r['elapsed_job_seconds']/60 for r in rows],y,color=COLORS[profile],lw=.8,label=LABELS[profile])
    axes[0].set_xlabel('Collected agent transitions (millions)')
    axes[1].set_xlabel('Elapsed job time, including startup (minutes)')
    axes[0].set_title('Complete 512-step episode return');axes[1].set_title('Episode return versus equal wall time')
    axes[1].set_xlim(0,max(j['budget_seconds'] for j in jobs)/60)
    for ax in axes:ax.set_ylabel('Mean agent return over 512 steps');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('One hour per method, seed 0; complete episodes, no smoothing',fontsize=11)
    fig.savefig(root/'episode-return-curves.png',dpi=180);fig.savefig(root/'episode-return-curves.pdf');plt.close(fig)
    evaluation=json.loads((root/'evaluation/results.json').read_text())
    speed=summaries['jointloom']['processed_samples_within_deadline']/summaries['sf-async']['processed_samples_within_deadline']
    summary=dict(budget='one hour per method including subprocess startup; finish current batch then save/exit',
        seed=0,num_envs=16,agents_per_env=128,cpu_affinity=state['cpu_affinity'],
        task_bank_episodes=state['bank_episodes'],profiles=summaries,
        within_deadline_processed_sample_ratio=speed,heldout_final_checkpoints=evaluation,
        timing='reward timestamps are collection completion; processed throughput is learner update completion; distinct counters',
        plot='only samples/episodes collected within own 3600-second deadline; all raw points including tail retained in CSV',
        caution='single training seed; native asynchronous policy lag retained; final checkpoints include separately recorded overshoot')
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['CostTracer: one-hour CPU/SF and JointLoom training results','',
        '16env x128agents; shared seed0 initial weights; RTX5080, eight physical CPU cores (CPU0..7) per method, sequential runs.',
        'Each3600-second budget starts at subprocess launch and includes initialization, compilation, training and checkpoint saves during the run.',
        f'Learner samples completed within3600 seconds (JointLoom / SF): {speed:.4f}.',
        'At the deadline, the current batch finishes before saving and exiting. The final batch and exit time are listed separately and excluded from that ratio.','']
    for profile,s in summaries.items():
        lines += [LABELS[profile]+':',
            f"  Samples completed within budget: {s['processed_samples_within_deadline']:,}; Adam updates: {s['adam_steps_within_deadline']:,}.",
            f"  Final-checkpoint samples: {s['checkpoint_processed_samples']:,}; extra samples after deadline: {s['processed_samples_after_deadline']:,}; full process: {s['full_job_seconds']:.3f}s.",
            f"  Mean per-step reward over the last16 windows={s['tail16_mean_progress_reward']:.7f}; mean return over the last16 complete episodes={s['last16_episode_mean_return']:.6f}.",
            f"  Complete episode groups within budget: {s['full_episodes_within_budget']}; task-bank cycles: {s['task_bank_cycles']}.",'']
    lines += ['Fixed unseen-map evaluation: two mazes, eight environments each,128agents/env,512 steps; final and initial models share tasks/RNG, with no best-checkpoint selection.',
        'Evaluation uses the actual final checkpoints, which may contain the small deadline-overrun batches listed above.']
    for r in evaluation:lines.append(f"{r['model']}: mean_agent_episode_return={r['mean_agent_episode_return']:.6f}；goals={r['completed_goals']:.0f}；goals/env-step={r['goals_per_joint_env_step']:.6f}。")
    lines += ['', 'Main curves stop at3600 seconds; reward curves show16-window means and raw points; episode curves are unsmoothed.',
        'Both sample-count and wall-clock axes are shown. These are single-seed measurements, without a claim of quality equivalence or statistical significance.',
        'Native asynchronous SF retains policy lag, different environment progress and in-flight samples. Collected samples and samples processed by the learner are recorded separately.',
        'The shared2048-episode bank stores randomized tasks, not trajectories, observations or rewards; actual cycling is reported above.',
        'Checkpoints and full raw logs: '+str(root),'']
    (root/'RESULTS.txt').write_text('\n'.join(lines))
    args.desktop.mkdir(parents=True,exist_ok=True)
    copies={'reward-curves.png':'training-reward-comparison.png','reward-curves.pdf':'training-reward-comparison.pdf',
        'episode-return-curves.png':'training-episode-return-comparison.png','episode-return-curves.pdf':'training-episode-return-comparison.pdf',
        'RESULTS.txt':'RESULTS.txt','summary.json':'summary.json','reward-points.csv':'reward-raw-records.csv',
        'episode-return-points.csv':'episode-return-raw-records.csv','TRAINING_PARAMETERS.txt':'TRAINING_PARAMETERS.txt'}
    for src,dst in copies.items():shutil.copy2(root/src,args.desktop/dst)
    print('\n'.join(lines[:18]))


if __name__=='__main__':main()
