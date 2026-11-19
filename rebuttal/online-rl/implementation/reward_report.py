"""Actual native SF and JointLoom reward curves with explicit sample clocks."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROFILES=('sf-sync','sf-async','jointloom')
LABELS={'sf-sync':'Native SF synchronous (CPU env)',
        'sf-async':'Native SF asynchronous (CPU env)',
        'jointloom':'JointLoom GPU resident'}
COLORS={'sf-sync':'#bc763b','sf-async':'#699448','jointloom':'#207da4'}


def clock_bridge():
    samples=[]
    for _ in range(20):
        before=time.perf_counter_ns();wall=time.time_ns();after=time.perf_counter_ns()
        samples.append(dict(bracket_ns=after-before,wall_minus_perf_ns=wall-(before+after)//2))
    return min(samples,key=lambda x:x['bracket_ns'])


def extract(root,job,bridge,max_samples):
    folder=root/job['run'];profile=job['profile'];seed=job['seed']
    start=json.loads((folder/'training-clock.json').read_text())['start_perf_counter_ns']
    job_start=int(job['started']*1e9)-bridge['wall_minus_perf_ns']
    rows=[]
    if profile=='jointloom':
        raw=[json.loads(s) for s in (folder/'iterations.jsonl').read_text().splitlines()]
        for r in raw:
            rows.append(dict(profile=profile,seed=seed,collected_agent_samples=r['agent_samples'],
                reward_sum=r['collected_reward_sum'],
                mean_progress_reward=r['collected_reward_sum']/r['rollout_agent_samples'],
                training_seconds=(r['sample_completed_perf_counter_ns']-start)/1e9,
                elapsed_job_seconds=(r['sample_completed_perf_counter_ns']-job_start)/1e9,
                source=str(folder/'iterations.jsonl')))
        episodes=[]
    else:
        files=[json.loads(p.read_text()) for p in folder.glob('reward-env-*-pid-*.json')]
        events=[r for f in files for r in f['records'] if r['kind']=='rollout']
        events.sort(key=lambda x:(x['completed_perf_counter_ns'],x['env_id'],x['env_joint_steps']))
        if len({(r['env_id'],r['env_joint_steps']) for r in events})!=len(events):raise ValueError('duplicate reward block')
        if any(r['agent_samples']!=4096 for r in events):raise ValueError('reward denominator')
        for index in range(len(events)//16):
            group=events[index*16:(index+1)*16]
            total=sum(r['reward_sum'] for r in group);end=max(r['completed_perf_counter_ns'] for r in group)
            rows.append(dict(profile=profile,seed=seed,collected_agent_samples=(index+1)*65536,
                reward_sum=total,mean_progress_reward=total/65536,
                training_seconds=(end-start)/1e9,elapsed_job_seconds=(end-job_start)/1e9,
                source=str(folder)))
        episodes=[dict(profile=profile,seed=seed,**r) for f in files for r in f['records'] if r['kind']=='episode']
    rows=[r for r in rows if r['collected_agent_samples']<=max_samples]
    if not rows or rows[-1]['collected_agent_samples']!=max_samples:raise ValueError('incomplete reward axis')
    if not all(np.isfinite(r['mean_progress_reward']) and 0<=r['mean_progress_reward']<=.100001 for r in rows):
        raise ValueError('invalid raw reward')
    return rows,episodes


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--smooth',type=int,default=4)
    args=p.parse_args();root=args.root
    jobs=[j for j in json.loads((root/'status.json').read_text()) if j['run'].startswith('formal-')]
    if len(jobs)!=9 or not all(j['status']=='complete' for j in jobs):raise RuntimeError('formal runs incomplete')
    bridge=clock_bridge();(root/'plot-clock-bridge.json').write_text(json.dumps(bridge,indent=2))
    budget=jobs[0]['iterations']*65536
    raw=[];episodes=[];grouped={};summaries=[]
    for job in jobs:
        rows,ep=extract(root,job,bridge,budget);raw.extend(rows);episodes.extend(ep)
        grouped[(job['profile'],job['seed'])]=rows
        tail=rows[-16:]
        summaries.append(dict(profile=job['profile'],seed=job['seed'],collected_agent_samples=budget,
            final_raw_reward=rows[-1]['mean_progress_reward'],
            tail_16_rollout_mean_reward=float(np.mean([r['mean_progress_reward'] for r in tail])),
            time_to_collect_budget_including_startup=rows[-1]['elapsed_job_seconds'],
            time_to_collect_budget_from_native_training_start=rows[-1]['training_seconds']))
    with (root/'reward-points.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=raw[0]);w.writeheader();w.writerows(raw)
    (root/'native-episode-returns.json').write_text(json.dumps(episodes,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.0),layout='constrained')
    kernel=np.ones(args.smooth)/args.smooth
    for profile in PROFILES:
        runs=[grouped[(profile,s)] for s in (0,1,2)];color=COLORS[profile]
        xs=[];ys=[];ts=[]
        for rows in runs:
            x=np.array([r['collected_agent_samples'] for r in rows])[args.smooth-1:]/1e6
            y=np.convolve([r['mean_progress_reward'] for r in rows],kernel,mode='valid')
            t=np.array([r['elapsed_job_seconds'] for r in rows])[args.smooth-1:]
            xs.append(x);ys.append(y);ts.append(t)
            axes[0].plot(x,y,color=color,alpha=.12,lw=.65)
            axes[1].plot(t,y,color=color,alpha=.12,lw=.65)
        values=np.stack(ys);mean=values.mean(0);sd=values.std(0,ddof=1)
        style='--' if profile=='sf-async' else '-'
        axes[0].plot(xs[0],mean,color=color,lw=1.9,ls=style,label=LABELS[profile])
        axes[0].fill_between(xs[0],mean-sd,mean+sd,color=color,alpha=.13)
        grid=np.linspace(max(t[0] for t in ts),min(t[-1] for t in ts),256)
        temporal=np.stack([np.interp(grid,t,y) for t,y in zip(ts,ys)])
        axes[1].plot(grid,temporal.mean(0),color=color,lw=1.9,ls=style,label=LABELS[profile])
        axes[1].fill_between(grid,temporal.mean(0)-temporal.std(0,ddof=1),
            temporal.mean(0)+temporal.std(0,ddof=1),color=color,alpha=.13)
    axes[0].set_xlabel('Collected agent transitions (millions)')
    axes[1].set_xlabel('Elapsed job time, including startup (seconds)')
    axes[0].set_title('Reward versus interactions')
    axes[1].set_title('Reward versus wall time')
    for ax in axes:
        ax.set_ylabel('Mean raw progress reward / agent-step')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc='best')
    fig.suptitle('CostTracer PPO: 16 environments × 128 agents, 3 seeds',fontsize=11)
    fig.savefig(root/'reward-curves.png',dpi=180);fig.savefig(root/'reward-curves.pdf')
    results=json.loads((root/'results.json').read_text())
    report=dict(metric='raw training-period progress reward per collected agent transition',
        common_collected_sample_budget=budget,profiles={},runs=summaries,
        cpu_affinity=list(range(8)),num_envs=16,agents_per_env=128,concurrent_agents=2048,
        smoothing=dict(rolling_rollouts=args.smooth,agent_transitions=args.smooth*65536,
            raw_data_retained=True),bands='sample standard deviation across 3 seeds; not confidence interval',
        sample_axis='collected transitions; CPU groups16 timestamp-ordered per-env32-step blocks; GPU native macro endpoints',
        wall_axis='full elapsed subprocess job time including imports/spawn/model init/compile/env init; monotonic timestamps bridged to launch wall clock',
        timing_note='native training-start excludes different initialization stages; recorded separately, not used as headline wall axis',
        async_note='original asynchronous SF policy lag and in-flight extra batch preserved; plotted collection axis clipped to common budget',
        no_claim='short training trace is not convergence, held-out policy quality, or an RL algorithm comparison')
    for profile in PROFILES:
        group=[r for r in summaries if r['profile']==profile]
        report['profiles'][profile]={}
        for key in ('tail_16_rollout_mean_reward','time_to_collect_budget_including_startup',
                    'time_to_collect_budget_from_native_training_start'):
            vals=[r[key] for r in group]
            report['profiles'][profile][key]=dict(mean=statistics.mean(vals),seed_sd=statistics.stdev(vals))
        report['profiles'][profile]['actual_checkpoints']=[r for r in results if r['phase']=='formal' and r['profile']==profile]
    (root/'reward-summary.json').write_text(json.dumps(report,indent=2))
    # A complete 512-step return integrates the reset-related raw-reward cycle.
    episode_rows=[]
    for job in jobs:
        profile=job['profile'];seed=job['seed'];folder=root/job['run']
        start=int(job['started']*1e9)-bridge['wall_minus_perf_ns']
        if profile=='jointloom':
            source=[json.loads(s) for s in (folder/'iterations.jsonl').read_text().splitlines()]
            for e in range(len(source)//16):
                chunk=source[e*16:(e+1)*16]
                episode_rows.append(dict(profile=profile,seed=seed,episode=e+1,
                    per_env_steps=(e+1)*512,
                    mean_agent_episode_return=sum(r['collected_reward_sum'] for r in chunk)/2048,
                    elapsed_job_seconds=(chunk[-1]['sample_completed_perf_counter_ns']-start)/1e9))
        else:
            relevant=[r for r in episodes if r['profile']==profile and r['seed']==seed]
            for e in sorted(set(r['episode'] for r in relevant)):
                chunk=[r for r in relevant if r['episode']==e]
                if len(chunk)!=16 or {r['env_id'] for r in chunk}!=set(range(16)):continue
                episode_rows.append(dict(profile=profile,seed=seed,episode=e,
                    per_env_steps=e*512,
                    mean_agent_episode_return=float(np.mean([r['mean_agent_episode_return'] for r in chunk])),
                    elapsed_job_seconds=(max(r['completed_perf_counter_ns'] for r in chunk)-start)/1e9))
    with (root/'episode-return-points.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=episode_rows[0]);w.writeheader();w.writerows(episode_rows)
    efig,eaxes=plt.subplots(1,2,figsize=(10.4,4.0),layout='constrained')
    final_returns={}
    for profile in PROFILES:
        group=[sorted([r for r in episode_rows if r['profile']==profile and r['seed']==s],key=lambda r:r['episode']) for s in (0,1,2)]
        common=min(len(r) for r in group);group=[r[:common] for r in group]
        x=np.array([r['per_env_steps'] for r in group[0]])
        y=np.array([[r['mean_agent_episode_return'] for r in rs] for rs in group])
        times=np.array([[r['elapsed_job_seconds'] for r in rs] for rs in group])
        style='--' if profile=='sf-async' else '-';color=COLORS[profile]
        for rs in range(3):
            eaxes[0].plot(x,y[rs],color=color,alpha=.15,lw=.7)
            eaxes[1].plot(times[rs],y[rs],color=color,alpha=.15,lw=.7)
        eaxes[0].plot(x,y.mean(0),color=color,ls=style,lw=1.9,marker='o',ms=3,label=LABELS[profile])
        eaxes[0].fill_between(x,y.mean(0)-y.std(0,ddof=1),y.mean(0)+y.std(0,ddof=1),color=color,alpha=.13)
        # Plot mean completion-time coordinates; each return averages all 16 envs.
        eaxes[1].plot(times.mean(0),y.mean(0),color=color,ls=style,lw=1.9,marker='o',ms=3,label=LABELS[profile])
        eaxes[1].fill_between(times.mean(0),y.mean(0)-y.std(0,ddof=1),y.mean(0)+y.std(0,ddof=1),color=color,alpha=.13)
        final_returns[profile]=dict(episodes=common,first_mean=float(y[:,0].mean()),
            last_mean=float(y[:,-1].mean()),last_seed_sd=float(y[:,-1].std(ddof=1)))
    eaxes[0].set_xlabel('Completed interaction steps per environment')
    eaxes[1].set_xlabel('Elapsed job time, including startup (seconds)')
    eaxes[0].set_title('Episode return versus interactions')
    eaxes[1].set_title('Episode return versus wall time')
    for ax in eaxes:
        ax.set_ylabel('Mean agent return over 512 steps');ax.grid(alpha=.2)
    eaxes[0].legend(fontsize=8,loc='lower right')
    efig.suptitle('CostTracer PPO: 16 environments × 128 agents, 3 seeds',fontsize=11)
    efig.savefig(root/'episode-return-curves.png',dpi=180);efig.savefig(root/'episode-return-curves.pdf')
    report.update(episode_returns=final_returns,
        episode_figure='all16 envs complete each512-step episode; mean±seed SD; no smoothing; time coordinates average seed completion times')
    (root/'reward-summary.json').write_text(json.dumps(report,indent=2))
    lines=['Native Sample Factory / JointLoom short-budget reward training results','',
        'Workload:16env,128agents/env,2048 concurrent agents; eight physical CPU cores per method, RTX5080.',
        '65536 samples and32 Adam updates per iteration;128 formal iterations per seed, three seeds and three profiles, totaling nine runs.',
        'Synchronous SF and JointLoom each process8388608 samples and4096 updates; native asynchronous SF in-flight batches are listed separately.',
        'Shared random initialization; fixed raw reward and PPO/GAE/Adam definitions; SF keeps its default unshuffled order and JointLoom uses the same actor-major grouping.',
        '', 'Mean reward over the last16 sampling windows (three-seed mean +/- seed standard deviation):']
    for profile in PROFILES:
        v=report['profiles'][profile]['tail_16_rollout_mean_reward']
        t=report['profiles'][profile]['time_to_collect_budget_including_startup']
        lines.append(f"{profile}: reward {v['mean']:.8f} +/- {v['seed_sd']:.8f}; collection-budget completion time including startup {t['mean']:.3f} +/- {t['seed_sd']:.3f}s.")
    lines += ['', '512-step episode return per agent, averaged over three seeds:']
    for profile in PROFILES:
        v=final_returns[profile]
        lines.append(f"{profile}: episode1 {v['first_mean']:.6f} -> episode{v['episodes']} {v['last_mean']:.6f} ± {v['last_seed_sd']:.6f}。")
    lines += ['', 'Left: equal cumulative collected interactions. Right: full-job wall-clock time including startup, compilation and environment initialization for both methods.',
        'Some CPU framework environment initialization occurs after on_start; GPU initialization occurs before the loop. Both main curves include startup to align the timing boundary.',
        'Reward is the raw +0.1 environment progress reward before timeout bootstrap and GAE adjustments; it is separate from value estimates, returns and losses.',
        f'The plot uses a {args.smooth}-window moving average ({args.smooth*65536} interactions); shading shows the three-seed standard deviation and all raw points are retained.',
        'Wall-clock curves are interpolated only over the time range shared by all three seeds per method, without extrapolation.',
        'Asynchronous SF retains different environment progress and policy lag; collected and processed samples are reported separately, with potentially different trajectories and update times.',
        'These are short-budget training reward measurements; convergence, held-out generalization and algorithm quality are separate evaluation questions.',
        'Per-step reward varies periodically after synchronized512-step resets. An additional unsmoothed complete-episode return plot and all raw records are provided.',
        'Each episode-plot point includes the same episode index from all16 environments; its time is when every environment has completed that episode, averaged across seeds.',
        '', 'Files: reward-curves.png/pdf, reward-points.csv, reward-summary.json, TRAINING_PARAMETERS.txt.','']
    (root/'REWARD_RESULTS.txt').write_text('\n'.join(lines))
    print('\n'.join(lines[:13]))


if __name__=='__main__':main()
