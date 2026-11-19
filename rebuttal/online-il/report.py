"""Replot the existing strong-CPU logging cohort without running new training."""
import bootstrap
from repository_paths import ARTIFACTS, input_path
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def lifelong(out):
    root=input_path('JOINTLOOM_ONLINE_ARTIFACTS', ARTIFACTS/'online-il')/'e2'
    rows=[]
    for family,name in [('Maze','formal-maze-w4'),('Random','formal-random-w4'),('Warehouse','formal-warehouse-w4')]:
        d=json.loads((root/name/'summary.json').read_text())
        cpu=[r for r in d['runs'] if r['path']=='cpu_pipeline'];gpu=[r for r in d['runs'] if r['path']=='jointloom']
        rows.append(dict(family=family,cpu_median_seconds=float(np.median([r['wall_seconds'] for r in cpu])),
            gpu_median_seconds=float(np.median([r['wall_seconds'] for r in gpu])),
            paired_median_speedup=d['median_paired_speedup'],
            speedup_min=min(p['speedup'] for p in d['pairs']),speedup_max=max(p['speedup'] for p in d['pairs']),
            completed_goals=cpu[0]['completed_goals'],
            exact_final_states=all(all(p['final_state_equal'].values()) for p in d['pairs']),
            goals_in_last_256_steps=cpu[0]['completed_goals']-cpu[0]['cumulative_goals_every_128_steps'][-3],
            source=str(root/name/'summary.json')))
    out.mkdir(parents=True,exist_ok=True)
    with (out/'table.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(rows,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(10.8,3),layout='constrained')
    for ax,row,name in zip(axes,rows,['formal-maze-w4','formal-random-w4','formal-warehouse-w4']):
        raw=json.loads((root/name/'summary.json').read_text())['runs'][0]
        values=[0]+raw['cumulative_goals_every_128_steps']
        ax.plot(np.arange(len(values))*128,values,'o-',ms=3)
        ax.set_title(row['family']);ax.set_xlabel('Decision steps per environment');ax.grid(alpha=.2)
    axes[0].set_ylabel('Completed goals across 16 environments')
    fig.savefig(out/'goal-progress.pdf');fig.savefig(out/'goal-progress.png',dpi=160);plt.close(fig)
    print(json.dumps(rows,indent=2))


def curves(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old=input_path('JOINTLOOM_LEGACY_ARTIFACTS', ARTIFACTS/'legacy')/'jointloom-training-progress-analysis-20260921-v1'
    cached=input_path('JOINTLOOM_LEGACY_ARTIFACTS', ARTIFACTS/'legacy')/'jointloom-strong-cpu-learning-20260921-v1/runs'
    observations=[];losses=[]
    with (old/'observations.csv').open() as f:
        for r in csv.DictReader(f):
            if r['cohort']=='august' and r['method'] in ('conventional_sync','proposed_async'):
                observations.append(dict(method=r['method'],seed=int(r['seed']),updates=int(r['updates']),
                    accuracy=float(r['val_accuracy_pct']),event=r['event'],source=r['source']))
    raw={}
    with (old/'raw-loss-by-update.csv').open() as f:
        for r in csv.DictReader(f):
            if r['cohort']=='august' and r['method'] in ('conventional_sync','proposed_async'):
                raw.setdefault((r['method'],int(r['seed'])),[]).append((int(r['optimizer_step']),float(r['loss'])))
    for seed in range(3):
        source=cached/f'seed-{seed}/result.json'
        result=json.loads(source.read_text())
        for r in result['checkpoint_history']:
            observations.append(dict(method='cpu_cached_prefetch',seed=seed,updates=r['optimizer_step'],
                accuracy=100*r['val_overall_accuracy'],event=r['event_kind'],source=str(source)))
        with (cached/f'seed-{seed}/learning-trace.csv').open() as f:
            raw['cpu_cached_prefetch',seed]=[(int(r['optimizer_step']),float(r['loss'])) for r in csv.DictReader(f)]
    for (method,seed),records in raw.items():
        records=sorted(records)
        for start in range(0,len(records),200):
            block=records[start:start+200]
            if len(block)==200:
                losses.append(dict(method=method,seed=seed,updates=block[-1][0],loss=float(np.mean([v for _,v in block]))))
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('validation-by-update.csv',observations),('loss-by-update.csv',losses)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.9))
    names={'conventional_sync':'CPU synchronous (Aug)', 'proposed_async':'JointLoom (Aug)',
           'cpu_cached_prefetch':'CPU cached prefetch (Sep logging)'}
    colors={'conventional_sync':'#d55e00','proposed_async':'#0072b2','cpu_cached_prefetch':'#009e73'}
    for method in names:
        for seed,style in enumerate(['-', '--', ':']):
            obs=sorted([r for r in observations if r['method']==method and r['seed']==seed and r['updates']>0],key=lambda r:r['updates'])
            ls=sorted([r for r in losses if r['method']==method and r['seed']==seed],key=lambda r:r['updates'])
            label=f'{names[method]}, seed {seed}'
            axes[0].plot([r['updates'] for r in obs],[r['accuracy'] for r in obs],style,color=colors[method],marker='.',ms=4,lw=1,label=label)
            axes[1].plot([r['updates'] for r in ls],[r['loss'] for r in ls],style,color=colors[method],lw=.9,alpha=.85)
    for ax in axes:
        ax.set_xscale('log');ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2)
    axes[0].set_ylabel('Frozen-validation action accuracy (%)')
    axes[1].set_ylabel('Training loss (disjoint 200-update means)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0,.22,1,1))
    fig.legend(handles,labels,fontsize=7,loc='lower center',ncol=3)
    fig.savefig(out/'learning-by-update.pdf',bbox_inches='tight');fig.savefig(out/'learning-by-update.png',dpi=180,bbox_inches='tight');plt.close(fig)
    summary=dict(runs=[dict(method=m,seed=s,raw_loss_count=len(v),last_update=max(u for u,_ in v)) for (m,s),v in raw.items()],
        validation_observations=len(observations),loss_blocks=len(losses),training_executed=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'README.md').write_text('''# Strong CPU update-axis reanalysis

Existing August JointLoom/synchronous-CPU runs and September cached-prefetch logging runs, all three seeds.
No training, model inference or expert search was executed by this renderer.
Validation uses actual recorded checkpoints; no interpolation or best-so-far envelope.
Loss uses disjoint 200-update means. Initial observations at update zero are retained in CSV but omitted on the log axis.
The weak cached-prefetch seed remains visible. Terminal evaluations retain their recorded event labels.
The cohorts share the generation protocol and validation tasks, not an identical full training record stream;
this is descriptive update alignment, not the new same-data ordered/shuffled control.
''')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['curves','lifelong']);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    {'curves':curves,'lifelong':lifelong}[args.command](args.out)
