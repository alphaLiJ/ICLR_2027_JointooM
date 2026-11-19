"""Summarize real SF callbacks and verify the framework's actual saved update counts."""
import bootstrap
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from sf_tasks import SF_ROOT


def main():
    root=SF_ROOT/'paired-v1';all_rows=[];runs=[]
    for seed in (0,1,2):
        for profile in ('sf-sync','sf-async','jointloom-sf2'):
            name=f'{profile}-s{seed}';folder=root/name
            if profile=='jointloom-sf2':
                summary=json.loads((folder/'summary.json').read_text())
                rows=[json.loads(x) for x in (folder/'iterations.jsonl').read_text().splitlines()]
                intervals=[r['iteration_seconds'] for r in rows]
                path=folder/'final.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                updates=sorted(set(float(x['step']) for x in checkpoint['optimizer']['state'].values()))
                trained=checkpoint['agent_samples']
                canonical=torch.load(SF_ROOT/f'initial-seed-{seed}.pt',map_location='cpu',weights_only=True)
                initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True)
                assert all(torch.equal(v,initial[k]) for k,v in canonical.items())
            else:
                summary=json.loads((folder/'native-timing.json').read_text());rows=summary['rows']
                intervals=[r['interval_seconds'] for r in rows]
                checkpoints=list((SF_ROOT/'runs'/name/'checkpoint_p0').glob('checkpoint_*.pth'))
                path=max(checkpoints,key=lambda p:p.stat().st_mtime)
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                updates=sorted(set(float(x['step']) for x in checkpoint['optimizer']['state'].values()))
                trained=checkpoint['env_steps']
            assert all(bool(torch.isfinite(v).all()) for v in checkpoint['model'].values())
            if profile!='sf-async':assert updates==[512.] and trained==1048576
            run=dict(profile=profile,seed=seed,observer_callbacks=len(rows),
                measured_callbacks=len(rows)-2,steady_iteration_seconds=float(np.mean(intervals[2:])),
                checkpoint_samples=trained,checkpoint_adam_steps=updates,checkpoint=str(path),
                stop_inflight_extra_batches=(trained-1048576)//65536)
            runs.append(run)
            all_rows.extend(dict(profile=profile,seed=seed,ordinal=i+1,interval_seconds=s,
                                 samples_per_callback=65536) for i,s in enumerate(intervals))
    result=dict(baseline='official Sample Factory 2.1.1 + Pogema 1.2.2, native core unchanged',
        customizations='official environment / encoder / model registration and logging observer',
        num_envs=16,agents_per_env=128,workers=8,envs_per_worker=2,
        rollout=32,minibatch=2048,updates_per_training_iteration=32,seeds=[0,1,2],
        common_initial_weights=True,fp32=True,tf32=False,normalize_input=False,normalize_returns=False,
        math_alignment=dict(advantage_min_std=1e-7,value_bootstrap='native SF2 current-value timeout approximation'),
        sampling='native batched sampler + native double buffering for both SF modes',
        statistics='mean and sample standard deviation of three seed means; first two batch completions excluded',
        profiles={},runs=runs)
    for profile in ('sf-sync','sf-async','jointloom-sf2'):
        values=np.array([r['steady_iteration_seconds'] for r in runs if r['profile']==profile])
        result['profiles'][profile]=dict(seconds=float(values.mean()),seed_std=float(values.std(ddof=1)),
                                        processed_agent_samples_per_second=65536/float(values.mean()))
    own=result['profiles']['jointloom-sf2']['seconds']
    result['speedups']=dict(vs_native_synchronous=result['profiles']['sf-sync']['seconds']/own,
                            vs_native_asynchronous=result['profiles']['sf-async']['seconds']/own)
    result['scope']='short efficiency pilot; synchronous budget matched; native async has documented in-flight overshoot and policy lag'
    (root/'results.json').write_text(json.dumps(result,indent=2))
    with (root/'intervals.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    source=root/'source';source.mkdir(exist_ok=True)
    for name in ('sf_native.py','sf_tasks.py','sf2_compat.py','sf_run_trials.py','sf_validate.py',
                 'sf_summarize.py','optimized_gpu.py','train_pipeline.py','model.py','environments.py'):
        shutil.copy2(Path(__file__).with_name(name),source/name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.6,3.8),layout='constrained')
    names=['sf-sync','sf-async','jointloom-sf2']
    values=[result['profiles'][n]['seconds'] for n in names]
    errors=[result['profiles'][n]['seed_std'] for n in names]
    ax.bar(range(3),values,yerr=errors,capsize=4,color=['#c57a38','#d99c6b','#207ca4'])
    ax.set_xticks(range(3),['Native SF\nsynchronous','Native SF\nasynchronous','JointLoom\nSF2-compatible'])
    ax.set_ylabel('Seconds per 65,536 processed agent samples')
    for i,v in enumerate(values):ax.text(i,v+.012,f'{v:.3f} s',ha='center')
    ax.set_ylim(0,.4);ax.set_title('Real Sample Factory baseline, 3 seeds')
    fig.savefig(root/'native-sf-comparison.png',dpi=180);fig.savefig(root/'native-sf-comparison.pdf')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
