"""Normalize scaling by processed samples and preserve native asynchronous overshoot."""
import bootstrap
import json
import csv
from pathlib import Path
import numpy as np
import torch
from sf_tasks import SF_ROOT


def main():
    root=SF_ROOT/'supply-scaling-8cores-v1';experiment=json.loads((root/'experiment.json').read_text())
    runs=[];rows=[]
    for ne in (16,32):
        samples=ne*128*32;updates=samples//2048
        for seed in (0,1,2):
            for profile in ('sf-async','jointloom'):
                name=f'{profile}-e{ne}-s{seed}';folder=root/name
                resources=json.loads((folder/'external-resources.json').read_text())
                assert resources['exit_code']==0 and resources['cpu_affinity']==experiment['cpu_affinity']
                if profile=='jointloom':
                    meta=json.loads((folder/'metadata.json').read_text())
                    summary=json.loads((folder/'summary.json').read_text())
                    data=[json.loads(x) for x in (folder/'iterations.jsonl').read_text().splitlines()]
                    intervals=[r['iteration_seconds'] for r in data];phase=summary['steady_mean']
                    checkpoint_path=folder/'final.pt';checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
                    processed=checkpoint['agent_samples']
                    assert summary['adam_step_counters']==[16*updates] and processed==16*samples
                    assert meta['cpu_affinity']==experiment['cpu_affinity']
                    allocated=max(r['peak_gpu_memory_bytes'] for r in data)
                else:
                    summary=json.loads((folder/'native-timing.json').read_text())
                    data=summary['rows'];intervals=[r['interval_seconds'] for r in data]
                    assert summary['samples_per_iteration']==samples
                    assert summary['cpu_affinity']==experiment['cpu_affinity']
                    config=json.loads((SF_ROOT/'runs'/f'{root.name}-{name}'/'config.json').read_text())
                    assert config['num_batches_per_epoch']==updates and config['num_envs_per_worker']==ne//8
                    assert config['async_rl'] and config['worker_num_splits']==2
                    checkpoints=list((SF_ROOT/'runs'/f'{root.name}-{name}'/'checkpoint_p0').glob('checkpoint_*.pth'))
                    checkpoint_path=max(checkpoints,key=lambda p:p.stat().st_mtime)
                    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
                    processed=checkpoint['env_steps'];phase=None;allocated=None
                steps=sorted(set(float(x['step']) for x in checkpoint['optimizer']['state'].values()))
                assert steps==[processed/2048]
                assert all(bool(torch.isfinite(v).all()) for v in checkpoint['model'].values())
                duration=float(np.mean(intervals[2:]))
                row=dict(profile=profile,num_envs=ne,seed=seed,samples_per_iteration=samples,
                    adam_updates_per_iteration=updates,complete_seconds=duration,agent_samples_per_second=samples/duration,
                    actual_checkpoint_samples=processed,actual_adam_steps=steps[0],
                    extra_inflight_batches=(processed-16*samples)//samples,
                    cpu_cores_used=resources['steady_cpu_cores_used_mean'],
                    gpu_utilization=resources['steady_gpu_utilization_mean'],
                    gpu_device_peak_bytes=resources['gpu_device_peak_bytes'],
                    steady_gpu_device_peak_bytes=resources['steady_gpu_memory_peak_bytes'],
                    torch_allocated_peak_bytes=allocated,phase=phase)
                runs.append(row)
                rows.extend(dict(profile=profile,num_envs=ne,seed=seed,ordinal=i+1,
                                 complete_seconds=value,samples=samples) for i,value in enumerate(intervals))
    result=dict(experiment=experiment,runs=runs,scales={})
    for ne in (16,32):
        values={}
        for profile in ('sf-async','jointloom'):
            selected=[r for r in runs if r['num_envs']==ne and r['profile']==profile];stats={}
            for key in ('complete_seconds','agent_samples_per_second','cpu_cores_used','gpu_utilization',
                        'gpu_device_peak_bytes','steady_gpu_device_peak_bytes'):
                array=np.array([r[key] for r in selected])
                stats[key]=dict(mean=float(array.mean()),seed_std=float(array.std(ddof=1)))
            stats['samples_per_cpu_second']=stats['agent_samples_per_second']['mean']/stats['cpu_cores_used']['mean']
            if profile=='jointloom':
                stats['phases']={k:float(np.mean([r['phase'][k] for r in selected])) for k in selected[0]['phase']}
                samples=ne*128*32
                stats['sampling_phase_samples_per_second']=samples/stats['phases']['sample_seconds']
                stats['update_phase_samples_per_second']=samples/stats['phases']['update_seconds']
                stats['update_fraction']=stats['phases']['update_seconds']/stats['phases']['iteration_seconds']
            values[profile]=stats
        values['complete_speedup']=values['sf-async']['complete_seconds']['mean']/values['jointloom']['complete_seconds']['mean']
        result['scales'][ne]=values
    result['scaling_16_to_32']={profile:result['scales'][32][profile]['agent_samples_per_second']['mean']/
        result['scales'][16][profile]['agent_samples_per_second']['mean'] for profile in ('sf-async','jointloom')}
    result['conclusion']='32x128 capacity verified for both; no widening speedup at this model/configuration; JointLoom uses less CPU time'
    result['limits']=['finite short pilots, not convergence/quality equivalence',
        'same rollout and epoch; macro batch and optimizer work double with environment count',
        'NVML device memory includes driver/context/allocator; GPU utilization is coarse busy-time, not SM occupancy',
        'CPU core-equivalent time is not proof of identical performance under a one-core affinity limit',
        'async SF can overshoot by in-flight batches; actual checkpoints recorded']
    (root/'results.json').write_text(json.dumps(result,indent=2))
    with (root/'intervals.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (root/'run-summary.csv').open('w',newline='') as f:
        flat=[{k:v for k,v in r.items() if k!='phase'} for r in runs]
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for profile,color,label in [('sf-async','#c57a38','Native SF async'),('jointloom','#207ca4','JointLoom')]:
        throughput=[result['scales'][n][profile]['agent_samples_per_second']['mean']/1000 for n in (16,32)]
        error=[result['scales'][n][profile]['agent_samples_per_second']['seed_std']/1000 for n in (16,32)]
        axes[0].errorbar([16,32],throughput,yerr=error,color=color,marker='o',capsize=4,label=label)
        cpu=[result['scales'][n][profile]['cpu_cores_used']['mean'] for n in (16,32)]
        offset=-.18 if profile=='sf-async' else .18
        axes[1].bar(np.arange(2)+offset,cpu,width=.36,color=color,label=label)
    axes[0].set_xticks([16,32]);axes[0].set_ylim(0,400);axes[0].set_xlabel('Concurrent environments (128 agents each)')
    axes[0].set_ylabel('Training throughput (thousand agent samples/s)');axes[0].legend(fontsize=8)
    axes[1].set_xticks([0,1],['16 environments','32 environments']);axes[1].set_ylim(0,8.6)
    axes[1].axhline(8,color='#777777',linestyle='--',linewidth=1,label='8-core affinity budget')
    axes[1].set_ylabel('Average process-tree CPU cores used');axes[1].legend(fontsize=8)
    fig.suptitle('Fixed 8 physical cores, three seeds, same PPO work per sample')
    fig.savefig(root/'supply-scaling.png',dpi=180);fig.savefig(root/'supply-scaling.pdf')
    print(json.dumps({k:result[k] for k in ('scales','scaling_16_to_32','conclusion')},indent=2))


if __name__=='__main__':main()
