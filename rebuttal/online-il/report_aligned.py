"""Human-readable aligned inference results, retaining every observed pair."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def run(root):
    rows=[];pairs=[]
    for n in [16,256]:
        path=root/f'l1-inference/n{n}/summary.json'
        if not path.exists():continue
        data=json.loads(path.read_text())
        for pair in data['pairs']:
            for phase,ratio in pair['speedup'].items():
                current={r['path']:r for r in data['runs'] if r['mode']==pair['mode']}
                # Preserve per-repetition times from their own result files.
                base=root/f"l1-inference/n{n}/{pair['mode']}/repeat-{pair['repetition']}"
                cpu=json.loads((base/'pogema_original/result.json').read_text())
                gpu=json.loads((base/'jointloom/result.json').read_text())
                rows.append(dict(envs=n,mode=pair['mode'],repeat=pair['repetition'],phase=phase,
                    cpu_seconds=cpu['metrics'][phase]['wall_seconds'],gpu_seconds=gpu['metrics'][phase]['wall_seconds'],
                    speedup=ratio,final_states_equal=all(pair['final_state_equal'].values()),
                    cpu_goals=cpu['completed_goals'],gpu_goals=gpu['completed_goals']))
            pairs.append((n,pair))
    out=root/'report';out.mkdir(exist_ok=True)
    if rows:
        with (out/'inference-pairs.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Aligned inference and order-verification status','',
        'Original A256 checkpoint and unchanged128x128 map pool. Both task modes use the same documented reachable initial-task resampling. CPU0–15; float32/TF32 off.',
        'Historical single-decision latencies are not sustained interaction measurements. No new JAX control is claimed.','',
        '| N | Mode | Paired repetitions | First ratio | After-one ratio | Sustained ratio | Equal final states |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for n in [16,256]:
        for mode in ['standard_mapf','lifelong']:
            selected=[p for count,p in pairs if count==n and p['mode']==mode]
            if not selected:continue
            med=[statistics.median(p['speedup'][phase] for p in selected) for phase in ['first','after_one','sustained']]
            equal=sum(all(p['final_state_equal'].values()) for p in selected)
            lines.append(f'| {n} | {mode} | {len(selected)}/3 | {med[0]:.3f}x | {med[1]:.3f}x | {med[2]:.3f}x | {equal}/{len(selected)} |')
    lines+=['','Goal totals include two individually timed lead-in decisions plus the256 sustained decisions; they are not a sustained-window goal-throughput numerator.',
        '', 'The N256 standard-mapf first-divergence diagnostic identifies step76, env98, agent200: node max error5.96e-8, identical edges/attributes, near-tied logits flip argmax. Lifelong endpoints also differ by one completed goal; its first divergence has not been separately localized. Final-state equality is reported per actual run, not assumed.',
        '', 'Order checks cover real ordered/window/global records, cross-episode goal changes, exact labels/edges, a513-stage sampler tail, and actual optimizer updates. These checks are not policy-quality results.',
        '', 'Full learning, replay, live expert and finite256 task results are tracked separately in execution-state.json; pending rows are not completed evidence.']
    lines+=['','## Original-shape fixed-record consumption','',
        'A512/N4,512 actual CE/Adam updates; original cached CPU H2/Q4 versus compact ring128/DMA. Expert generation excluded.','',
        '| Workload | Pairs | CPU median s | JointLoom median s | Paired median ratio |',
        '| --- | ---: | ---: | ---: | ---: |']
    for family in ['l2-fixed','l2-lifelong']:
        cpus=[];gpus=[];ratios=[]
        for seed in [42,43,44]:
            a=root/family/f'seed-{seed}/cpu_cached_prefetch/result.json'
            b=root/family/f'seed-{seed}/jointloom/result.json'
            if not a.exists() or not b.exists():continue
            cpu=json.loads(a.read_text());gpu=json.loads(b.read_text())
            cpus.append(cpu['wall_seconds']);gpus.append(gpu['wall_seconds']);ratios.append(cpus[-1]/gpus[-1])
        if ratios:lines.append(f'| {family} | {len(ratios)}/3 | {statistics.median(cpus):.3f} | {statistics.median(gpus):.3f} | {statistics.median(ratios):.3f}x |')
    lines+=['','## Live expert training (partial until all three seeds complete)','',
        'A512/E8,batch4096,original expert/prefetch separation,H8/Q16,ring512,300s. Ratios use actual agent-sample rates; each run has its own timed expert stream. Source worker time is not an additive wall-clock breakdown.','',
        'V1 blocking acquisition is retained as a diagnostic: an expert wait crossed the target300s boundary. Deadline-aware runs are a separate cohort, not pooled with V1.','',
        '| Cohort; mode/order | Paired seeds | CPU rate k/s | JointLoom rate k/s | Paired median ratio |',
        '| --- | ---: | ---: | ---: | ---: |']
    for family in ['live-deadline','live']:
        for mode,order in [('standard_mapf','ordered'),('lifelong','ordered'),('standard_mapf','window_shuffle_128')]:
            cpus=[];gpus=[];ratios=[]
            for seed in [0,1,2]:
                a=root/f'{family}/{mode}/{order}/seed-{seed}/cpu_cached_prefetch/result.json'
                b=root/f'{family}/{mode}/{order}/seed-{seed}/jointloom/result.json'
                if not a.exists() or not b.exists():continue
                cpu=json.loads(a.read_text());gpu=json.loads(b.read_text())
                cpus.append(cpu['samples_per_second']/1000);gpus.append(gpu['samples_per_second']/1000)
                if cpus[-1]>0:ratios.append(gpus[-1]/cpus[-1])
            if ratios:lines.append(f'| {family}; {mode}/{order} | {len(ratios)}/3 | {statistics.median(cpus):.3f} | {statistics.median(gpus):.3f} | {statistics.median(ratios):.3f}x |')
    state_file=root/'execution-state.json'
    if state_file.exists():
        state=json.loads(state_file.read_text());active=[j['name'] for j in state['jobs'] if j['status']=='running']
        learned=sum((root/f'order/learning/seed-{s}/{o}/result.json').exists() for s in [0,1,2]
            for o in ['ordered','window_shuffle_128','global_shuffle'])
        lines+=['',f"Queue state: {state['status']}; active: {', '.join(active) or 'none'}. Completed100k learning runs: {learned}/9."]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
