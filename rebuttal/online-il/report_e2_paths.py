"""Summarize original-Pogema lifelong inference and the two IL timing scopes."""
import bootstrap
from repository_paths import ARTIFACTS, input_path
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main(root,out):
    rows=[];details={}
    cases=[('inference-maze',root/'e2-original/formal-maze'),
           ('inference-random',root/'e2-original/formal-random'),
           ('inference-warehouse',root/'e2-original/formal-warehouse'),
           ('IL-consumption',root/'e2-il/consume-128'),
           ('IL-online',root/'e2-il/online-64')]
    for name,directory in cases:
        d=json.loads((directory/'summary.json').read_text());details[name]=d
        a=[r for r in d['runs'] if r['path']=='pogema_original']
        b=[r for r in d['runs'] if r['path']=='jointloom']
        inference=name.startswith('inference')
        same=[all(p['final_state_equal'].values()) if inference else p['consumed_records_equal'] for p in d['pairs']]
        rows.append(dict(case=name,window=d['config']['measured_steps'] if inference else d['config']['updates'],
            original_median_seconds=float(np.median([r['wall_seconds'] for r in a])),
            jointloom_median_seconds=float(np.median([r['wall_seconds'] for r in b])),
            median_paired_speedup=d['median_paired_speedup'],
            speedup_min=min(p['speedup'] for p in d['pairs']),speedup_max=max(p['speedup'] for p in d['pairs']),
            matched_pairs=sum(same),pair_count=len(same),
            match_scope='final positions/goals/cursors' if inference else 'all consumed compact records and labels',
            expert_in_timer=not inference and d['phase']=='online',source=str(directory/'summary.json')))
    out.mkdir(parents=True,exist_ok=True)
    with (out/'table.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(rows,indent=2))
    md=['# Original Pogema lifelong execution and existing-IL comparison','',
        'All cases: 65x65 maps, A128, N16, RTX5080, CPU0–15, float32/TF32 off.',
        'Inference uses the frozen checkpoint; training uses the same full 2048-node CE/Adam update on both paths.',
        'Three fresh-process timing repetitions are not three independent learning seeds.','',
        '| Case | Window | Original median s | JointLoom median s | Paired median speedup | Range | Matched pairs |',
        '| --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for r in rows:
        md.append(f"| {r['case']} | {r['window']} | {r['original_median_seconds']:.3f} | {r['jointloom_median_seconds']:.3f} | {r['median_paired_speedup']:.3f}x | {r['speedup_min']:.3f}–{r['speedup_max']:.3f}x | {r['matched_pairs']}/{r['pair_count']} |")
    md.extend(['','IL consumption excludes expert generation and uses distinct real recorded stages.',
        'IL online includes live LaCAM/Pogema producers, real waiting/overlap, all updates and the final source successor.',
        'Source worker times overlap GPU work and each other; their sum is not a wall-clock decomposition.',
        'Native-extent observation adaptation is enabled and timed. Earlier unadapted profile-padding runs remain separate.',
        'These short training windows establish functionality and cost, not convergence or policy-quality improvement.'])
    for name in ['IL-consumption','IL-online']:
        md.extend(['',f'## {name} repetitions','',
            '| Path | Wall s | First loss | Last loss | H2D MiB | Source goals completed | Expert worker-time sum s |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for r in details[name]['runs']:
            s=r['source'];goals=s.get('completed_goals','n/a')
            md.append(f"| {r['path']} | {r['wall_seconds']:.3f} | {r['first_training_loss']:.6f} | {r['last_training_loss']:.6f} | {r['h2d_bytes']/2**20:.2f} | {goals} | {s.get('expert_seconds_sum',0.):.3f} |")
    (out/'RESULTS.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=input_path('JOINTLOOM_ONLINE_ARTIFACTS', ARTIFACTS/'online-il'))
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();main(args.root,args.out)
