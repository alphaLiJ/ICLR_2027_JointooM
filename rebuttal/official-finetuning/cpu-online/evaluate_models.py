"""Existing best/final checkpoints: official heldout tasks and shared 128-agent tasks."""
import official_bootstrap
from repository_paths import ARTIFACTS, input_path
import official_bootstrap as bootstrap
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from configuration import configuration


def atomic(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False));temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--desktop',type=Path,required=True);cli=parser.parse_args()
    from magat_plus.magat.agents import get_model
    from magat_plus.run_model import run_model_on_grid
    from magat_plus.train_imitation_learning_pyg import aux_func
    from grid_config_generator import grid_config_generator_factory
    torch.set_num_threads(1)
    out=cli.run/'evaluation';out.mkdir(exist_ok=False)
    best=cli.run/'checkpoints/best.pt'
    if not best.exists():best=cli.run/'checkpoints/best_low_val.pt'
    paths=[('cpu-online-validation-best',best),('cpu-online-final',cli.run/'checkpoints/last.pt'),
           ('official-pretrained',Path(configuration().pretrain_weights_path))]
    atomic(out/'selected-checkpoints.json',dict(checkpoints=[dict(name=n,path=str(p)) for n,p in paths],
        selection='public validation-selected best and fixed final; no test selection'))
    summaries=[];instances=[];started=time.perf_counter()
    for setting in ['official-heldout150','shared128-warehouse32']:
        args=configuration()
        if setting=='official-heldout150':
            seeds=np.random.default_rng(args.dataset_seed).integers(10**10,size=args.num_samples)
            ids=json.loads((cli.run/'input-corpus.json').read_text())['test_case_ids']
            seeds=[int(seeds[i]) for i in ids]
        else:
            args.num_agents='128+128'
            args.map_dir=str(input_path('JOINTLOOM_WAREHOUSE128', ARTIFACTS/'maps/warehouse32'))
            ids=list(range(128));seeds=list(range(2037000000,2037000128))
        factory=grid_config_generator_factory(args)
        for name,path in paths:
            torch.manual_seed(args.model_seed);np.random.seed(args.model_seed)
            model,kwargs=get_model(args,torch.device('cuda'))
            model.load_state_dict(torch.load(path,map_location='cuda',weights_only=False));model.eval()
            rows=[]
            for j,(case_id,seed) in enumerate(zip(ids,seeds)):
                ok,env,_=run_model_on_grid(model,torch.device('cuda'),factory(seed,map_id=case_id),args,kwargs,aux_func=aux_func)
                row=dict(setting=setting,checkpoint=name,case_id=int(case_id),seed=int(seed),
                    agents=env.num_agents,success=bool(ok),agent_arrival=float(np.mean(env.was_on_goal)),
                    soc=float(np.sum(aux_func.costs)) if ok else None,makespan=aux_func.makespan if ok else None)
                rows.append(row);instances.append(row)
                with (out/'instances.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                atomic(cli.desktop/'evaluation-progress.json',dict(status='running',setting=setting,checkpoint=name,
                    completed=j+1,total=len(ids),elapsed_seconds=time.perf_counter()-started))
            successes=[r for r in rows if r['success']]
            summaries.append(dict(setting=setting,checkpoint=name,tasks=len(rows),successes=len(successes),
                success_rate=len(successes)/len(rows),mean_agent_arrival=float(np.mean([v['agent_arrival'] for v in rows])),
                soc_mean=float(np.mean([v['soc'] for v in successes])) if successes else None,
                makespan_mean=float(np.mean([v['makespan'] for v in successes])) if successes else None))
            atomic(out/'summary.json',dict(status='running',summaries=summaries))
            print(summaries[-1],flush=True)
            del model
    value=dict(status='complete',summaries=summaries,elapsed_seconds=time.perf_counter()-started)
    atomic(out/'summary.json',value);atomic(cli.desktop/'independent-evaluation.json',value)
    for name,rows in [('summary.csv',summaries),('instances.csv',instances)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Official workflow with online CPU input construction: independent evaluation','',
        'CPU inputs are built online from existing raw expert states. Checkpoints follow the official validation selection rule; test tasks do not participate in training or selection. Both groups use official CPU runtime inputs, naive/probabilistic/T1, unclipped CTG and 512 steps.','',
        '| setting | model | successes/total | success rate | agent arrival rate |','|---|---|---:|---:|---:|']
    for r in summaries:lines.append(f'| {r["setting"]} | {r["checkpoint"]} | {r["successes"]}/{r["tasks"]} | {r["success_rate"]*100:.2f}% | {r["mean_agent_arrival"]*100:.3f}% |')
    lines+=['','shared128-warehouse32 uses the same task set2037000000-2037000127 as the previous full-adaptation models (119/128 and117/128) and query-sampled models. Online correction corpora can differ between training runs; checkpoint equality is not an assumption of this table.']
    report='\n'.join(lines)+'\n';(out/'RESULTS.md').write_text(report);(cli.desktop/'independent-evaluation.md').write_text(report)
    atomic(cli.desktop/'evaluation-progress.json',dict(status='complete',elapsed_seconds=value['elapsed_seconds']))


if __name__=='__main__':main()
