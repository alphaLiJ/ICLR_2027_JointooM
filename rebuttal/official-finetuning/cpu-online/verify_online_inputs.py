"""Numerical comparison only; expanded references never feed the training run."""
import official_bootstrap
from repository_paths import ARTIFACTS, input_path
import official_bootstrap as bootstrap
import json
from pathlib import Path
import time
import torch
from configuration import configuration
from cpu_online_inputs import OnlineGraphDataset
from data_store import Frame,case_frames,cpu_frame
from train_official import dense_references


ART=input_path('JOINTLOOM_MAGAT_CPU_ARTIFACTS', ARTIFACTS/'magat-cpu-online')


def main():
    torch.set_num_threads(1);ART.mkdir(parents=True,exist_ok=True)
    args=configuration()
    from magat_plus.magat.agents import get_model
    model,kwargs=get_model(args,torch.device('cpu'));del model
    frames=[];references=[]
    for i in range(16):
        available=case_frames(bootstrap.ART/'dataset/cases'/f'{i:04d}')
        for step in sorted({0,len(available)//2,len(available)-1}):
            frames.append(available[step]);references.append(available[step])
    compact=bootstrap.ART/'online-correction-probe/compact'
    expanded=bootstrap.ART/'online-correction-probe/cpu'
    if (compact/'result.json').exists() and (expanded/'result.json').exists():
        available=case_frames(compact)
        for step in sorted({0,len(available)//2,len(available)-1}):
            frames.append(available[step]);references.append(Frame(str(expanded),step,available[step].case_id))
    OnlineGraphDataset.configure(args)
    dataset=OnlineGraphDataset(dense_references(frames),**kwargs)
    errors={k:0. for k in ('x','edge_attr','edge_weight')}
    exact={k:True for k in ('edge_index','y','terminated')}
    started=time.perf_counter();agents=0
    for i,frame in enumerate(references):
        actual=dataset[i]
        expected=cpu_frame(frame)
        agents+=len(actual.y)
        for key in errors:
            a,b=getattr(actual,key),getattr(expected,key)
            error=float((a-b).abs().max()) if a.numel() else 0.
            errors[key]=max(errors[key],error)
        for key in exact:exact[key]=exact[key] and torch.equal(getattr(actual,key),getattr(expected,key))
    result=dict(status='passed' if max(errors.values())<=1e-6 and all(exact.values()) else 'failed',
        frames=len(frames),agents=agents,maximum_errors=errors,exact_fields=exact,
        cpu_build_seconds=dataset.build_seconds,elapsed_seconds=time.perf_counter()-started,
        mode='independent check process only; formal training opens raw sources and computes inputs online')
    (ART/'input-verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if result['status']!='passed':raise RuntimeError('Online CPU conversion differs from official data')


if __name__=='__main__':main()
