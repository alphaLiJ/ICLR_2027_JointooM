"""Two real original maps, arrivals and graph/cache fidelity in both modes."""
import bootstrap
import json
import os
from pathlib import Path
import numpy as np
import torch
from aligned_tasks import load_paper_tasks, AlignedResident, PAPER_CHECKPOINT
from pogema_path import OriginalPogemaLifelong
from inputs import HostTransfer
from native_extent_inputs import NativeExtentBatchBuilder
from policies import MagatPolicy


def main():
    os.sched_setaffinity(0,set(range(16)))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    tasks=load_paper_tasks(2)
    tasks.goals[:,:4]=tasks.positions[:,:4]
    tasks.initial_arrived[:,:4]=1
    policy=MagatPolicy(str(PAPER_CHECKPOINT),'cuda:0')
    policy.builder=NativeExtentBatchBuilder([g.shape for g in tasks.grids],256,'cuda:0')
    transfer=HostTransfer(2,256,'cuda:0')
    rows=[]
    for mode in ['standard_mapf','lifelong']:
        cpu=OriginalPogemaLifelong(tasks,prepare=False,mode=mode)
        gpu=AlignedResident(tasks,mode)
        maximum=0.;logit_error=0.;argmax_diff=0
        rng=np.random.default_rng(17)
        for step in range(12):
            arrays=cpu.build();gpu.sim.update_derived_state()
            data=policy.builder.build_from_stateful(gpu.sim)
            error=float(np.abs(arrays[0]-data.x.cpu().numpy()).max());maximum=max(maximum,error)
            if error>=1e-6:raise AssertionError((mode,step,'nodes',error))
            if not np.array_equal(arrays[1],data.edge_index.cpu().numpy()):raise AssertionError('edges')
            if not np.array_equal(arrays[2],data.edge_attr.cpu().numpy()):raise AssertionError('attrs')
            if step in [0,1,11]:
                a=transfer.send(arrays)
                with torch.no_grad():
                    x=policy.model(a.x,a);y=policy.model(data.x,data)
                logit_error=max(logit_error,float((x-y).abs().max()))
                argmax_diff+=int((x.argmax(-1)!=y.argmax(-1)).sum())
            actions=np.zeros((2,256),dtype=np.uint8) if step==0 else rng.integers(0,5,(2,256),dtype=np.uint8)
            cpu.step(actions);gpu.sim.update_actions(torch.as_tensor(actions,device='cuda:0'));gpu.sim.step_sim_only()
            expected=cpu.result();actual=gpu.result()
            for key in ['positions','goals','cursors','arrived']:
                if not np.array_equal(expected[key],actual[key]):raise AssertionError((mode,step,key))
        rows.append(dict(mode=mode,steps=12,max_input_error=maximum,max_logit_error=logit_error,
            argmax_differences=argmax_diff,goal_events=int(cpu.cursors.sum()),state_equal=True))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
