"""Locate first input/action/state divergence; never a throughput measurement."""
import bootstrap
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from aligned_tasks import load_paper_tasks,AlignedResident,PAPER_CHECKPOINT
from envs import CpuPool
from inputs import HostTransfer,combine_inputs
from native_extent_inputs import NativeExtentBatchBuilder
from policies import MagatPolicy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--envs',type=int,default=256)
    p.add_argument('--steps',type=int,default=258);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    os.sched_setaffinity(0,set(range(16)));torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tasks=load_paper_tasks(args.envs);gpu=AlignedResident(tasks,'standard_mapf')
    policy=MagatPolicy(str(PAPER_CHECKPOINT),'cuda:0')
    policy.builder=NativeExtentBatchBuilder([g.shape for g in tasks.grids],256,'cuda:0')
    transfer=HostTransfer(args.envs,256,'cuda:0')
    pool=CpuPool(tasks,8,list(range(16)),backend='original',mode='standard_mapf',prepare=False)
    began=time.perf_counter();rows=[];maximum=0.
    try:
        for step in range(args.steps):
            for c in pool.conns:c.send(('build',None))
            arrays=combine_inputs([pool.receive(c) for c in pool.conns],pool.env_batch*256)
            gpu.sim.update_derived_state();resident=policy.builder.build_from_stateful(gpu.sim)
            gx=resident.x.cpu().numpy();difference=np.abs(arrays[0]-gx)
            err=float(difference.max());maximum=max(maximum,err)
            edges_equal=np.array_equal(arrays[1],resident.edge_index.cpu().numpy())
            attrs_equal=np.array_equal(arrays[2],resident.edge_attr.cpu().numpy())
            original=transfer.send(arrays)
            with torch.no_grad():
                a=policy.model(original.x,original);b=policy.model(resident.x,resident)
            la=a.cpu().numpy();lb=b.cpu().numpy();aa=la.argmax(-1);ab=lb.argmax(-1)
            different=np.flatnonzero(aa!=ab)
            current=dict(step=step,max_input_error=err,max_logit_error=float(np.abs(la-lb).max()),
                edges_equal=edges_equal,attrs_equal=attrs_equal,action_differences=len(different))
            rows.append(current)
            if err>=1e-6 or len(different) or not edges_equal or not attrs_equal:
                nodes=np.flatnonzero(difference.reshape(len(aa),-1).max(1)>=1e-6)
                selected=np.unique(np.r_[different,nodes])[:16]
                details=[]
                positions=torch.stack((gpu.sim.cur_x,gpu.sim.cur_y),-1).cpu().numpy()
                goals=torch.stack((gpu.sim.goal_x,gpu.sim.goal_y),-1).cpu().numpy()
                for node in selected:
                    e,agent=divmod(int(node),256)
                    details.append(dict(env=e,agent=agent,position=positions[e,agent].tolist(),goal=goals[e,agent].tolist(),
                        cpu_logits=la[node].tolist(),gpu_logits=lb[node].tolist(),
                        max_input_error=float(difference[node].max()),cpu_x=arrays[0][node].tolist(),gpu_x=gx[node].tolist()))
                result=dict(first_divergence=current,details=details,history=rows,elapsed_seconds=time.perf_counter()-began,
                    scope='same states before the first divergence; no different actions applied; diagnostic only')
                (args.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(current),flush=True);return
            actions=aa.astype(np.uint8).reshape(args.envs,256)
            for e,c in enumerate(pool.conns):c.send(('step',(actions[e*pool.env_batch:(e+1)*pool.env_batch],False)))
            gpu.sim.update_actions(torch.as_tensor(actions,device='cuda:0'));gpu.sim.step_sim_only()
            for c in pool.conns:pool.receive(c)
            if step%16==0:print(json.dumps(current),flush=True)
        (args.out/'result.json').write_text(json.dumps(dict(history=rows,max_input_error=maximum,
            first_divergence=None,elapsed_seconds=time.perf_counter()-began),indent=2))
    finally:pool.close()


if __name__=='__main__':main()
