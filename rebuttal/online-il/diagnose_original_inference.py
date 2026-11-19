"""Locate the first long-rollout divergence; no performance timing or hashes."""
import bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from tasks import make_tasks
from envs import CpuPool,ResidentLifelong
from inputs import HostTransfer,combine_inputs
from policies import MagatPolicy
from native_extent_inputs import NativeExtentBatchBuilder


def main(out):
    c=yaml.safe_load((Path(__file__).parent/'configs/e2.yaml').read_text())
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tasks=make_tasks('random-e2-seed20261108',128,16,20261108)
    pool=CpuPool(tasks,8,c['cpu_affinity'],backend='original');env=ResidentLifelong(tasks,'cuda:0',4096)
    policy=MagatPolicy(c['checkpoint'],'cuda:0');policy.builder=NativeExtentBatchBuilder([tasks.grid.shape]*16,128,'cuda:0')
    transfer=HostTransfer(16,128,'cuda:0');result={}
    try:
        for con in pool.conns:con.send(('build',None))
        with torch.no_grad():
            for step in range(1024):
                arrays=combine_inputs([pool.receive(con) for con in pool.conns],256)
                env.sim.update_derived_state();a=transfer.send(arrays);b=policy.builder.build_from_stateful(env.sim)
                error=(a.x-b.x).abs();maximum=float(error.max().item())
                la=policy.model(a.x,a);lb=policy.model(b.x,b);ca=la.argmax(-1);cb=lb.argmax(-1)
                different=(ca!=cb).nonzero().flatten()
                if len(different) or maximum>1e-6:
                    result=dict(step=step,max_input_error=maximum,
                        max_error_by_channel=error.amax(dim=(0,2,3)).cpu().tolist(),
                        edges_equal=bool(torch.equal(a.edge_index,b.edge_index)),
                        attributes_equal=bool(torch.equal(a.edge_attr,b.edge_attr)),
                        differing_actions=len(different),max_logit_error=float((la-lb).abs().max().item()),agents=[])
                    for node in different[:8].cpu().tolist():
                        top=la[node].topk(2).values
                        result['agents'].append(dict(env=node//128,agent=node%128,
                            original_logits=la[node].cpu().tolist(),jointloom_logits=lb[node].cpu().tolist(),
                            original_top_two_gap=float((top[0]-top[1]).item()),
                            max_agent_input_error=float(error[node].max().item())))
                    break
                commands=ca.reshape(16,128).to(torch.uint8)
                host=commands.cpu().numpy()
                for i,con in enumerate(pool.conns):con.send(('step',(host[2*i:2*i+2],True)))
                env.sim.update_actions(commands);env.sim.step_sim_only()
                if (step+1)%128==0:print('compared steps',step+1,flush=True)
            else:result=dict(compared_steps=1024,no_divergence=True)
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
    finally:pool.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
