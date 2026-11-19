"""Few real state/action checks for the added lifelong task source and CPU builder."""
import bootstrap
from repository_paths import ARTIFACTS, input_path
import json
import numpy as np
import torch
from tasks import make_tasks
from envs import CpuLifelong,ResidentLifelong
from inputs import HostTransfer
from policies import MagatPolicy


def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tasks=make_tasks('mazes-s0_wc8_od55',8,2,20261108,128)
    # Two agents reach their own initial goal together; following steps must move again.
    tasks.goals[:,0:2]=tasks.positions[:,0:2]
    cpu=CpuLifelong(tasks);gpu=ResidentLifelong(tasks,'cuda:0',256)
    transfer=HostTransfer(2,8,'cuda:0')
    checkpoint=str(input_path('JOINTLOOM_DECISION_CHECKPOINT', ARTIFACTS/'decision-checkpoint.pt'))
    policy=MagatPolicy(checkpoint,'cuda:0')
    rng=np.random.default_rng(40);max_input=0.;max_logit=0.
    for step in range(24):
        if step:gpu.sim.update_derived_state()
        data=transfer.send(cpu.build())
        resident=policy.builder.build_from_stateful(gpu.sim)
        delta=float((data.x-resident.x).abs().max())
        max_input=max(max_input,delta)
        if delta>1e-6:
            idx=torch.nonzero((data.x-resident.x).abs()>1e-6)[:12].tolist()
            raise AssertionError(('inputs',step,delta,idx))
        assert torch.equal(data.edge_index,resident.edge_index),('edges',step)
        assert torch.equal(data.edge_attr,resident.edge_attr),('attrs',step)
        with torch.inference_mode():
            a=policy.model(data.x,data);b=policy.model(resident.x,resident)
        max_logit=max(max_logit,float((a-b).abs().max()))
        assert torch.equal(a.argmax(-1),b.argmax(-1)),('actions',step)
        actions=np.zeros((2,8),dtype=np.uint8) if step==0 else rng.integers(0,5,size=(2,8),dtype=np.uint8)
        cpu.step(actions)
        gpu.sim.update_actions(torch.as_tensor(actions,device='cuda:0'))
        gpu.sim.step_sim_only()
        result=gpu.result()
        for k,v in [('positions',cpu.positions()),('goals',cpu.goals),('cursors',cpu.cursors)]:
            assert np.array_equal(v,result[k]),(k,step,v,result[k])
    assert int(cpu.cursors.sum())>=4
    print(json.dumps(dict(steps=24,goals_completed=int(cpu.cursors.sum()),max_input_error=max_input,max_logit_error=max_logit,
        goal_and_transition_match=True,action_match=True)))


if __name__=='__main__':main()
