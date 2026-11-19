"""Small actual-goal-change check for the newly connected original path."""
import bootstrap
import numpy as np
import torch
from tasks import make_tasks
from pogema_path import OriginalPogemaLifelong
from envs import ResidentLifelong
from inputs import HostTransfer
from policies import MagatPolicy
from pathlib import Path
import yaml


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tasks=make_tasks('mazes-s0_wc8_od55',8,2,20261108)
    tasks.positions[0,0]=[3,64];tasks.positions[1,0]=[62,4]
    tasks.goals[:,:2]=tasks.positions[:,:2]
    cpu=OriginalPogemaLifelong(tasks);gpu=ResidentLifelong(tasks,'cuda:0',256)
    c=yaml.safe_load((Path(__file__).parent/'configs/e2.yaml').read_text())
    policy=MagatPolicy(c['checkpoint'],'cuda:0');transfer=HostTransfer(2,8,'cuda:0')
    from native_extent_inputs import NativeExtentBatchBuilder
    policy.builder=NativeExtentBatchBuilder([tasks.grid.shape]*2,8,'cuda:0')
    rng=np.random.default_rng(11);maximum=0.;logits_error=0.
    for step in range(24):
        arrays=cpu.build();gpu.sim.update_derived_state();data=policy.builder.build_from_stateful(gpu.sim)
        err=float(np.abs(arrays[0]-data.x.cpu().numpy()).max());maximum=max(maximum,err)
        assert err<1e-6,err
        assert np.array_equal(arrays[1],data.edge_index.cpu().numpy())
        assert np.array_equal(arrays[2],data.edge_attr.cpu().numpy())
        if step in (0,1,23):
            with torch.no_grad():
                a=policy.model(transfer.send(arrays).x,transfer.data);b=policy.model(data.x,data)
                logits_error=max(logits_error,float((a-b).abs().max().item()))
                assert torch.equal(a.argmax(-1),b.argmax(-1))
        actions=np.zeros((2,8),dtype=np.uint8) if step==0 else rng.integers(0,5,(2,8),dtype=np.uint8)
        cpu.step(actions);gpu.sim.update_actions(torch.as_tensor(actions,device='cuda:0'));gpu.sim.step_sim_only()
        state=gpu.result()
        for key,value in [('positions',cpu.positions()),('goals',cpu.goals),('cursors',cpu.cursors)]:
            assert np.array_equal(value,state[key]),(step,key)
    print(dict(check='PASS',steps=24,goal_updates=int(cpu.cursors.sum()),max_input_error=maximum,max_logit_error=logits_error))


if __name__=='__main__':main()
