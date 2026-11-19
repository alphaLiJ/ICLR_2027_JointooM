"""One fixed held-out lifelong evaluation; no PPO, tuning, or MCTS."""
import bootstrap
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from model import CostTracer,configure_torch
from environments import CudaEnvironment
from tasks import make_tasks
from sf_tasks import SF_ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root;out=root/'evaluation';out.mkdir(exist_ok=False)
    configure_torch(43000);torch.backends.cudnn.benchmark=True
    maps=['test-mazes-s40_wc4_od30','test-mazes-s41_wc5_od50']
    tasks=make_tasks(seed=43000,envs=16,agents=128,bank=1,horizon=512,map_ids=maps)
    np.savez(out/'fixed-evaluation-tasks.npz',positions=tasks['positions'],goals=tasks['goals'],
        streams=tasks['streams'],grids=np.stack(tasks['grids']))
    cpu_dir=SF_ROOT/'runs'/(root.name+'-formal-sf-async-s0')/'checkpoint_p0'
    cpu=max(cpu_dir.glob('checkpoint_*.pth'),key=lambda p:p.stat().st_mtime)
    paths={'initial':SF_ROOT/'initial-seed-0.pt','sf-async':cpu,'jointloom':root/'training/formal-jointloom-s0/final.pt'}
    results=[]
    for name,path in paths.items():
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        state=checkpoint['model'] if 'model' in checkpoint else checkpoint
        model=CostTracer().cuda();model.load_state_dict(state,strict=True);model.eval()
        env=CudaEnvironment(tasks);rng=torch.Generator(device='cuda').manual_seed(91000)
        rewards=torch.zeros((16,128),device='cuda');goals=torch.zeros_like(rewards)
        torch.cuda.synchronize();begin=time.perf_counter()
        with torch.no_grad():
            for step in range(512):
                logits,_=model(env.obs.reshape(2048,2,11,11))
                act=torch.multinomial(logits.softmax(-1),1,generator=rng).reshape(16,128)
                _,r,hit,_=env.step(act);rewards.add_(r);goals.add_(hit)
        torch.cuda.synchronize()
        rs=rewards.cpu().numpy();gs=goals.cpu().numpy()
        per_env=[dict(env_id=e,map_name=tasks['names'][e],mean_agent_episode_return=float(rs[e].mean()),
            completed_goals=float(gs[e].sum()),goals_per_joint_env_step=float(gs[e].sum()/512)) for e in range(16)]
        result=dict(model=name,checkpoint=str(path),selection='fixed initial/final, no best-checkpoint selection',
            agents_per_env=128,num_envs=16,steps=512,action_mode='categorical, same fixed evaluation RNG',
            mean_agent_episode_return=float(rs.mean()),completed_goals=float(gs.sum()),
            goals_per_joint_env_step=float(gs.sum()/(16*512)),per_env=per_env,
            evaluation_seconds=time.perf_counter()-begin)
        results.append(result);(out/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({k:result[k] for k in ('model','mean_agent_episode_return','completed_goals','goals_per_joint_env_step')}),flush=True)
        del model,env;torch.cuda.empty_cache()
    (out/'README.txt').write_text('Fixed two held-out maps, 16env x128agents, 512-step lifelong categorical execution.\n'
        'One training seed, 16 evaluation tasks. Same tasks/RNG and GPU evaluator for all three networks.\n'
        'No planner/MCTS; goals per environment step is lifelong task throughput, not all-agent success rate.\n'
        'Final checkpoints selected by fixed training budget. This does not reproduce full MATS-LP performance.\n')


if __name__=='__main__':main()
