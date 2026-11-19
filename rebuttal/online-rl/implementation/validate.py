"""Independent released-code, network, loss, and CPU/CUDA transition comparisons."""
import bootstrap
import ast
import json
import sysconfig
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from model import CostTracer,configure_torch
from tasks import make_tasks
from environments import CpuEnvironment,CudaEnvironment,encode_numpy
from ppo import gae,loss


def released_input_oracle():
    out=bootstrap.ARTIFACTS/'reference-oracle';out.mkdir(exist_ok=True)
    source=out/'oracle.cpp';lib=out/('released_input'+sysconfig.get_config_var('EXT_SUFFIX'))
    source.write_text('''#include <cmath>
#include <algorithm>
#include "environment.h"
PYBIND11_MODULE(released_input,m) {
 m.def("encode",[](std::vector<std::vector<int>> positions,
     std::vector<std::vector<std::vector<int16_t>>> distances,std::vector<int> best) {
   Environment env;env.agents.resize(positions.size());
   for(size_t i=0;i<positions.size();i++) {
     env.agents[i].cur_position={positions[i][0]+5,positions[i][1]+5};
     env.agents[i].goal={int(i),0};env.agents[i].furthest_reached=best[i];
     env.cost2go[env.agents[i].goal]=distances[i];
   }
   std::vector<std::vector<float>> out;
   for(size_t i=0;i<positions.size();i++) out.push_back(env.generate_input(i,5));
   return out;
 });
}
''')
    subprocess.run(['c++','-shared','-std=c++17','-O2','-fPIC',str(source),'-o',str(lib),
        '-I'+sysconfig.get_paths()['include'],'-I'+str(Path(torch.__file__).parent/'include'),
        '-I'+str(bootstrap.REFERENCE)],check=True,capture_output=True)
    import sys
    sys.path.insert(0,str(out));import released_input
    return released_input


def forward_test():
    import onnxruntime as ort
    opts=ort.SessionOptions();opts.intra_op_num_threads=1
    session=ort.InferenceSession(str(bootstrap.REFERENCE/'cost-tracer.onnx'),sess_options=opts,
                                providers=['CPUExecutionProvider'])
    model=CostTracer();model.load_released();model.eval();gpu=model.cuda()
    rng=np.random.default_rng(0);maximum=[0.,0.];cuda_max=[0.,0.]
    cpu=CostTracer();cpu.load_state_dict(model.state_dict());cpu.eval()
    for _ in range(16):
        obs=rng.uniform(-1,1,size=(1,2,11,11)).astype(np.float32)
        ref=session.run(['action_logits','values'],dict(obs=obs,rnn_state=np.zeros((1,1),np.float32)))
        with torch.no_grad():
            actual=cpu(torch.from_numpy(obs));dev=gpu(torch.from_numpy(obs).cuda())
        for k in range(2):
            np.testing.assert_allclose(actual[k].numpy(),ref[k],atol=5e-5,rtol=2e-6)
            np.testing.assert_allclose(dev[k].cpu().numpy(),ref[k],atol=3e-5,rtol=2e-6)
            maximum[k]=max(maximum[k],float(np.max(np.abs(ref[k]-actual[k].numpy()))))
            cuda_max[k]=max(cuda_max[k],float(np.max(np.abs(ref[k]-dev[k].cpu().numpy()))))
    return dict(parameters=sum(p.numel() for p in model.parameters()),samples=16,
                cpu_max_logits_value_error=maximum,cuda_max_logits_value_error=cuda_max)


def loss_test():
    tree=ast.parse((bootstrap.REFERENCE/'sample-factory-v1.121.4/sample_factory/algorithms/appo/learner.py').read_text())
    functions={}
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef) and node.name in ('_policy_loss','_value_loss'):
            node.decorator_list=[];scope={'torch':torch}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                         '<released Sample Factory loss>','exec'),scope)
            functions[node.name]=scope[node.name]
    worst=0.
    for seed in range(8):
        torch.manual_seed(seed);n=257
        logits=torch.randn(n,5,dtype=torch.float64,requires_grad=True)
        values=torch.randn(n,dtype=torch.float64,requires_grad=True)
        actions=torch.randint(5,(n,));old=torch.randn(n,dtype=torch.float64)
        oldv=torch.randn(n,dtype=torch.float64);adv=torch.randn(n,dtype=torch.float64)
        target=torch.randn(n,dtype=torch.float64)
        ours,_=loss(logits,values,actions,old,oldv,adv,target)
        d=torch.distributions.Categorical(logits=logits)
        ratio=(d.log_prob(actions)-old).exp().clamp(.05,20.)
        norm=(adv-adv.mean())/adv.std().clamp_min(.001)
        valid=torch.ones(n,dtype=torch.bool)
        actor=functions['_policy_loss'](ratio,norm,1/1.2,1.2,valid)
        fake=SimpleNamespace(cfg=SimpleNamespace(value_loss_coeff=.5))
        value=functions['_value_loss'](fake,values,oldv,target,1.,valid)
        reference=actor+value-.06878*d.entropy().mean()
        gradients=torch.autograd.grad(ours,(logits,values),retain_graph=True)
        refgrad=torch.autograd.grad(reference,(logits,values))
        worst=max(worst,abs(float(ours.detach()-reference.detach())),
                  *(float((a-b).abs().max()) for a,b in zip(gradients,refgrad)))
    assert worst<1e-10,worst
    # Independent hand-computed time-limit bootstrap, with no cross-reset GAE.
    r=torch.tensor([[1.],[2.],[100.]])
    v=torch.tensor([[3.],[4.],[5.]])
    nv=torch.tensor([[4.],[7.],[6.]])
    reset=torch.tensor([[False],[True],[False]])
    actual,_=gae(r,v,nv,reset)
    expected=torch.tensor([[1+.96*4-3+.96*.95*(2+.96*7-4)],
                           [2+.96*7-4],[100+.96*6-5]])
    assert torch.allclose(actual,expected,atol=1e-5),(actual,expected)
    return dict(sample_factory_random_cases=8,max_loss_gradient_error=worst,gae_time_limit='passed')


def trajectory_test():
    oracle=released_input_oracle();rng=np.random.default_rng(42)
    tasks=make_tasks(envs=4,agents=128,bank=4,horizon=48)
    cpu=CpuEnvironment(tasks);gpu=CudaEnvironment(tasks)
    worst=0.;oracle_error=0.;hits=0;rewards=0.;resets=0
    def compare_obs(a,b):
        nonlocal worst
        worst=max(worst,float(np.max(np.abs(a-b))))
        assert worst<2e-6,worst
    compare_obs(cpu.observe(),gpu.obs.cpu().numpy())
    for step in range(144):
        actions=[]
        for p,d in zip(cpu.positions(),cpu.distance):
            costs=[]
            for dx,dy in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):
                x=p[:,0]+dx;y=p[:,1]+dy
                valid=(x>=0)&(x<d.shape[1])&(y>=0)&(y<d.shape[2])
                score=d[np.arange(len(p)),x.clip(0,d.shape[1]-1),y.clip(0,d.shape[2]-1)].copy()
                score[(score<0)|(~valid)]=100000;costs.append(score)
            a=np.argmin(costs,axis=0)
            # Some forced waits / detours exercise non-repeated progress rewards.
            random=rng.random(len(a))<.15;a[random]=rng.integers(5,size=random.sum())
            actions.append(a)
        actions=np.stack(actions)
        co,cr,ch,ct=cpu.step(actions)
        go,gr,gh,gt=gpu.step(torch.as_tensor(actions,device='cuda'))
        cs=cpu.state();gs=gpu.state()
        for key in cs:
            assert np.array_equal(cs[key],gs[key]),(step,key,np.argwhere(cs[key]!=gs[key])[:8])
        assert np.array_equal(ct,gt.cpu().numpy())
        assert np.array_equal(cr,gr.cpu().numpy())
        assert np.array_equal(ch,gh.cpu().numpy())
        compare_obs(co,go.cpu().numpy())
        if step%24==0:
            for e in range(tasks['envs']):
                dist=np.pad(cpu.distance[e],((0,0),(5,5),(5,5)),constant_values=-1)
                ref=np.asarray(oracle.encode(cs['positions'][e].tolist(),dist.tolist(),cs['best'][e].tolist()),np.float32)
                error=float(np.max(np.abs(ref.reshape(-1,2,11,11)-co[e])))
                oracle_error=max(oracle_error,error)
                assert error<2e-6,error
        hits+=int(ch.sum());rewards+=float(cr.sum())
        if ct.any():
            co=cpu.reset_due();go=gpu.reset_due();resets+=len(ct)
            compare_obs(co,go.cpu().numpy())
            for key,a in cpu.state().items():assert np.array_equal(a,gpu.state()[key]),('reset',step,key)
    assert hits>0 and resets>0 and rewards>0,(hits,resets,rewards)
    # Long corridor exceeds MAGAT uint8 distance capacity: the CostTracer cache must stay exact.
    serp=np.ones((128,128),np.int32)
    for x in range(0,127,2):
        serp[x,:]=0
        if x<126:serp[x+1,127 if (x//2)%2==0 else 0]=0
    from environments import Topology
    p=np.array([[[0,0]]],np.int16);g=np.array([[[126,0]]],np.int16)
    special=dict(grids=[serp],positions=np.stack([p,p]),goals=np.stack([g,g]),
        streams=np.broadcast_to(g[None,:,:,None,:],(2,1,1,5,2)).copy(),names=['serpentine exact-distance check'],
        horizon=4,seed=0,agents=1,envs=1)
    resident=CudaEnvironment(special)
    exact=Topology(serp).distances(g[0])
    native=resident.builder.cost_to_go.cpu().numpy().astype(np.int32)
    native[native==-1]=-1
    assert np.array_equal(exact[0],native[0,0])
    assert exact[0,0,0]>255
    return dict(environments=4,agents=128,steps=144,agent_transitions=4*128*144,
        goal_reassignments=hits,environment_resets=resets,progress_reward_sum=rewards,
        max_cpu_cuda_observation_error=worst,max_released_cpp_observation_error=oracle_error,
        long_corridor_distance=int(exact[0,0,0]),diagnostics=gpu.builder.diagnostics.cpu().tolist())


def lifecycle_test():
    grid=np.ones((7,12),np.int32);grid[3,1:11]=0
    positions=np.array([[[3,1]],[[3,2]]],np.int16)
    goals=np.array([[[3,9]],[[3,9]]],np.int16)
    tasks=dict(grids=[grid,grid],positions=np.stack([positions]*3),goals=np.stack([goals]*3),
        streams=np.broadcast_to(goals[None,:,:,None,:],(3,2,1,5,2)).copy(),
        names=['lifecycle','lifecycle'],horizon=4,seed=0,agents=1,envs=2)
    cpu=CpuEnvironment(tasks);gpu=CudaEnvironment(tasks)
    # Environment 1 is older; its time reset must leave environment 0 untouched.
    cpu.steps[1]=2;gpu.sim.step_counts.copy_(torch.tensor([0,2],dtype=torch.int32,device='cuda'))
    expected=[.1,0.,0.,0.];observed=[];reset_masks=[]
    for action in (4,3,4,0):
        commands=np.full((2,1),action,np.int64)
        co,cr,ch,ct=cpu.step(commands);go,gr,gh,gt=gpu.step(torch.as_tensor(commands,device='cuda'))
        observed.append(float(cr[0,0]));reset_masks.append(ct.tolist())
        np.testing.assert_allclose(cr,gr.cpu().numpy(),atol=0,rtol=0)
        np.testing.assert_allclose(co,go.cpu().numpy(),atol=1e-6,rtol=0)
        if ct.any():
            co=cpu.reset_due();go=gpu.reset_due()
            np.testing.assert_allclose(co,go.cpu().numpy(),atol=1e-6,rtol=0)
        for key,a in cpu.state().items():assert np.array_equal(a,gpu.state()[key]),key
    np.testing.assert_allclose(observed,expected,atol=1e-8)
    assert reset_masks==[[False,False],[False,True],[False,False],[True,False]],reset_masks
    # Completing an already-current target earns no artificial progress reward;
    # it still assigns the next target, including an allowed self-target.
    task=dict(tasks,grids=[grid],positions=tasks['positions'][:,:1].copy(),
        goals=tasks['positions'][:,:1].copy(),streams=tasks['streams'][:,:1].copy(),
        envs=1,names=['self target'])
    task['streams'][:]=task['positions'][0,0,0]
    cpu=CpuEnvironment(task);gpu=CudaEnvironment(task)
    for _ in range(2):
        co,cr,ch,ct=cpu.step(np.zeros((1,1),np.int64))
        go,gr,gh,gt=gpu.step(torch.zeros((1,1),dtype=torch.long,device='cuda'))
        assert cr.sum()==0 and float(gr.sum())==0 and ch.sum()==1 and float(gh.sum())==1
        np.testing.assert_allclose(co,go.cpu().numpy(),atol=1e-6,rtol=0)
        assert cpu.best[0][0]==0 and int(gpu.best[0,0])==0
    return dict(reward_forward_back_repeat_wait=observed,asynchronous_reset_masks=reset_masks,
                repeated_self_target='passed')


def main():
    configure_torch(0)
    results={}
    for name,fn in [('network',forward_test),('ppo',loss_test),('trajectory',trajectory_test),('lifecycle',lifecycle_test)]:
        results[name]=fn();print(name,json.dumps(results[name]),flush=True)
    path=bootstrap.ARTIFACTS/'validation.json';path.write_text(json.dumps(results,indent=2))
    print('saved',path,flush=True)

if __name__=='__main__':main()
