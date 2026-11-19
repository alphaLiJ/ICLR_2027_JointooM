"""Warmed CUDA event attribution, with optional trace; not a speedup benchmark."""
import bootstrap
import argparse
import json
import time
from types import SimpleNamespace
import torch
import train
from model import CostTracer
from environments import CudaEnvironment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trace',action='store_true')
    parser.add_argument('--iterations',type=int,default=6)
    options=parser.parse_args()
    root=bootstrap.ARTIFACTS/('diagnostic-'+time.strftime('%H%M%S'))
    root.mkdir()
    active=False;updates=0;events=[]
    profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA])

    def timed(name,fn,*args,**kwargs):
        if not active:return fn(*args,**kwargs)
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        before=time.perf_counter();start.record()
        with torch.profiler.record_function(name):result=fn(*args,**kwargs)
        end.record();events.append((name,start,end,time.perf_counter()-before))
        return result

    original_forward=CostTracer.forward
    def forward(self,*args,**kwargs):
        return timed('train_forward' if torch.is_grad_enabled() else 'policy_forward',
                     original_forward,self,*args,**kwargs)
    CostTracer.forward=forward

    def observe(self):
        s=self.sim;b=self.builder
        timed('distance_refresh',b.refresh_distances,s.cur_x,s.cur_y,s.goal_x,s.goal_y)
        timed('costtracer_observation',self.ext.costtracer_observe,s,b,self.best,self.obs)
        return self.obs
    CudaEnvironment.observe=observe

    def step(self,actions):
        timed('action_copy',self.sim.update_actions,actions.to(torch.uint8).reshape(self.n,self.a))
        timed('simulator_step',self.sim.step_sim_only)
        timed('reward_goal_assignment',self.ext.costtracer_reward_goals,self.sim,self.builder,
              self.best,self.stream,self.cursor,self.reward,self.reached)
        self.observe()
        return self.obs,self.reward,self.reached,self.sim.step_counts>=self.horizon
    CudaEnvironment.step=step

    original_gae=train.gae
    train.gae=lambda *a,**kw:timed('GAE',original_gae,*a,**kw)
    original_update=train.update
    def update(*args,**kwargs):
        nonlocal updates,active
        result=timed('PPO_update_total',original_update,*args,**kwargs)
        updates+=1
        if updates==2:
            torch.cuda.synchronize()
            if options.trace:profiler.start()
            active=True
        elif updates==options.iterations:
            torch.cuda.synchronize();active=False
            if options.trace:profiler.stop()
        return result
    train.update=update
    train.run(SimpleNamespace(seed=0,output=str(root/'run'),backend='cuda',envs=16,agents=128,
              horizon=512,workers=8,rollout=32,iterations=options.iterations,minibatch=2048))
    torch.cuda.synchronize()
    totals={}
    for name,start,end,host in events:
        d=totals.setdefault(name,dict(calls=0,cuda_event_ms=0.,host_call_ms=0.))
        d['calls']+=1;d['cuda_event_ms']+=start.elapsed_time(end);d['host_call_ms']+=host*1000
    result=dict(trace=options.trace,measured_iterations=options.iterations-2,components=totals,
                mean_component_cuda_ms={k:v['cuda_event_ms']/(options.iterations-2) for k,v in totals.items()})
    (root/'component-events.json').write_text(json.dumps(result,indent=2))
    if options.trace:
        profiler.export_chrome_trace(str(root/'trace.json'))
        avg=profiler.key_averages()
        (root/'operators-cuda.txt').write_text(avg.table(sort_by='self_cuda_time_total',row_limit=35))
        (root/'operators-cpu.txt').write_text(avg.table(sort_by='self_cpu_time_total',row_limit=35))
    print('DIAGNOSTIC',root,flush=True)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
