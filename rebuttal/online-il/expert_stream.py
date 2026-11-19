"""Finite, bounded original-Pogema/LaCAM producers for lifelong IL.

One worker owns persistent environment shards and their unchanged LaCAM
adapter. Compact and expanded online paths differ only in policy-input work
performed after the common expert label is available.
"""
import bootstrap
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback
import numpy as np
from pogema_path import OriginalPogemaLifelong
from inputs import combine_inputs


class LifelongExpert:
    def __init__(self,tasks,ids,config,expanded=False):
        from expert.expert_running import LacamExpertPolicy,configure_cpu_thread_limits
        configure_cpu_thread_limits()
        self.env=OriginalPogemaLifelong(tasks.subset(ids),prepare=False,with_inputs=expanded)
        self.ids=ids;self.expanded=expanded
        self.policies=[LacamExpertPolicy(lib_path=config['lacam_library'],timeouts=config['expert_timeouts']) for _ in ids]
        for p,e in zip(self.policies,self.env.envs):p.reset_states(e)
        self.previous_goals=None;self.expert_seconds=0.;self.environment_seconds=0.;self.builder_seconds=0.
        self.new_goal_calls=0;self.calls=0;self.nonwait_labels=0

    def packet(self,step):
        started=time.perf_counter()
        actions=np.asarray([p.act(obs) for p,obs in zip(self.policies,self.env.observations)],dtype=np.int16)
        self.expert_seconds+=time.perf_counter()-started;self.calls+=len(self.policies)
        changed=np.ones(self.env.goals.shape[:2],dtype=bool) if self.previous_goals is None else np.any(self.previous_goals!=self.env.goals,axis=-1)
        self.new_goal_calls+=int(np.any(changed,axis=1).sum());self.nonwait_labels+=int((actions!=0).sum())
        raw=self.env.rows(actions,self.ids);raw[:,7]=changed.reshape(-1)
        self.previous_goals=self.env.goals.copy()
        packet=dict(step=step,raw=raw,completed_before=int(self.env.cursors.sum()))
        if self.expanded:
            started=time.perf_counter();packet['arrays']=self.env.build()
            self.builder_seconds+=time.perf_counter()-started
        return packet,actions

    def advance(self,actions):
        started=time.perf_counter();self.env.step(actions)
        self.environment_seconds+=time.perf_counter()-started

    def result(self):
        result=self.env.result()
        result.update(expert_seconds=self.expert_seconds,environment_seconds=self.environment_seconds,
            input_builder_seconds=self.builder_seconds,expert_calls=self.calls,
            calls_with_new_goals=self.new_goal_calls,nonwait_labels=self.nonwait_labels)
        return result


def _producer(output,control,start,tasks,ids,config,expanded,steps,trace_file):
    try:
        os.sched_setaffinity(0,set(config['cpu_affinity']))
        if trace_file:
            env=OriginalPogemaLifelong(tasks.subset(ids),prepare=False)
            trace=np.load(trace_file,mmap_mode='r').reshape(-1,len(tasks.positions),tasks.positions.shape[1],8)
        else:expert=LifelongExpert(tasks,ids,config,expanded)
        control.send(dict(ready=True));start.wait();put_seconds=0.;builder_seconds=0.
        for step in range(steps):
            if trace_file:
                raw=np.ascontiguousarray(trace[step,ids].reshape(-1,8))
                began=time.perf_counter();env.load_rows(raw);arrays=env.build();builder_seconds+=time.perf_counter()-began
                packet=dict(step=step,raw=raw,arrays=arrays)
            else:packet,actions=expert.packet(step)
            began=time.perf_counter();output.put(packet);put_seconds+=time.perf_counter()-began
            if not trace_file:expert.advance(actions)
        result=env.result() if trace_file else expert.result()
        result['queue_put_seconds']=put_seconds
        if trace_file:result['restore_and_builder_seconds']=builder_seconds
        control.send(dict(result=result))
        output.close();output.join_thread()
    except BaseException:
        control.send(dict(error=traceback.format_exc()))
    finally:control.close()


class ExpertPool:
    def __init__(self,tasks,config,expanded,steps,trace_file=None):
        n=len(tasks.positions);workers=config['workers']
        if n%workers:raise ValueError('envs must divide worker count')
        self.group_envs=n//workers;self.agents=tasks.positions.shape[1]
        self.outputs=[];self.conns=[];self.processes=[]
        ctx=mp.get_context('spawn');self.start_event=ctx.Event()
        for ids in np.split(np.arange(n),workers):
            q=ctx.Queue(maxsize=config['prefetch_depth']);parent,child=ctx.Pipe()
            p=ctx.Process(target=_producer,args=(q,child,self.start_event,tasks,ids,config,expanded,steps,trace_file))
            p.start();child.close();self.outputs.append(q);self.conns.append(parent);self.processes.append(p)
        try:
            for c in self.conns:self._control(c)
        except BaseException:
            self.close();raise

    @staticmethod
    def _control(c):
        if not c.poll(180):raise TimeoutError('expert worker did not answer')
        message=c.recv()
        if 'error' in message:raise RuntimeError(message['error'])
        return message

    def start(self):self.start_event.set()

    def get(self,step):
        packets=[]
        for q,c in zip(self.outputs,self.conns):
            try:p=q.get(timeout=180)
            except queue.Empty:
                if c.poll():self._control(c)
                raise TimeoutError('expert/input producer timed out')
            if p['step']!=step:raise RuntimeError('unexpected source order')
            packets.append(p)
        result=dict(raw=np.concatenate([p['raw'] for p in packets]))
        if 'arrays' in packets[0]:
            result['arrays']=combine_inputs([p['arrays'] for p in packets],self.group_envs*self.agents)
        return result

    def finish(self):return [self._control(c)['result'] for c in self.conns]

    def close(self):
        for p in self.processes:
            p.join(2)
            if p.is_alive():p.terminate();p.join()
        for q in self.outputs:q.cancel_join_thread();q.close()
        for c in self.conns:c.close()


def save_worker_reports(out,reports):
    import json
    arrays={k:np.concatenate([r[k] for r in reports]) for k in ['positions','goals','cursors']}
    np.savez_compressed(out/'source-final-state.npz',**arrays)
    def convert(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    (out/'source-workers.json').write_text(json.dumps(reports,indent=2,default=convert))
    return dict(completed_goals=int(arrays['cursors'].sum()),
        expert_seconds_sum=sum(r.get('expert_seconds',0.) for r in reports),
        environment_seconds_sum=sum(r.get('environment_seconds',0.) for r in reports),
        source_input_seconds_sum=sum(r.get('input_builder_seconds',r.get('restore_and_builder_seconds',0.)) for r in reports),
        nonwait_labels=sum(r.get('nonwait_labels',0) for r in reports),
        expert_calls=sum(r.get('expert_calls',0) for r in reports),
        calls_with_new_goals=sum(r.get('calls_with_new_goals',0) for r in reports))
