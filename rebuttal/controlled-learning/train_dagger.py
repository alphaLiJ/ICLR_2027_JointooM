"""Sampled DAgger with matched online work and device-retained training inputs."""
import os
import sys
# CUDA reads this before its first context is created, including spawned workers.
if '--deterministic' in sys.argv:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import runtime
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import time
import json
import numpy as np
import torch
from torch_geometric.data import Batch
from collector import EnvironmentPool, Inputs, own_graph
from dagger_expert import initialize_worker, label_state


def background_replay(count,agents,out):
    if count<=0:return []
    from data_store import Frame,cpu_frame
    index=json.loads((runtime.legacy.ART/'dataset/index.json').read_text())
    candidates=[c for c in index if c['case_id']<700 and c['agents']==agents and c['success']]
    rng=np.random.default_rng(20261117)
    selected=rng.choice(len(candidates),size=min(len(candidates),(count+7)//8),replace=False)
    frames=[]
    for i in selected:
        case=candidates[i]
        steps=np.unique(np.linspace(0,case['steps']-1,8,dtype=int))
        frames.extend(Frame(case['directory'],int(t),case['case_id']) for t in steps)
    frames=frames[:count]
    runtime.json_write(out/'background-frames.json',[dict(directory=f.directory,step=f.step,case_id=f.case_id) for f in frames])
    # Common historical replay; includes no validation or test task.
    # Its initial GPU materialization is counted in process startup.
    batch=Batch.from_data_list([cpu_frame(f) for f in frames]).to('cuda')
    return batch.to_data_list()


def cpu_time(pids):
    total=0.
    for pid in pids:
        try:
            fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            total+=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')
        except FileNotFoundError:pass
    return total


def query_schedule(rng,steps,envs,queries,strategy,bins):
    count=min(queries,steps*envs)
    if strategy=='uniform':
        chosen=rng.choice(steps*envs,size=count,replace=False).tolist()
    else:
        if count%(envs*bins):
            raise ValueError('stratified queries must divide evenly across environments and time bins')
        per_cell=count//(envs*bins);chosen=[]
        for env in range(envs):
            for bucket in range(bins):
                lo=bucket*steps//bins;hi=(bucket+1)*steps//bins
                selected=rng.choice(np.arange(lo,hi),size=per_cell,replace=False)
                chosen.extend(int(step)*envs+env for step in selected)
    schedule={}
    for flat in sorted(chosen):schedule.setdefault(flat//envs,[]).append(flat%envs)
    return schedule


def replay_indices(rng,size,count,episode_bins,strategy,bins):
    if strategy=='uniform':return rng.choice(size,size=count,replace=False)
    selected=[];quota,remainder=divmod(count,bins)
    values=np.asarray(episode_bins,dtype=np.int8)
    for rank,bucket in enumerate(rng.permutation(bins)):
        candidates=np.flatnonzero(values==bucket)
        take=min(len(candidates),quota+int(rank<remainder))
        if take:selected.extend(rng.choice(candidates,size=take,replace=False).tolist())
    if len(selected)<count:
        remaining=np.setdiff1d(np.arange(size),selected,assume_unique=True)
        selected.extend(rng.choice(remaining,size=count-len(selected),replace=False).tolist())
    return rng.permutation(np.asarray(selected,dtype=np.int64))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['cpu','jointloom'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--agents',type=int,default=80)
    parser.add_argument('--envs',type=int,default=16)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--queries',type=int,default=64)
    parser.add_argument('--updates',type=int,default=32)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--expert-timeout',type=float,default=2.0)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--checkpoint',default=None)
    parser.add_argument('--freeze-batchnorm',action='store_true')
    parser.add_argument('--background-size',type=int,default=0)
    parser.add_argument('--background-agents',type=int,default=None)
    parser.add_argument('--online-per-batch',type=int,default=16)
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--independent-rngs',action='store_true')
    parser.add_argument('--expert-iterations',type=int,default=0)
    parser.add_argument('--save-round-states',action='store_true')
    parser.add_argument('--record-trajectory',action='store_true')
    parser.add_argument('--query-sampling',choices=['uniform','stratified'],default='uniform')
    parser.add_argument('--query-time-bins',type=int,default=8)
    parser.add_argument('--replay-sampling',choices=['uniform','episode-stratified'],default='uniform')
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False)
    if args.deterministic:
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
    runtime.json_write(out/'configuration.json',vars(args)|dict(out=str(out),
        protocol='sampled DAgger; student actions; joint expert labels; common GPU input retention',
        cpu_affinity=sorted(__import__('os').sched_getaffinity(0))))
    runtime.json_write(out/'numerical-settings.json',dict(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        default_dtype=str(torch.get_default_dtype())))
    startup=time.perf_counter()
    model,kwargs=runtime.load_model(args.checkpoint,args.seed)
    initial_parameters=[p.detach().cpu().clone() for p in model.parameters()]
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate,
                               weight_decay=runtime.configuration().weight_decay)
    background=background_replay(args.background_size,args.background_agents or args.agents,out)
    pool=EnvironmentPool(args.backend,args.agents,args.envs,args.workers,kwargs)
    cfg=runtime.task_config(333_000_000+args.seed,args.agents)
    grid=np.asarray(cfg.map,dtype=np.uint8)
    inputs=Inputs(args.backend,pool,grid)
    expert_pool=ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),
        initializer=initialize_worker,initargs=(4,))
    replay=[];replay_episode_bins=[];corpus=[];rounds=[];optimizer_updates=0;sampled_agents=0
    rng=np.random.default_rng(71_000+args.seed)
    query_rng=rng
    replay_rng=np.random.default_rng(81_000+args.seed) if args.independent_rngs else rng
    update_trace=[];trajectory=[]
    try:
        # Common setup on separate warmup tasks; actual task goal caches are cold.
        pool.reset([333_000_000+args.seed*1000+e for e in range(args.envs)])
        for _ in range(3):
            data=inputs.build()
            with torch.no_grad():logits=model(data.x,data)
            pool.step(pool.sample_actions(logits))
        warm=[dict(index=-e,grid=grid,rows=pool.arrays['rows'][e].copy(),timeout=args.expert_timeout,
                   iterations=args.expert_iterations)
              for e in range(min(4,args.envs))]
        if args.queries:list(expert_pool.map(label_state,warm))
        torch.cuda.synchronize();startup_seconds=time.perf_counter()-startup
        torch.cuda.reset_peak_memory_stats()
        pids=[os.getpid()]+[p.pid for p in pool.children]+list(expert_pool._processes)
        initial_cpu=cpu_time(pids)
        online_start=time.perf_counter();initial_bytes=inputs.h2d_bytes
        for round_id in range(args.rounds):
            start=time.perf_counter();stage=time.perf_counter()
            seeds=[900_000_000+args.seed*1_000_000+round_id*10_000+e for e in range(args.envs)]
            pool.reset(seeds)
            initialization_seconds=time.perf_counter()-stage
            schedule=query_schedule(query_rng,args.steps,args.envs,args.queries,args.query_sampling,args.query_time_bins)
            futures=[];snapshots=[]
            model.eval();collection_start=time.perf_counter()
            for step in range(args.steps):
                data=inputs.build()
                sample_indices={}
                for e in schedule.get(step,[]):
                    state=pool.arrays['rows'][e].copy()
                    index=len(snapshots)
                    sample_indices[e]=index
                    snapshots.append(dict(rows=state,graph=own_graph(data,e,args.agents),
                        round=round_id+1,step=step,env=e,executed_actions=None,
                        episode_step=int(pool.episode_steps[e]),
                        episode_bin=min(int(pool.episode_steps[e])*args.query_time_bins//cfg.max_episode_steps,args.query_time_bins-1),
                        time_bin=step*args.query_time_bins//args.steps,
                        arrived_agents=int(np.sum(pool.arrays['terminated'][e]))))
                    futures.append(expert_pool.submit(label_state,dict(index=index,grid=grid,rows=state,
                        timeout=args.expert_timeout,iterations=args.expert_iterations)))
                with torch.no_grad():logits=model(data.x,data)
                actions=pool.sample_actions(logits)
                if args.record_trajectory:trajectory.append(actions.astype(np.uint8))
                for e,index in sample_indices.items():
                    # Snapshot action is kept separate from the later expert label.
                    snapshots[index]['executed_actions']=actions[e].copy()
                pool.step(actions,full=step+1<args.steps)
            torch.cuda.synchronize();collection_seconds=time.perf_counter()-collection_start
            wait_start=time.perf_counter();results=[future.result() for future in futures]
            expert_drain_seconds=time.perf_counter()-wait_start
            expert_times=[];added=0
            disagreements=0;compared_actions=0
            for sample,result in zip(snapshots,results):
                expert_times.append(result['seconds'])
                record={k:v for k,v in sample.items() if k!='graph'}
                record.update(expert=result)
                corpus.append(record)
                if result['success']:
                    disagreements+=int(np.sum(sample['executed_actions']!=np.asarray(result['labels'])))
                    compared_actions+=args.agents
                    sample['graph'].y=torch.tensor(result['labels'],dtype=torch.long,device='cuda')
                    replay.append(sample['graph']);replay_episode_bins.append(sample['episode_bin']);added+=1
            torch.cuda.synchronize();training_start=time.perf_counter();losses=[]
            model.train()
            if args.freeze_batchnorm:
                # Preserve pretrained inference statistics with a small,
                # correlated online corpus; affine BN parameters still learn.
                for module in model.modules():
                    if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
            if replay:
                for update in range(args.updates):
                    count=min(args.online_per_batch if background else args.batch_size,len(replay))
                    ids=replay_indices(replay_rng,len(replay),count,replay_episode_bins,
                        args.replay_sampling,args.query_time_bins)
                    samples=[replay[i] for i in ids]
                    old_ids=[]
                    if background:
                        old_ids=replay_rng.choice(len(background),size=args.batch_size-count,replace=False)
                        samples.extend(background[i] for i in old_ids)
                    batch=Batch.from_data_list(samples)
                    dropout_seed=91_000+args.seed*10_000+optimizer_updates if args.independent_rngs else None
                    if dropout_seed is not None:torch.manual_seed(dropout_seed)
                    optimizer.zero_grad(set_to_none=True)
                    value=torch.nn.functional.cross_entropy(model(batch.x,batch),batch.y)
                    value.backward();optimizer.step()
                    losses.append(value.detach())
                    optimizer_updates+=1;sampled_agents+=len(batch.y)
                    update_trace.append(dict(round=round_id+1,update=optimizer_updates,
                        online_indices=list(map(int,ids)),background_indices=list(map(int,old_ids)),
                        dropout_seed=dropout_seed,online_episode_bins=[replay_episode_bins[i] for i in ids]))
            torch.cuda.synchronize();training_seconds=time.perf_counter()-training_start
            row=dict(round=round_id+1,backend=args.backend,agents=args.agents,envs=args.envs,
                joint_transitions=args.steps*args.envs,agent_transitions=args.steps*args.envs*args.agents,
                queries=len(results),labels=added,replay_graphs=len(replay),
                expert_student_action_disagreement=disagreements/compared_actions if compared_actions else None,
                optimizer_updates=optimizer_updates,learner_agent_samples=sampled_agents,
                initialization_seconds=initialization_seconds,collection_seconds=collection_seconds,
                expert_drain_seconds=expert_drain_seconds,expert_sum_seconds=sum(expert_times),
                expert_max_seconds=max(expert_times,default=0),training_seconds=training_seconds,
                wall_seconds=time.perf_counter()-start,loss_mean=float(torch.stack(losses).mean()) if losses else None,
                completed_episodes=len(pool.completed_episodes),
                successful_episodes=sum(r['success'] for r in pool.completed_episodes),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                queries_by_time_bin=[sum(s['time_bin']==b for s in snapshots) for b in range(args.query_time_bins)],
                labels_by_episode_bin=[sum(s['episode_bin']==b and r['success'] for s,r in zip(snapshots,results)) for b in range(args.query_time_bins)],
                queries_late_episode=sum(s['episode_step']>=256 for s in snapshots),
                queries_near_completion=sum(s['arrived_agents']>=args.agents*3/4 for s in snapshots),
                h2d_bytes=inputs.h2d_bytes-initial_bytes)
            rounds.append(row)
            for item,value in zip(update_trace[-len(losses):] if losses else [],losses):
                item['loss']=float(value)
            if args.save_round_states:
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),
                           out/f'round-{round_id+1:02d}.pt')
            with (out/'rounds.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            runtime.json_write(out/'progress.json',dict(status='running',latest=row,
                elapsed_seconds=time.perf_counter()-online_start))
            runtime.publish('dagger_training',status='running',run=out.name,backend=args.backend,
                completed_rounds=round_id+1,total_rounds=args.rounds,latest=row)
        online_seconds=time.perf_counter()-online_start
        runtime.json_write(out/'update-trace.json',update_trace)
        if args.record_trajectory:
            np.save(out/'executed-actions.npy',np.asarray(trajectory,dtype=np.uint8))
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            configuration=vars(args),rounds=rounds),out/'training-state.pt')
        torch.save(model.state_dict(),out/'last.pt')
        torch.save(corpus,out/'visited-labeled-states.pt')
        runtime.json_write(out/'coverage.json',dict(
            query_sampling=args.query_sampling,replay_sampling=args.replay_sampling,
            episode_horizon=cfg.max_episode_steps,collection_steps=args.steps,
            time_bin_width=args.steps/args.query_time_bins,
            queries_by_time_bin=[sum(s['time_bin']==b for s in corpus) for b in range(args.query_time_bins)],
            queries_by_episode_bin=[sum(s['episode_bin']==b for s in corpus) for b in range(args.query_time_bins)],
            labels_by_episode_bin=[sum(s['episode_bin']==b and s['expert']['success'] for s in corpus) for b in range(args.query_time_bins)],
            queries_late_episode=sum(s['episode_step']>=256 for s in corpus),
            queries_near_completion=sum(s['arrived_agents']>=args.agents*3/4 for s in corpus),
            learner_online_graphs_by_episode_bin=[sum(item['online_episode_bins'].count(b) for item in update_trace) for b in range(args.query_time_bins)]))
        retained=sum(v.numel()*v.element_size() for graph in replay for v in graph.values() if torch.is_tensor(v))
        result=dict(status='complete',backend=args.backend,rounds=rounds,
            online_seconds=online_seconds,startup_seconds=startup_seconds,
            joint_transitions=args.rounds*args.steps*args.envs,
            agent_transitions=args.rounds*args.steps*args.envs*args.agents,
            queries=len(corpus),labels=len(replay),optimizer_updates=optimizer_updates,
            learner_agent_samples=sampled_agents,retained_device_bytes=retained,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            background_graphs=len(background),
            h2d_bytes=inputs.h2d_bytes-initial_bytes,process_seconds=time.perf_counter()-startup,
            cpu_seconds=cpu_time(pids)-initial_cpu,
            mean_cpu_core_equivalents=(cpu_time(pids)-initial_cpu)/online_seconds,
            parameter_max_change=max(float((p.detach().cpu()-q).abs().max())
                                     for p,q in zip(model.parameters(),initial_parameters)))
        runtime.json_write(out/'result.json',result)
        runtime.json_write(out/'progress.json',result)
    finally:
        pool.close();expert_pool.shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':main()
