"""Run the published trainer with interchangeable data storage/delivery adapters.

The public model/loss/Adam/schedule, validation and checkpoint selection are
reused. A common CPU expert pool overlaps ordered policy rollouts and experts.
"""
import official_bootstrap as bootstrap
import argparse
from dataclasses import dataclass
import inspect
import json
import multiprocessing as mp
import os
import resource
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGLoader

from configuration import configuration,official_arguments,official_parser
from data_store import Frame,PublishedGraphDataset,case_frames,save_case,ensure_occupancy_context


class PreviewComplete(Exception):pass


@dataclass
class ObservationReference:
    directory: str
    steps: int


def scalars(value):
    if isinstance(value,dict):return {k:scalars(v) for k,v in value.items()}
    if torch.is_tensor(value):return value.detach().cpu().item() if value.numel()==1 else value.detach().cpu().tolist()
    if isinstance(value,np.generic):return value.item()
    return value


class RunLog:
    def __init__(self,out,backend,preview=0):
        self.out,self.backend,self.preview=out,backend,preview
        self.started=int(os.environ.get('JOINTLOOM_RUN_START_NS',time.perf_counter_ns()))/1e9
        self.training_started=None
        self.updates=self.samples=self.graphs=0;self.last_progress=0.;self.model=None
        self.completed_epochs=0
        self.logged_epochs=set();self.phase='initializing_official_trainer'
        self.validation_counts={}
        self.wall_saved=set();self.last_update_elapsed=None
        self.args=None
        self.file=(out/'updates.jsonl').open('w',buffering=64*1024)
        self.state()

    def state(self,**extra):
        value=dict(status='running',backend=self.backend,phase=self.phase,pid=os.getpid(),
            optimizer_updates=self.updates,learner_samples=self.samples,learner_graphs=self.graphs,
            completed_epochs=self.completed_epochs,
            elapsed_seconds=time.perf_counter()-self.started,
            open_file_descriptors=len(os.listdir('/proc/self/fd')),
            file_descriptor_limit=resource.getrlimit(resource.RLIMIT_NOFILE)[0],
            cpu_affinity=sorted(os.sched_getaffinity(0)),**extra)
        temporary=self.out/'progress.tmp';temporary.write_text(json.dumps(value,indent=2));temporary.replace(self.out/'progress.json')
        return value

    def record_update(self,data,model):
        self.phase='supervised_learning';self.model=model
        self.updates+=1;self.samples+=int(data.x.shape[0]);self.graphs+=len(data.ptr)-1
        row=dict(update=self.updates,samples=self.samples,graphs=self.graphs,
            elapsed_seconds=time.perf_counter()-self.started)
        self.last_update_elapsed=row['elapsed_seconds']
        self.file.write(json.dumps(row)+'\n')
        if self.updates%64==0:self.file.flush()
        if time.perf_counter()-self.last_progress>15:
            self.last_progress=time.perf_counter();self.file.flush();self.state()
        if self.preview and self.updates>=self.preview:
            self.file.flush();torch.save(model.state_dict(),self.out/'preview-model.pt')
            raise PreviewComplete()
        self.wall_checkpoint(model)

    def wall_checkpoint(self,model):
        if self.preview:return
        elapsed=time.perf_counter()-self.started
        for budget in [1800,3600,7200,14400]:
            if elapsed<budget or budget in self.wall_saved:continue
            directory=self.out/'wall-checkpoints';directory.mkdir(exist_ok=True)
            torch.save(model.state_dict(),directory/f'budget-{budget}.pt')
            row=dict(nominal_budget_seconds=budget,saved_elapsed_seconds=elapsed,
                last_model_update_seconds=self.last_update_elapsed,updates=self.updates,samples=self.samples)
            with (self.out/'wall-checkpoints.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            self.wall_saved.add(budget)

    def epoch_result(self,parent):
        self.file.flush()
        epoch=parent.f_locals.get('epoch')
        if epoch is None or epoch in self.logged_epochs:return
        results=parent.f_locals.get('results')
        if not results or 'validation_success_rate' not in results:return
        self.logged_epochs.add(epoch)
        row=dict(epoch=int(epoch)+1,updates=self.updates,samples=self.samples,
            elapsed_seconds=time.perf_counter()-self.started,
            validation_evaluations=self.validation_counts.get(int(epoch)+1,0),**scalars(results))
        with (self.out/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        self.state(latest_epoch=row)


class CachedCPUInputs:
    def __init__(self,dataset,batch_size,log,persistent=True):
        self.log=log;self.first=True
        self.persistent=persistent
        self.loader=PyGLoader(dataset,batch_size=batch_size,shuffle=False,
            num_workers=8,prefetch_factor=2,persistent_workers=persistent,pin_memory=True)

    def __len__(self):return len(self.loader)

    def __iter__(self):
        # Match the published single-process loader's per-iterator seed draw
        # while retaining persistent CPU workers for the optimized reference.
        if self.persistent and not self.first:torch.empty((),dtype=torch.int64).random_().item()
        self.first=False
        stream=torch.cuda.Stream();iterator=iter(self.loader)
        def next_batch():
            try:host=next(iterator)
            except StopIteration:return None
            with torch.cuda.stream(stream):
                batch=host.to('cuda',non_blocking=True)
                event=torch.cuda.Event();event.record(stream)
            return host,batch,event
        try:
            future=next_batch()
            while future is not None:
                owner,batch,event=future
                torch.cuda.current_stream().wait_event(event)
                for value in batch.values():
                    if torch.is_tensor(value):value.record_stream(torch.cuda.current_stream())
                future=next_batch()
                yield batch
        finally:
            if not self.persistent:
                stream.synchronize()
                iterator._shutdown_workers()


class ResidentInputs:
    def __init__(self,dataset,batch_size,builder,log):
        self.dataset,self.batch_size,self.builder,self.log=dataset,batch_size,builder,log

    def __len__(self):return (len(self.dataset)+self.batch_size-1)//self.batch_size

    def __iter__(self):
        torch.empty((),dtype=torch.int64).random_().item()
        for i in range(0,len(self.dataset),self.batch_size):
            yield self.builder.build(self.dataset.frames[i:i+self.batch_size])


def dense_references(frames,with_positions=False):
    result=tuple(list(frames) for _ in range(4))+([f.case_id for f in frames],)
    return result+(list(frames),) if with_positions else result


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['cpu_cached','cpu_online','jointloom'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--preview-updates',type=int,default=0)
    p.add_argument('--preview-cases',type=int,default=8)
    p.add_argument('--expert-workers',type=int,default=4)
    p.add_argument('--smoke-online',action='store_true')
    cli=p.parse_args();cli.out.mkdir(parents=True,exist_ok=False)
    os.chdir(bootstrap.UPSTREAM);torch.set_num_threads(1)
    mp.set_start_method('fork',force=True)
    log=RunLog(cli.out,cli.backend,cli.preview_updates)
    import magat_plus.train_imitation_learning_pyg as official
    import magat_plus.run_model as run_module
    from magat_plus.run_expert import run_expert_algorithm as native_expert
    run_arguments=official_arguments(cli.out/'checkpoints')
    if cli.smoke_online:
        run_arguments+=['--num_epochs','2','--validation_every_epochs','1',
                        '--initial_val_size','2','--num_run_oe','8']
    args=official_parser().parse_args(run_arguments);log.args=args
    (cli.out/'official-configuration.json').write_text(json.dumps(vars(args),indent=2))
    if cli.preview_updates or cli.smoke_online:
        index=[json.loads((bootstrap.ART/'dataset/cases'/f'{i:04d}'/'result.json').read_text())
               for i in range(cli.preview_cases)]
    else:
        index=json.loads((bootstrap.ART/'dataset/index.json').read_text())
    frames=[f for case in index if case['success'] for f in case_frames(case['directory'])]
    if cli.backend!='cpu_online':
        for directory in {f.directory for f in frames}:ensure_occupancy_context(directory)
    valid_ids=sorted(c['case_id'] for c in index if c['success'])
    train_count=int(len(valid_ids)*(1-args.validation_fraction-args.test_fraction))
    validation_count=int(len(valid_ids)*args.validation_fraction)
    (cli.out/'input-corpus.json').write_text(json.dumps(dict(generated_cases=len(index),
        solved_cases=sum(c['success'] for c in index),joint_graphs=len(frames),
        agents_per_case=sorted({c['agents'] for c in index}),
        training_case_ids=valid_ids[:train_count],
        validation_case_ids=valid_ids[train_count:train_count+validation_count],
        test_case_ids=valid_ids[train_count+validation_count:]),indent=2))

    builder=None
    if cli.backend=='jointloom':
        from gpu_inputs import JointLoomInputs,RuntimeInputs
        builder=JointLoomInputs()
        run_module.get_runtime_data_generator=lambda **kw:RuntimeInputs(**kw,builder=builder)

    def load_dataset(funcs,dir_name,unused_args):
        if dir_name=='processed_dataset':return dense_references(frames)
        if dir_name=='positions':return list(frames)
        if dir_name=='additional_data':return [None]*len(frames)
        raise ValueError(dir_name)

    cpu_loader_calls=0
    def data_loader(dataset,batch_size):
        nonlocal cpu_loader_calls
        if log.training_started is None:log.training_started=time.perf_counter()
        if cli.backend=='jointloom':return ResidentInputs(dataset,batch_size,builder,log)
        cpu_loader_calls+=1
        # The first two loaders (base train/validation) are reused. The public
        # loop rebuilds each OE loader, so its workers must end with that iterator.
        if cli.backend=='cpu_online':
            dataset.phase='initial_train' if cpu_loader_calls==1 else ('validation_accuracy' if cpu_loader_calls==2 else 'correction_train')
        return CachedCPUInputs(dataset,batch_size,log,persistent=cpu_loader_calls<=2)

    def store_online_expert(expert,*unused,**kwargs):
        log.phase='online_expert_correction';log.state()
        actions,observations,terminated=native_expert(expert,**kwargs)
        if not all(terminated[-1]):return actions,None,terminated
        path=cli.out/'online-data'/f'{os.getpid()}-{time.monotonic_ns()}'
        log.phase='online_expert_input_preparation';log.state()
        save_case(path,os.getpid(),kwargs['grid_config'],actions,observations,terminated,args,
                  expanded=(cli.backend=='cpu_cached'))
        return actions,ObservationReference(str(path),len(actions)),terminated

    def new_graphs(dataset,**kwargs):
        refs=[]
        for obs,actions,terminated in dataset:refs.extend(case_frames(obs.directory))
        return dense_references(refs,with_positions=kwargs.get('use_edge_attr',False))

    original_loss=official.get_loss_function
    def logged_loss(unused_args):
        loss=original_loss(unused_args);accuracy=loss.get_accuracies
        def get_accuracies(out,data,model,split='train'):
            result=accuracy(out,data,model,split)
            if split=='train':log.record_update(data,model)
            return result
        loss.get_accuracies=get_accuracies
        return loss

    original_run=official.run_model_on_grid
    source=Path(official.__file__).read_text().splitlines()
    oe_start=next(i+1 for i,s in enumerate(source) if 'if args.run_online_expert and' in s)
    def logged_run(*a,**kw):
        parent=inspect.currentframe().f_back
        phase='online_expert_policy_rollout' if parent.f_lineno>oe_start else 'validation_rollout'
        log.phase=phase
        log.wall_checkpoint(kw['model'])
        if phase.startswith('online'):log.epoch_result(parent)
        log.state(epoch=int(parent.f_locals.get('epoch',-1))+1)
        success,env,observations=original_run(*a,**kw)
        current_epoch=int(parent.f_locals.get('epoch',-1))+1
        if phase=='validation_rollout':
            log.validation_counts[current_epoch]=log.validation_counts.get(current_epoch,0)+1
        row=dict(phase=phase,epoch=current_epoch,
            case_id=int(parent.f_locals.get('graph_id',-1)),
            success=bool(success),agent_arrival=float(np.mean(env.was_on_goal)),
            agents=int(env.num_agents),elapsed_seconds=time.perf_counter()-log.started)
        with (cli.out/'rollouts.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return success,env,observations

    official.load_dataset=load_dataset
    if cli.backend=='cpu_online':
        from cpu_online_inputs import OnlineGraphDataset
        OnlineGraphDataset.configure(args,cli.out/'input-builders')
        official.MAPFGraphDataset=OnlineGraphDataset
    else:
        official.MAPFGraphDataset=PublishedGraphDataset
    official.DataLoader=data_loader
    official.generate_additional_data=lambda **kw:[None]*len(kw['all_actions'])
    official.generate_graph_dataset=new_graphs
    official.run_expert_algorithm=store_online_expert
    official.get_loss_function=logged_loss
    official.run_model_on_grid=logged_run
    if cli.expert_workers > 1:
        from parallel_online_expert import install, run_round, rng_state
        def parallel_round(*a):
            log.epoch_result(inspect.currentframe().f_back)
            return run_round(*a, run_policy=original_run, log=log,
                workers=cli.expert_workers, expanded=cli.backend=='cpu_cached', out=cli.out)
        def save_epoch_state(context):
            log.completed_epochs=int(context['epoch'])+1
            log.file.flush()
            scheduler=context['lr_scheduler']
            payload=dict(completed_epochs=log.completed_epochs,
                model=context['model'].state_dict(),optimizer=context['optimizer'].state_dict(),
                scheduler=scheduler.scheduler.state_dict(),
                scheduler_wrapper={k:v for k,v in vars(scheduler).items() if k!='scheduler'},
                rng=rng_state(),args=vars(args),
                online_state={k:context[k] for k in ('oe_graph_dataset','oe_grid_configs',
                    'oe_additional_data','best_validation_success_rate','best_validation_accuracy',
                    'best_val_file_name','cur_validation_id_max')},
                log_state={k:getattr(log,k) for k in ('updates','samples','graphs','logged_epochs',
                    'validation_counts','wall_saved','last_update_elapsed')},
                elapsed_seconds=time.perf_counter()-log.started)
            temporary=cli.out/'training-state.tmp'
            torch.save(payload,temporary)
            temporary.replace(cli.out/'training-state.pt')
            log.state()
        install(official, parallel_round, save_epoch_state)
    # Trace only the official final checkpoint write to retain its final epoch
    # summary. The saved official state_dict itself is not changed.
    native_save=torch.save
    def save(obj,path,*a,**kw):
        result=native_save(obj,path,*a,**kw)
        if isinstance(path,(str,os.PathLike)) and Path(path).name=='last.pt':
            log.epoch_result(inspect.currentframe().f_back)
        return result
    torch.save=save
    sys.argv=[official.__file__]+run_arguments
    try:
        official.main()
    except PreviewComplete:
        result=log.state(status_note='preview_completed')
        result['status']='preview_complete'
    except BaseException:
        result=log.state(error=traceback.format_exc());result['status']='failed'
        (cli.out/'result.json').write_text(json.dumps(result,indent=2))
        (cli.out/'progress.json').write_text(json.dumps(result,indent=2))
        raise
    else:
        result=log.state();result['status']='complete'
    log.file.flush();log.file.close()
    result.update(official_epochs=args.num_epochs,backend=cli.backend,
        learning_protocol='public learning configuration; common ordered parallel expert schedule',
        input_materialization='CPU official runtime generators at sample access' if cli.backend=='cpu_online' else cli.backend,
        expert_workers=cli.expert_workers,
        torch_version=torch.__version__,cuda_version=torch.version.cuda)
    (cli.out/'result.json').write_text(json.dumps(result,indent=2))
    (cli.out/'progress.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
