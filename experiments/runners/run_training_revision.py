"""JointLoom R1/R2: reuse the existing ring, GPU builders and model runtimes.

record/gate/fixed-work/train are separate phases. No historic runner defaults
are changed. Run from the repository root with PYTHONPATH=src:.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np

from mapf_cuda.training.frozen_stage_source import load_records, save_records, sha256_file, RingStageSource
from mapf_cuda.training.host_inputs import CachedHostBuilder, HostPrefetch, ResourceMonitor, memory_estimate

MODES = ("cpu_cached_sync", "cpu_cached_prefetch", "compact_stream")
MAPS = {
    "magat": ["mazes-s0_wc8_od55", "mazes-s1_wc3_od70", "mazes-s2_wc8_od25", "mazes-s3_wc6_od35"],
    "mapf_gpt": ["mazes-s0_wc8_od55", "mazes-s1_wc3_od70", "mazes-s27_wc7_od60", "mazes-s28_wc2_od35"],
}


def make_config(backend, **overrides):
    if backend not in MAPS:
        raise ValueError("unsupported backend")
    magat = backend == "magat"
    c = dict(backend=backend, device="cuda:0", seed=0, num_agents=512 if magat else 256,
             map_names=MAPS[backend], maps_path="maps/maps.yaml", num_steps=100000 if magat else 10000,
             max_episode_steps=256, ring_depth=128 if magat else 4, host_workers=1, host_depth=2,
             checkpoint_interval_s=300., validation_steps=[250,500,1000,2000], archive_steps=[1000,2000],
             wall_budget_s=3035.921933, validation=[], lr_start=1e-3, lr_end=1e-6,
             lr_scheduler="cosine-annealing" if magat else None, scheduler_total_steps=100000,
             train_on_arrived_agents=True, model_size="2M", microbatch_size=256,
             shuffle_capacity=8192, timeout_s=120., capture_stage_hashes=False,
             validation_batch_size=256, top_k_checkpoints=3, grad_clip_norm=None, progress_interval_s=30.)
    unknown=set(overrides)-set(c)
    if unknown: raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    c.update(overrides)
    c["train_batch_size"] = c["num_agents"] * len(c["map_names"])
    if not c["map_names"] or len(set(c["map_names"]))!=len(c["map_names"]):
        raise ValueError("map_names must be nonempty and unique")
    if c["num_steps"] < 1 or c["num_agents"] < 1 or c["ring_depth"] < 1:
        raise ValueError("steps, agents and ring depth must be positive")
    if c["wall_budget_s"] is not None and c["wall_budget_s"] <= 0:
        raise ValueError("wall budget must be positive")
    if c["checkpoint_interval_s"] <= 0 or c["timeout_s"] <= 0:
        raise ValueError("invalid timing interval")
    if not 1 <= c["host_workers"] <= c["host_depth"]:
        raise ValueError("require 1 <= host_workers <= host_depth")
    return c


def write_json(path, data):
    path=Path(path)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n")
    os.replace(temporary,path)


def provenance(config):
    import torch
    root = Path(__file__).resolve().parents[2]
    files = [p for base in (root/"src", root/"experiments", root/"third_party")
             for p in base.rglob("*") if p.is_file() and p.suffix in (".py", ".cu", ".h", ".cpp", ".so")]
    return dict(git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
                source_manifest=[dict(path=str(p.relative_to(root)), sha256=sha256_file(p)) for p in sorted(files)],
                python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(config["device"]),
                maps_sha256=sha256_file(config["maps_path"]),
                validation=[dict(path=str(Path(p).resolve()),sha256=sha256_file(p)) for p in config["validation"]])


def selected_grids(c, records=None):
    from expert.training_baseline_engines import _selected_maps
    _, grids = _selected_maps(c["maps_path"], c["map_names"])
    if records:
        m, frozen_grids, stages = load_records(records)
        if m["backend"] != c["backend"] or m["agents"] != c["num_agents"] or not np.array_equal(frozen_grids,grids):
            raise ValueError("frozen records do not match backend/agents/maps")
        if len(stages) < c["num_steps"]:
            raise ValueError("record archive shorter than requested work")
    return grids


def record(c, output):
    from expert.training_baseline_engines import _generate_sequential_stages, _selected_maps
    selected, grids = _selected_maps(c["maps_path"],c["map_names"])
    stages = _generate_sequential_stages(c["backend"], maps_path=c["maps_path"], selected=selected,
                                      num_agents=c["num_agents"], num_steps=c["num_steps"], seed=c["seed"],
                                      max_episode_steps=c["max_episode_steps"])
    return save_records(output, backend=c["backend"], grids=grids, stages=stages, metadata=c)


def validation_from_records(c, records, output):
    """Rebuild a lost validation artifact from a separately frozen dev split."""
    import torch
    from expert.mapf_gpt_schema import tokenize_stage_reference
    from expert.mapf_gpt_validation import save_frozen_mapf_gpt_validation
    if c['backend']!='mapf_gpt' or records is None: raise ValueError('validation requires GPT development records')
    if Path(output).exists(): raise FileExistsError(output)
    m,grids,stages=load_records(records)
    selected_grids(c,records)
    if set(c['map_names']) & set(MAPS['mapf_gpt']): raise ValueError('development maps overlap training')
    builder=gpu_builder(c,grids); tokens=[]; labels=[]
    for i,rows in enumerate(stages):
        raw=torch.from_numpy(rows.astype(np.int16)).to(c['device']);builder.build_tokens(raw)
        t=builder.tokens.cpu().numpy().copy();y=builder.labels.cpu().numpy().copy()
        if i in (0,len(stages)-1):
            ref_t,ref_y=tokenize_stage_reference(grids,rows,num_agents=c['num_agents'])
            np.testing.assert_array_equal(t,ref_t);np.testing.assert_array_equal(y,ref_y)
        tokens.append(t);labels.append(y)
    save_frozen_mapf_gpt_validation(output,tokens=np.concatenate(tokens),labels=np.concatenate(labels),
        metadata=dict(kind='revision-development-validation',maps=c['map_names'],seed=c['seed'],records_sha256=sha256_file(Path(records)/'manifest.json'),
                      records_manifest=m,not_identical_to_lost_historical_validation=True))
    return dict(status='ok',samples=sum(len(y) for y in labels),sha256=sha256_file(output))


def runtime_for(c):
    from expert.training_baseline_engines import _make_runtime
    return _make_runtime(c["backend"],device=c["device"],seed=c["seed"],kwargs=c)


def gpu_builder(c, grids):
    import grid_world_cpp as ext
    from mapf_cuda.simulation.grids import stack_grids_for_compiled_simulator
    padded, extents = stack_grids_for_compiled_simulator(grids,device=c["device"])
    if c["backend"] == "magat":
        builder = ext.StatelessGridWorldSimulator(padded,c["num_agents"],3)
        builder.set_real_map_extents(extents)
        builder.pyg_builder_mode = "local_gather"
        builder.pyg_local_gather_impl = "auto"
        return builder
    return ext.MapfGPTObservationBuilder(padded,c["num_agents"])


def train_compact(c, runtime, builder, raw):
    if c["backend"] == "magat":
        reset = raw[raw[:,7] != 0]
        if len(reset):
            builder.update_energy_maps(reset,len(reset))
        builder.refresh_compact_state_from_raw_batch(raw)
        return float(runtime.train_step(builder,raw).item()), len(raw)
    metrics = runtime.train_step(builder,raw)
    if metrics["skipped"]:
        raise RuntimeError("unexpected skipped stage under train-on-arrived protocol")
    return float(metrics["loss"]), int(metrics["samples"])


def train_host(c, runtime, prepared):
    obj = prepared.to_device(c["device"])
    return train_device(c,runtime,obj)


def train_device(c,runtime,obj):
    if c["backend"] == "magat":
        return float(runtime.train_step_from_batch(obj).item()),len(obj.y)
    metrics = runtime.train_step(obj,None)
    if metrics["skipped"]:
        raise RuntimeError("unexpected skipped host stage")
    return float(metrics["loss"]),int(metrics["samples"])


def run_training(c, output, *, mode, phase, records=None, gate_report=None, smoke=False):
    import torch
    from expert.expert_running import ExtremeMAPFPipeline, RingBuffer
    from expert.checkpoint_manager import TopKCheckpointManager, model_state_sha256
    from expert.training_baseline_engines import _make_checkpoint_manager
    if mode not in MODES or phase not in ("fixed-work", "train", "online-throughput"):
        raise ValueError("unsupported training row")
    if phase == "fixed-work" and not records:
        raise ValueError("fixed-work requires frozen expert records")
    if phase == "train" and (records or not c["validation"]):
        raise ValueError("online train requires validation and disallows frozen records")
    if phase == "online-throughput" and (records or c["validation"]):
        raise ValueError("online-throughput disallows replay and validation")
    if not smoke and gate_report is None:
        raise ValueError("formal runs require --gate-report; use --smoke only for implementation checks")
    gate = None
    if gate_report is not None:
        from experiments.runners.training_revision_gate import check_gate
        gate = check_gate(gate_report,c,records)
    if c['capture_stage_hashes'] and not smoke:
        raise ValueError('stage hashing is untimed diagnostics only')
    if not records:
        from real_expert_alg.lacam import inference  # prebuild once before spawning
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    grids = selected_grids(c,records)
    memory_preflight=memory_estimate(c["backend"],grids,c["num_agents"],c["ring_depth"],c["host_workers"],c["host_depth"])
    torch.cuda.set_device(c["device"])
    torch.set_num_threads(1)
    runtime = runtime_for(c)
    initial_hash = model_state_sha256(runtime.model)
    features = 8 if c["backend"]=="magat" else 13
    stage_rows = c["train_batch_size"]
    pipeline = source = prefetch = None
    dma_thread = None
    builder = None
    if mode == "compact_stream":
        builder = gpu_builder(c,grids)
        if c["backend"] == "mapf_gpt":
            from expert.mapf_gpt_online import MapfGPTOnlinePipeline
            pipeline = MapfGPTOnlinePipeline(builder,runtime,capacity=stage_rows*c["ring_depth"],batch_threshold=stage_rows,device=c["device"])
        else:
            pipeline = ExtremeMAPFPipeline(builder,capacity=stage_rows*c["ring_depth"],batch_threshold=stage_rows,device=c["device"])
        pipeline.initialize()
        ring = pipeline.ring_buffer
    else:
        ring = RingBuffer(stage_rows*c["ring_depth"],features,num_envs=len(grids),agents_per_env=c["num_agents"])
        builder = CachedHostBuilder(c["backend"],grids,c["num_agents"])
    manager_args = {**c, "checkpoint_dir":str(output/"checkpoints") if phase=="train" else None,
                    "checkpoint_selection_mode":"validation_accuracy" if phase=="train" else "latest",
                    "checkpoint_interval":c["num_steps"]+1,
                    "validation_trajectories":c["validation"],"validation_datasets":c["validation"]}
    manager = _make_checkpoint_manager(c["backend"],manager_args)
    protocol = dict(optimizer_config=runtime.optimizer.defaults,
                    runtime_training_config=runtime.training_config() if c["backend"]=="magat" else runtime.checkpoint_metadata(),memory_preflight=memory_preflight,smoke=smoke,gate=gate,schema_version=1,experiment="JointLoom-training-revision",phase=phase,mode=mode,config=c,
                    timing_scope=("initialized_pipeline_no_validation_excluding_terminal_io_teardown" if phase=="online-throughput" else "initialized_pipeline_including_periodic_validation_io_excluding_terminal_io_teardown"),
                    records=None if not records else dict(path=str(Path(records).resolve()),sha256=sha256_file(Path(records)/"manifest.json")),
                    provenance=provenance(c))
    write_json(output/"protocol.json",protocol)
    losses, stage_hashes = [], []
    samples = arriving = payload = 0
    next_wall = c["checkpoint_interval_s"]
    wait_s = 0.
    last_progress = time.perf_counter()
    monitor=None
    trace_file = (output/"learning-trace.csv").open("x", newline="")
    trace = csv.writer(trace_file)
    trace.writerow(["optimizer_step", "training_elapsed_s", "samples_processed", "loss", "learning_rate", "scheduler_step"])
    trace_file.flush()
    trace_flush_at = time.perf_counter()
    write_json(output/"progress.json",dict(status="initializing",num_optimizer_steps=0,updated_unix_s=time.time()))
    try:
        source = RingStageSource(ring,c["backend"],c,records=records,timeout=c["timeout_s"])
        if mode == "cpu_cached_prefetch":
            prefetch = HostPrefetch(c["backend"],grids,c["num_agents"],workers=c["host_workers"],depth=c["host_depth"],timeout=c["timeout_s"],capture_hashes=c["capture_stage_hashes"])
        if phase == "train":
            manager.maybe_save(loss=0.,runtime=runtime,optimizer_step=0,force=True,
                               extra_meta=dict(training_elapsed_s=0.,event_kind="initial",samples_processed=0))
        torch.cuda.synchronize(c["device"])
        torch.cuda.reset_peak_memory_stats(c["device"])
        monitor=ResourceMonitor()
        monitor.start()
        start = time.perf_counter()
        last_report=start
        source.start()
        if pipeline:
            dma_thread = pipeline.dma_worker.start()
        submitted = 0
        device_ready=None
        h2d_lookahead_stages=0
        copy_stream=torch.cuda.Stream(device=c["device"]) if prefetch else None
        while len(losses) < c["num_steps"]:
            source.check()
            if c["wall_budget_s"] is not None and time.perf_counter()-start >= c["wall_budget_s"]:
                break
            waited = time.perf_counter()
            if mode == "compact_stream":
                if pipeline.dma_worker.dma_read_ptr-pipeline.compute_ptr < stage_rows:
                    if time.perf_counter()-last_progress > c["timeout_s"]:
                        raise TimeoutError("compact consumer stalled")
                    time.sleep(.0005)
                    wait_s += time.perf_counter()-waited
                    continue
                pipeline.dma_event.synchronize()
                raw = pipeline.extract_env_aligned_batch(stage_rows,c["num_agents"],len(grids))
                if raw is None:
                    continue
                wait_s += time.perf_counter()-waited
                if c["capture_stage_hashes"]:
                    stage_hashes.append(hashlib.sha256(raw.detach().cpu().numpy().tobytes()).hexdigest())
                loss,count = train_compact(c,runtime,builder,raw)
                payload += stage_rows*features*2
            else:
                if prefetch:
                    while submitted < min(c["num_steps"],len(losses)+c["host_depth"]):
                        rows = source.next_rows()
                        prefetch.submit(rows)
                        submitted += 1
                    if device_ready is None:
                        prepared = prefetch.get()
                        obj=prepared.to_device(c['device'])
                    else:
                        prepared,obj,event=device_ready
                        torch.cuda.current_stream(c['device']).wait_event(event)
                        device_ready=None
                    if prepared.stage_sequence != len(losses):
                        raise RuntimeError('prefetch consumer stage sequence mismatch')
                    if c['capture_stage_hashes']:
                        stage_hashes.append(prepared.stage_sha256)
                    upcoming=prefetch.try_get()
                    if upcoming is not None:
                        with torch.cuda.stream(copy_stream):
                            next_obj=upcoming.to_device(c['device'])
                            event=torch.cuda.Event()
                            event.record(copy_stream)
                        device_ready=(upcoming,next_obj,event)
                        h2d_lookahead_stages+=1
                else:
                    rows = source.next_rows()
                    if c["capture_stage_hashes"]:
                        stage_hashes.append(hashlib.sha256(rows.tobytes()).hexdigest())
                    prepared = builder.build(rows)
                wait_s += time.perf_counter()-waited
                loss,count = train_device(c,runtime,obj) if prefetch else train_host(c,runtime,prepared)
                payload += prepared.nbytes
            if not np.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at update {len(losses)+1}")
            losses.append(loss)
            samples += count
            arriving += stage_rows
            last_progress = time.perf_counter()
            step = len(losses)
            elapsed = last_progress-start
            scheduler = getattr(runtime, "scheduler", None)
            trace.writerow([step, elapsed, samples, loss, runtime.optimizer.param_groups[0]["lr"],
                            "" if scheduler is None else scheduler.last_epoch])
            if last_progress-trace_flush_at >= 5.:
                trace_file.flush()
                trace_flush_at = last_progress
            if phase=="train" and (elapsed >= next_wall or step in c["validation_steps"]):
                manager.maybe_save(loss=loss,runtime=runtime,optimizer_step=step,force=True,
                    extra_meta=dict(training_elapsed_s=elapsed,event_kind="periodic",samples_processed=samples,
                                    arriving_records=arriving,archive_checkpoint=step in c["archive_steps"]))
                while next_wall <= time.perf_counter()-start:
                    next_wall += c["checkpoint_interval_s"]
            last_progress = time.perf_counter()
            if last_progress-last_report >= c['progress_interval_s']:
                write_json(output/'progress.json',dict(status='running',num_optimizer_steps=step,
                    elapsed_s=last_progress-start,samples_processed=samples,last_loss=loss,updated_unix_s=time.time()))
                last_report=last_progress
        torch.cuda.synchronize(c["device"])
        wall = time.perf_counter()-start
        resources=monitor.finish()
        monitor=None
        if not losses:
            raise RuntimeError("budget exhausted without an optimizer update")
        if phase=="train":
            manager.maybe_save(loss=losses[-1],runtime=runtime,optimizer_step=len(losses),force=True,
                extra_meta=dict(training_elapsed_s=wall,event_kind="terminal_outside_timer",samples_processed=samples,arriving_records=arriving))
        result = dict(resources=resources,schema_version=1,status="ok",phase=phase,backend=c["backend"],mode=mode,seed=c["seed"],
                      protocol=protocol,num_optimizer_steps=len(losses),samples_processed=samples,arriving_records=arriving,
                      budget_overrun_s=0. if c["wall_budget_s"] is None else max(0.,wall-c["wall_budget_s"]),
                      total_wall_s=wall,samples_s=samples/wall,h2d_bytes=payload,consumer_input_wait_s=wait_s,
                      wait_scope="consumer_input_acquisition_includes_CPU_build_or_prefetch_wait; not_expert_only",
                      stop_reason="update_cap" if len(losses)==c["num_steps"] else "wall_budget",
                      initial_model_state_sha256=initial_hash,final_model_state_sha256=model_state_sha256(runtime.model),
                      checkpoint_history=manager.checkpoint_history(),saved_checkpoints=manager.snapshot(),
                      losses=losses,learning_trace="learning-trace.csv",stage_sha256s=stage_hashes[:len(losses)],
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(c["device"]),
                      peak_host_parent_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      raw_ring_bytes=stage_rows*c["ring_depth"]*features*2,
                      h2d_lookahead_stages=h2d_lookahead_stages,
                      host_prefetch_workers=c["host_workers"] if prefetch else 0,
                      host_prefetch_depth=c["host_depth"] if prefetch else 0)
        if phase == "online-throughput":
            torch.save(dict(model=runtime.model.state_dict(),optimizer=runtime.optimizer.state_dict(),
                scheduler=None if getattr(runtime,"scheduler",None) is None else runtime.scheduler.state_dict(),
                optimizer_step=len(losses), config=c), output/"terminal.pt")
        write_json(output/"result.json",result)
        write_json(output/"progress.json",dict(status="complete",num_optimizer_steps=len(losses),elapsed_s=wall,updated_unix_s=time.time()))
        return result
    except BaseException as exc:
        write_json(output/"failure.json",dict(status="failed",type=type(exc).__name__,error=str(exc)))
        raise
    finally:
        trace_file.flush()
        trace_file.close()
        if monitor: monitor.finish()
        if pipeline:
            pipeline.dma_worker.stop()
            if dma_thread:
                dma_thread.join(timeout=5)
            torch.cuda.synchronize(c["device"])
        if prefetch:
            prefetch.close()
        if source:
            source.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=["record","validation","gate","fixed-work","train","online-throughput"],required=True)
    p.add_argument("--backend",choices=list(MAPS),required=True)
    p.add_argument("--mode",choices=MODES,default="compact_stream")
    p.add_argument("--config",type=Path)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--records",type=Path)
    p.add_argument("--gate-report",type=Path)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--seed",type=int)
    p.add_argument("--num-steps",type=int)
    p.add_argument("--wall-budget-s",type=float)
    p.add_argument("--validation",nargs="+",type=str)
    args=p.parse_args(argv)
    mp.set_start_method("spawn",force=True)
    overrides={} if not args.config else json.loads(args.config.read_text())
    for key in ("seed","num_steps","wall_budget_s","validation"):
        if getattr(args,key) is not None:
            overrides[key]=getattr(args,key)
    if args.phase in ("record","validation","gate","fixed-work"):
        overrides["wall_budget_s"]=None
    c=make_config(args.backend,**overrides)
    if args.phase=="record":
        result=record(c,args.output_dir)
    elif args.phase=="validation":
        result=validation_from_records(c,args.records,args.output_dir)
    elif args.phase=="gate":
        from experiments.runners.training_revision_gate import run_gate
        result=run_gate(c,args.records,args.output_dir)
    else:
        result=run_training(c,args.output_dir,mode=args.mode,phase=args.phase,records=args.records,gate_report=args.gate_report,smoke=args.smoke)
    print(json.dumps({k:result.get(k) for k in ("status","num_optimizer_steps","total_wall_s","samples_s")},sort_keys=True))


if __name__=="__main__":
    main()
