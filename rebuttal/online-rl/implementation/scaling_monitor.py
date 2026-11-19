"""Low-frequency, identical external resource measurements for both training paths."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import time
import psutil


def physical_core_cpus(count=8):
    chosen=[];seen=set()
    for cpu in sorted(os.sched_getaffinity(0)):
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=((p/'physical_package_id').read_text().strip(),(p/'core_id').read_text().strip())
        if key not in seen:seen.add(key);chosen.append(cpu)
        if len(chosen)==count:break
    return chosen


class NVML:
    class Memory(ctypes.Structure):
        _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
    class Utilization(ctypes.Structure):
        _fields_=[('gpu',ctypes.c_uint),('memory',ctypes.c_uint)]
    def __init__(self):
        self.lib=ctypes.CDLL('libnvidia-ml.so.1');self.handle=ctypes.c_void_p()
        assert self.lib.nvmlInit_v2()==0
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle))==0
    def sample(self):
        memory=self.Memory();util=self.Utilization()
        a=self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(memory))
        b=self.lib.nvmlDeviceGetUtilizationRates(self.handle,ctypes.byref(util))
        return dict(gpu_memory_bytes=int(memory.used) if a==0 else None,
                    gpu_utilization=int(util.gpu) if b==0 else None)


def run_monitored(command,log_path,output_dir,cpus,iterations):
    nvml=NVML();idle=nvml.sample();rows=[];previous={};previous_time=None
    log_path=Path(log_path);out=Path(output_dir)
    with log_path.open('w') as log:
        process=subprocess.Popen(['taskset','-c',','.join(map(str,cpus)),*map(str,command)],
                                 stdout=log,stderr=subprocess.STDOUT)
        root=psutil.Process(process.pid)
        while True:
            now=time.perf_counter();current={}
            try:processes=[root,*root.children(recursive=True)]
            except psutil.NoSuchProcess:processes=[]
            cpu_delta=0.
            for p in processes:
                try:
                    c=p.cpu_times();value=c.user+c.system;current[p.pid]=value
                    if p.pid in previous:cpu_delta+=max(0.,value-previous[p.pid])
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            stats=nvml.sample()
            progress=out/'native-iterations.jsonl'
            if not progress.exists():progress=out/'iterations.jsonl'
            completed=len(progress.read_text().splitlines()) if progress.exists() else 0
            row=dict(elapsed=now-(rows[0]['monotonic'] if rows else now),monotonic=now,
                completed_iterations=completed,steady=2<=completed<iterations,
                cpu_cores_used=cpu_delta/(now-previous_time) if previous_time else None,**stats)
            rows.append(row);previous=current;previous_time=now
            code=process.poll()
            if code is not None:break
            time.sleep(.2)
    out.mkdir(parents=True,exist_ok=True)
    summary=dict(exit_code=code,cpu_affinity=cpus,interval_seconds=.2,idle_gpu=idle,
        gpu_device_peak_bytes=max(r['gpu_memory_bytes'] for r in rows if r['gpu_memory_bytes'] is not None),
        steady_samples=sum(r['steady'] for r in rows),rows=rows)
    steady=[r for r in rows if r['steady']]
    if steady:
        summary['steady_gpu_utilization_mean']=sum(r['gpu_utilization'] for r in steady)/len(steady)
        summary['steady_cpu_cores_used_mean']=sum(r['cpu_cores_used'] or 0 for r in steady)/len(steady)
        summary['steady_gpu_memory_peak_bytes']=max(r['gpu_memory_bytes'] for r in steady)
    (out/'external-resources.json').write_text(json.dumps(summary,indent=2))
    return code
