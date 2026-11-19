"""Device-clock neural intervals, including CUDA graph replays.

Opt-in instrumentation only. Each timestamp is a tiny stream-ordered CUDA
kernel, so captured graphs append fresh timestamps on every replay. All GPU
processes use the same device globaltimer. Analysis subtracts the UNION of
neural intervals from measured wall windows, not a sum of process timers.
"""
import bootstrap
from contextlib import contextmanager
import json
import multiprocessing.util
import os
import time
from pathlib import Path

os.environ.setdefault('TRITON_CACHE_DIR',str(bootstrap.ARTIFACTS/'pipeline-v2-cache'/'triton'))
import torch
import triton
import triton.language as tl

RECORDER = None
ORIGINAL_BACKWARD = None
NATIVE_STAGE_PATCHED = False


@triton.jit
def _timestamp(RECORDS, COUNT, TAG: tl.constexpr, CAPACITY: tl.constexpr):
    index = tl.atomic_add(COUNT, 1)
    stamp = tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;',
        constraints='=l', args=[], dtype=tl.uint64, is_pure=False, pack=1)
    tl.store(RECORDS+index*2,stamp,index<CAPACITY)
    tl.store(RECORDS+index*2+1,TAG,index<CAPACITY)


class NeuralRecorder:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True,exist_ok=True)
        self.capacity = 65536
        self.records = torch.empty((self.capacity,2),dtype=torch.int64,device='cuda')
        self.count = torch.zeros((),dtype=torch.int32,device='cuda')
        # Compile and warm outside both training and timing windows.
        self.mark(99)
        torch.cuda.synchronize()
        self.count.zero_()
        self.calibration = []
        for index in range(12):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            self.mark(99)
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            gpu_ns = int(self.records[index,0].item())
            self.calibration.append(dict(bracket_ns=end-start,
                cpu_minus_gpu_ns=(start+end)//2-gpu_ns))
        best = min(self.calibration,key=lambda r:r['bracket_ns'])
        self.cpu_minus_gpu_ns = best['cpu_minus_gpu_ns']
        self.count.zero_()
        self.dumped = False
        multiprocessing.util.Finalize(None,self.dump,exitpriority=10)

    def mark(self, tag):
        _timestamp[(1,)](self.records,self.count,tag,self.capacity,num_warps=1)

    def wrap_forward(self, function):
        def measured(*args,**kwargs):
            self.mark(0)
            output = function(*args,**kwargs)
            self.mark(1)
            return output
        return measured

    @contextmanager
    def stage(self, tag):
        self.mark(tag)
        try:
            yield
        finally:
            self.mark(tag+1)

    def wrap_stage(self, function, tag):
        def measured(*args,**kwargs):
            with self.stage(tag):
                return function(*args,**kwargs)
        return measured

    def dump(self):
        if self.dumped:
            return
        torch.cuda.synchronize()
        n = int(self.count.item())
        if n>self.capacity:
            raise RuntimeError(f'neural timing capacity {n}>{self.capacity}')
        data = dict(pid=os.getpid(),device=torch.cuda.get_device_name(),
            cpu_minus_gpu_ns=self.cpu_minus_gpu_ns,calibration=self.calibration,
            records=self.records[:n].cpu().tolist(),
            tags={'0':'neural forward begin','1':'neural forward end',
                  '2':'total backward begin','3':'total backward end',
                  '4':'learner stage begin','5':'learner stage end',
                  '6':'retained shuffle/gather begin','7':'retained shuffle/gather end',
                  '10':'inference/model invocation begin','11':'inference/model invocation end'},
            backward_scope='entire loss.backward, excluding grad clipping and optimizer')
        (self.folder/f'gpu-neural-{os.getpid()}.json').write_text(json.dumps(data))
        self.dumped = True


def recorder():
    global RECORDER
    if RECORDER is None:
        folder = os.environ.get('JOINTLOOM_NEURAL_PROFILE_DIR')
        if not folder:
            raise RuntimeError('profiling directory not configured')
        RECORDER = NeuralRecorder(folder)
    return RECORDER


def instrument_backward():
    global ORIGINAL_BACKWARD
    if ORIGINAL_BACKWARD is not None:
        return
    ORIGINAL_BACKWARD = torch.Tensor.backward
    r = recorder()
    def measured_backward(tensor,*args,**kwargs):
        r.mark(2)
        result = ORIGINAL_BACKWARD(tensor,*args,**kwargs)
        r.mark(3)
        return result
    torch.Tensor.backward = measured_backward


def instrument_native_model(model):
    r = recorder()
    # These are exactly the CostTracer neural modules. Distribution creation,
    # action sampling and PPO forward loss stay outside the excluded intervals.
    for module in (model.encoder,model.critic_linear,
                   model.action_parameterization.distribution_linear):
        module.forward = r.wrap_forward(module.forward)
    instrument_backward()
    if os.environ.get('JOINTLOOM_MODEL_STAGE_PROFILE'):
        model.forward = r.wrap_stage(model.forward,10)
        instrument_native_stages()


def instrument_native_stages():
    """Wrap public Python scopes, without replacing SF's learner implementation."""
    global NATIVE_STAGE_PATCHED
    if NATIVE_STAGE_PATCHED:
        return
    NATIVE_STAGE_PATCHED = True
    r = recorder()
    from sample_factory.algo.learning.learner import Learner
    from sample_factory.utils.timing import Timing
    Learner._train = r.wrap_stage(Learner._train,4)
    Learner._get_minibatches = r.wrap_stage(Learner._get_minibatches,6)
    Learner._get_minibatch = staticmethod(r.wrap_stage(Learner._get_minibatch,6))
    original_add_time = Timing.add_time
    def measured_add_time(self,key):
        context = original_add_time(self,key)
        if key not in ('forward','update_model'):
            return context
        @contextmanager
        def model_scope():
            with r.stage(10),context:
                yield
        return model_scope()
    Timing.add_time = measured_add_time


def instrument_jointloom_forward(function):
    instrument_backward()
    return recorder().wrap_forward(function)


def flush():
    if RECORDER is not None:
        RECORDER.dump()


def merge_intervals(intervals):
    result = []
    for start,end in sorted(intervals):
        if result and start<=result[-1][1]:
            result[-1][1] = max(result[-1][1],end)
        else:
            result.append([start,end])
    return result


def subtract_intervals(intervals, holes):
    """Remove data preparation from its process's learner envelope first."""
    holes = merge_intervals(holes)
    result = []
    for start,end in intervals:
        at = start
        for lo,hi in holes:
            if hi<=at:
                continue
            if lo>=end:
                break
            if lo>at:
                result.append((at,min(lo,end)))
            at = max(at,hi)
            if at>=end:
                break
        if at<end:
            result.append((at,end))
    return result


def analyze(output, native):
    output = Path(output)
    all_intervals = []
    model_intervals = []
    data_holes = []
    calibration = []
    for path in output.glob('gpu-neural-*.json'):
        data = json.loads(path.read_text())
        offset = data['cpu_minus_gpu_ns']
        calibration.append(min(v['bracket_ns'] for v in data['calibration']))
        stack = []
        process_intervals = []
        for stamp,tag in data['records']:
            if tag in (0,2,4,6,10):
                stack.append((stamp,tag))
            elif tag in (1,3,5,7,11):
                begin,begin_tag = stack.pop()
                if tag!=begin_tag+1 or stamp<begin:
                    raise AssertionError((path,begin_tag,tag,begin,stamp))
                process_intervals.append((begin+offset,stamp+offset,begin_tag))
        if stack:
            raise AssertionError((path,'unfinished neural intervals',stack))
        all_intervals.extend((a,b,t) for a,b,t in process_intervals if t in (0,2))
        envelopes = [(a,b) for a,b,t in process_intervals if t==4]
        holes = [(a,b) for a,b,t in process_intervals if t==6]
        data_holes.extend(holes)
        model_intervals.extend(subtract_intervals(envelopes,holes))
        model_intervals.extend((a,b) for a,b,t in process_intervals if t in (0,2,10))
    if not all_intervals:
        raise RuntimeError('No neural timing records: '+str(output))
    if native:
        rows = json.loads((output/'native-timing.json').read_text())['rows']
        windows = [(int(rows[i-1]['timestamp_perf_counter']*1e9),
                    int(rows[i]['timestamp_perf_counter']*1e9)) for i in range(2,len(rows))]
    else:
        rows = [json.loads(s) for s in (output/'iterations.jsonl').read_text().splitlines()]
        windows = [(r['begin_perf_counter_ns'],r['end_perf_counter_ns']) for r in rows[2:]]
    measured = []
    for begin,end in windows:
        clipped = [(max(begin,a),min(end,b),tag) for a,b,tag in all_intervals if a<end and b>begin]
        union = merge_intervals([(a,b) for a,b,_ in clipped])
        neural = sum(b-a for a,b in union)
        forward = sum(b-a for a,b in merge_intervals([(a,b) for a,b,t in clipped if t==0]))
        backward = sum(b-a for a,b in merge_intervals([(a,b) for a,b,t in clipped if t==2]))
        measured.append(dict(wall_seconds=(end-begin)/1e9,
            neural_union_seconds=neural/1e9,forward_union_seconds=forward/1e9,
            backward_union_seconds=backward/1e9,residual_seconds=(end-begin-neural)/1e9,
            neural_process_sum_seconds=sum(b-a for a,b,_ in clipped)/1e9))
        if any(a<end and b>begin for a,b in model_intervals):
            model_union=merge_intervals([(max(begin,a),min(end,b)) for a,b in model_intervals if a<end and b>begin])
            model_ns=sum(b-a for a,b in model_union)
            measured[-1].update(model_stage_union_seconds=model_ns/1e9,
                system_residual_seconds=(end-begin-model_ns)/1e9,
                shuffle_gather_hole_seconds=sum(min(end,b)-max(begin,a) for a,b in data_holes if a<end and b>begin)/1e9)
    means = {k:sum(r[k] for r in measured)/len(measured) for k in measured[0]}
    result = dict(metric='wall duration minus union of device-clock forward/backward intervals',
        mean=means,iterations=measured,raw_neural_intervals=len(all_intervals),
        min_calibration_brackets_ns=calibration,
        backward='entire loss.backward; gradient clipping and optimizer are retained',
        interpretation='residual accounting on actual overlapping run, not a zero-cost-network counterfactual throughput')
    if any(t==4 for path in output.glob('gpu-neural-*.json')
           for _,t in json.loads(path.read_text())['records']):
        result['full_model_stage_scope']='learner loop minus shuffle/gather holes; inference evaluation/action sampling/model invocation and parameter synchronization'
    (output/'residual-timing.json').write_text(json.dumps(result,indent=2))
    return result
