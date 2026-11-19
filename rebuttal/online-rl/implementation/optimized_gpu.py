"""JointLoom-only execution optimizations; FP32 and PPO definitions are preserved."""
import bootstrap
import os
from pathlib import Path

CACHE=bootstrap.ARTIFACTS/'pipeline-v2-cache'
CACHE.mkdir(exist_ok=True)
os.environ.setdefault('TRITON_CACHE_DIR',str(CACHE/'triton'))
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(CACHE/'inductor'))
os.environ.setdefault('TMPDIR',str(bootstrap.ARTIFACTS/'tmp'))
import torch
import triton
import triton.language as tl
from configuration import SPEC
from ppo import loss


@triton.jit
def _gae(R,V,NV,D,A,RET,N:tl.constexpr,T:tl.constexpr,G:tl.constexpr,L:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    carry=tl.full((B,),0,tl.float32)
    for t in range(T-1,-1,-1):
        at=t*N+i
        r=tl.load(R+at,i<N,0);v=tl.load(V+at,i<N,0);nv=tl.load(NV+at,i<N,0)
        reset=tl.load(D+at,i<N,0)
        delta=(r+G*nv)-v
        carry=delta+((G*L)*(1-reset.to(tl.float32)))*carry
        tl.store(A+at,carry,i<N);tl.store(RET+at,carry+v,i<N)


def fused_gae(rewards,values,next_values,reset,advantages=None,returns=None):
    if advantages is None:advantages=torch.empty_like(rewards)
    if returns is None:returns=torch.empty_like(rewards)
    t,n=rewards.shape
    _gae[(triton.cdiv(n,128),)](rewards,values,next_values,reset,advantages,returns,n,t,
        SPEC['ppo']['gamma'],SPEC['ppo']['gae_lambda'],128,enable_fp_fusion=False)
    return advantages,returns


@triton.jit
def _gather_obs(X,ORDER,Y,START,M:tl.constexpr,CL:tl.constexpr,B:tl.constexpr):
    o=tl.program_id(0)*B+tl.arange(0,B);agent=o//242;local=o%242
    idx=tl.load(ORDER+START+agent,agent<M,0)
    src_local=(local%2)*121+local//2 if CL else local
    x=tl.load(X+idx*242+src_local,agent<M,0)
    tl.store(Y+o,x,agent<M)


@triton.jit
def _gather_scalars(ORDER,ACT,LP,V,A,R,OA,OUT,START,M:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);idx=tl.load(ORDER+START+i,i<M,0)
    tl.store(OA+i,tl.load(ACT+idx,i<M,0),i<M)
    tl.store(OUT+i,tl.load(LP+idx,i<M,0),i<M)
    tl.store(OUT+M+i,tl.load(V+idx,i<M,0),i<M)
    tl.store(OUT+2*M+i,tl.load(A+idx,i<M,0),i<M)
    tl.store(OUT+3*M+i,tl.load(R+idx,i<M,0),i<M)


class StaticLearner:
    """Capture fixed-shape forward/backward/Adam; replay each actual minibatch once."""
    def __init__(self,model,n=2048,minibatch=2048,channels_last=False,benchmark=True,graphs=True,
                 compile_loss=False,compile_model=False,loss_function=None):
        self.model=model;self.n=n;self.m=minibatch;self.channels_last=channels_last;self.graphs=graphs
        torch.backends.cudnn.benchmark=benchmark
        fmt=torch.channels_last if channels_last else torch.contiguous_format
        if channels_last:model.to(memory_format=torch.channels_last)
        self.forward=torch.compile(model,fullgraph=True,options={'triton.cudagraphs':False,
            'layout_optimization':False}) if compile_model else model
        if os.environ.get('JOINTLOOM_NEURAL_PROFILE_DIR'):
            from neural_timing import instrument_jointloom_forward
            self.forward=instrument_jointloom_forward(self.forward)
        self.stage_recorder=None
        if os.environ.get('JOINTLOOM_MODEL_STAGE_PROFILE'):
            from neural_timing import recorder
            self.stage_recorder=recorder()
        chosen_loss=loss if loss_function is None else loss_function
        self.loss_fn=torch.compile(chosen_loss,fullgraph=True,options={'triton.cudagraphs':False}) if compile_loss else chosen_loss
        self.actor_input=torch.empty((n,2,11,11),device='cuda',memory_format=fmt)
        self.mb_obs=torch.empty((minibatch,2,11,11),device='cuda',memory_format=fmt)
        self.mb_actions=torch.zeros(minibatch,dtype=torch.long,device='cuda')
        self.mb_scalars=torch.zeros((4,minibatch),device='cuda')
        self.mb_obs.zero_();self.actor_input.zero_()
        opt=SPEC['optimizer']
        self.optimizer=torch.optim.Adam(model.parameters(),lr=opt['learning_rate'],betas=opt['betas'],
            eps=opt['epsilon'],weight_decay=opt['weight_decay'],fused=True,capturable=True)
        initial={k:v.detach().clone() for k,v in model.state_dict().items()}
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                with torch.no_grad():self.forward(self.actor_input)
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        if graphs:
            self.actor_graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.actor_graph):
                with torch.no_grad():self.actor_logits,self.actor_value=self.forward(self.actor_input)
            self.train_graph=torch.cuda.CUDAGraph()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.graph(self.train_graph):self._step()
        # Warmup/capture never counts as training. Keep storage addresses for graph references.
        with torch.no_grad():
            for k,v in model.state_dict().items():v.copy_(initial[k])
            for state in self.optimizer.state.values():
                for v in state.values():
                    if isinstance(v,torch.Tensor):v.zero_()
        self.warmup_initial_weights_restored=all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
        self.warmup_optimizer_zero=all(bool((v==0).all()) for s in self.optimizer.state.values()
                                      for v in s.values() if isinstance(v,torch.Tensor))

    def _step(self):
        self.optimizer.zero_grad(set_to_none=True)
        logits,values=self.forward(self.mb_obs)
        total,stats=self.loss_fn(logits,values,self.mb_actions,*self.mb_scalars)
        total.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),SPEC['optimizer']['max_global_grad_norm'],foreach=True)
        self.optimizer.step()
        self.stats=torch.cat((total.detach()[None],stats,norm.detach()[None]))

    def policy(self,obs):
        self.actor_input.copy_(obs.reshape(self.n,2,11,11))
        if self.stage_recorder:
            with self.stage_recorder.stage(10):
                if self.graphs:
                    self.actor_graph.replay();return self.actor_logits,self.actor_value
                with torch.no_grad():return self.forward(self.actor_input)
        if self.graphs:
            self.actor_graph.replay();return self.actor_logits,self.actor_value
        with torch.no_grad():return self.forward(self.actor_input)

    def sample_order(self,count,generator,mode='shuffle'):
        if mode=='shuffle':return torch.randperm(count,device='cuda',generator=generator)
        if mode=='sf_env_major':
            # Storage is [time, actor]; native SF flattens [actor, time].
            return torch.arange(count,device='cuda').reshape(-1,self.n).T.contiguous().flatten()
        raise ValueError(mode)

    def update(self,data,generator,order_mode='shuffle'):
        if self.stage_recorder:
            return self._profiled_update(data,generator,order_mode)
        n=len(data['actions'])
        order=self.sample_order(n,generator,order_mode)
        recorded=torch.empty((n//self.m,7),device='cuda')
        for start in range(0,n,self.m):
            _gather_obs[(triton.cdiv(self.m*242,1024),)](data['obs'],order,self.mb_obs,start,
                self.m,self.channels_last,1024)
            _gather_scalars[(triton.cdiv(self.m,256),)](order,data['actions'],data['logp'],data['values'],
                data['advantages'],data['returns'],self.mb_actions,self.mb_scalars,start,self.m,256)
            if self.graphs:self.train_graph.replay()
            else:self._step()
            recorded[start//self.m].copy_(self.stats)
        return recorded.mean(0)

    def _profiled_update(self,data,generator,order_mode='shuffle'):
        r=self.stage_recorder
        with r.stage(4):
            with r.stage(6):
                n=len(data['actions'])
                order=self.sample_order(n,generator,order_mode)
                recorded=torch.empty((n//self.m,7),device='cuda')
            for start in range(0,n,self.m):
                with r.stage(6):
                    _gather_obs[(triton.cdiv(self.m*242,1024),)](data['obs'],order,self.mb_obs,start,
                        self.m,self.channels_last,1024)
                    _gather_scalars[(triton.cdiv(self.m,256),)](order,data['actions'],data['logp'],data['values'],
                        data['advantages'],data['returns'],self.mb_actions,self.mb_scalars,start,self.m,256)
                if self.graphs:self.train_graph.replay()
                else:self._step()
                recorded[start//self.m].copy_(self.stats)
            result=recorded.mean(0)
        return result

    def load_minibatch(self,data):
        self.mb_obs.copy_(data['obs']);self.mb_actions.copy_(data['actions'])
        for i,k in enumerate(('logp','values','advantages','returns')):self.mb_scalars[i].copy_(data[k])

    def step_loaded(self):
        if self.graphs:self.train_graph.replay()
        else:self._step()
        return self.stats.clone()
