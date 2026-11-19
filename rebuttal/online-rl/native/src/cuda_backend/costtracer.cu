#include "costtracer.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__global__ void progress_goals(GridWorldContext s, MapfGPTCUDAContext b, int* best,
    const int16_t* targets, int* cursor, float* rewards, float* reached, int length) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=s.n_envs*s.n_agents) return;
    int d=b.d_cost_to_go[(int64_t)i*GRID_SIZE+s.cur_x[i]*MAP_H+s.cur_y[i]];
    rewards[i]=(d<best[i]) ? .1f : 0.f;
    best[i]=min(best[i],d);
    bool hit=s.cur_x[i]==s.goals_x[i] && s.cur_y[i]==s.goals_y[i];
    reached[i]=hit ? 1.f:0.f;
    if(hit) {
        int k=cursor[i]++;
        // length > horizon: an agent cannot exhaust a stream before reset.
        s.goals_x[i]=targets[((int64_t)i*length+k)*2];
        s.goals_y[i]=targets[((int64_t)i*length+k)*2+1];
        best[i]=-1; // Even if the sampled target equals the old target.
    }
}

__global__ void observe(GridWorldContext s, MapfGPTCUDAContext b, int* best, float* output) {
    int i=blockIdx.x,t=threadIdx.x,env=i/s.n_agents;
    __shared__ int vals[128];
    int x=s.cur_x[i]+t/11-5, y=s.cur_y[i]+t%11-5;
    int d=-1;
    if(t<121 && x>=0 && x<MAP_W && y>=0 && y<MAP_H) {
        int raw=b.d_cost_to_go[(int64_t)i*GRID_SIZE+x*MAP_H+y];
        d=raw==65535 ? -1:raw;
    }
    vals[t]=t<121? max(0,d):0;
    __syncthreads();
    for(int n=64;n;n>>=1) {if(t<n) vals[t]=max(vals[t],vals[t+n]); __syncthreads();}
    int v=d<0?d+5+vals[0]:d;
    __syncthreads();
    vals[t]=t<121?v:1000000;
    __syncthreads();
    for(int n=64;n;n>>=1) {if(t<n) vals[t]=min(vals[t],vals[t+n]); __syncthreads();}
    v-=vals[0];
    __syncthreads();
    vals[t]=t<121?v:0;
    __syncthreads();
    for(int n=64;n;n>>=1) {if(t<n) vals[t]=max(vals[t],vals[t+n]); __syncthreads();}
    if(t==0 && (best[i]<0 || b.d_states[i].reset_flag))
        best[i]=b.d_cost_to_go[(int64_t)i*GRID_SIZE+s.cur_x[i]*MAP_H+s.cur_y[i]];
    __syncthreads();
    if(t<121) {
        bool occupied=false;
        if(x>=0 && x<MAP_W && y>=0 && y<MAP_H)
            occupied=(s.d_grid_ocp[env*MAP_OFFSET+x*COL_OFFSET+(y>>5)]>>(y&31))&1;
        output[(int64_t)i*242+t]=occupied?1.f:0.f;
        output[(int64_t)i*242+121+t]=vals[0]>0 ? float(vals[0]-v)/vals[0]:0.f;
        if(t==60) {
            int current=b.d_cost_to_go[(int64_t)i*GRID_SIZE+s.cur_x[i]*MAP_H+s.cur_y[i]];
            output[(int64_t)i*242+181]=-fminf(1.f,float(current-best[i]+1)/64.f);
        }
    }
}

__global__ void reset_selected(GridWorldContext s, const int16_t* pos, const int16_t* goal,
    int* episodes, int* cursors, int* best, int bank, int horizon) {
    int e=blockIdx.x,t=threadIdx.x;
    if(s.d_step_counts[e]<horizon) return;
    __shared__ int episode;
    if(t==0) {episode=(++episodes[e])%bank;s.d_step_counts[e]=0;}
    __syncthreads();
    for(int j=t;j<MAP_OFFSET;j+=blockDim.x) s.d_grid_ocp[e*MAP_OFFSET+j]=0;
    __syncthreads();
    for(int a=t;a<s.n_agents;a+=blockDim.x) {
        int i=e*s.n_agents+a;
        int64_t k=((int64_t)episode*s.n_envs*s.n_agents+i)*2;
        s.cur_x[i]=pos[k];s.cur_y[i]=pos[k+1];
        s.goals_x[i]=goal[k];s.goals_y[i]=goal[k+1];
        cursors[i]=0;best[i]=-1;
        atomicOr(s.d_grid_ocp+e*MAP_OFFSET+s.cur_x[i]*COL_OFFSET+(s.cur_y[i]>>5),1u<<(s.cur_y[i]&31));
    }
}

static void check(const torch::Tensor& t, torch::ScalarType dtype, int64_t n,
                  const torch::Device& device) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type()==dtype &&
                t.numel()==n && t.device()==device,"CostTracer tensor shape/dtype/device mismatch");
}

void costtracer_reward_goals(GridWorldSimulator& s, MapfGPTObservationBuilder& b,
    torch::Tensor best, torch::Tensor targets, torch::Tensor cursor,
    torch::Tensor rewards, torch::Tensor reached) {
    int n=s.ctx.n_envs*s.ctx.n_agents;
    auto dev=s.t_cur_x.device();
    check(best,torch::kInt32,n,dev);check(cursor,torch::kInt32,n,dev);
    check(rewards,torch::kFloat32,n,dev);check(reached,torch::kFloat32,n,dev);
    TORCH_CHECK(targets.dim()==4 && targets.size(0)==s.ctx.n_envs &&
        targets.size(1)==s.ctx.n_agents && targets.size(2)>s.ctx.max_episode_steps && targets.size(3)==2,
        "goal stream must be [N,A,K>horizon,2]");
    check(targets,torch::kInt16,targets.numel(),dev);
    c10::cuda::CUDAGuard guard(dev);
    progress_goals<<<(n+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(s.ctx,b.ctx,
        best.data_ptr<int>(),targets.data_ptr<int16_t>(),cursor.data_ptr<int>(),
        rewards.data_ptr<float>(),reached.data_ptr<float>(),targets.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void costtracer_observe(GridWorldSimulator& s, MapfGPTObservationBuilder& b,
    torch::Tensor best, torch::Tensor output) {
    int n=s.ctx.n_envs*s.ctx.n_agents;
    auto dev=s.t_cur_x.device();
    check(best,torch::kInt32,n,dev);check(output,torch::kFloat32,n*242,dev);
    c10::cuda::CUDAGuard guard(dev);
    observe<<<n,128,0,at::cuda::getCurrentCUDAStream()>>>(s.ctx,b.ctx,best.data_ptr<int>(),output.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void costtracer_reset(GridWorldSimulator& s, torch::Tensor pos, torch::Tensor goal,
    torch::Tensor episodes, torch::Tensor cursors, torch::Tensor best, torch::Tensor /*reserved*/, int horizon) {
    int n=s.ctx.n_envs*s.ctx.n_agents;auto dev=s.t_cur_x.device();
    TORCH_CHECK(pos.dim()==4 && pos.size(1)==s.ctx.n_envs && pos.size(2)==s.ctx.n_agents && pos.size(3)==2,
        "reset bank must be [episodes,N,A,2]");
    check(pos,torch::kInt16,pos.numel(),dev);check(goal,torch::kInt16,pos.numel(),dev);
    check(episodes,torch::kInt32,s.ctx.n_envs,dev);check(cursors,torch::kInt32,n,dev);check(best,torch::kInt32,n,dev);
    c10::cuda::CUDAGuard guard(dev);
    reset_selected<<<s.ctx.n_envs,128,0,at::cuda::getCurrentCUDAStream()>>>(s.ctx,
        pos.data_ptr<int16_t>(),goal.data_ptr<int16_t>(),episodes.data_ptr<int>(),
        cursors.data_ptr<int>(),best.data_ptr<int>(),pos.size(0),horizon);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
