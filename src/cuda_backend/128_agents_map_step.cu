#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <cstdint>
#include <stdio.h>
#include <torch/extension.h>
#include "grid_world_cuda.h"

// 首先，保证 MAP_H 为 32 的倍数
// 障碍物和观测智能体的存储方式:每连续 32 个状态信息被压缩为一个 uint32_t

__device__ bool is_wall(const uint32_t* __restrict__ s_map_static, int x, int y) {
    if (x < 0 || x >= MAP_W || y < 0 || y >= MAP_H) return true;
    uint32_t row_data = s_map_static[x * COL_OFFSET + (y>>5)];
    return (row_data >> (y & 31)) & 1;
}

__device__ __forceinline__ float fast_sign_step(float x) {
    if (x == 0.0f) return 0.0f;
    return copysignf(1.0f, x);
}

// Atmoic Grid 占位优化可用于更小的 grid_size
// 因为在设计过程中进行了 kernel fusion, 所以显得很大
__global__ void step_kernel(GridWorldContext ctx) {
    int env_id = blockIdx.x;
    int agent_count = ctx.n_agents;
    int tid = threadIdx.x;

    if (env_id >= ctx.n_envs) return;
    if (ctx.task_mode == TASK_MODE_STANDARD_MAPF &&
        (ctx.d_terminated[env_id] != 0 || ctx.d_truncated[env_id] != 0)) {
        return;
    }

    const int dx[5] = {0, -1, 1, 0, 0};
    const int dy[5] = {0, 0, 0, -1, 1};
    extern __shared__ uint8_t s_shared_mem[];
    uint8_t* s_curr_x = s_shared_mem;
    uint8_t* s_curr_y = s_shared_mem + agent_count;
    uint8_t* s_next_x = s_shared_mem + 2 * agent_count;
    uint8_t* s_next_y = s_shared_mem + 3 * agent_count;
    uint8_t* s_was_arrived = s_shared_mem + 4 * agent_count;

    __shared__ uint32_t s_map_static[MAP_OFFSET];  // 缓存 d_maps
    __shared__ uint32_t s_map_dynamic[MAP_OFFSET]; // 缓存 d_grid_ocp

    for (int i = tid; i < MAP_OFFSET; i += blockDim.x) {
        s_map_static[i] = ctx.d_maps[env_id * MAP_OFFSET + i];
    }
    
    __syncthreads(); 

    for (int i = tid; i < agent_count; i += blockDim.x) {
        int global_idx = env_id * agent_count + i;
        
        int cx = ctx.cur_x[global_idx];
        int cy = ctx.cur_y[global_idx];
        uint8_t act = ctx.actions[global_idx];

        int nx, ny;

        if (act > 4) act = 0;
        int tx = cx + dx[act];
        int ty = cy + dy[act];

        bool was_arrived =
            ctx.task_mode == TASK_MODE_STANDARD_MAPF && ctx.d_arrived[global_idx] != 0;
        s_was_arrived[i] = was_arrived ? 1 : 0;
        if (was_arrived) {
            nx = cx; ny = cy;
        } else if (is_wall(s_map_static, tx, ty)) {
            nx = cx; ny = cy;
        } else {
            nx = tx; ny = ty;
        }
        
        s_curr_x[i] = (uint8_t)cx;
        s_curr_y[i] = (uint8_t)cy;
        s_next_x[i] = (uint8_t)nx;
        s_next_y[i] = (uint8_t)ny;
    }
    
    uint8_t* s_temp_result_x = (uint8_t*)s_map_dynamic;
    uint8_t* s_temp_result_y = &s_temp_result_x[agent_count];
    __shared__ int s_collision_counter;
    
    __syncthreads();
    
    // 在每轮中，仅意图者需要被检测
    for (int iter = 0; iter < agent_count; ++iter) {
        
        if (tid == 0) s_collision_counter = 0;
        __syncthreads();

        for (int i = tid; i < agent_count; i += blockDim.x) {
            uint8_t my_cx = s_curr_x[i];
            uint8_t my_cy = s_curr_y[i];
            
            uint8_t my_nx = s_next_x[i];
            uint8_t my_ny = s_next_y[i];

            uint8_t final_nx = my_nx;
            uint8_t final_ny = my_ny;
            bool changed = false;

            if (my_nx != my_cx || my_ny != my_cy) {
                bool collision = false;

                for (int k = 0; k < agent_count; ++k) {
                    if (k == i) continue;
                    
                    // 曼哈顿距离粗筛
                    if (abs((int)my_cx - s_curr_x[k]) + abs((int)my_cy - s_curr_y[k]) > 2) continue;

                    // 读取对方的意图 (Input, 也是上一轮的结果)
                    uint8_t other_nx = s_next_x[k]; 
                    uint8_t other_ny = s_next_y[k];
                    uint8_t other_cx = s_curr_x[k];
                    uint8_t other_cy = s_curr_y[k];

                    // 1. Swap Conflict (交换位置冲突)
                    if (my_nx == other_cx && my_ny == other_cy &&
                        other_nx == my_cx && other_ny == my_cy) {
                        collision = true; break;
                    }

                    // 2. Vertex Conflict (同一目标冲突)
                    if (my_nx == other_nx && my_ny == other_ny) {
                        if (i > k) { collision = true; break; } // ID 大的让路
                    }

                    // 3. Chain Conflict (依赖链冲突)
                    if (my_nx == other_cx && my_ny == other_cy) {
                        if (other_nx == other_cx && other_ny == other_cy) {
                            collision = true; break;
                        }
                    }
                }

                if (collision) {
                    final_nx = my_cx;
                    final_ny = my_cy;
                    changed = true;
                }
            }

            // 写入 Output Buffer (s_temp)
            // 注意：此时绝对不能修改 s_next，因为其他线程可能还在读它
            s_temp_result_x[i] = final_nx;
            s_temp_result_y[i] = final_ny;

            if (changed) {
                atomicAdd(&s_collision_counter, 1);
            }
        }

        __syncthreads();
        // 快速退出机制：如果没有人改变决定，说明系统已达到稳态 (Fixed Point)
        if (s_collision_counter == 0) break;

        // Buffer Swap: 将 s_temp (Output) 拷回 s_next (Input) 为下一轮做准备
        // Shared Memory 内部拷贝，极快 (可以用乒乓搬运进行优化)
        for (int i = tid; i < agent_count; i += blockDim.x) {
            s_next_x[i] = s_temp_result_x[i];
            s_next_y[i] = s_temp_result_y[i];
        }
         __syncthreads(); 
    }
    
    uint8_t* s_goal_x = s_curr_x;
    uint8_t* s_goal_y = s_curr_y;
    // --- 迭代结束，写入 Global Memory ---
    // 此时 s_next 中存储的是最终收敛的无冲突状态
    
    // 写入 Global Memory (最终状态)
    for (int i = tid; i < agent_count; i += blockDim.x) {
        int global_idx = env_id * agent_count + i;
        
        // 直接从 s_next 读取最终结果
        uint8_t final_nx = s_next_x[i];
        uint8_t final_ny = s_next_y[i];

        ctx.cur_x[global_idx] = final_nx;
        ctx.cur_y[global_idx] = final_ny;
        
        uint8_t gx = ctx.goals_x[global_idx];
        uint8_t gy = ctx.goals_y[global_idx];
        s_goal_x[i] = gx;
        s_goal_y[i] = gy;

        ctx.rewards[global_idx] = (final_nx == gx && final_ny == gy) ? 1.0f : 0.0f;
        // A standard-MAPF goal is fixed.  The resident energy map is already
        // valid after the pre-step derived-state update, so keep it clean.
        ctx.d_goal_changed_flags[global_idx] = 0;
    }

    __syncthreads(); 
    for (int i = tid; i < MAP_OFFSET; i += blockDim.x) {
        s_map_dynamic[i] = 0;
    }
    __syncthreads();

    for (int i = tid; i < agent_count; i += blockDim.x) {
        uint8_t final_nx = s_next_x[i];
        uint8_t final_ny = s_next_y[i];
        
        int word_idx = (int)final_nx * COL_OFFSET + (final_ny >> 5);
        uint32_t mask = 1u << (final_ny & 31);
        
        atomicOr(&s_map_dynamic[word_idx], mask);
    }
    __syncthreads();
    for (int i = tid; i < MAP_OFFSET; i += blockDim.x) {
        ctx.d_grid_ocp[env_id * MAP_OFFSET + i] = s_map_dynamic[i];
    }
    __syncthreads();

    if (ctx.task_mode == TASK_MODE_STANDARD_MAPF) {
        for (int i = tid; i < agent_count; i += blockDim.x) {
            int global_idx = env_id * agent_count + i;
            if (s_next_x[i] == s_goal_x[i] && s_next_y[i] == s_goal_y[i]) {
                ctx.d_arrived[global_idx] = 1;
            }
        }
        __syncthreads();

        if (tid == 0) {
            bool all_arrived = true;
            for (int i = 0; i < agent_count; ++i) {
                if (ctx.d_arrived[env_id * agent_count + i] == 0) {
                    all_arrived = false;
                    break;
                }
            }
            int next_step = ctx.d_step_counts[env_id] + 1;
            ctx.d_step_counts[env_id] = next_step;
            if (all_arrived) {
                ctx.d_terminated[env_id] = 1;
                ctx.d_truncated[env_id] = 0;
            } else if (next_step >= ctx.max_episode_steps) {
                ctx.d_truncated[env_id] = 1;
            }
        }
        __syncthreads();

        for (int i = tid; i < agent_count; i += blockDim.x) {
            int global_idx = env_id * agent_count + i;
            AgentState state;
            state.env_id = static_cast<uint16_t>(env_id);
            state.agent_id = static_cast<uint16_t>(i);
            state.pos_x = static_cast<uint16_t>(s_next_x[i]);
            state.pos_y = static_cast<uint16_t>(s_next_y[i]);
            state.target_x = static_cast<uint16_t>(s_goal_x[i]);
            state.target_y = static_cast<uint16_t>(s_goal_y[i]);
            state.action = s_was_arrived[i] != 0
                ? 0
                : static_cast<uint16_t>(ctx.actions[global_idx]);
            state.reset_flag = 0;
            ctx.d_packed_states[global_idx] = state;
        }
        return;
    }

    uint32_t* s_goal_bitmap = s_map_dynamic; 

    // 2. 初始化 Bitmap (清零)
    for (int i = tid; i < MAP_OFFSET; i += blockDim.x) {
        s_goal_bitmap[i] = 0;
    }
    __syncthreads();

    for (int i = tid; i < agent_count; i += blockDim.x) {
        int global_idx = env_id * agent_count + i;
        
        // 未改变目标的智能体占据对应的空格
        if (ctx.rewards[global_idx] <= 0.5f) {
            uint8_t gx = s_goal_x[i];
            uint8_t gy = s_goal_y[i];
            
            int cell_idx = (int)gx * COL_OFFSET + (gy >> 5);
            uint32_t mask = 1u << (gy & 31);
            
            atomicOr(&s_goal_bitmap[cell_idx], mask);
        }
    }
    __syncthreads();

    __shared__ int s_env_offset;
    __shared__ int s_env_free_count;
    
    if (tid == 0) {
        s_env_offset = ctx.d_env_offsets[env_id];
        s_env_free_count = ctx.d_env_counts[env_id];
    }
    __syncthreads();

    for (int i = tid; i < agent_count; i += blockDim.x) {
        int global_idx = env_id * agent_count + i;

        if (ctx.rewards[global_idx] > 0.5f) {
            curandState local_state = ctx.rng_states[global_idx];
            
            int valid_gx = -1;
            int valid_gy = -1;
            
            int start_k = curand(&local_state) % s_env_free_count;
            int curr_k = start_k;
            
            for (int attempt = 0; attempt < s_env_free_count; ++attempt) {
                
                int list_idx = s_env_offset + curr_k;
                
                uint32_t flat_coord = __ldg(&ctx.d_free_cell_list[list_idx]);
                uint16_t try_x = (uint16_t)(flat_coord >> 16);
                uint16_t try_y = (uint16_t)(flat_coord & 0xFFFF);
                int bmp_idx = try_x * COL_OFFSET + (try_y >> 5);
                uint32_t mask = 1u << (try_y & 31);
                
                if ((s_goal_bitmap[bmp_idx] & mask) == 0) {
                    uint32_t old_val = atomicOr(&s_goal_bitmap[bmp_idx], mask);
                    
                    if ((old_val & mask) == 0) {
                        valid_gx = try_x;
                        valid_gy = try_y;
                        break; 
                    }
                }
                curr_k++;
                if (curr_k >= s_env_free_count) curr_k = 0;
            }
            
            if (valid_gx != -1) {
                ctx.goals_x[global_idx] = (uint16_t)valid_gx;
                ctx.goals_y[global_idx] = (uint16_t)valid_gy;
                s_goal_x[i] = static_cast<uint8_t>(valid_gx);
                s_goal_y[i] = static_cast<uint8_t>(valid_gy);
                ctx.d_goal_changed_flags[global_idx] = 1;
            }
            // 写回随机状态
            ctx.rng_states[global_idx] = local_state;
        }
    }

    __syncthreads();

    for (int i = tid; i < agent_count; i += blockDim.x) {
        int global_idx = env_id * agent_count + i;
        int my_x = static_cast<int>(s_next_x[i]);
        int my_y = static_cast<int>(s_next_y[i]);
        int gx = static_cast<int>(s_goal_x[i]);
        int gy = static_cast<int>(s_goal_y[i]);

        AgentState state;
        state.env_id = static_cast<uint16_t>(env_id);
        state.agent_id = static_cast<uint16_t>(i);
        state.pos_x = static_cast<uint16_t>(my_x);
        state.pos_y = static_cast<uint16_t>(my_y);
        state.target_x = static_cast<uint16_t>(gx);
        state.target_y = static_cast<uint16_t>(gy);
        state.action = static_cast<uint16_t>(ctx.actions[global_idx]);
        state.reset_flag = static_cast<uint16_t>(ctx.d_goal_changed_flags[global_idx]);
        ctx.d_packed_states[global_idx] = state;
    }
}



void launch_step_kernel(GridWorldContext ctx, cudaStream_t stream){
    int n_envs = ctx.n_envs;
    int n_agents = ctx.n_agents;
    int blocks = n_envs;
    
    int threads = 256; 
    size_t shm_size = 5 * n_agents * sizeof(uint8_t);
    step_kernel<<<blocks, threads, shm_size, stream>>>(ctx);
}

__global__ void load_state_kernel(
    GridWorldContext ctx,
    const int16_t* __restrict__ positions,
    const int16_t* __restrict__ goals)
{
    int env_id = blockIdx.x;
    int tid = threadIdx.x;
    if (env_id >= ctx.n_envs) return;

    for (int word = tid; word < MAP_OFFSET; word += blockDim.x) {
        ctx.d_grid_ocp[env_id * MAP_OFFSET + word] = 0;
    }
    __syncthreads();

    for (int agent_id = tid; agent_id < ctx.n_agents; agent_id += blockDim.x) {
        int global_idx = env_id * ctx.n_agents + agent_id;
        int coordinate_idx = 2 * global_idx;
        uint16_t x = static_cast<uint16_t>(positions[coordinate_idx]);
        uint16_t y = static_cast<uint16_t>(positions[coordinate_idx + 1]);
        uint16_t goal_x = static_cast<uint16_t>(goals[coordinate_idx]);
        uint16_t goal_y = static_cast<uint16_t>(goals[coordinate_idx + 1]);

        ctx.cur_x[global_idx] = x;
        ctx.cur_y[global_idx] = y;
        ctx.goals_x[global_idx] = goal_x;
        ctx.goals_y[global_idx] = goal_y;
        // External state loading bypasses the normal goal-sampling path.  Mark
        // every loaded goal dirty so the first update_derived_state() rebuilds
        // its energy map instead of consuming the zero-initialized buffer.
        ctx.d_goal_changed_flags[global_idx] = 1;

        AgentState state;
        state.env_id = static_cast<uint16_t>(env_id);
        state.agent_id = static_cast<uint16_t>(agent_id);
        state.pos_x = x;
        state.pos_y = y;
        state.target_x = goal_x;
        state.target_y = goal_y;
        state.action = 0;
        state.reset_flag = 1;
        ctx.d_packed_states[global_idx] = state;

        int word_idx = static_cast<int>(x) * COL_OFFSET + (y >> 5);
        atomicOr(
            &ctx.d_grid_ocp[env_id * MAP_OFFSET + word_idx],
            1u << (y & 31));
    }
}

void launch_load_state_kernel(
    GridWorldContext ctx,
    const torch::Tensor& positions,
    const torch::Tensor& goals,
    cudaStream_t stream)
{
    int threads = 256;
    load_state_kernel<<<ctx.n_envs, threads, 0, stream>>>(
        ctx,
        positions.data_ptr<int16_t>(),
        goals.data_ptr<int16_t>());
}

__global__ void pack_agent_state_kernel(GridWorldContext ctx, AgentState* d_states) {
    int env_id = blockIdx.x;
    int tid = threadIdx.x;
    if (env_id >= ctx.n_envs || tid >= ctx.n_agents) return;

    int global_idx = env_id * ctx.n_agents + tid;
    AgentState state;
    state.env_id = static_cast<uint16_t>(env_id);
    state.agent_id = static_cast<uint16_t>(tid);
    state.pos_x = ctx.cur_x[global_idx];
    state.pos_y = ctx.cur_y[global_idx];
    state.target_x = ctx.goals_x[global_idx];
    state.target_y = ctx.goals_y[global_idx];
    state.action = ctx.actions[global_idx];
    state.reset_flag = ctx.d_goal_changed_flags[global_idx];
    d_states[global_idx] = state;
}

void launch_pack_agent_state_kernel(GridWorldContext ctx, const torch::Tensor& packed_states) {
    TORCH_CHECK(packed_states.is_cuda(), "packed_states must be CUDA");
    TORCH_CHECK(packed_states.dtype() == torch::kInt16, "packed_states must be int16");
    TORCH_CHECK(packed_states.dim() == 2 && packed_states.size(1) == 8, "packed_states must be [N, 8]");
    int threads = min(ctx.n_agents, 256);
    int blocks = ctx.n_envs;
    auto d_states = reinterpret_cast<AgentState*>((uint16_t*)packed_states.data_ptr<int16_t>());
    pack_agent_state_kernel<<<blocks, threads>>>(ctx, d_states);
}

__global__ void pack_changed_agent_state_kernel(
    GridWorldContext ctx,
    const int64_t* __restrict__ d_goal_changed_prefix,
    AgentState* d_states)
{
    int env_id = blockIdx.x;
    int tid = threadIdx.x;
    if (env_id >= ctx.n_envs || tid >= ctx.n_agents) return;

    int global_idx = env_id * ctx.n_agents + tid;
    if (ctx.d_goal_changed_flags[global_idx] == 0) return;

    int64_t write_idx = d_goal_changed_prefix[global_idx] - 1;
    AgentState state;
    state.env_id = static_cast<uint16_t>(env_id);
    state.agent_id = static_cast<uint16_t>(tid);
    state.pos_x = ctx.cur_x[global_idx];
    state.pos_y = ctx.cur_y[global_idx];
    state.target_x = ctx.goals_x[global_idx];
    state.target_y = ctx.goals_y[global_idx];
    state.action = ctx.actions[global_idx];
    state.reset_flag = 1;
    d_states[write_idx] = state;
}

void launch_pack_changed_agent_state_kernel(
    GridWorldContext ctx,
    const torch::Tensor& goal_changed_prefix,
    const torch::Tensor& packed_states)
{
    TORCH_CHECK(goal_changed_prefix.is_cuda(), "goal_changed_prefix must be CUDA");
    TORCH_CHECK(goal_changed_prefix.dtype() == torch::kInt64, "goal_changed_prefix must be int64");
    TORCH_CHECK(packed_states.is_cuda(), "packed_states must be CUDA");
    TORCH_CHECK(packed_states.dtype() == torch::kInt16, "packed_states must be int16");
    TORCH_CHECK(packed_states.dim() == 2 && packed_states.size(1) == 8, "packed_states must be [N, 8]");
    int threads = min(ctx.n_agents, 256);
    int blocks = ctx.n_envs;
    auto d_states = reinterpret_cast<AgentState*>((uint16_t*)packed_states.data_ptr<int16_t>());
    pack_changed_agent_state_kernel<<<blocks, threads>>>(
        ctx,
        goal_changed_prefix.data_ptr<int64_t>(),
        d_states);
}
