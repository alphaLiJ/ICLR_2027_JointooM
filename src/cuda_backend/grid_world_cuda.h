// grid_world_cuda.h
#pragma once

#include <stdint.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifndef MAP_W
#define MAP_W 128
#endif

#ifndef MAP_H
#define MAP_H 128
#endif

#if (MAP_H % 32) != 0
#error "MAP_H must be a multiple of 32 for compressed-column storage"
#endif

#define GRID_SIZE (MAP_W * MAP_H)
#define MAP_OFFSET (GRID_SIZE / 32)
#define COL_OFFSET (MAP_H / 32)
#define RADIUS 5
#define PYG_OBS_DIAM (2 * RADIUS + 3)
#define PYG_OBS_AREA (PYG_OBS_DIAM * PYG_OBS_DIAM)
#define MAGAT_COMM_RADIUS 7
#define MAGAT_COMM_RADIUS_SQ (MAGAT_COMM_RADIUS * MAGAT_COMM_RADIUS)

enum PygBuilderMode : int {
    PYG_BUILDER_MODE_AUTO = 0,
    PYG_BUILDER_MODE_LEGACY_FULLMAP = 1,
    PYG_BUILDER_MODE_LOCAL_GATHER = 2,
};

enum PygLocalGatherImplMode : int {
    PYG_LOCAL_GATHER_IMPL_AUTO = 0,
    PYG_LOCAL_GATHER_IMPL_SCALAR = 1,
    PYG_LOCAL_GATHER_IMPL_ASYNC_SM89PLUS = 2,
};

enum TaskMode : int {
    TASK_MODE_LIFELONG = 0,
    TASK_MODE_STANDARD_MAPF = 1,
};

// 需要有两个初始化函数: 1.有状态 2.无状态

struct StepOutput
{
    // size =[n_envs * n_agents, 4 * (2*obs_radius+1)^2]
    float* d_pyg_x;
    // size =[n_envs * n_agents, 2]
    float* d_pyg_pos;
};

struct AgentState;

struct EnergyMapContext
{
    // size =[n_envs, n_agents, MAP_W, MAP_H]
    uint32_t* d_maps;               // 地图数据，压缩为32位整数，每位表示一个格子是否被占用
    uint8_t* d_global_energy_maps;
};

struct GridWorldContext {
    // 1. 地图相关 (Read-only on progress generation)
    uint32_t* d_maps;               // 地图数据，压缩为32位整数，每位表示一个格子是否被占用
    uint32_t* d_grid_ocp;           // 当前占用状态，动态更新
    uint32_t* d_free_cell_list;     // 空闲位置列表，预先分配足够空间
    int* d_global_pool_ptr;         // 全局空闲位置池指针，指向d_free_cell_list中的下一个可用位置
    int* d_env_offsets;             // 每个环境起始偏移
    int* d_env_counts;              // 每个环境空闲数
    int* d_real_map_extents;        // [n_envs, 2] 每个环境的真实有效范围 (x_limit, y_limit)
    // cost-to-go 原图: size = [n_envs, n_agents, MAP_W, MAP_H]
    uint8_t* d_global_energy_maps; 

    // 2. 智能体状态 (Read-write)
    uint16_t* cur_x;
    uint16_t* cur_y;
    uint16_t* goals_x;
    uint16_t* goals_y;

    // 3. 压缩观测与奖励
    uint16_t* d_agents_obs;
    float* rewards;
    uint8_t* actions;
    uint8_t* d_goal_changed_flags;
    uint8_t* d_arrived;
    uint8_t* d_terminated;
    uint8_t* d_truncated;
    int32_t* d_step_counts;
    AgentState* d_packed_states;
    StepOutput step_output;
    int* d_edge_counts;

    // 4. 系统相关
    curandState* rng_states;
    int n_envs;
    uint32_t* max_free_cell_count;
    int n_agents;
    int pool_capacity;
    int current_step;
    int task_mode;
    int max_episode_steps;
    float clamp_value;

};

struct StatelessGridWorldContext {
    EnergyMapContext energy_map_ctx;  // 包含地图和能量图相关的指针
    StepOutput step_output;

    uint32_t* d_grid_ocp;
    AgentState* d_packed_states;
    uint16_t* goals_x;
    uint16_t* goals_y;
    int n_envs;
    int n_agents;
    float clamp_value;
    int* d_edge_counts;          // [n_envs * n_agents] per-agent edge count workspace
    int builder_mode;
    int local_gather_impl_mode;
    int cuda_cc_major;
    int cuda_cc_minor;
    int supports_async_local_gather;
};

struct AgentState {
    uint16_t env_id;
    uint16_t agent_id;
    uint16_t pos_x;
    uint16_t pos_y;
    uint16_t target_x;
    uint16_t target_y;
    uint16_t action;
    uint16_t reset_flag;
};

// Kernel 声明
__global__ void cache_map_free_cells_kernel(GridWorldContext ctx);
__global__ void init_agents_from_cache_kernel(GridWorldContext ctx);
__global__ void step_kernel(GridWorldContext ctx);
__global__ void setup_rng_kernel(
    curandState* rng_states, 
    unsigned long seed, 
    int total_agents
);
__global__ void init_agents_kernel(GridWorldContext ctx);
__global__ void decode_to_uint8_kernel(
    const uint32_t* __restrict__ compressed_obs, 
    uint8_t* __restrict__ output,               
    int total_pixels,                           
    int diameter                              
);
__global__ void generate_imitation_obs_fullmap_shared_kernel(
    const AgentState* __restrict__ d_states, 
    StatelessGridWorldContext ctx);

__global__ void generate_energy_map_kernel(
    EnergyMapContext ctx,
    int num_agents,
    int length,
    const AgentState* __restrict__ d_states,
    bool incremental_only);
