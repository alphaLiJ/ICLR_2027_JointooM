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
    PYG_LOCAL_GATHER_IMPL_WARP_COOPERATIVE = 3,
};

enum TaskMode : int {
    TASK_MODE_LIFELONG = 0,
    TASK_MODE_STANDARD_MAPF = 1,
    TASK_MODE_COSTTRACER = 2,
};

// Two initialization functions are needed: stateful and stateless

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
    uint32_t* d_maps;               // Map packed into 32-bit integers, one occupancy bit per cell
    uint8_t* d_global_energy_maps;
};

struct GridWorldContext {
    // 1. Map data (read-only during progress generation)
    uint32_t* d_maps;               // Map packed into 32-bit integers, one occupancy bit per cell
    uint32_t* d_grid_ocp;           // Current occupancy, updated dynamically
    uint32_t* d_free_cell_list;     // Free-cell list with sufficient preallocated capacity
    int* d_global_pool_ptr;         // Global free-cell pool pointer: the next available entry in d_free_cell_list
    int* d_env_offsets;             // Starting offset for each environment
    int* d_env_counts;              // Number of free cells in each environment
    int* d_real_map_extents;        // [n_envs, 2] Valid extent of each environment (x_limit, y_limit)
    // Original cost-to-go map: size = [n_envs, n_agents, MAP_W, MAP_H]
    uint8_t* d_global_energy_maps; 

    // 2. Agent states (read/write)
    uint16_t* cur_x;
    uint16_t* cur_y;
    uint16_t* goals_x;
    uint16_t* goals_y;

    // 3. Compressed observations and rewards
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

    // 4. System state
    curandState* rng_states;
    int n_envs;
    uint32_t* max_free_cell_count;
    int n_agents;
    int pool_capacity;
    int current_step;
    int task_mode;
    int max_episode_steps;
    float clamp_value;
    const int* d_goal_candidates = nullptr;
    const uint16_t* d_candidate_cells = nullptr;
    const int* d_candidate_counts = nullptr;
    int* d_goal_cursors = nullptr;
    int candidate_length = 0;
    int candidate_cell_stride = 0;

};

struct StatelessGridWorldContext {
    EnergyMapContext energy_map_ctx;  // Pointers for map and energy-map data
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

// Kernel declarations
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
