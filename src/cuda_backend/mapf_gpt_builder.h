#pragma once

#include <torch/extension.h>

#include "grid_world_cuda.h"


struct MapfGPTCUDAContext {
    const uint32_t* d_maps;
    AgentState* d_states;
    uint16_t* d_histories;
    int64_t* d_labels;
    uint16_t* d_cost_to_go;
    int32_t* d_agent_at_cell;
    uint16_t* d_goal_cache;
    int32_t* d_tokens;
    bool* d_active_mask;
    int32_t* d_diagnostics;
    int n_envs;
    int n_agents;
};


class MapfGPTObservationBuilder {
public:
    int n_envs;
    int n_agents;

    torch::Tensor t_grid_compressed;
    torch::Tensor t_state_packed;
    torch::Tensor t_histories;
    torch::Tensor t_labels;
    torch::Tensor t_tokens;
    torch::Tensor t_active_mask;
    torch::Tensor t_cost_to_go;
    torch::Tensor t_agent_at_cell;
    torch::Tensor t_goal_cache;
    torch::Tensor t_diagnostics;
    torch::Tensor t_pyg_ptr;
    MapfGPTCUDAContext ctx;

    MapfGPTObservationBuilder(torch::Tensor grids, int n_agents);
    void build_tokens(const torch::Tensor& raw_rows);
    void build_tokens_from_state(
        const torch::Tensor& cur_x,
        const torch::Tensor& cur_y,
        const torch::Tensor& goal_x,
        const torch::Tensor& goal_y);
    void append_actions(const torch::Tensor& actions);
    void reset_histories();
};

void launch_mapf_gpt_unpack_kernel(
    const torch::Tensor& raw_rows,
    MapfGPTCUDAContext ctx,
    cudaStream_t stream);
void launch_mapf_gpt_state_unpack_kernel(
    const torch::Tensor& cur_x,
    const torch::Tensor& cur_y,
    const torch::Tensor& goal_x,
    const torch::Tensor& goal_y,
    MapfGPTCUDAContext ctx,
    cudaStream_t stream);
void launch_mapf_gpt_append_actions_kernel(
    const torch::Tensor& actions,
    MapfGPTCUDAContext ctx,
    cudaStream_t stream);
void launch_mapf_gpt_cost_to_go_kernel(
    MapfGPTCUDAContext ctx,
    cudaStream_t stream);
void launch_mapf_gpt_token_kernel(
    MapfGPTCUDAContext ctx,
    cudaStream_t stream);
