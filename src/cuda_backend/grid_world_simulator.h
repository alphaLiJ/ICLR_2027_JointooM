// grid_world_simulator.h
#pragma once

#include <cstddef>
#include <string>
#include <torch/extension.h>
#include "grid_world_cuda.h" 

class GridWorldSimulator {
public:
    GridWorldContext ctx;  // 注意：这个 struct 是纯 POD，可在 host 初始化
    StatelessGridWorldContext pyg_ctx;  // reuse stateless PyG generation path on stateful state

    // Host 端 Tensor 引用（用于内存管理 & RAII）
    torch::Tensor t_grid_compressed;
    torch::Tensor t_free_cell_list, t_pool_ptr, t_offsets, t_counts, t_grid_ocp;
    torch::Tensor t_real_map_extents;
    torch::Tensor t_cur_x, t_cur_y, t_goals_x, t_goals_y;
    torch::Tensor t_agents_obs, t_rewards, t_actions, t_rng_states;
    torch::Tensor t_max_free_cell_count;
    torch::Tensor t_goal_changed_flags;
    torch::Tensor t_arrived;
    torch::Tensor t_terminated;
    torch::Tensor t_truncated;
    torch::Tensor t_step_counts;
    torch::Tensor t_goal_changed_prefix;
    torch::Tensor t_changed_state_packed;
    torch::Tensor t_goal_changed_count;

    // Stateful PyG / energy-map tensors
    torch::Tensor t_energy_maps;
    torch::Tensor t_state_packed;
    torch::Tensor t_pyg_x;
    torch::Tensor t_pyg_pos;
    torch::Tensor t_pyg_edge_index;
    torch::Tensor t_pyg_edge_attr;
    torch::Tensor t_pyg_batch;
    torch::Tensor t_pyg_ptr;
    torch::Tensor t_pyg_num_edges;
    torch::Tensor t_edge_counts;
    torch::Tensor t_pyg_edge_index_storage;
    torch::Tensor t_pyg_edge_attr_storage;
    torch::Tensor t_pyg_edge_prefix;
    torch::Tensor t_scan_temp_storage;
    size_t scan_temp_storage_bytes = 0;
    int max_edges_per_env;

    GridWorldSimulator(
        torch::Tensor grids,
        int n_agents,
        int num_features,
        int pool_capacity,
        unsigned long seed,
        const std::string& task_mode = "lifelong",
        int max_episode_steps = 256);
    void update_actions(const torch::Tensor& new_actions);
    void init_rng(unsigned long seed);
    void run_initialization();
    void step_sim_only();
    void step_compact_only();
    void update_derived_state();
    void build_magat_plus_nodes();
    void finalize_magat_plus_graph();
    void build_magat_plus_inputs();
    void materialize_pyg_inputs();
    void step_and_build_pyg();
    void step();
    void set_real_map_extents(const torch::Tensor& real_map_extents);
    void load_state(
        const torch::Tensor& positions,
        const torch::Tensor& goals,
        const torch::Tensor& arrived,
        const torch::Tensor& step_counts);
    std::string get_task_mode() const;
    int get_max_episode_steps() const;
    void set_pyg_builder_mode(const std::string& mode);
    std::string get_pyg_builder_mode() const;
    void set_pyg_local_gather_impl(const std::string& mode);
    std::string get_pyg_local_gather_impl() const;
    std::string get_resolved_pyg_builder_impl() const;
    
    torch::Tensor decode_to_uint8();

    // print functions
    void print_t_grid_compressed();
    void print_t_free_cell_list();
    void print_t_pool_ptr();
    void print_t_offsets();
    void print_t_counts();
    void print_t_grid_ocp();
    void print_t_cur_x();
    void print_t_cur_y();
    void print_t_goals_x();
    void print_t_goals_y();
    void print_t_agents_obs();
    void print_t_rewards();
    void print_t_actions();
    void print_t_max_free_cell_count();
};

class StatelessGridWorldSimulator{
public:
    StatelessGridWorldContext ctx;
    torch::Tensor t_grid_compressed;
    torch::Tensor t_grid_ocp;
    torch::Tensor t_state_packed;
    torch::Tensor t_goals_x, t_goals_y;
    torch::Tensor t_energy_maps;  // [n_envs, n_agents, MAP_W, MAP_H] uint8

    // PyG node tensors
    torch::Tensor t_pyg_x;          // [n_envs * n_agents, 484] float32
    torch::Tensor t_pyg_pos;        // [n_envs * n_agents, 2] float32

    // PyG-ready graph tensors
    torch::Tensor t_pyg_edge_index;          // [2, E_valid] int64
    torch::Tensor t_pyg_edge_attr;           // [E_valid, 3] float32
    torch::Tensor t_pyg_batch;               // [n_envs * n_agents] int64
    torch::Tensor t_pyg_ptr;                 // [n_envs + 1] int64
    torch::Tensor t_pyg_num_edges;           // [1] int64

    // Internal storage / workspace
    torch::Tensor t_edge_counts;             // [n_envs * n_agents] int32
    torch::Tensor t_pyg_edge_index_storage;  // [2, TotalEdges] int64
    torch::Tensor t_pyg_edge_attr_storage;   // [TotalEdges, 3] float32
    torch::Tensor t_pyg_edge_prefix;         // [n_envs * n_agents + 1] int64
    torch::Tensor t_scan_temp_storage;       // CUB exclusive-scan temp storage
    size_t scan_temp_storage_bytes = 0;
    int max_edges_per_env;                   // Pre-allocated max edges per env

    StatelessGridWorldSimulator(torch::Tensor grids, int n_agents, int num_features);
    void refresh_compact_state_from_raw_batch(const torch::Tensor& states_tensor);
    void materialize_pyg_inputs();
    void update_derived_state(const torch::Tensor& states_tensor, const int length);
    void build_magat_plus_nodes(const torch::Tensor& states_tensor);
    void finalize_magat_plus_graph();
    void build_magat_plus_inputs(const torch::Tensor& states_tensor);
    void setup_imitation_obs(const torch::Tensor& states_tensor);
    void update_energy_maps(const torch::Tensor& states_tensor, const int length);
    void set_pyg_builder_mode(const std::string& mode);
    std::string get_pyg_builder_mode() const;
    void set_pyg_local_gather_impl(const std::string& mode);
    std::string get_pyg_local_gather_impl() const;
    std::string get_resolved_pyg_builder_impl() const;
};

void launch_energy_map_kernel(
    EnergyMapContext ctx,
    int num_agents,
    int length,
    const torch::Tensor& d_states,
    bool incremental_only = false);
void launch_setup_rng_kernel(curandState* states, unsigned long seed, int total_agents);
void launch_init_agents_kernel(GridWorldContext ctx, dim3 blocks, dim3 threads);
void launch_cache_map_free_cells_kernel(GridWorldContext ctx, dim3 blocks, dim3 threads);
void launch_decode_to_uint8_kernel(uint16_t* compressed_obs, int blocks, int threads,
                                    uint8_t* output, int total_pixels, int diameter);
void launch_step_kernel(GridWorldContext ctx, cudaStream_t stream);
void launch_load_state_kernel(
    GridWorldContext ctx,
    const torch::Tensor& positions,
    const torch::Tensor& goals,
    cudaStream_t stream);
void launch_encode_full_map_kernel(const uint32_t* input_map, uint32_t* compressed_map, 
                                   int batch_size, int blockSize = 256);
void launch_encode_full_map_kernel_on_stream(
    const uint32_t* input_map,
    uint32_t* compressed_map,
    int batch_size,
    int block_size,
    cudaStream_t stream);

void launch_imitation_obs_generator(const torch::Tensor& states_tensor, StatelessGridWorldContext ctx);
void launch_imitation_obs_generator_from_packed_states(
    const AgentState* d_states,
    StatelessGridWorldContext ctx);
void launch_refresh_occ_bitmap_from_packed_states(
    const torch::Tensor& packed_states,
    StatelessGridWorldContext ctx);
void launch_fill_pyg_ctg_channel(const torch::Tensor& states_tensor, StatelessGridWorldContext ctx);
void launch_generate_pyg_edges_kernel(
    const torch::Tensor& node_prefix,
    const torch::Tensor& node_pos,
    torch::Tensor& pyg_edge_index,
    torch::Tensor& pyg_edge_attr,
    int agent_count,
    int n_envs,
    int total_nodes);
void launch_pack_agent_state_kernel(GridWorldContext ctx, const torch::Tensor& packed_states);
void launch_pack_changed_agent_state_kernel(
    GridWorldContext ctx,
    const torch::Tensor& goal_changed_prefix,
    const torch::Tensor& packed_states);
size_t query_exclusive_scan_edge_counts_temp_storage_bytes(
    const torch::Tensor& edge_counts,
    const torch::Tensor& edge_prefix);
void launch_exclusive_scan_edge_counts(
    const torch::Tensor& edge_counts,
    torch::Tensor& edge_prefix,
    torch::Tensor& num_edges,
    const torch::Tensor& temp_storage);
