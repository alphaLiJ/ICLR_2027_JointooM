#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <array>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <stdexcept>
#include <vector>
#include "utils.h"
#include "grid_world_simulator.h"

namespace {

constexpr size_t kLegacyFullmapSharedSmemLimitBytes = 96 * 1024;
constexpr bool kHasAsyncLocalGatherKernel = true;

int parse_pyg_builder_mode_string(const std::string& mode) {
    if (mode == "auto") {
        return PYG_BUILDER_MODE_AUTO;
    }
    if (mode == "legacy_fullmap") {
        return PYG_BUILDER_MODE_LEGACY_FULLMAP;
    }
    if (mode == "local_gather") {
        return PYG_BUILDER_MODE_LOCAL_GATHER;
    }
    throw std::invalid_argument(
        "Unknown pyg builder mode '" + mode + "'. Expected one of: auto, legacy_fullmap, local_gather");
}

int parse_task_mode_string(const std::string& mode) {
    if (mode == "lifelong") {
        return TASK_MODE_LIFELONG;
    }
    if (mode == "standard_mapf") {
        return TASK_MODE_STANDARD_MAPF;
    }
    throw std::invalid_argument(
        "Unknown task mode '" + mode + "'. Expected one of: lifelong, standard_mapf");
}

std::string task_mode_to_string(int mode) {
    switch (mode) {
        case TASK_MODE_LIFELONG:
            return "lifelong";
        case TASK_MODE_STANDARD_MAPF:
            return "standard_mapf";
        default:
            throw std::invalid_argument("Unknown internal task mode value");
    }
}

std::string pyg_builder_mode_to_string(int mode) {
    switch (mode) {
        case PYG_BUILDER_MODE_AUTO:
            return "auto";
        case PYG_BUILDER_MODE_LEGACY_FULLMAP:
            return "legacy_fullmap";
        case PYG_BUILDER_MODE_LOCAL_GATHER:
            return "local_gather";
        default:
            throw std::invalid_argument("Unknown internal pyg builder mode value");
    }
}

int parse_pyg_local_gather_impl_string(const std::string& mode) {
    if (mode == "auto") {
        return PYG_LOCAL_GATHER_IMPL_AUTO;
    }
    if (mode == "scalar") {
        return PYG_LOCAL_GATHER_IMPL_SCALAR;
    }
    if (mode == "async_sm89plus") {
        return PYG_LOCAL_GATHER_IMPL_ASYNC_SM89PLUS;
    }
    throw std::invalid_argument(
        "Unknown pyg local gather impl '" + mode + "'. Expected one of: auto, scalar, async_sm89plus");
}

std::string pyg_local_gather_impl_to_string(int mode) {
    switch (mode) {
        case PYG_LOCAL_GATHER_IMPL_AUTO:
            return "auto";
        case PYG_LOCAL_GATHER_IMPL_SCALAR:
            return "scalar";
        case PYG_LOCAL_GATHER_IMPL_ASYNC_SM89PLUS:
            return "async_sm89plus";
        default:
            throw std::invalid_argument("Unknown internal local-gather impl mode value");
    }
}

void initialize_local_gather_capability(StatelessGridWorldContext& ctx) {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDevice failed: ") + cudaGetErrorString(err));
    }

    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(err));
    }

    ctx.cuda_cc_major = prop.major;
    ctx.cuda_cc_minor = prop.minor;
    ctx.supports_async_local_gather =
        (prop.major > 8 || (prop.major == 8 && prop.minor >= 9)) ? 1 : 0;
    ctx.local_gather_impl_mode = PYG_LOCAL_GATHER_IMPL_AUTO;
}

size_t legacy_fullmap_builder_smem_bytes_host(int agent_count) {
    size_t temp_map_bytes = MAP_W * (MAP_H + 1) * sizeof(int16_t);
    size_t agent_bytes = 4 * static_cast<size_t>(agent_count) * sizeof(int16_t);
    return temp_map_bytes + agent_bytes;
}

bool can_use_legacy_fullmap_builder_host(const StatelessGridWorldContext& ctx) {
    return MAP_W == 128
        && MAP_H == 128
        && ctx.n_agents <= 256
        && legacy_fullmap_builder_smem_bytes_host(ctx.n_agents) <= kLegacyFullmapSharedSmemLimitBytes;
}

void validate_requested_pyg_builder_mode(const StatelessGridWorldContext& ctx, int builder_mode) {
    if (builder_mode == PYG_BUILDER_MODE_LEGACY_FULLMAP && !can_use_legacy_fullmap_builder_host(ctx)) {
        throw std::runtime_error(
            "legacy_fullmap builder mode requires n_agents <= 256 and shared memory usage within the legacy budget");
    }
}

int resolve_local_gather_impl_mode(const StatelessGridWorldContext& ctx) {
    if (ctx.local_gather_impl_mode == PYG_LOCAL_GATHER_IMPL_SCALAR) {
        return PYG_LOCAL_GATHER_IMPL_SCALAR;
    }
    if (ctx.local_gather_impl_mode == PYG_LOCAL_GATHER_IMPL_ASYNC_SM89PLUS) {
        if (ctx.supports_async_local_gather && kHasAsyncLocalGatherKernel) {
            return PYG_LOCAL_GATHER_IMPL_ASYNC_SM89PLUS;
        }
        return PYG_LOCAL_GATHER_IMPL_SCALAR;
    }

    if (ctx.supports_async_local_gather && kHasAsyncLocalGatherKernel) {
        return PYG_LOCAL_GATHER_IMPL_ASYNC_SM89PLUS;
    }
    return PYG_LOCAL_GATHER_IMPL_SCALAR;
}

std::string resolved_pyg_builder_impl_to_string(const StatelessGridWorldContext& ctx) {
    switch (ctx.builder_mode) {
        case PYG_BUILDER_MODE_AUTO:
            if (can_use_legacy_fullmap_builder_host(ctx)) {
                return "legacy_fullmap";
            }
            return resolve_local_gather_impl_mode(ctx) == PYG_LOCAL_GATHER_IMPL_ASYNC_SM89PLUS
                ? "local_gather_async_sm89plus"
                : "local_gather_scalar";
        case PYG_BUILDER_MODE_LEGACY_FULLMAP:
            return "legacy_fullmap";
        case PYG_BUILDER_MODE_LOCAL_GATHER:
            return resolve_local_gather_impl_mode(ctx) == PYG_LOCAL_GATHER_IMPL_ASYNC_SM89PLUS
                ? "local_gather_async_sm89plus"
                : "local_gather_scalar";
        default:
            throw std::invalid_argument("Unknown internal pyg builder mode value");
    }
}

}  // namespace

// 注意: 在cuda 上运行的代码中, 有取地址操作的变量一定要位于 cuda 上
GridWorldSimulator::GridWorldSimulator(
    torch::Tensor grids,
    int n_agents,
    int num_features,
    int pool_capacity,
    unsigned long seed,
    const std::string& task_mode,
    int max_episode_steps)
{
    TORCH_CHECK(grids.is_cuda(), "grids must be CUDA");
    TORCH_CHECK(grids.dtype() == torch::kInt32, "grids must be int32");
    TORCH_CHECK(grids.is_contiguous(), "grids must be contiguous");
    TORCH_CHECK(
        grids.dim() == 3 && grids.size(1) == MAP_W && grids.size(2) == MAP_H,
        "grids must have shape [n_envs, ", MAP_W, ", ", MAP_H, "]"
    );
    c10::cuda::CUDAGuard device_guard(grids.device());

    int n_envs = grids.size(0);
    
    ctx.n_envs = n_envs;
    // ctx.max_free_cell_count = 0;
    ctx.n_agents = n_agents;
    ctx.pool_capacity = pool_capacity;
    ctx.current_step = 0; // 初始化步数
    ctx.task_mode = parse_task_mode_string(task_mode);
    TORCH_CHECK(max_episode_steps > 0, "max_episode_steps must be positive");
    ctx.max_episode_steps = max_episode_steps;
    ctx.clamp_value = 1.0f;
    initialize_local_gather_capability(pyg_ctx);
    
    // 定义 Tensor 配置 (CUDA, 对应的类型)
    auto opt_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(grids.device());  // for uint8_t
    auto opt_i16 = torch::TensorOptions().dtype(torch::kInt16).device(grids.device());  // for int16_t
    auto opt_u16 = torch::TensorOptions().dtype(torch::kInt16).device(grids.device());   // for uint16_t (use Int16, same byte width)
    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(grids.device());  // for int, uint32_t
    auto opt_u32 = torch::TensorOptions().dtype(torch::kInt32).device(grids.device());   // for uint32_t (use Int32, same byte width)
    auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(grids.device()); // for float
    auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64).device(grids.device());
    
    t_max_free_cell_count = torch::zeros({1}, opt_u32);
    ctx.max_free_cell_count = (uint32_t*)t_max_free_cell_count.data_ptr<int32_t>();

    t_grid_compressed = torch::zeros({n_envs, MAP_OFFSET}, opt_u32);
    ctx.d_maps = (uint32_t*)t_grid_compressed.data_ptr<int32_t>();

    t_free_cell_list = torch::empty({pool_capacity}, opt_u32);
    ctx.d_free_cell_list = (uint32_t*)t_free_cell_list.data_ptr<int32_t>();

    t_pool_ptr = torch::zeros({1}, opt_i32);
    ctx.d_global_pool_ptr = t_pool_ptr.data_ptr<int>();

    t_offsets = torch::full({n_envs}, -1, opt_i32);
    ctx.d_env_offsets = t_offsets.data_ptr<int>();

    t_counts = torch::zeros({n_envs}, opt_i32);
    ctx.d_env_counts = t_counts.data_ptr<int>();

    t_real_map_extents = torch::empty({n_envs, 2}, opt_i32);
    t_real_map_extents.select(1, 0).fill_(MAP_W);
    t_real_map_extents.select(1, 1).fill_(MAP_H);
    ctx.d_real_map_extents = t_real_map_extents.data_ptr<int>();

    t_grid_ocp = torch::zeros({n_envs, MAP_OFFSET}, opt_i32);
    ctx.d_grid_ocp = (uint32_t*)t_grid_ocp.data_ptr<int>();

    // 2. 智能体状态初始化 (Agent States)
    // 注意：Torch C++ API 中 kInt16 对应 short，与 uint16_t 字节数相同(2bytes)
    t_cur_x = torch::empty({n_envs, n_agents}, opt_u16);
    ctx.cur_x = (uint16_t*)t_cur_x.data_ptr<int16_t>();
    
    t_cur_y = torch::empty({n_envs, n_agents}, opt_u16);
    ctx.cur_y = (uint16_t*)t_cur_y.data_ptr<int16_t>();

    t_goals_x = torch::empty({n_envs, n_agents}, opt_u16);
    ctx.goals_x = (uint16_t*)t_goals_x.data_ptr<int16_t>();

    t_goals_y = torch::empty({n_envs, n_agents}, opt_u16);
    ctx.goals_y = (uint16_t*)t_goals_y.data_ptr<int16_t>();

    // 3. 观测、奖励与动作 (Obs, Reward, Action)

    // 补充一个 初始化 ctx.d_agents_obs 的操作, 每个智能体拥有 16 个(注意关注 step 中逻辑是否正确)
    t_agents_obs = torch::zeros({n_envs*n_agents, num_features, RADIUS * 2 + 1}, opt_u16); 
    ctx.d_agents_obs = (uint16_t*)t_agents_obs.data_ptr<int16_t>();

    t_rewards = torch::zeros({n_envs, n_agents}, opt_f32);
    ctx.rewards = t_rewards.data_ptr<float>();

    t_actions = torch::zeros({n_envs, n_agents}, opt_u8);
    ctx.actions = t_actions.data_ptr<uint8_t>();
    t_goal_changed_flags = torch::zeros({n_envs, n_agents}, opt_u8);
    ctx.d_goal_changed_flags = t_goal_changed_flags.data_ptr<uint8_t>();
    t_arrived = torch::zeros({n_envs, n_agents}, opt_u8);
    ctx.d_arrived = t_arrived.data_ptr<uint8_t>();
    t_terminated = torch::zeros({n_envs}, opt_u8);
    ctx.d_terminated = t_terminated.data_ptr<uint8_t>();
    t_truncated = torch::zeros({n_envs}, opt_u8);
    ctx.d_truncated = t_truncated.data_ptr<uint8_t>();
    t_step_counts = torch::zeros({n_envs}, opt_i32);
    ctx.d_step_counts = t_step_counts.data_ptr<int32_t>();

    // Stateful PyG / energy-map path
    t_energy_maps = torch::zeros({n_envs, n_agents, MAP_W, MAP_H}, opt_u8);
    ctx.d_global_energy_maps = t_energy_maps.data_ptr<uint8_t>();
    pyg_ctx.energy_map_ctx.d_global_energy_maps = t_energy_maps.data_ptr<uint8_t>();
    pyg_ctx.energy_map_ctx.d_maps = ctx.d_maps;
    pyg_ctx.d_grid_ocp = ctx.d_grid_ocp;
    pyg_ctx.n_envs = n_envs;
    pyg_ctx.n_agents = n_agents;
    pyg_ctx.goals_x = ctx.goals_x;
    pyg_ctx.goals_y = ctx.goals_y;
    pyg_ctx.clamp_value = ctx.clamp_value;
    pyg_ctx.builder_mode = PYG_BUILDER_MODE_AUTO;

    int total_agents = n_envs * n_agents;
    t_state_packed = torch::zeros({total_agents, 8}, opt_i16);
    ctx.d_packed_states = reinterpret_cast<AgentState*>((uint16_t*)t_state_packed.data_ptr<int16_t>());
    t_pyg_x = torch::zeros({total_agents, 4 * PYG_OBS_AREA}, opt_f32);
    ctx.step_output.d_pyg_x = t_pyg_x.data_ptr<float>();
    pyg_ctx.step_output.d_pyg_x = t_pyg_x.data_ptr<float>();
    t_pyg_pos = torch::zeros({total_agents, 2}, opt_f32);
    ctx.step_output.d_pyg_pos = t_pyg_pos.data_ptr<float>();
    pyg_ctx.step_output.d_pyg_pos = t_pyg_pos.data_ptr<float>();

    max_edges_per_env = n_agents * n_agents;
    int total_max_edges = n_envs * max_edges_per_env;
    t_edge_counts = torch::zeros({total_agents}, opt_i32);
    ctx.d_edge_counts = t_edge_counts.data_ptr<int>();
    pyg_ctx.d_edge_counts = t_edge_counts.data_ptr<int>();
    t_pyg_edge_index_storage = torch::zeros({2, total_max_edges}, opt_i64);
    t_pyg_edge_attr_storage = torch::zeros({total_max_edges, 3}, opt_f32);
    t_pyg_edge_index = t_pyg_edge_index_storage.narrow(1, 0, 0);
    t_pyg_edge_attr = t_pyg_edge_attr_storage.narrow(0, 0, 0);
    t_pyg_edge_prefix = torch::zeros({total_agents + 1}, opt_i64);
    t_pyg_num_edges = torch::zeros({1}, opt_i64);
    scan_temp_storage_bytes = query_exclusive_scan_edge_counts_temp_storage_bytes(
        t_edge_counts, t_pyg_edge_prefix.narrow(0, 0, total_agents));
    t_scan_temp_storage = torch::empty({static_cast<long>(scan_temp_storage_bytes)}, opt_u8);
    t_goal_changed_prefix = torch::zeros({total_agents}, opt_i64);
    t_changed_state_packed = torch::zeros({total_agents, 8}, opt_i16);
    t_goal_changed_count = torch::zeros({1}, opt_i64);

    std::vector<int64_t> batch_host(total_agents);
    for (int env = 0; env < n_envs; ++env) {
        for (int agent = 0; agent < n_agents; ++agent) {
            batch_host[env * n_agents + agent] = env;
        }
    }
    t_pyg_batch = torch::empty({total_agents}, opt_i64);
    cudaMemcpy(t_pyg_batch.data_ptr<int64_t>(), batch_host.data(),
               total_agents * sizeof(int64_t), cudaMemcpyHostToDevice);

    std::vector<int64_t> ptr_host(n_envs + 1);
    for (int env = 0; env <= n_envs; ++env) {
        ptr_host[env] = static_cast<int64_t>(env) * n_agents;
    }
    t_pyg_ptr = torch::empty({n_envs + 1}, opt_i64);
    cudaMemcpy(t_pyg_ptr.data_ptr<int64_t>(), ptr_host.data(),
               (n_envs + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
    // -------------------------------------------------
    // 4. RNG 状态
    // -------------------------------------------------
    int64_t rng_size = n_envs * std::max(n_agents, 32);
    t_rng_states = torch::empty({rng_size * (int64_t)sizeof(curandState)}, opt_u8);
    ctx.rng_states = (curandState*)t_rng_states.data_ptr();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Data init failed: ") + cudaGetErrorString(err));
    }
    launch_encode_full_map_kernel((uint32_t*)grids.data_ptr<int32_t>(), ctx.d_maps, n_envs);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("encode_full_map kernel failed: ") + cudaGetErrorString(err));
    }
    init_rng(seed);
}

StatelessGridWorldSimulator::StatelessGridWorldSimulator(torch::Tensor grids, int n_agents, int num_features = 3)
{
    TORCH_CHECK(
        grids.dim() == 3 && grids.size(1) == MAP_W && grids.size(2) == MAP_H,
        "grids must have shape [n_envs, ", MAP_W, ", ", MAP_H, "]"
    );

    int n_envs = grids.size(0);
    
    ctx.n_envs = n_envs;
    ctx.n_agents = n_agents;
    ctx.clamp_value = 1.0f;
    ctx.builder_mode = PYG_BUILDER_MODE_AUTO;
    initialize_local_gather_capability(ctx);
    
    // 定义 Tensor 配置 (CUDA, 对应的类型)
    auto opt_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);  // for uint8_t
    auto opt_i16 = torch::TensorOptions().dtype(torch::kInt16).device(torch::kCUDA);  // for int16_t
    auto opt_u16 = torch::TensorOptions().dtype(torch::kInt16).device(torch::kCUDA);   // for uint16_t (use Int16, same byte width)
    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);  // for int, uint32_t
    auto opt_u32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);   // for uint32_t (use Int32, same byte width)
    auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA); // for float
    auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    
    t_grid_compressed = torch::zeros({n_envs, MAP_OFFSET}, opt_u32);
    ctx.energy_map_ctx.d_maps = (uint32_t*)t_grid_compressed.data_ptr<int32_t>();
    t_grid_ocp = torch::zeros({n_envs, MAP_OFFSET}, opt_i32);
    ctx.d_grid_ocp = (uint32_t*)t_grid_ocp.data_ptr<int>();

    // Allocate energy maps: [n_envs, n_agents, MAP_W, MAP_H] uint8
    t_energy_maps = torch::zeros({n_envs, n_agents, MAP_W, MAP_H}, opt_u8);
    ctx.energy_map_ctx.d_global_energy_maps = t_energy_maps.data_ptr<uint8_t>();

    t_goals_x = torch::empty({n_envs, n_agents}, opt_u16);
    ctx.goals_x = (uint16_t*)t_goals_x.data_ptr<int16_t>();

    t_goals_y = torch::empty({n_envs, n_agents}, opt_u16);
    ctx.goals_y = (uint16_t*)t_goals_y.data_ptr<int16_t>();

    // Allocate StepOutput tensors for Python access
    int total_agents = n_envs * n_agents;
    t_state_packed = torch::zeros({total_agents, 8}, opt_i16);
    ctx.d_packed_states = reinterpret_cast<AgentState*>((uint16_t*)t_state_packed.data_ptr<int16_t>());
    t_pyg_x = torch::zeros({total_agents, 4 * PYG_OBS_AREA}, opt_f32);
    ctx.step_output.d_pyg_x = t_pyg_x.data_ptr<float>();
    t_pyg_pos = torch::zeros({total_agents, 2}, opt_f32);
    ctx.step_output.d_pyg_pos = t_pyg_pos.data_ptr<float>();

    // Internal edge-count workspace and final PyG edge storage
    max_edges_per_env = n_agents * n_agents;  // worst case: all-to-all
    int total_max_edges = n_envs * max_edges_per_env;
    t_edge_counts = torch::zeros({n_envs * n_agents}, opt_i32);
    ctx.d_edge_counts = t_edge_counts.data_ptr<int>();

    t_pyg_edge_index_storage = torch::zeros({2, total_max_edges}, opt_i64);
    t_pyg_edge_attr_storage = torch::zeros({total_max_edges, 3}, opt_f32);
    t_pyg_edge_index = t_pyg_edge_index_storage.narrow(1, 0, 0);
    t_pyg_edge_attr = t_pyg_edge_attr_storage.narrow(0, 0, 0);
    t_pyg_edge_prefix = torch::zeros({total_agents + 1}, opt_i64);
    t_pyg_num_edges = torch::zeros({1}, opt_i64);
    scan_temp_storage_bytes = query_exclusive_scan_edge_counts_temp_storage_bytes(
        t_edge_counts, t_pyg_edge_prefix.narrow(0, 0, total_agents));
    t_scan_temp_storage = torch::empty({static_cast<long>(scan_temp_storage_bytes)}, opt_u8);

    std::vector<int64_t> batch_host(total_agents);
    for (int env = 0; env < n_envs; ++env) {
        for (int agent = 0; agent < n_agents; ++agent) {
            batch_host[env * n_agents + agent] = env;
        }
    }
    t_pyg_batch = torch::empty({total_agents}, opt_i64);
    cudaMemcpy(t_pyg_batch.data_ptr<int64_t>(), batch_host.data(),
               total_agents * sizeof(int64_t), cudaMemcpyHostToDevice);

    std::vector<int64_t> ptr_host(n_envs + 1);
    for (int env = 0; env <= n_envs; ++env) {
        ptr_host[env] = static_cast<int64_t>(env) * n_agents;
    }
    t_pyg_ptr = torch::empty({n_envs + 1}, opt_i64);
    cudaMemcpy(t_pyg_ptr.data_ptr<int64_t>(), ptr_host.data(),
               (n_envs + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);

    launch_encode_full_map_kernel((uint32_t*)grids.data_ptr<int32_t>(), ctx.energy_map_ctx.d_maps, n_envs);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("encode_full_map kernel failed: ") + cudaGetErrorString(err));
    }
}

void StatelessGridWorldSimulator::refresh_compact_state_from_raw_batch(const torch::Tensor& states_tensor) {
    TORCH_CHECK(states_tensor.is_cuda(), "states_tensor must be on CUDA device");
    TORCH_CHECK(states_tensor.dtype() == torch::kInt16, "states_tensor must be int16");
    TORCH_CHECK(states_tensor.dim() == 2, "states_tensor must be 2D (N, 8)");
    TORCH_CHECK(states_tensor.size(1) == 8, "states_tensor feature dimension must be 8");
    TORCH_CHECK(states_tensor.is_contiguous(), "states_tensor must be contiguous");
    TORCH_CHECK(
        states_tensor.size(0) <= t_state_packed.size(0),
        "states_tensor length exceeds preallocated stateless capacity"
    );

    t_state_packed.fill_(-1);
    t_state_packed.narrow(0, 0, states_tensor.size(0)).copy_(states_tensor);
    t_grid_ocp.zero_();
    launch_refresh_occ_bitmap_from_packed_states(t_state_packed, ctx);
}

void StatelessGridWorldSimulator::materialize_pyg_inputs() {
    launch_imitation_obs_generator_from_packed_states(ctx.d_packed_states, ctx);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Packed-state imitation obs generator kernel failed: ") + cudaGetErrorString(err));
    }
    finalize_magat_plus_graph();
}

void GridWorldSimulator::set_pyg_builder_mode(const std::string& mode) {
    int builder_mode = parse_pyg_builder_mode_string(mode);
    validate_requested_pyg_builder_mode(pyg_ctx, builder_mode);
    pyg_ctx.builder_mode = builder_mode;
}

void GridWorldSimulator::set_real_map_extents(const torch::Tensor& real_map_extents) {
    TORCH_CHECK(real_map_extents.dtype() == torch::kInt32, "real_map_extents must be int32");
    TORCH_CHECK(real_map_extents.dim() == 2, "real_map_extents must be 2D [n_envs, 2]");
    TORCH_CHECK(real_map_extents.size(0) == ctx.n_envs, "real_map_extents must have one row per env");
    TORCH_CHECK(real_map_extents.size(1) == 2, "real_map_extents must have shape [n_envs, 2]");

    auto extents_cuda = real_map_extents.contiguous().to(
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto extents_cpu = extents_cuda.cpu();
    const int* host_ptr = extents_cpu.data_ptr<int>();
    for (int env_id = 0; env_id < ctx.n_envs; ++env_id) {
        int real_map_w = host_ptr[env_id * 2];
        int real_map_h = host_ptr[env_id * 2 + 1];
        TORCH_CHECK(real_map_w > 0 && real_map_w <= MAP_W,
            "real_map_extents[:, 0] must be in [1, ", MAP_W, "]");
        TORCH_CHECK(real_map_h > 0 && real_map_h <= MAP_H,
            "real_map_extents[:, 1] must be in [1, ", MAP_H, "]");
    }

    t_real_map_extents = extents_cuda.clone();
    ctx.d_real_map_extents = t_real_map_extents.data_ptr<int>();
}

std::string GridWorldSimulator::get_task_mode() const {
    return task_mode_to_string(ctx.task_mode);
}

int GridWorldSimulator::get_max_episode_steps() const {
    return ctx.max_episode_steps;
}

void GridWorldSimulator::load_state(
    const torch::Tensor& positions,
    const torch::Tensor& goals,
    const torch::Tensor& arrived,
    const torch::Tensor& step_counts)
{
    const auto expected_positions = std::vector<int64_t>{ctx.n_envs, ctx.n_agents, 2};
    const auto expected_agents = std::vector<int64_t>{ctx.n_envs, ctx.n_agents};
    const auto expected_envs = std::vector<int64_t>{ctx.n_envs};

    TORCH_CHECK(positions.is_cuda(), "positions must be CUDA");
    TORCH_CHECK(positions.dtype() == torch::kInt16, "positions must be int16");
    TORCH_CHECK(
        positions.sizes() == expected_positions,
        "positions must have shape [n_envs, n_agents, 2]");
    TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
    TORCH_CHECK(
        positions.device() == t_cur_x.device(),
        "positions and simulator tensors must use the same CUDA device");
    TORCH_CHECK(goals.is_cuda(), "goals must be CUDA");
    TORCH_CHECK(goals.dtype() == torch::kInt16, "goals must be int16");
    TORCH_CHECK(
        goals.sizes() == expected_positions,
        "goals must have shape [n_envs, n_agents, 2]");
    TORCH_CHECK(goals.is_contiguous(), "goals must be contiguous");
    TORCH_CHECK(
        goals.device() == t_cur_x.device(),
        "goals and simulator tensors must use the same CUDA device");
    TORCH_CHECK(arrived.is_cuda(), "arrived must be CUDA");
    TORCH_CHECK(arrived.dtype() == torch::kUInt8, "arrived must be uint8");
    TORCH_CHECK(
        arrived.sizes() == expected_agents,
        "arrived must have shape [n_envs, n_agents]");
    TORCH_CHECK(arrived.is_contiguous(), "arrived must be contiguous");
    TORCH_CHECK(
        arrived.device() == t_cur_x.device(),
        "arrived and simulator tensors must use the same CUDA device");
    TORCH_CHECK(step_counts.is_cuda(), "step_counts must be CUDA");
    TORCH_CHECK(step_counts.dtype() == torch::kInt32, "step_counts must be int32");
    TORCH_CHECK(step_counts.sizes() == expected_envs, "step_counts must have shape [n_envs]");
    TORCH_CHECK(step_counts.is_contiguous(), "step_counts must be contiguous");
    TORCH_CHECK(
        step_counts.device() == t_cur_x.device(),
        "step_counts and simulator tensors must use the same CUDA device");

    c10::cuda::CUDAGuard device_guard(t_cur_x.device());

    auto positions_x = positions.select(2, 0);
    auto positions_y = positions.select(2, 1);
    auto goals_x = goals.select(2, 0);
    auto goals_y = goals.select(2, 1);
    bool coordinates_out_of_bounds =
        positions_x.lt(0).any().item<bool>() || positions_x.ge(MAP_W).any().item<bool>() ||
        positions_y.lt(0).any().item<bool>() || positions_y.ge(MAP_H).any().item<bool>() ||
        goals_x.lt(0).any().item<bool>() || goals_x.ge(MAP_W).any().item<bool>() ||
        goals_y.lt(0).any().item<bool>() || goals_y.ge(MAP_H).any().item<bool>();
    TORCH_CHECK(
        !coordinates_out_of_bounds,
        "position and goal coordinates must be within map bounds");
    TORCH_CHECK(!arrived.gt(1).any().item<bool>(), "arrived values must be 0 or 1");
    auto on_goal = positions.eq(goals).all(2);
    auto invalid_arrived = torch::logical_and(
        arrived.to(torch::kBool), torch::logical_not(on_goal));
    TORCH_CHECK(!invalid_arrived.any().item<bool>(), "arrived agents must be at their goals");
    TORCH_CHECK(
        !step_counts.lt(0).any().item<bool>() &&
            !step_counts.gt(ctx.max_episode_steps).any().item<bool>(),
        "step_counts must be in [0, max_episode_steps]");

    t_arrived.copy_(arrived);
    t_step_counts.copy_(step_counts);
    auto all_arrived = t_arrived.to(torch::kBool).all(1);
    t_terminated.copy_(all_arrived.to(torch::kUInt8));
    auto horizon_reached = t_step_counts.ge(ctx.max_episode_steps);
    t_truncated.copy_(
        torch::logical_and(torch::logical_not(all_arrived), horizon_reached)
            .to(torch::kUInt8));
    t_rewards.zero_();
    t_actions.zero_();
    ctx.current_step = 0;

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(t_cur_x.get_device());
    launch_load_state_kernel(ctx, positions, goals, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("load_state kernel failed: ") + cudaGetErrorString(err));
    }
}

std::string GridWorldSimulator::get_pyg_builder_mode() const {
    return pyg_builder_mode_to_string(pyg_ctx.builder_mode);
}

void GridWorldSimulator::set_pyg_local_gather_impl(const std::string& mode) {
    pyg_ctx.local_gather_impl_mode = parse_pyg_local_gather_impl_string(mode);
}

std::string GridWorldSimulator::get_pyg_local_gather_impl() const {
    return pyg_local_gather_impl_to_string(pyg_ctx.local_gather_impl_mode);
}

std::string GridWorldSimulator::get_resolved_pyg_builder_impl() const {
    return resolved_pyg_builder_impl_to_string(pyg_ctx);
}

void StatelessGridWorldSimulator::set_pyg_builder_mode(const std::string& mode) {
    int builder_mode = parse_pyg_builder_mode_string(mode);
    validate_requested_pyg_builder_mode(ctx, builder_mode);
    ctx.builder_mode = builder_mode;
}

std::string StatelessGridWorldSimulator::get_pyg_builder_mode() const {
    return pyg_builder_mode_to_string(ctx.builder_mode);
}

void StatelessGridWorldSimulator::set_pyg_local_gather_impl(const std::string& mode) {
    ctx.local_gather_impl_mode = parse_pyg_local_gather_impl_string(mode);
}

std::string StatelessGridWorldSimulator::get_pyg_local_gather_impl() const {
    return pyg_local_gather_impl_to_string(ctx.local_gather_impl_mode);
}

std::string StatelessGridWorldSimulator::get_resolved_pyg_builder_impl() const {
    return resolved_pyg_builder_impl_to_string(ctx);
}


void GridWorldSimulator::init_rng(unsigned long seed) {
    int total_agents = ctx.n_envs * ctx.n_agents;
    // setup_rng_kernel<<<blocks, threads>>>(ctx.rng_states, seed, total_agents);
    launch_setup_rng_kernel(ctx.rng_states, seed, total_agents);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("setup_rng_kernel kernel failed: ") + cudaGetErrorString(err));
    }
    cudaDeviceSynchronize(); 
}

// warp 级调度 结合 SM数量 vs env 数量
// 综合考量:(n_envs, n_agents, map_size), 其中, n_envs 可以随着 n_agents 及 map_size 改变
// 综合考虑 SM 的个数以及 shared_memory 大小
void GridWorldSimulator::run_initialization() {
    c10::cuda::CUDAGuard device_guard(t_cur_x.device());
    ctx.current_step = 0;
    t_arrived.zero_();
    t_terminated.zero_();
    t_truncated.zero_();
    t_step_counts.zero_();
    t_goal_changed_flags.zero_();
    t_actions.zero_();
    t_pool_ptr.zero_(); 
    dim3 blocks(ctx.n_envs);

    int _num_threads = ctx.n_agents;
    if(_num_threads >= 256)   _num_threads = 256;
    else _num_threads = next_pow2_efficient(_num_threads);
    if (_num_threads < 32) _num_threads = 32;
    dim3 threads(_num_threads);   // 根据智能体数量而定, 最大设置为 256(智能体数量更大时, 对应的 kernel 内部循环处理)

    launch_cache_map_free_cells_kernel(ctx, blocks, threads);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("free_cells kernel failed: ") + cudaGetErrorString(err));
    }
    else std::cout<<"free_cells kernel success!"<<std::endl;

    launch_init_agents_kernel(ctx, blocks, threads);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("init_agents kernel failed: ") + cudaGetErrorString(err));
    }
}

void GridWorldSimulator::update_actions(const torch::Tensor& new_actions) {
    TORCH_CHECK(new_actions.dtype() == torch::kUInt8, "new_actions must be uint8");
    TORCH_CHECK(new_actions.device().is_cuda(), "new_actions must be CUDA");
    TORCH_CHECK(new_actions.sizes() == t_actions.sizes(), "new_actions shape mismatch");
    TORCH_CHECK(new_actions.is_contiguous(), "new_actions must be contiguous");
    TORCH_CHECK(
        new_actions.device() == t_actions.device(),
        "new_actions and simulator tensors must use the same CUDA device");
    c10::cuda::CUDAGuard device_guard(t_actions.device());
    t_actions.copy_(new_actions);
}

void GridWorldSimulator::step_compact_only() {
    step_sim_only();
}

void GridWorldSimulator::step_sim_only() {
    c10::cuda::CUDAGuard device_guard(t_cur_x.device());
    ctx.current_step += 1;

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(t_cur_x.get_device());
    launch_step_kernel(this->ctx, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Step kernel failed: ") + cudaGetErrorString(err));
    }
}

void GridWorldSimulator::update_derived_state() {
    cudaError_t err = cudaGetLastError();
    int total_agents = ctx.n_envs * ctx.n_agents;
    auto flat_goal_changed_flags = t_goal_changed_flags.reshape({total_agents}).to(torch::kInt64);
    if (total_agents == 0) {
        t_goal_changed_prefix.zero_();
        t_goal_changed_count.zero_();
        return;
    }

    t_goal_changed_prefix.copy_(torch::cumsum(flat_goal_changed_flags, 0));
    t_goal_changed_count.copy_(t_goal_changed_prefix.narrow(0, total_agents - 1, 1));

    if (ctx.current_step == 1) {
        launch_energy_map_kernel(pyg_ctx.energy_map_ctx, ctx.n_agents, total_agents, t_state_packed);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Stateful energy map kernel failed: ") + cudaGetErrorString(err));
        }
    } else {
        int64_t changed_count = t_goal_changed_count.cpu().item<int64_t>();
        if (changed_count == 0) {
            t_changed_state_packed.zero_();
            return;
        }

        t_changed_state_packed.zero_();
        launch_pack_changed_agent_state_kernel(
            ctx,
            t_goal_changed_prefix,
            t_changed_state_packed);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Changed-state pack kernel failed: ") + cudaGetErrorString(err));
        }

        launch_energy_map_kernel(
            pyg_ctx.energy_map_ctx,
            ctx.n_agents,
            static_cast<int>(changed_count),
            t_changed_state_packed);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Incremental energy map kernel failed: ") + cudaGetErrorString(err));
        }
    }
}

void GridWorldSimulator::build_magat_plus_nodes() {
    cudaError_t err = cudaGetLastError();
    launch_imitation_obs_generator_from_packed_states(ctx.d_packed_states, pyg_ctx);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Stateful imitation obs builder kernel failed: ") + cudaGetErrorString(err));
    }
}

void GridWorldSimulator::finalize_magat_plus_graph() {
    cudaError_t err = cudaGetLastError();
    int total_agents = ctx.n_envs * ctx.n_agents;
    launch_exclusive_scan_edge_counts(
        t_edge_counts,
        t_pyg_edge_prefix,
        t_pyg_num_edges,
        t_scan_temp_storage);

    launch_generate_pyg_edges_kernel(
        t_pyg_edge_prefix,
        t_pyg_pos,
        t_pyg_edge_index_storage,
        t_pyg_edge_attr_storage,
        ctx.n_agents,
        ctx.n_envs,
        total_agents);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Stateful PyG edge builder kernel failed: ") + cudaGetErrorString(err));
    }
}

void GridWorldSimulator::build_magat_plus_inputs() {
    build_magat_plus_nodes();
    finalize_magat_plus_graph();
}

void GridWorldSimulator::materialize_pyg_inputs() {
    build_magat_plus_inputs();
}

void GridWorldSimulator::step_and_build_pyg() {
    step_sim_only();
    update_derived_state();
    materialize_pyg_inputs();
}

void GridWorldSimulator::step() {
    step_and_build_pyg();
}

void StatelessGridWorldSimulator::build_magat_plus_nodes(const torch::Tensor& states_tensor) {
    refresh_compact_state_from_raw_batch(states_tensor);
    launch_imitation_obs_generator_from_packed_states(ctx.d_packed_states, ctx);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Packed-state imitation obs generator kernel failed: ") + cudaGetErrorString(err));
    }
}

void StatelessGridWorldSimulator::finalize_magat_plus_graph() {
    cudaError_t err = cudaGetLastError();
    int total_nodes = ctx.n_envs * ctx.n_agents;
    launch_exclusive_scan_edge_counts(
        t_edge_counts,
        t_pyg_edge_prefix,
        t_pyg_num_edges,
        t_scan_temp_storage);

    launch_generate_pyg_edges_kernel(
        t_pyg_edge_prefix,
        t_pyg_pos,
        t_pyg_edge_index_storage,
        t_pyg_edge_attr_storage,
        ctx.n_agents,
        ctx.n_envs,
        total_nodes);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("PyG edge generation kernel failed: ") + cudaGetErrorString(err));
    }
}

void StatelessGridWorldSimulator::build_magat_plus_inputs(const torch::Tensor& states_tensor) {
    refresh_compact_state_from_raw_batch(states_tensor);
    materialize_pyg_inputs();
}

void StatelessGridWorldSimulator::setup_imitation_obs(const torch::Tensor& states_tensor) {
    refresh_compact_state_from_raw_batch(states_tensor);
    materialize_pyg_inputs();
}

void StatelessGridWorldSimulator::update_derived_state(const torch::Tensor& states_tensor, int length) {
    update_energy_maps(states_tensor, length);
}

void StatelessGridWorldSimulator::update_energy_maps(const torch::Tensor& states_tensor, int length) {
    // 更新能量图
    launch_energy_map_kernel(ctx.energy_map_ctx, ctx.n_agents, length, states_tensor);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Energy map kernel failed: ") + cudaGetErrorString(err));
    }
}

torch::Tensor GridWorldSimulator::decode_to_uint8() {
    int diameter = 2 * RADIUS + 1;
    int n_batch = ctx.n_envs * ctx.n_agents;
    
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    torch::Tensor out_tensor = torch::empty({n_batch, 2, diameter, diameter}, options);

    int total_pixels = n_batch * 2 * diameter * diameter;
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;
    
    launch_decode_to_uint8_kernel(ctx.d_agents_obs, blocks, threads, 
            out_tensor.data_ptr<uint8_t>(), total_pixels, diameter);

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Decode kernel failed: ") + cudaGetErrorString(err));
    }

    return out_tensor;
}

// generate_sparse_edges removed - now fused into setup_imitation_obs
