#include "mapf_gpt_builder.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "grid_world_cuda.h"


void launch_encode_full_map_kernel_on_stream(
    const uint32_t* input_map,
    uint32_t* compressed_map,
    int batch_size,
    int block_size,
    cudaStream_t stream);


MapfGPTObservationBuilder::MapfGPTObservationBuilder(
    torch::Tensor grids,
    int requested_n_agents)
    : n_envs(0), n_agents(requested_n_agents) {
    TORCH_CHECK(grids.is_cuda(), "grids must be on a CUDA device");
    TORCH_CHECK(grids.dtype() == torch::kInt32, "grids must use int32 dtype");
    TORCH_CHECK(grids.is_contiguous(), "grids must be contiguous");
    TORCH_CHECK(
        grids.dim() == 3 && grids.size(1) == MAP_W && grids.size(2) == MAP_H,
        "grids shape must be [n_envs, ", MAP_W, ", ", MAP_H, "]");
    TORCH_CHECK(requested_n_agents > 0, "n_agents must be positive");
    c10::cuda::CUDAGuard device_guard(grids.device());

    n_envs = static_cast<int>(grids.size(0));
    TORCH_CHECK(n_envs > 0, "grids must contain at least one environment");
    const int64_t total_agents = static_cast<int64_t>(n_envs) * n_agents;
    const auto device = grids.device();
    const auto opt_i16 = torch::TensorOptions().dtype(torch::kInt16).device(device);
    const auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    const auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
    const auto opt_bool = torch::TensorOptions().dtype(torch::kBool).device(device);

    t_grid_compressed = torch::zeros({n_envs, MAP_OFFSET}, opt_i32);
    t_state_packed = torch::zeros({total_agents, 8}, opt_i16);
    t_histories = torch::full({total_agents, 5}, 5, opt_i16);
    t_labels = torch::zeros({total_agents}, opt_i64);
    t_tokens = torch::full({total_agents, 256}, 66, opt_i32);
    t_active_mask = torch::zeros({total_agents}, opt_bool);
    t_cost_to_go = torch::full(
        {n_envs, n_agents, MAP_W, MAP_H}, -1, opt_i16);
    t_agent_at_cell = torch::full({n_envs, MAP_W, MAP_H}, -1, opt_i32);
    t_goal_cache = torch::full({total_agents, 2}, -1, opt_i16);
    t_diagnostics = torch::zeros({4}, opt_i32);
    t_pyg_ptr = torch::arange(n_envs + 1, opt_i64) * n_agents;

    ctx.d_maps = reinterpret_cast<const uint32_t*>(
        t_grid_compressed.data_ptr<int32_t>());
    ctx.d_states = reinterpret_cast<AgentState*>(
        t_state_packed.data_ptr<int16_t>());
    ctx.d_histories = reinterpret_cast<uint16_t*>(
        t_histories.data_ptr<int16_t>());
    ctx.d_labels = t_labels.data_ptr<int64_t>();
    ctx.d_cost_to_go = reinterpret_cast<uint16_t*>(
        t_cost_to_go.data_ptr<int16_t>());
    ctx.d_agent_at_cell = t_agent_at_cell.data_ptr<int32_t>();
    ctx.d_goal_cache = reinterpret_cast<uint16_t*>(
        t_goal_cache.data_ptr<int16_t>());
    ctx.d_tokens = t_tokens.data_ptr<int32_t>();
    ctx.d_active_mask = t_active_mask.data_ptr<bool>();
    ctx.d_diagnostics = t_diagnostics.data_ptr<int32_t>();
    ctx.n_envs = n_envs;
    ctx.n_agents = n_agents;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(grids.get_device());
    launch_encode_full_map_kernel_on_stream(
        reinterpret_cast<const uint32_t*>(grids.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(t_grid_compressed.data_ptr<int32_t>()),
        n_envs,
        256,
        stream);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("MAPF-GPT map encoding failed: ") +
            cudaGetErrorString(error));
    }
}


void MapfGPTObservationBuilder::build_tokens(const torch::Tensor& raw_rows) {
    TORCH_CHECK(raw_rows.is_cuda(), "raw_rows must be on a CUDA device");
    TORCH_CHECK(raw_rows.dtype() == torch::kInt16, "raw_rows must use int16 dtype");
    TORCH_CHECK(raw_rows.dim() == 2, "raw_rows must be a 2D tensor");
    TORCH_CHECK(raw_rows.size(1) == 13, "raw_rows feature dimension must be 13");
    TORCH_CHECK(
        raw_rows.size(0) == static_cast<int64_t>(n_envs) * n_agents,
        "raw_rows row count must equal n_envs * n_agents");
    TORCH_CHECK(raw_rows.is_contiguous(), "raw_rows must be contiguous");
    TORCH_CHECK(
        raw_rows.device() == t_tokens.device(),
        "raw_rows and builder tensors must use the same CUDA device");
    c10::cuda::CUDAGuard device_guard(raw_rows.device());
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(raw_rows.get_device());
    t_diagnostics.zero_();
    launch_mapf_gpt_unpack_kernel(raw_rows, ctx, stream);
    launch_mapf_gpt_cost_to_go_kernel(ctx, stream);
    launch_mapf_gpt_token_kernel(ctx, stream);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("MAPF-GPT cost-to-go generation failed: ") +
            cudaGetErrorString(error));
    }
}


namespace {

void check_resident_state_tensor(
    const torch::Tensor& tensor,
    const char* name,
    int n_envs,
    int n_agents,
    const torch::Device& expected_device) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be on a CUDA device");
    TORCH_CHECK(tensor.dtype() == torch::kInt16, name, " must use int16 dtype");
    TORCH_CHECK(
        tensor.dim() == 2 && tensor.size(0) == n_envs &&
            tensor.size(1) == n_agents,
        name, " shape must be [n_envs, n_agents]");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        tensor.device() == expected_device,
        name, " must use the builder CUDA device");
}

}  // namespace


void MapfGPTObservationBuilder::build_tokens_from_state(
    const torch::Tensor& cur_x,
    const torch::Tensor& cur_y,
    const torch::Tensor& goal_x,
    const torch::Tensor& goal_y) {
    check_resident_state_tensor(
        cur_x, "cur_x", n_envs, n_agents, t_tokens.device());
    check_resident_state_tensor(
        cur_y, "cur_y", n_envs, n_agents, t_tokens.device());
    check_resident_state_tensor(
        goal_x, "goal_x", n_envs, n_agents, t_tokens.device());
    check_resident_state_tensor(
        goal_y, "goal_y", n_envs, n_agents, t_tokens.device());
    c10::cuda::CUDAGuard device_guard(cur_x.device());
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(cur_x.get_device());
    t_diagnostics.zero_();
    launch_mapf_gpt_state_unpack_kernel(
        cur_x, cur_y, goal_x, goal_y, ctx, stream);
    launch_mapf_gpt_cost_to_go_kernel(ctx, stream);
    launch_mapf_gpt_token_kernel(ctx, stream);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("MAPF-GPT resident token generation failed: ") +
            cudaGetErrorString(error));
    }
}


void MapfGPTObservationBuilder::append_actions(
    const torch::Tensor& actions) {
    TORCH_CHECK(actions.is_cuda(), "actions must be on a CUDA device");
    TORCH_CHECK(actions.dtype() == torch::kUInt8, "actions must use uint8 dtype");
    TORCH_CHECK(
        actions.dim() == 2 && actions.size(0) == n_envs &&
            actions.size(1) == n_agents,
        "actions shape must be [n_envs, n_agents]");
    TORCH_CHECK(actions.is_contiguous(), "actions must be contiguous");
    TORCH_CHECK(
        actions.device() == t_tokens.device(),
        "actions must use the builder CUDA device");
    c10::cuda::CUDAGuard device_guard(actions.device());
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(actions.get_device());
    launch_mapf_gpt_append_actions_kernel(actions, ctx, stream);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("MAPF-GPT action-history update failed: ") +
            cudaGetErrorString(error));
    }
}


void MapfGPTObservationBuilder::reset_histories() {
    t_histories.fill_(5);
}
