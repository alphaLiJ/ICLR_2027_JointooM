#include "mapf_gpt_builder.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdexcept>
#include <string>


namespace {

constexpr int MAPFGPT_RAW_FEATURE_DIM = 13;
constexpr int MAPFGPT_HISTORY_LEN = 5;
constexpr int MAPFGPT_MAX_FRONTIER = 4096;
constexpr uint16_t MAPFGPT_UNREACHABLE = 0xFFFFu;
constexpr int MAPFGPT_CONTEXT_SIZE = 256;
constexpr int MAPFGPT_OBS_RADIUS = 5;
constexpr int MAPFGPT_OBS_DIAMETER = 11;
constexpr int MAPFGPT_COST_TOKENS = 121;
constexpr int MAPFGPT_VISIBLE_AGENTS = 13;
constexpr int MAPFGPT_AGENT_TOKENS = 10;
constexpr int MAPFGPT_COST_LIMIT = 20;
constexpr int MAPFGPT_TOKEN_UNREACHABLE = 41;
constexpr int MAPFGPT_TOKEN_NEGATIVE_CLIP = 42;
constexpr int MAPFGPT_TOKEN_POSITIVE_CLIP = 43;
constexpr int MAPFGPT_TOKEN_EMPTY_ACTION = 44;
constexpr int MAPFGPT_TOKEN_WAIT = 45;
constexpr int MAPFGPT_TOKEN_GREEDY_BASE = 50;
constexpr int MAPFGPT_TOKEN_PADDING = 66;


int cost_to_go_shared_bytes() {
    const int visited_words = (GRID_SIZE + 31) / 32;
    return GRID_SIZE * static_cast<int>(sizeof(uint16_t)) +
           visited_words * static_cast<int>(sizeof(uint32_t)) +
           2 * MAPFGPT_MAX_FRONTIER * static_cast<int>(sizeof(uint16_t));
}

}  // namespace


__device__ __forceinline__ int mapf_gpt_coordinate_token(int value) {
    value = value < -MAPFGPT_COST_LIMIT ? -MAPFGPT_COST_LIMIT : value;
    value = value > MAPFGPT_COST_LIMIT ? MAPFGPT_COST_LIMIT : value;
    return value + MAPFGPT_COST_LIMIT;
}


__device__ __forceinline__ int mapf_gpt_cost_token(
    uint16_t value,
    uint16_t center) {
    if (value == MAPFGPT_UNREACHABLE) {
        return MAPFGPT_TOKEN_UNREACHABLE;
    }
    const int difference = static_cast<int>(value) - static_cast<int>(center);
    if (difference > MAPFGPT_COST_LIMIT) {
        return MAPFGPT_TOKEN_POSITIVE_CLIP;
    }
    if (difference < -MAPFGPT_COST_LIMIT) {
        return MAPFGPT_TOKEN_NEGATIVE_CLIP;
    }
    return difference + MAPFGPT_COST_LIMIT;
}


__device__ __forceinline__ bool mapf_gpt_key_less(
    int lhs_distance,
    int lhs_id,
    int rhs_distance,
    int rhs_id) {
    return lhs_distance < rhs_distance ||
           (lhs_distance == rhs_distance && lhs_id < rhs_id);
}


__global__ void mapf_gpt_unpack_kernel(
    const int16_t* __restrict__ raw_rows,
    MapfGPTCUDAContext ctx,
    int total_agents) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_agents) {
        return;
    }

    const uint16_t* row = reinterpret_cast<const uint16_t*>(
        raw_rows + static_cast<int64_t>(index) * MAPFGPT_RAW_FEATURE_DIM);
    const uint16_t env_id = row[0];
    const uint16_t agent_id = row[1];
    if (env_id >= ctx.n_envs || agent_id >= ctx.n_agents ||
        index != static_cast<int>(env_id) * ctx.n_agents + agent_id) {
        atomicAdd(ctx.d_diagnostics + 1, 1);
        return;
    }

    AgentState state;
    state.env_id = env_id;
    state.agent_id = agent_id;
    state.pos_x = row[2];
    state.pos_y = row[3];
    state.target_x = row[4];
    state.target_y = row[5];
    state.action = row[6];

    uint16_t* cached_goal = ctx.d_goal_cache + static_cast<int64_t>(index) * 2;
    const bool goal_changed =
        cached_goal[0] != state.target_x || cached_goal[1] != state.target_y;
    state.reset_flag = (row[7] != 0 || goal_changed) ? 1 : 0;
    cached_goal[0] = state.target_x;
    cached_goal[1] = state.target_y;
    ctx.d_states[index] = state;

    uint16_t* history =
        ctx.d_histories + static_cast<int64_t>(index) * MAPFGPT_HISTORY_LEN;
    #pragma unroll
    for (int slot = 0; slot < MAPFGPT_HISTORY_LEN; ++slot) {
        history[slot] = row[8 + slot];
    }
    ctx.d_labels[index] = static_cast<int64_t>(state.action);
}


__global__ void mapf_gpt_state_unpack_kernel(
    const int16_t* __restrict__ cur_x,
    const int16_t* __restrict__ cur_y,
    const int16_t* __restrict__ goal_x,
    const int16_t* __restrict__ goal_y,
    MapfGPTCUDAContext ctx,
    int total_agents) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_agents) {
        return;
    }

    const int env_id = index / ctx.n_agents;
    const int agent_id = index - env_id * ctx.n_agents;
    AgentState state;
    state.env_id = static_cast<uint16_t>(env_id);
    state.agent_id = static_cast<uint16_t>(agent_id);
    state.pos_x = reinterpret_cast<const uint16_t*>(cur_x)[index];
    state.pos_y = reinterpret_cast<const uint16_t*>(cur_y)[index];
    state.target_x = reinterpret_cast<const uint16_t*>(goal_x)[index];
    state.target_y = reinterpret_cast<const uint16_t*>(goal_y)[index];
    state.action = 0;

    uint16_t* cached_goal = ctx.d_goal_cache + static_cast<int64_t>(index) * 2;
    const bool goal_changed =
        cached_goal[0] != state.target_x || cached_goal[1] != state.target_y;
    state.reset_flag = goal_changed ? 1 : 0;
    cached_goal[0] = state.target_x;
    cached_goal[1] = state.target_y;
    ctx.d_states[index] = state;
    ctx.d_labels[index] = 0;
}


__global__ void mapf_gpt_append_actions_kernel(
    const uint8_t* __restrict__ actions,
    MapfGPTCUDAContext ctx,
    int total_agents) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_agents) {
        return;
    }

    uint16_t action = ctx.d_active_mask[index]
        ? static_cast<uint16_t>(actions[index])
        : 0;
    if (action >= 5) {
        atomicAdd(ctx.d_diagnostics + 1, 1);
        action = 0;
    }
    uint16_t* history =
        ctx.d_histories + static_cast<int64_t>(index) * MAPFGPT_HISTORY_LEN;
    #pragma unroll
    for (int slot = 0; slot < MAPFGPT_HISTORY_LEN - 1; ++slot) {
        history[slot] = history[slot + 1];
    }
    history[MAPFGPT_HISTORY_LEN - 1] = action;
}


__global__ void mapf_gpt_cost_to_go_kernel(MapfGPTCUDAContext ctx) {
    const int global_agent = blockIdx.x;
    const int total_agents = ctx.n_envs * ctx.n_agents;
    if (global_agent >= total_agents ||
        ctx.d_states[global_agent].reset_flag == 0) {
        return;
    }

    const AgentState state = ctx.d_states[global_agent];
    const int map_offset = static_cast<int>(state.env_id) * MAP_OFFSET;
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    extern __shared__ uint8_t dynamic_shared[];
    uint16_t* distances = reinterpret_cast<uint16_t*>(dynamic_shared);
    uint32_t* visited = reinterpret_cast<uint32_t*>(
        distances + GRID_SIZE);
    const int visited_words = (GRID_SIZE + 31) / 32;
    uint16_t* queue_a = reinterpret_cast<uint16_t*>(visited + visited_words);
    uint16_t* queue_b = queue_a + MAPFGPT_MAX_FRONTIER;

    __shared__ int current_size;
    __shared__ int next_size;
    __shared__ int queue_selector;
    __shared__ int current_distance;

    for (int cell = tid; cell < GRID_SIZE; cell += blockDim.x) {
        distances[cell] = MAPFGPT_UNREACHABLE;
    }
    for (int word = tid; word < visited_words; word += blockDim.x) {
        visited[word] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        current_size = 0;
        next_size = 0;
        queue_selector = 0;
        current_distance = 0;
        const int goal_x = state.target_x;
        const int goal_y = state.target_y;
        if (goal_x >= 0 && goal_x < MAP_W && goal_y >= 0 && goal_y < MAP_H) {
            const int goal_index = goal_x * MAP_H + goal_y;
            const uint32_t obstacle_word =
                ctx.d_maps[map_offset + goal_index / 32];
            if ((obstacle_word & (1u << (goal_index & 31))) == 0) {
                distances[goal_index] = 0;
                visited[goal_index / 32] |= 1u << (goal_index & 31);
                queue_a[0] = static_cast<uint16_t>(goal_index);
                current_size = 1;
            } else {
                atomicAdd(ctx.d_diagnostics + 3, 1);
            }
        } else {
            atomicAdd(ctx.d_diagnostics + 3, 1);
        }
    }
    __syncthreads();

    while (current_size > 0) {
        uint16_t* read_queue = queue_selector == 0 ? queue_a : queue_b;
        uint16_t* write_queue = queue_selector == 0 ? queue_b : queue_a;
        const int rounded_size = (current_size + 31) & ~31;

        for (int queue_index = tid;
             queue_index < rounded_size;
             queue_index += blockDim.x) {
            int local_count = 0;
            int local_nodes[4];
            if (queue_index < current_size) {
                const int cell = read_queue[queue_index];
                const int x = cell / MAP_H;
                const int y = cell % MAP_H;
                const int dx[4] = {-1, 1, 0, 0};
                const int dy[4] = {0, 0, -1, 1};
                #pragma unroll
                for (int direction = 0; direction < 4; ++direction) {
                    const int nx = x + dx[direction];
                    const int ny = y + dy[direction];
                    if (nx < 0 || nx >= MAP_W || ny < 0 || ny >= MAP_H) {
                        continue;
                    }
                    const int neighbor = nx * MAP_H + ny;
                    const int word_index = neighbor / 32;
                    const uint32_t mask = 1u << (neighbor & 31);
                    if ((ctx.d_maps[map_offset + word_index] & mask) != 0) {
                        continue;
                    }
                    const uint32_t previous = atomicOr(visited + word_index, mask);
                    if ((previous & mask) == 0) {
                        distances[neighbor] = static_cast<uint16_t>(
                            current_distance + 1);
                        local_nodes[local_count++] = neighbor;
                    }
                }
            }

            int prefix = local_count;
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                const int other = __shfl_up_sync(0xFFFFFFFFu, prefix, offset);
                if (lane >= offset) {
                    prefix += other;
                }
            }
            const int warp_offset = prefix - local_count;
            const int warp_total = __shfl_sync(0xFFFFFFFFu, prefix, 31);
            int base = 0;
            if (lane == 31 && warp_total > 0) {
                base = atomicAdd(&next_size, warp_total);
            }
            base = __shfl_sync(0xFFFFFFFFu, base, 31);

            #pragma unroll
            for (int item = 0; item < 4; ++item) {
                if (item < local_count) {
                    const int write_index = base + warp_offset + item;
                    if (write_index < MAPFGPT_MAX_FRONTIER) {
                        write_queue[write_index] =
                            static_cast<uint16_t>(local_nodes[item]);
                    }
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            if (next_size > MAPFGPT_MAX_FRONTIER) {
                atomicAdd(ctx.d_diagnostics + 0, next_size - MAPFGPT_MAX_FRONTIER);
                next_size = MAPFGPT_MAX_FRONTIER;
            }
            current_size = next_size;
            next_size = 0;
            queue_selector ^= 1;
            ++current_distance;
        }
        __syncthreads();
    }

    uint16_t* output =
        ctx.d_cost_to_go + static_cast<int64_t>(global_agent) * GRID_SIZE;
    for (int cell = tid; cell < GRID_SIZE; cell += blockDim.x) {
        output[cell] = distances[cell];
    }
}


__global__ void mapf_gpt_scatter_agents_kernel(
    MapfGPTCUDAContext ctx,
    int total_agents) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_agents) {
        return;
    }
    const AgentState state = ctx.d_states[index];
    if (state.pos_x >= MAP_W || state.pos_y >= MAP_H) {
        atomicAdd(ctx.d_diagnostics + 3, 1);
        return;
    }
    int32_t* cell = ctx.d_agent_at_cell +
        static_cast<int64_t>(state.env_id) * GRID_SIZE +
        static_cast<int>(state.pos_x) * MAP_H + state.pos_y;
    const int32_t previous = atomicCAS(cell, -1, static_cast<int32_t>(state.agent_id));
    if (previous != -1) {
        atomicAdd(ctx.d_diagnostics + 2, 1);
    }
}


__global__ void mapf_gpt_materialize_tokens_kernel(
    MapfGPTCUDAContext ctx,
    int total_agents) {
    constexpr int warps_per_block = 4;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int ego_global = blockIdx.x * warps_per_block + warp_in_block;
    if (ego_global >= total_agents) {
        return;
    }

    int32_t* output = ctx.d_tokens +
        static_cast<int64_t>(ego_global) * MAPFGPT_CONTEXT_SIZE;
    for (int token = lane; token < MAPFGPT_CONTEXT_SIZE; token += 32) {
        output[token] = MAPFGPT_TOKEN_PADDING;
    }
    __syncwarp();

    const AgentState ego = ctx.d_states[ego_global];
    const bool active =
        ego.pos_x != ego.target_x || ego.pos_y != ego.target_y;
    if (lane == 0) {
        ctx.d_active_mask[ego_global] = active;
    }
    if (ego.pos_x >= MAP_W || ego.pos_y >= MAP_H) {
        return;
    }
    const uint16_t* ego_cost = ctx.d_cost_to_go +
        static_cast<int64_t>(ego_global) * GRID_SIZE;
    const int ego_cell = static_cast<int>(ego.pos_x) * MAP_H + ego.pos_y;
    const uint16_t center = ego_cost[ego_cell];

    for (int token = lane; token < MAPFGPT_COST_TOKENS; token += 32) {
        const int dx = token / MAPFGPT_OBS_DIAMETER - MAPFGPT_OBS_RADIUS;
        const int dy = token % MAPFGPT_OBS_DIAMETER - MAPFGPT_OBS_RADIUS;
        const int x = static_cast<int>(ego.pos_x) + dx;
        const int y = static_cast<int>(ego.pos_y) + dy;
        uint16_t value = MAPFGPT_UNREACHABLE;
        if (x >= 0 && x < MAP_W && y >= 0 && y < MAP_H) {
            value = ego_cost[x * MAP_H + y];
        }
        output[token] = mapf_gpt_cost_token(value, center);
    }
    __syncwarp();

    if (lane != 0) {
        return;
    }

    int selected_ids[MAPFGPT_VISIBLE_AGENTS];
    int selected_distances[MAPFGPT_VISIBLE_AGENTS];
    int selected_count = 0;
    const int32_t* env_agents = ctx.d_agent_at_cell +
        static_cast<int64_t>(ego.env_id) * GRID_SIZE;

    for (int dx = -MAPFGPT_OBS_RADIUS; dx <= MAPFGPT_OBS_RADIUS; ++dx) {
        for (int dy = -MAPFGPT_OBS_RADIUS; dy <= MAPFGPT_OBS_RADIUS; ++dy) {
            const int x = static_cast<int>(ego.pos_x) + dx;
            const int y = static_cast<int>(ego.pos_y) + dy;
            if (x < 0 || x >= MAP_W || y < 0 || y >= MAP_H) {
                continue;
            }
            const int agent_id = env_agents[x * MAP_H + y];
            if (agent_id < 0) {
                continue;
            }
            const int distance = abs(dx) + abs(dy);
            if (selected_count == MAPFGPT_VISIBLE_AGENTS &&
                !mapf_gpt_key_less(
                    distance,
                    agent_id,
                    selected_distances[MAPFGPT_VISIBLE_AGENTS - 1],
                    selected_ids[MAPFGPT_VISIBLE_AGENTS - 1])) {
                continue;
            }

            int insertion = selected_count < MAPFGPT_VISIBLE_AGENTS
                ? selected_count
                : MAPFGPT_VISIBLE_AGENTS - 1;
            if (selected_count < MAPFGPT_VISIBLE_AGENTS) {
                ++selected_count;
            }
            while (insertion > 0 &&
                   mapf_gpt_key_less(
                       distance,
                       agent_id,
                       selected_distances[insertion - 1],
                       selected_ids[insertion - 1])) {
                if (insertion < MAPFGPT_VISIBLE_AGENTS) {
                    selected_distances[insertion] =
                        selected_distances[insertion - 1];
                    selected_ids[insertion] = selected_ids[insertion - 1];
                }
                --insertion;
            }
            selected_distances[insertion] = distance;
            selected_ids[insertion] = agent_id;
        }
    }

    for (int slot = 0; slot < selected_count; ++slot) {
        const int agent_id = selected_ids[slot];
        const int selected_global =
            static_cast<int>(ego.env_id) * ctx.n_agents + agent_id;
        const AgentState selected = ctx.d_states[selected_global];
        const int base = MAPFGPT_COST_TOKENS + slot * MAPFGPT_AGENT_TOKENS;
        output[base + 0] = mapf_gpt_coordinate_token(
            static_cast<int>(selected.pos_x) - ego.pos_x);
        output[base + 1] = mapf_gpt_coordinate_token(
            static_cast<int>(selected.pos_y) - ego.pos_y);
        output[base + 2] = mapf_gpt_coordinate_token(
            static_cast<int>(selected.target_x) - ego.pos_x);
        output[base + 3] = mapf_gpt_coordinate_token(
            static_cast<int>(selected.target_y) - ego.pos_y);

        const uint16_t* history = ctx.d_histories +
            static_cast<int64_t>(selected_global) * MAPFGPT_HISTORY_LEN;
        #pragma unroll
        for (int item = 0; item < MAPFGPT_HISTORY_LEN; ++item) {
            const int action = history[item];
            output[base + 4 + item] = action == 5
                ? MAPFGPT_TOKEN_EMPTY_ACTION
                : MAPFGPT_TOKEN_WAIT + action;
        }

        const uint16_t* selected_cost = ctx.d_cost_to_go +
            static_cast<int64_t>(selected_global) * GRID_SIZE;
        const int selected_cell =
            static_cast<int>(selected.pos_x) * MAP_H + selected.pos_y;
        const uint16_t selected_center = selected_cost[selected_cell];
        const int move_x[4] = {-1, 1, 0, 0};
        const int move_y[4] = {0, 0, -1, 1};
        int greedy_mask = 0;
        #pragma unroll
        for (int direction = 0; direction < 4; ++direction) {
            const int x = static_cast<int>(selected.pos_x) + move_x[direction];
            const int y = static_cast<int>(selected.pos_y) + move_y[direction];
            uint16_t neighbor = MAPFGPT_UNREACHABLE;
            if (x >= 0 && x < MAP_W && y >= 0 && y < MAP_H) {
                neighbor = selected_cost[x * MAP_H + y];
            }
            if (neighbor != MAPFGPT_UNREACHABLE && selected_center > neighbor) {
                greedy_mask |= 1 << (3 - direction);
            }
        }
        output[base + 9] = MAPFGPT_TOKEN_GREEDY_BASE + greedy_mask;
    }
}


void launch_mapf_gpt_unpack_kernel(
    const torch::Tensor& raw_rows,
    MapfGPTCUDAContext ctx,
    cudaStream_t stream) {
    const int total_agents = ctx.n_envs * ctx.n_agents;
    constexpr int threads = 256;
    const int blocks = (total_agents + threads - 1) / threads;
    mapf_gpt_unpack_kernel<<<blocks, threads, 0, stream>>>(
        raw_rows.data_ptr<int16_t>(), ctx, total_agents);
}


void launch_mapf_gpt_state_unpack_kernel(
    const torch::Tensor& cur_x,
    const torch::Tensor& cur_y,
    const torch::Tensor& goal_x,
    const torch::Tensor& goal_y,
    MapfGPTCUDAContext ctx,
    cudaStream_t stream) {
    const int total_agents = ctx.n_envs * ctx.n_agents;
    constexpr int threads = 256;
    const int blocks = (total_agents + threads - 1) / threads;
    mapf_gpt_state_unpack_kernel<<<blocks, threads, 0, stream>>>(
        cur_x.data_ptr<int16_t>(),
        cur_y.data_ptr<int16_t>(),
        goal_x.data_ptr<int16_t>(),
        goal_y.data_ptr<int16_t>(),
        ctx,
        total_agents);
}


void launch_mapf_gpt_append_actions_kernel(
    const torch::Tensor& actions,
    MapfGPTCUDAContext ctx,
    cudaStream_t stream) {
    const int total_agents = ctx.n_envs * ctx.n_agents;
    constexpr int threads = 256;
    const int blocks = (total_agents + threads - 1) / threads;
    mapf_gpt_append_actions_kernel<<<blocks, threads, 0, stream>>>(
        actions.data_ptr<uint8_t>(), ctx, total_agents);
}


void launch_mapf_gpt_cost_to_go_kernel(
    MapfGPTCUDAContext ctx,
    cudaStream_t stream) {
    const int shared_bytes = cost_to_go_shared_bytes();
    int device = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaGetDevice failed for MAPF-GPT cost kernel: ") +
            cudaGetErrorString(error));
    }
    int opt_in_limit = 0;
    error = cudaDeviceGetAttribute(
        &opt_in_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("failed to query CUDA shared-memory limit: ") +
            cudaGetErrorString(error));
    }
    if (shared_bytes > opt_in_limit) {
        throw std::runtime_error(
            "MAPF-GPT uint16 cost kernel requires " +
            std::to_string(shared_bytes) +
            " bytes of shared memory, exceeding device limit " +
            std::to_string(opt_in_limit));
    }
    error = cudaFuncSetAttribute(
        mapf_gpt_cost_to_go_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes);
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("failed to opt in MAPF-GPT shared memory: ") +
            cudaGetErrorString(error));
    }

    const int blocks = ctx.n_envs * ctx.n_agents;
    mapf_gpt_cost_to_go_kernel<<<blocks, 256, shared_bytes, stream>>>(ctx);
}


void launch_mapf_gpt_token_kernel(
    MapfGPTCUDAContext ctx,
    cudaStream_t stream) {
    const int total_agents = ctx.n_envs * ctx.n_agents;
    cudaError_t error = cudaMemsetAsync(
        ctx.d_agent_at_cell,
        0xFF,
        static_cast<size_t>(ctx.n_envs) * GRID_SIZE * sizeof(int32_t),
        stream);
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("failed to clear MAPF-GPT agent identity grid: ") +
            cudaGetErrorString(error));
    }
    constexpr int scatter_threads = 256;
    const int scatter_blocks =
        (total_agents + scatter_threads - 1) / scatter_threads;
    mapf_gpt_scatter_agents_kernel<<<scatter_blocks, scatter_threads, 0, stream>>>(
        ctx, total_agents);

    constexpr int warps_per_block = 4;
    constexpr int token_threads = warps_per_block * 32;
    const int token_blocks = (total_agents + warps_per_block - 1) /
        warps_per_block;
    mapf_gpt_materialize_tokens_kernel<<<token_blocks, token_threads, 0, stream>>>(
        ctx, total_agents);
}
