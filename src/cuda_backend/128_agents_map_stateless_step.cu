#include <cuda_runtime.h>
#include <math_constants.h> // 引入 CUDART_PI_F
#include <cstdint>
#include <torch/extension.h>
#include <stdio.h>
#include "grid_world_cuda.h"

void launch_refresh_occ_bitmap_from_packed_states(
    const torch::Tensor& packed_states,
    StatelessGridWorldContext ctx);

namespace {

constexpr int LOCAL_GATHER_THREADS = 128;
constexpr int LOCAL_GATHER_PATCH_DIAM = 2 * RADIUS + 1;
constexpr size_t LEGACY_FULLMAP_SHARED_SMEM_LIMIT_BYTES = 96 * 1024;

inline int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

inline size_t legacy_fullmap_builder_smem_bytes(int agent_count) {
    size_t temp_map_bytes = MAP_W * (MAP_H + 1) * sizeof(int16_t);
    size_t agent_bytes = 4 * static_cast<size_t>(agent_count) * sizeof(int16_t);
    return temp_map_bytes + agent_bytes;
}

inline bool can_use_legacy_fullmap_builder(const StatelessGridWorldContext& ctx) {
    return ctx.n_agents <= 256
        && legacy_fullmap_builder_smem_bytes(ctx.n_agents) <= LEGACY_FULLMAP_SHARED_SMEM_LIMIT_BYTES;
}

}  // namespace

// 解析静态墙壁的 helper
__device__ inline bool is_wall(const uint32_t* __restrict__ s_map_static, int x, int y) {
    if (x < 0 || x >= MAP_W || y < 0 || y >= MAP_H) return true;
    uint32_t row_data = s_map_static[x * COL_OFFSET + (y >> 5)];
    return (row_data >> (y & 31)) & 1;
}

__device__ __forceinline__ uint32_t low_bits_mask(int bit_count) {
    if (bit_count <= 0) {
        return 0u;
    }
    if (bit_count >= 32) {
        return 0xFFFFFFFFu;
    }
    return (1u << bit_count) - 1u;
}

__device__ __forceinline__ uint32_t gather_row_window_bits(
    const uint32_t* __restrict__ d_bitmap,
    int x,
    int y_start)
{
    if (x < 0 || x >= MAP_W) {
        return 0u;
    }

    int valid_y0 = max(y_start, 0);
    int valid_y1 = min(y_start + LOCAL_GATHER_PATCH_DIAM - 1, MAP_H - 1);
    if (valid_y0 > valid_y1) {
        return 0u;
    }

    int first_local_col = valid_y0 - y_start;
    int valid_bit_count = valid_y1 - valid_y0 + 1;
    int word_idx = valid_y0 >> 5;
    int bit_shift = valid_y0 & 31;

    const uint32_t* row_ptr = d_bitmap + x * COL_OFFSET;
    uint32_t gathered_bits = __ldg(&row_ptr[word_idx]) >> bit_shift;
    if (bit_shift + valid_bit_count > 32 && (word_idx + 1) < COL_OFFSET) {
        gathered_bits |= __ldg(&row_ptr[word_idx + 1]) << (32 - bit_shift);
    }

    return (gathered_bits & low_bits_mask(valid_bit_count)) << first_local_col;
}

__device__ __forceinline__ uint32_t gather_row_window_bits_shared(
    const uint32_t* __restrict__ s_bitmap,
    int x,
    int y_start)
{
    if (x < 0 || x >= MAP_W) {
        return 0u;
    }

    int valid_y0 = max(y_start, 0);
    int valid_y1 = min(y_start + LOCAL_GATHER_PATCH_DIAM - 1, MAP_H - 1);
    if (valid_y0 > valid_y1) {
        return 0u;
    }

    int first_local_col = valid_y0 - y_start;
    int valid_bit_count = valid_y1 - valid_y0 + 1;
    int word_idx = valid_y0 >> 5;
    int bit_shift = valid_y0 & 31;

    const uint32_t* row_ptr = s_bitmap + x * COL_OFFSET;
    uint32_t gathered_bits = row_ptr[word_idx] >> bit_shift;
    if (bit_shift + valid_bit_count > 32 && (word_idx + 1) < COL_OFFSET) {
        gathered_bits |= row_ptr[word_idx + 1] << (32 - bit_shift);
    }

    return (gathered_bits & low_bits_mask(valid_bit_count)) << first_local_col;
}

// 替换了原来的带分支 sign，利用硬件 copysign 指令，速度极快
__device__ inline float fast_sign(float x) {
    if (x == 0.0f) return 0.0f;
    return copysignf(1.0f, x);
}

// NumPy's round uses nearest-even ties. Goal projection is an integer ratio;
// evaluating that ratio in float under --use_fast_math can move an exact
// negative half tie below the boundary (for example -150 / 60). Resolve the
// ratio exactly so every builder path preserves the host's discrete cell.
__device__ __forceinline__ int round_ratio_to_nearest_even(
    long long numerator,
    int positive_denominator) {
    const bool negative = numerator < 0;
    const unsigned long long magnitude = negative
        ? static_cast<unsigned long long>(-numerator)
        : static_cast<unsigned long long>(numerator);
    const unsigned long long denominator =
        static_cast<unsigned long long>(positive_denominator);
    unsigned long long quotient = magnitude / denominator;
    const unsigned long long remainder = magnitude % denominator;
    const unsigned long long twice_remainder = remainder * 2ULL;
    if (twice_remainder > denominator ||
        (twice_remainder == denominator && (quotient & 1ULL) != 0ULL)) {
        ++quotient;
    }
    const int rounded = static_cast<int>(quotient);
    return negative ? -rounded : rounded;
}

__global__ void refresh_occ_bitmap_from_packed_states_kernel(
    const AgentState* __restrict__ d_states,
    StatelessGridWorldContext ctx)
{
    int job_idx = blockIdx.x;
    int tid = threadIdx.x;
    int agent_count = ctx.n_agents;
    int job_base_offset = job_idx * agent_count;

    __shared__ uint16_t real_env_id;
    if (tid == 0) {
        real_env_id = d_states[job_base_offset].env_id;
    }
    __syncthreads();

    if (real_env_id >= ctx.n_envs) return;

    uint32_t* env_occ = ctx.d_grid_ocp + real_env_id * MAP_OFFSET;
    for (int i = tid; i < MAP_OFFSET; i += blockDim.x) {
        env_occ[i] = 0;
    }
    __syncthreads();

    for (int i = tid; i < agent_count; i += blockDim.x) {
        AgentState state = d_states[job_base_offset + i];
        if (state.env_id >= ctx.n_envs) {
            continue;
        }

        int word_idx = static_cast<int>(state.pos_x) * COL_OFFSET + (static_cast<int>(state.pos_y) >> 5);
        uint32_t mask = 1u << (state.pos_y & 31);
        atomicOr(&env_occ[word_idx], mask);
    }
}

// Optimized kernel with all 7 optimizations:
// 1. Bank conflict padding (MAP_H + 1)
// 2. Vectorized memory access (32-bit loads)
// 3. Fused node-feature generation + per-node edge counting
// 4. __launch_bounds__ for occupancy
// 5. __ldg__ for texture cache

__global__ void __launch_bounds__(256, 2)
generate_imitation_obs_fullmap_shared_kernel(
    const AgentState* __restrict__ d_states, 
    StatelessGridWorldContext ctx)
{
    int job_idx = blockIdx.x;
    int tid = threadIdx.x;
    int agent_count = ctx.n_agents;
    int job_base_offset = job_idx * agent_count;
    int env_node_base = 0;

    StepOutput& _step_output = ctx.step_output;

    __shared__ int real_env_id;
    if (tid == 0) {
        real_env_id = d_states[job_base_offset].env_id; 
    }
    __syncthreads();

    if (real_env_id >= ctx.n_envs || tid >= agent_count) return;
    env_node_base = real_env_id * agent_count;

    int obs_r = RADIUS;
    int obs_diam = 2 * obs_r + 1;
    int pyg_obs_diam = PYG_OBS_DIAM;
    int pyg_obs_area = PYG_OBS_AREA;

    // Optimization 1: Bank conflict padding (MAP_H + 1 instead of MAP_H)
    // Optimization 2: Use int32_t for vectorized access where possible
    extern __shared__ int16_t s_mem[];
    // Padded: MAP_W * (MAP_H + 1) to avoid bank conflict on column access
    int16_t* s_temp_map = s_mem;
    
    // Agent coordinates in shared memory
    int16_t* s_agent_x = &s_temp_map[MAP_W * (MAP_H + 1)];
    int16_t* s_agent_y = &s_agent_x[agent_count];
    int16_t* s_goal_x = &s_agent_y[agent_count];
    int16_t* s_goal_y = &s_goal_x[agent_count];

    __shared__ uint32_t s_map_static[MAP_OFFSET];

    // 1. 并行读取静态地图
    #pragma unroll 4
    for (int i = tid; i < MAP_OFFSET; i += blockDim.x) {
        // Optimization 7: __ldg__ for texture cache
        s_map_static[i] = __ldg(&ctx.energy_map_ctx.d_maps[real_env_id * MAP_OFFSET + i]);
    }
    __syncthreads();

    // 2. 初始化临时地图 with padding
    // Optimization 2: Vectorized initialization (process 2 cells at a time)
    for (int i = tid; i < MAP_W * MAP_H; i += blockDim.x) {
        int x = i / MAP_H;
        int y = i % MAP_H;
        // Use padded index: x * (MAP_H + 1) + y
        s_temp_map[x * (MAP_H + 1) + y] = is_wall(s_map_static, x, y) ? -1 : 0;
    }
    __syncthreads();

    // 3. 读取并填充自身状态
    int global_agent_idx = env_node_base + tid;
    AgentState my_state = d_states[global_agent_idx];
    
    s_agent_x[tid] = my_state.pos_x;
    s_agent_y[tid] = my_state.pos_y;
    s_goal_x[tid]  = my_state.target_x;
    s_goal_y[tid]  = my_state.target_y;

    // Use padded index for s_temp_map
    s_temp_map[my_state.pos_x * (MAP_H + 1) + my_state.pos_y] = tid + 1;
    __syncthreads();

    // ==========================================
    // 自身观测生成 + edge count
    // ==========================================
    int cx = s_agent_x[tid];
    int cy = s_agent_y[tid];
    int gx = s_goal_x[tid];
    int gy = s_goal_y[tid];

    int pyg_base  = global_agent_idx * (4 * pyg_obs_area);
    
    // Optimization 7: __ldg__ for texture cache
    float center_ctg = 0.0f;
    int global_map_base = global_agent_idx * MAP_W * MAP_H;
    if (cx >= 0 && cx < MAP_W && cy >= 0 && cy < MAP_H) {
        center_ctg = __ldg(&ctx.energy_map_ctx.d_global_energy_maps[global_map_base + cx * MAP_H + cy]);
    }

    // Initialize output arrays for this node
    #pragma unroll
    for (int k = 0; k < pyg_obs_area; ++k) {
        _step_output.d_pyg_x[pyg_base + 0 * pyg_obs_area + k] = 0.0f;
        _step_output.d_pyg_x[pyg_base + 1 * pyg_obs_area + k] = 0.0f;
        _step_output.d_pyg_x[pyg_base + 2 * pyg_obs_area + k] = 0.0f;
        _step_output.d_pyg_x[pyg_base + 3 * pyg_obs_area + k] = 0.0f;
    }

    // 4. 生成 11x11 局部视野 (using pre-computed offsets)
    // Optimization 7: __ldg__ for texture cache
    #pragma unroll 4
    for (int dx = -obs_r; dx <= obs_r; ++dx) {
        #pragma unroll
        for (int dy = -obs_r; dy <= obs_r; ++dy) {
            int tx = cx + dx;
            int ty = cy + dy;
            int local_idx = (dx + obs_r + 1) * pyg_obs_diam + (dy + obs_r + 1);

            float obs_val = 0.0f;
            float ctg_val = 0.0f;

            if (tx < -1 || tx > MAP_W || ty < -1 || ty > MAP_H) {
                obs_val = 0.0f;
                ctg_val = ctx.clamp_value;
            } else if (tx == -1 || tx == MAP_W || ty == -1 || ty == MAP_H) {
                obs_val = 1.0f;
                ctg_val = ctx.clamp_value;
            } else {
                // Optimization 1: Use padded index for bank conflict avoidance
                int padded_idx = tx * (MAP_H + 1) + ty;
                int16_t map_val = s_temp_map[padded_idx];
                if (map_val == -1) {
                    obs_val = 1.0f;
                    ctg_val = ctx.clamp_value;
                } else if (map_val > 0) {
                    _step_output.d_pyg_x[pyg_base + 1 * pyg_obs_area + local_idx] = 1.0f;
                }
                if (map_val != -1) {
                    // Optimization 7: __ldg__ for texture cache
                    float target_ctg = __ldg(&ctx.energy_map_ctx.d_global_energy_maps[global_map_base + tx * MAP_H + ty]);
                    ctg_val = (center_ctg - target_ctg) / (2.0f * obs_r);
                    ctg_val = fmaxf(-ctx.clamp_value, fminf(ctx.clamp_value, ctg_val));
                }
            }

            _step_output.d_pyg_x[pyg_base + 0 * pyg_obs_area + local_idx] = obs_val;
            _step_output.d_pyg_x[pyg_base + 3 * pyg_obs_area + local_idx] = ctg_val;
        }
    }

    // 5. 目标通道投影
    int rel_gx = gx - cx;
    int rel_gy = gy - cy;
    
    if (abs(rel_gx) <= obs_r && abs(rel_gy) <= obs_r) {
        int local_idx = (rel_gx + obs_r + 1) * pyg_obs_diam + (rel_gy + obs_r + 1);
        _step_output.d_pyg_x[pyg_base + 2 * pyg_obs_area + local_idx] = 1.0f;
    } else {
        float angle = atan2f((float)rel_gy, (float)rel_gx);
        float dist = static_cast<float>(pyg_obs_diam / 2);
        float sign_x = fast_sign((float)rel_gx);
        float sign_y = fast_sign((float)rel_gy);

        int goalX_FOV, goalY_FOV;
        
        if ((angle >= CUDART_PI_F / 4.0f && angle <= CUDART_PI_F * 3.0f / 4.0f) ||
            (angle >= -CUDART_PI_F * 3.0f / 4.0f && angle <= -CUDART_PI_F / 4.0f)) {
            goalY_FOV = static_cast<int>(dist * (sign_y + 1.0f));
            goalX_FOV = (pyg_obs_diam / 2) + round_ratio_to_nearest_even(
                static_cast<long long>(pyg_obs_diam / 2) * rel_gx, abs(rel_gy));
        } else {
            goalX_FOV = static_cast<int>(dist * (sign_x + 1.0f));
            goalY_FOV = (pyg_obs_diam / 2) + round_ratio_to_nearest_even(
                static_cast<long long>(pyg_obs_diam / 2) * rel_gy, abs(rel_gx));
        }

        goalX_FOV = max(0, min(pyg_obs_diam - 1, goalX_FOV));
        goalY_FOV = max(0, min(pyg_obs_diam - 1, goalY_FOV));
        _step_output.d_pyg_x[pyg_base + 2 * pyg_obs_area + goalX_FOV * pyg_obs_diam + goalY_FOV] = 1.0f;
    }

    int edge_count = 0;
    for (int other = 0; other < agent_count; ++other) {
        if (other == tid) continue;
        int dx = cx - s_agent_x[other];
        int dy = cy - s_agent_y[other];
        if (dx * dx + dy * dy <= MAGAT_COMM_RADIUS_SQ) {
            edge_count += 1;
        }
    }

    _step_output.d_pyg_pos[global_agent_idx * 2 + 0] = (float)cx;
    _step_output.d_pyg_pos[global_agent_idx * 2 + 1] = (float)cy;
    ctx.d_edge_counts[global_agent_idx] = edge_count;
}

__global__ void materialize_pyg_nodes_and_count_edges_local_gather_kernel(
    const AgentState* __restrict__ d_states,
    StatelessGridWorldContext ctx)
{
    int env_id = blockIdx.x;
    int tid = threadIdx.x;
    int agent_count = ctx.n_agents;
    int env_base = env_id * agent_count;
    int target_agent_id = blockIdx.y * blockDim.x + tid;

    if (env_id >= ctx.n_envs) return;

    extern __shared__ int16_t s_agent_data[];
    int16_t* s_agent_x = s_agent_data;
    int16_t* s_agent_y = s_agent_x + agent_count;
    int16_t* s_goal_x = s_agent_y + agent_count;
    int16_t* s_goal_y = s_goal_x + agent_count;

    for (int i = tid; i < agent_count; i += blockDim.x) {
        AgentState state = d_states[env_base + i];
        s_agent_x[i] = static_cast<int16_t>(state.pos_x);
        s_agent_y[i] = static_cast<int16_t>(state.pos_y);
        s_goal_x[i] = static_cast<int16_t>(state.target_x);
        s_goal_y[i] = static_cast<int16_t>(state.target_y);
    }
    __syncthreads();

    if (target_agent_id >= agent_count) return;

    int global_agent_idx = env_base + target_agent_id;
    AgentState my_state = d_states[global_agent_idx];
    if (my_state.env_id >= ctx.n_envs) {
        return;
    }

    const uint32_t* env_static = ctx.energy_map_ctx.d_maps + env_id * MAP_OFFSET;
    const uint32_t* env_occ = ctx.d_grid_ocp + env_id * MAP_OFFSET;
    StepOutput& step_output = ctx.step_output;

    int cx = static_cast<int>(s_agent_x[target_agent_id]);
    int cy = static_cast<int>(s_agent_y[target_agent_id]);
    int gx = static_cast<int>(s_goal_x[target_agent_id]);
    int gy = static_cast<int>(s_goal_y[target_agent_id]);

    int pyg_obs_diam = PYG_OBS_DIAM;
    int pyg_obs_area = PYG_OBS_AREA;
    int pyg_base = global_agent_idx * (4 * pyg_obs_area);

    #pragma unroll
    for (int k = 0; k < pyg_obs_area; ++k) {
        step_output.d_pyg_x[pyg_base + 0 * pyg_obs_area + k] = 0.0f;
        step_output.d_pyg_x[pyg_base + 1 * pyg_obs_area + k] = 0.0f;
        step_output.d_pyg_x[pyg_base + 2 * pyg_obs_area + k] = 0.0f;
        step_output.d_pyg_x[pyg_base + 3 * pyg_obs_area + k] = 0.0f;
    }

    float center_ctg = 0.0f;
    int global_map_base = global_agent_idx * MAP_W * MAP_H;
    if (cx >= 0 && cx < MAP_W && cy >= 0 && cy < MAP_H) {
        center_ctg = __ldg(&ctx.energy_map_ctx.d_global_energy_maps[global_map_base + cx * MAP_H + cy]);
    }

    const int window_y_start = cy - RADIUS;
    #pragma unroll 4
    for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
        int tx = cx + dx;
        int local_row = dx + RADIUS + 1;
        int local_row_base = local_row * pyg_obs_diam;
        bool row_is_far_oob = (tx < -1 || tx > MAP_W);
        bool row_is_border = (tx == -1 || tx == MAP_W);
        uint32_t static_row_bits = 0u;
        uint32_t occ_row_bits = 0u;
        const uint8_t* energy_row = nullptr;

        if (tx >= 0 && tx < MAP_W) {
            static_row_bits = gather_row_window_bits(env_static, tx, window_y_start);
            occ_row_bits = gather_row_window_bits(env_occ, tx, window_y_start);
            energy_row = ctx.energy_map_ctx.d_global_energy_maps + global_map_base + tx * MAP_H;
        }

        #pragma unroll
        for (int local_col = 0; local_col < LOCAL_GATHER_PATCH_DIAM; ++local_col) {
            int ty = window_y_start + local_col;
            int local_idx = local_row_base + (local_col + 1);

            float obs_val = 0.0f;
            float occ_val = 0.0f;
            float ctg_val = 0.0f;

            if (row_is_far_oob || ty < -1 || ty > MAP_H) {
                ctg_val = ctx.clamp_value;
            } else if (row_is_border || ty == -1 || ty == MAP_H) {
                obs_val = 1.0f;
                ctg_val = ctx.clamp_value;
            } else {
                uint32_t local_mask = 1u << local_col;
                if ((static_row_bits & local_mask) != 0u) {
                    obs_val = 1.0f;
                    ctg_val = ctx.clamp_value;
                } else if ((occ_row_bits & local_mask) != 0u) {
                    occ_val = 1.0f;
                }
                if ((static_row_bits & local_mask) == 0u) {
                    float target_ctg = static_cast<float>(__ldg(&energy_row[ty]));
                    ctg_val = (center_ctg - target_ctg) / (2.0f * RADIUS);
                    ctg_val = fmaxf(-ctx.clamp_value, fminf(ctx.clamp_value, ctg_val));
                }
            }

            step_output.d_pyg_x[pyg_base + 0 * pyg_obs_area + local_idx] = obs_val;
            step_output.d_pyg_x[pyg_base + 1 * pyg_obs_area + local_idx] = occ_val;
            step_output.d_pyg_x[pyg_base + 3 * pyg_obs_area + local_idx] = ctg_val;
        }
    }

    int rel_gx = gx - cx;
    int rel_gy = gy - cy;
    if (abs(rel_gx) <= RADIUS && abs(rel_gy) <= RADIUS) {
        int local_idx = (rel_gx + RADIUS + 1) * pyg_obs_diam + (rel_gy + RADIUS + 1);
        step_output.d_pyg_x[pyg_base + 2 * pyg_obs_area + local_idx] = 1.0f;
    } else {
        float angle = atan2f(static_cast<float>(rel_gy), static_cast<float>(rel_gx));
        float dist = static_cast<float>(pyg_obs_diam / 2);
        float sign_x = fast_sign(static_cast<float>(rel_gx));
        float sign_y = fast_sign(static_cast<float>(rel_gy));

        int goal_x_fov;
        int goal_y_fov;
        if ((angle >= CUDART_PI_F / 4.0f && angle <= CUDART_PI_F * 3.0f / 4.0f) ||
            (angle >= -CUDART_PI_F * 3.0f / 4.0f && angle <= -CUDART_PI_F / 4.0f)) {
            goal_y_fov = static_cast<int>(dist * (sign_y + 1.0f));
            goal_x_fov = (pyg_obs_diam / 2) + round_ratio_to_nearest_even(
                static_cast<long long>(pyg_obs_diam / 2) * rel_gx, abs(rel_gy));
        } else {
            goal_x_fov = static_cast<int>(dist * (sign_x + 1.0f));
            goal_y_fov = (pyg_obs_diam / 2) + round_ratio_to_nearest_even(
                static_cast<long long>(pyg_obs_diam / 2) * rel_gy, abs(rel_gx));
        }

        goal_x_fov = max(0, min(pyg_obs_diam - 1, goal_x_fov));
        goal_y_fov = max(0, min(pyg_obs_diam - 1, goal_y_fov));
        step_output.d_pyg_x[pyg_base + 2 * pyg_obs_area + goal_x_fov * pyg_obs_diam + goal_y_fov] = 1.0f;
    }

    int edge_count = 0;
    for (int other = 0; other < agent_count; ++other) {
        if (other == target_agent_id) continue;
        int dx = cx - static_cast<int>(s_agent_x[other]);
        int dy = cy - static_cast<int>(s_agent_y[other]);
        if (dx * dx + dy * dy <= MAGAT_COMM_RADIUS_SQ) {
            edge_count += 1;
        }
    }

    step_output.d_pyg_pos[global_agent_idx * 2 + 0] = static_cast<float>(cx);
    step_output.d_pyg_pos[global_agent_idx * 2 + 1] = static_cast<float>(cy);
    ctx.d_edge_counts[global_agent_idx] = edge_count;
}

__global__ void __launch_bounds__(128, 2)
materialize_pyg_nodes_and_count_edges_local_gather_staged_kernel(
    const AgentState* __restrict__ d_states,
    StatelessGridWorldContext ctx)
{
    int env_id = blockIdx.x;
    int tid = threadIdx.x;
    int agent_count = ctx.n_agents;
    int env_base = env_id * agent_count;
    int target_agent_id = blockIdx.y * blockDim.x + tid;

    if (env_id >= ctx.n_envs) return;

    extern __shared__ int16_t s_agent_data[];
    int16_t* s_agent_x = s_agent_data;
    int16_t* s_agent_y = s_agent_x + agent_count;
    int16_t* s_goal_x = s_agent_y + agent_count;
    int16_t* s_goal_y = s_goal_x + agent_count;

    __shared__ uint32_t s_env_static[MAP_OFFSET];
    __shared__ uint32_t s_env_occ[MAP_OFFSET];

    const uint32_t* env_static_global = ctx.energy_map_ctx.d_maps + env_id * MAP_OFFSET;
    const uint32_t* env_occ_global = ctx.d_grid_ocp + env_id * MAP_OFFSET;

    for (int i = tid; i < MAP_OFFSET; i += blockDim.x) {
        s_env_static[i] = env_static_global[i];
        s_env_occ[i] = env_occ_global[i];
    }

    for (int i = tid; i < agent_count; i += blockDim.x) {
        AgentState state = d_states[env_base + i];
        s_agent_x[i] = static_cast<int16_t>(state.pos_x);
        s_agent_y[i] = static_cast<int16_t>(state.pos_y);
        s_goal_x[i] = static_cast<int16_t>(state.target_x);
        s_goal_y[i] = static_cast<int16_t>(state.target_y);
    }
    __syncthreads();

    if (target_agent_id >= agent_count) return;

    int global_agent_idx = env_base + target_agent_id;
    AgentState my_state = d_states[global_agent_idx];
    if (my_state.env_id >= ctx.n_envs) {
        return;
    }

    StepOutput& step_output = ctx.step_output;

    int cx = static_cast<int>(s_agent_x[target_agent_id]);
    int cy = static_cast<int>(s_agent_y[target_agent_id]);
    int gx = static_cast<int>(s_goal_x[target_agent_id]);
    int gy = static_cast<int>(s_goal_y[target_agent_id]);

    int pyg_obs_diam = PYG_OBS_DIAM;
    int pyg_obs_area = PYG_OBS_AREA;
    int pyg_base = global_agent_idx * (4 * pyg_obs_area);

    #pragma unroll
    for (int k = 0; k < pyg_obs_area; ++k) {
        step_output.d_pyg_x[pyg_base + 0 * pyg_obs_area + k] = 0.0f;
        step_output.d_pyg_x[pyg_base + 1 * pyg_obs_area + k] = 0.0f;
        step_output.d_pyg_x[pyg_base + 2 * pyg_obs_area + k] = 0.0f;
        step_output.d_pyg_x[pyg_base + 3 * pyg_obs_area + k] = 0.0f;
    }

    float center_ctg = 0.0f;
    int global_map_base = global_agent_idx * MAP_W * MAP_H;
    if (cx >= 0 && cx < MAP_W && cy >= 0 && cy < MAP_H) {
        center_ctg = __ldg(&ctx.energy_map_ctx.d_global_energy_maps[global_map_base + cx * MAP_H + cy]);
    }

    const int window_y_start = cy - RADIUS;
    #pragma unroll 4
    for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
        int tx = cx + dx;
        int local_row = dx + RADIUS + 1;
        int local_row_base = local_row * pyg_obs_diam;
        bool row_is_far_oob = (tx < -1 || tx > MAP_W);
        bool row_is_border = (tx == -1 || tx == MAP_W);
        uint32_t static_row_bits = 0u;
        uint32_t occ_row_bits = 0u;
        const uint8_t* energy_row = nullptr;

        if (tx >= 0 && tx < MAP_W) {
            static_row_bits = gather_row_window_bits_shared(s_env_static, tx, window_y_start);
            occ_row_bits = gather_row_window_bits_shared(s_env_occ, tx, window_y_start);
            energy_row = ctx.energy_map_ctx.d_global_energy_maps + global_map_base + tx * MAP_H;
        }

        #pragma unroll
        for (int local_col = 0; local_col < LOCAL_GATHER_PATCH_DIAM; ++local_col) {
            int ty = window_y_start + local_col;
            int local_idx = local_row_base + (local_col + 1);

            float obs_val = 0.0f;
            float occ_val = 0.0f;
            float ctg_val = 0.0f;

            if (row_is_far_oob || ty < -1 || ty > MAP_H) {
                ctg_val = ctx.clamp_value;
            } else if (row_is_border || ty == -1 || ty == MAP_H) {
                obs_val = 1.0f;
                ctg_val = ctx.clamp_value;
            } else {
                uint32_t local_mask = 1u << local_col;
                if ((static_row_bits & local_mask) != 0u) {
                    obs_val = 1.0f;
                    ctg_val = ctx.clamp_value;
                } else if ((occ_row_bits & local_mask) != 0u) {
                    occ_val = 1.0f;
                }
                if ((static_row_bits & local_mask) == 0u) {
                    float target_ctg = static_cast<float>(__ldg(&energy_row[ty]));
                    ctg_val = (center_ctg - target_ctg) / (2.0f * RADIUS);
                    ctg_val = fmaxf(-ctx.clamp_value, fminf(ctx.clamp_value, ctg_val));
                }
            }

            step_output.d_pyg_x[pyg_base + 0 * pyg_obs_area + local_idx] = obs_val;
            step_output.d_pyg_x[pyg_base + 1 * pyg_obs_area + local_idx] = occ_val;
            step_output.d_pyg_x[pyg_base + 3 * pyg_obs_area + local_idx] = ctg_val;
        }
    }

    int rel_gx = gx - cx;
    int rel_gy = gy - cy;
    if (abs(rel_gx) <= RADIUS && abs(rel_gy) <= RADIUS) {
        int local_idx = (rel_gx + RADIUS + 1) * pyg_obs_diam + (rel_gy + RADIUS + 1);
        step_output.d_pyg_x[pyg_base + 2 * pyg_obs_area + local_idx] = 1.0f;
    } else {
        float angle = atan2f(static_cast<float>(rel_gy), static_cast<float>(rel_gx));
        float dist = static_cast<float>(pyg_obs_diam / 2);
        float sign_x = fast_sign(static_cast<float>(rel_gx));
        float sign_y = fast_sign(static_cast<float>(rel_gy));

        int goal_x_fov;
        int goal_y_fov;
        if ((angle >= CUDART_PI_F / 4.0f && angle <= CUDART_PI_F * 3.0f / 4.0f) ||
            (angle >= -CUDART_PI_F * 3.0f / 4.0f && angle <= -CUDART_PI_F / 4.0f)) {
            goal_y_fov = static_cast<int>(dist * (sign_y + 1.0f));
            goal_x_fov = (pyg_obs_diam / 2) + round_ratio_to_nearest_even(
                static_cast<long long>(pyg_obs_diam / 2) * rel_gx, abs(rel_gy));
        } else {
            goal_x_fov = static_cast<int>(dist * (sign_x + 1.0f));
            goal_y_fov = (pyg_obs_diam / 2) + round_ratio_to_nearest_even(
                static_cast<long long>(pyg_obs_diam / 2) * rel_gy, abs(rel_gx));
        }

        goal_x_fov = max(0, min(pyg_obs_diam - 1, goal_x_fov));
        goal_y_fov = max(0, min(pyg_obs_diam - 1, goal_y_fov));
        step_output.d_pyg_x[pyg_base + 2 * pyg_obs_area + goal_x_fov * pyg_obs_diam + goal_y_fov] = 1.0f;
    }

    int edge_count = 0;
    for (int other = 0; other < agent_count; ++other) {
        if (other == target_agent_id) continue;
        int dx = cx - static_cast<int>(s_agent_x[other]);
        int dy = cy - static_cast<int>(s_agent_y[other]);
        if (dx * dx + dy * dy <= MAGAT_COMM_RADIUS_SQ) {
            edge_count += 1;
        }
    }

    step_output.d_pyg_pos[global_agent_idx * 2 + 0] = static_cast<float>(cx);
    step_output.d_pyg_pos[global_agent_idx * 2 + 1] = static_cast<float>(cy);
    ctx.d_edge_counts[global_agent_idx] = edge_count;
}

inline bool should_use_async_local_gather_kernel(const StatelessGridWorldContext& ctx) {
    if (!ctx.supports_async_local_gather) {
        return false;
    }
    if (ctx.local_gather_impl_mode == PYG_LOCAL_GATHER_IMPL_SCALAR) {
        return false;
    }
    return true;
}

void launch_materialize_pyg_nodes_from_packed_states(
    const AgentState* d_states,
    StatelessGridWorldContext ctx)
{
    const bool legacy_supported = can_use_legacy_fullmap_builder(ctx);
    if (ctx.builder_mode == PYG_BUILDER_MODE_LEGACY_FULLMAP) {
        TORCH_CHECK(
            legacy_supported,
            "legacy_fullmap builder mode requires n_agents <= 256 and shared memory usage within the legacy budget");
    }

    const bool use_legacy =
        (ctx.builder_mode == PYG_BUILDER_MODE_LEGACY_FULLMAP) ||
        (ctx.builder_mode == PYG_BUILDER_MODE_AUTO && legacy_supported);

    if (use_legacy) {
        int blocks = ctx.n_envs;
        int threads = min(ctx.n_agents, 256);
        size_t shm_size = legacy_fullmap_builder_smem_bytes(ctx.n_agents);
        generate_imitation_obs_fullmap_shared_kernel<<<blocks, threads, shm_size>>>(d_states, ctx);
    } else {
        int threads = LOCAL_GATHER_THREADS;
        dim3 blocks(ctx.n_envs, ceil_div_int(ctx.n_agents, threads));
        size_t shm_size = 4 * static_cast<size_t>(ctx.n_agents) * sizeof(int16_t);
        if (should_use_async_local_gather_kernel(ctx)) {
            materialize_pyg_nodes_and_count_edges_local_gather_staged_kernel<<<blocks, threads, shm_size>>>(d_states, ctx);
        } else {
            materialize_pyg_nodes_and_count_edges_local_gather_kernel<<<blocks, threads, shm_size>>>(d_states, ctx);
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void launch_imitation_obs_generator(
    const torch::Tensor& states_tensor, 
    StatelessGridWorldContext ctx) 
{
    TORCH_CHECK(states_tensor.is_cuda(), "states_tensor must be on CUDA device");
    TORCH_CHECK(states_tensor.dtype() == torch::kInt16, "states_tensor must be int16");
    TORCH_CHECK(states_tensor.dim() == 2, "states_tensor must be 2D (cap, 8)");
    TORCH_CHECK(states_tensor.size(1) == 8, "states_tensor feature dimension must be 8");
    TORCH_CHECK(states_tensor.is_contiguous(), "states_tensor must be contiguous");

    launch_refresh_occ_bitmap_from_packed_states(states_tensor, ctx);
    auto d_states = reinterpret_cast<const AgentState*>((uint16_t*)states_tensor.data_ptr<int16_t>());
    launch_materialize_pyg_nodes_from_packed_states(d_states, ctx);
}

void launch_imitation_obs_generator_from_packed_states(
    const AgentState* d_states,
    StatelessGridWorldContext ctx)
{
    launch_materialize_pyg_nodes_from_packed_states(d_states, ctx);
}

void launch_refresh_occ_bitmap_from_packed_states(
    const torch::Tensor& packed_states,
    StatelessGridWorldContext ctx)
{
    TORCH_CHECK(packed_states.is_cuda(), "packed_states must be on CUDA device");
    TORCH_CHECK(packed_states.dtype() == torch::kInt16, "packed_states must be int16");
    TORCH_CHECK(packed_states.dim() == 2, "packed_states must be 2D (N, 8)");
    TORCH_CHECK(packed_states.size(1) == 8, "packed_states feature dimension must be 8");
    TORCH_CHECK(packed_states.is_contiguous(), "packed_states must be contiguous");

    auto d_states = reinterpret_cast<const AgentState*>((uint16_t*)packed_states.data_ptr<int16_t>());
    int blocks = ctx.n_envs;
    int threads = min(ctx.n_agents, 256);
    refresh_occ_bitmap_from_packed_states_kernel<<<blocks, threads>>>(d_states, ctx);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void fill_pyg_ctg_channel_kernel(
    const AgentState* __restrict__ d_states,
    StatelessGridWorldContext ctx)
{
    int env_id = blockIdx.x;
    int tid = threadIdx.x;
    if (env_id >= ctx.n_envs || tid >= ctx.n_agents) return;

    int global_agent_idx = env_id * ctx.n_agents + tid;
    AgentState my_state = d_states[global_agent_idx];
    int cx = my_state.pos_x;
    int cy = my_state.pos_y;

    int obs_r = RADIUS;
    int obs_diam = 2 * obs_r + 1;
    int pyg_obs_diam = PYG_OBS_DIAM;
    int pyg_obs_area = PYG_OBS_AREA;
    int pyg_base = global_agent_idx * (4 * pyg_obs_area) + 3 * pyg_obs_area;

    float center_ctg = 0.0f;
    int global_map_base = global_agent_idx * MAP_W * MAP_H;
    if (cx >= 0 && cx < MAP_W && cy >= 0 && cy < MAP_H) {
        center_ctg = __ldg(&ctx.energy_map_ctx.d_global_energy_maps[global_map_base + cx * MAP_H + cy]);
    }

    #pragma unroll 4
    for (int dx = -obs_r; dx <= obs_r; ++dx) {
        #pragma unroll
        for (int dy = -obs_r; dy <= obs_r; ++dy) {
            int tx = cx + dx;
            int ty = cy + dy;
            int local_idx = (dx + obs_r + 1) * pyg_obs_diam + (dy + obs_r + 1);
            float ctg_val = 0.0f;

            if (tx >= 0 && tx < MAP_W && ty >= 0 && ty < MAP_H) {
                float target_ctg = __ldg(&ctx.energy_map_ctx.d_global_energy_maps[global_map_base + tx * MAP_H + ty]);
                ctg_val = (center_ctg - target_ctg) / (2.0f * obs_r);
                ctg_val = fmaxf(-ctx.clamp_value, fminf(ctx.clamp_value, ctg_val));
            } else {
                ctg_val = ctx.clamp_value;
            }

            ctx.step_output.d_pyg_x[pyg_base + local_idx] = ctg_val;
        }
    }
}

void launch_fill_pyg_ctg_channel(
    const torch::Tensor& states_tensor,
    StatelessGridWorldContext ctx)
{
    TORCH_CHECK(states_tensor.is_cuda(), "states_tensor must be on CUDA device");
    TORCH_CHECK(states_tensor.dtype() == torch::kInt16, "states_tensor must be int16");
    TORCH_CHECK(states_tensor.dim() == 2, "states_tensor must be 2D (cap, 8)");
    TORCH_CHECK(states_tensor.size(1) == 8, "states_tensor feature dimension must be 8");
    TORCH_CHECK(states_tensor.is_contiguous(), "states_tensor must be contiguous");

    auto d_states = reinterpret_cast<const AgentState*>((uint16_t*)states_tensor.data_ptr<int16_t>());
    int blocks = ctx.n_envs;
    int threads = min(ctx.n_agents, 256);
    fill_pyg_ctg_channel_kernel<<<blocks, threads>>>(d_states, ctx);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void generate_pyg_edges_kernel(
    const int64_t* __restrict__ d_node_prefix,
    const float* __restrict__ d_node_pos,
    int64_t* __restrict__ d_pyg_edge_index,
    float* __restrict__ d_pyg_edge_attr,
    int agent_count,
    int64_t total_edges_capacity,
    int n_envs)
{
    int env_id = blockIdx.x;
    int source_agent_id = blockIdx.y * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (env_id >= n_envs) return;

    extern __shared__ int16_t s_agent_pos[];
    int16_t* s_agent_x = s_agent_pos;
    int16_t* s_agent_y = s_agent_pos + agent_count;
    int env_base = env_id * agent_count;
    for (int i = tid; i < agent_count; i += blockDim.x) {
        int node_idx = env_base + i;
        s_agent_x[i] = static_cast<int16_t>(d_node_pos[node_idx * 2 + 0]);
        s_agent_y[i] = static_cast<int16_t>(d_node_pos[node_idx * 2 + 1]);
    }
    __syncthreads();

    if (source_agent_id >= agent_count) return;

    int node_idx = env_base + source_agent_id;
    int cx = static_cast<int>(s_agent_x[source_agent_id]);
    int cy = static_cast<int>(s_agent_y[source_agent_id]);
    int64_t out_idx = d_node_prefix[node_idx];
    for (int other = 0; other < agent_count; ++other) {
        if (other == source_agent_id) continue;
        int dx = cx - static_cast<int>(s_agent_x[other]);
        int dy = cy - static_cast<int>(s_agent_y[other]);
        if (dx * dx + dy * dy <= MAGAT_COMM_RADIUS_SQ) {
            int dst = env_base + other;
            float edge_dx = static_cast<float>(dx);
            float edge_dy = static_cast<float>(dy);
            d_pyg_edge_index[out_idx] = static_cast<int64_t>(node_idx);
            d_pyg_edge_index[total_edges_capacity + out_idx] = static_cast<int64_t>(dst);
            d_pyg_edge_attr[out_idx * 3 + 0] = edge_dx;
            d_pyg_edge_attr[out_idx * 3 + 1] = edge_dy;
            d_pyg_edge_attr[out_idx * 3 + 2] = fabsf(edge_dx) + fabsf(edge_dy);
            out_idx++;
        }
    }
}

void launch_generate_pyg_edges_kernel(
    const torch::Tensor& node_prefix,
    const torch::Tensor& node_pos,
    torch::Tensor& pyg_edge_index,
    torch::Tensor& pyg_edge_attr,
    int agent_count,
    int n_envs,
    int total_nodes)
{
    TORCH_CHECK(node_prefix.is_cuda(), "node_prefix must be CUDA");
    TORCH_CHECK(node_pos.is_cuda(), "node_pos must be CUDA");
    TORCH_CHECK(pyg_edge_index.is_cuda() && pyg_edge_attr.is_cuda(), "pyg outputs must be CUDA");

    int threads = min(agent_count, 256);
    dim3 blocks(n_envs, ceil_div_int(agent_count, threads));
    int64_t total_edges_capacity = pyg_edge_attr.size(0);
    int64_t* d_pyg_edge_index_base = pyg_edge_index.data_ptr<int64_t>();
    size_t shm_size = 2 * agent_count * sizeof(int16_t);

    generate_pyg_edges_kernel<<<blocks, threads, shm_size>>>(
        node_prefix.data_ptr<int64_t>(),
        node_pos.data_ptr<float>(),
        d_pyg_edge_index_base,
        pyg_edge_attr.data_ptr<float>(),
        agent_count,
        total_edges_capacity,
        n_envs);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}
