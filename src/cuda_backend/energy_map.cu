#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <stdexcept>
#include <string>
#include "grid_world_cuda.h"
#include <torch/extension.h>
// 根据最大可能周长设定，256x256地图最大波前通常不超过 4000
#define MAX_QUEUE_SIZE 4096

namespace {

int compute_energy_map_shared_mem_size() {
    return GRID_SIZE * static_cast<int>(sizeof(uint8_t)) +
           ((GRID_SIZE + 31) / 32) * static_cast<int>(sizeof(uint32_t)) +
           2 * MAX_QUEUE_SIZE * static_cast<int>(sizeof(uint16_t));
}

void configure_energy_map_dynamic_smem_if_needed(int shared_mem_size) {
    int device_id = 0;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaGetDevice failed before energy map launch: ") +
            cudaGetErrorString(err));
    }

    int default_limit = 0;
    err = cudaDeviceGetAttribute(
        &default_limit,
        cudaDevAttrMaxSharedMemoryPerBlock,
        device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaDeviceGetAttribute(max shared per block) failed: ") +
            cudaGetErrorString(err));
    }

    if (shared_mem_size <= default_limit) {
        return;
    }

    int optin_limit = 0;
    err = cudaDeviceGetAttribute(
        &optin_limit,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaDeviceGetAttribute(max shared per block opt-in) failed: ") +
            cudaGetErrorString(err));
    }

    if (shared_mem_size > optin_limit) {
        throw std::runtime_error(
            "Energy map kernel shared-memory requirement (" +
            std::to_string(shared_mem_size) +
            " bytes) exceeds device opt-in limit (" +
            std::to_string(optin_limit) + " bytes)");
    }

    err = cudaFuncSetAttribute(
        generate_energy_map_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaFuncSetAttribute(max dynamic shared memory) failed: ") +
            cudaGetErrorString(err));
    }
}

}  // namespace

/**
 * 高性能 BFS 能量图生成 Kernel
 * Grid配置: <<< n_agents, 128或256, dynamic_smem_size >>>
 */
 // 输入只需要地图和目标位置，输出每个智能体的能量图
__global__ void generate_energy_map_kernel(
    EnergyMapContext ctx,
    int num_agents,
    int length,
    const AgentState* __restrict__ d_states,
    bool incremental_only)
{
    int block_id = blockIdx.x;
    if (block_id >= length) return;
    if (incremental_only && d_states[block_id].reset_flag == 0) return;
    
    int env_id = d_states[block_id].env_id;
    int local_agent_id = d_states[block_id].agent_id;
    int global_agent_idx = env_id * num_agents + local_agent_id;

    int map_word_offset = env_id * MAP_OFFSET;

    extern __shared__ uint8_t smem[];
    
    // 1. 距离/能量数组
    uint8_t* s_dist = smem; 
    
    // 2. 访问标记位图 (用 bitmask 压缩，避免 queue 重复入队，大幅降低内存冲突)
    // 256x256 只需要 8KB shared memory; 128x128 只需要 2KB
    uint32_t* s_visited_bits = (uint32_t*)(s_dist + GRID_SIZE); 
    int num_bit_words = (GRID_SIZE + 31) / 32;

    // 3. 技术 1: 双队列波前推进 (Two-Queue Frontier)
    uint16_t* s_q_curr = (uint16_t*)(s_visited_bits + num_bit_words);
    uint16_t* s_q_next = s_q_curr + MAX_QUEUE_SIZE;

    __shared__ int q_curr_sz;
    __shared__ int q_next_sz;
    __shared__ int current_dist;
    __shared__ int curr_queue_id;

    int tid = threadIdx.x;
    int lane_id = tid % 32; // Warp 内部通道 ID

    // --- 初始化 Shared Memory ---
    for (int i = tid; i < GRID_SIZE; i += blockDim.x) {
        s_dist[i] = 255; // 255 表示不可达
    }
    for (int i = tid; i < num_bit_words; i += blockDim.x) {
        s_visited_bits[i] = 0;
    }
    __syncthreads();

    // --- 线程 0 初始化起点 (以 Goal 作为 BFS 起点) ---
    if (tid == 0) {
        int gx = d_states[block_id].target_x;
        int gy = d_states[block_id].target_y;

        // 与现有项目一致: x 是行索引, y 是列索引
        int start_idx = gx * MAP_H + gy;
        
        s_dist[start_idx] = 0;
        // 标记起点已访问
        s_visited_bits[start_idx / 32] |= (1u << (start_idx % 32));
        
        s_q_curr[0] = start_idx;
        q_curr_sz = 1;
        q_next_sz = 0;
        current_dist = 0;
        curr_queue_id = 0;
    }
    __syncthreads();

    // ==========================================================
    // 技术 4: 混合粒度策略 (通过 grid-stride loop 自适应波前大小)
    // 当 q_curr_sz 很小时，只有少数 warp 活跃，避免了过早分支导致的性能惩罚
    // ==========================================================
    while (q_curr_sz > 0 && current_dist < 254) {
        uint16_t* read_queue = (curr_queue_id == 0) ? s_q_curr : s_q_next;
        uint16_t* write_queue = (curr_queue_id == 0) ? s_q_next : s_q_curr;
        
        // --- 核心 BFS 扩展循环 ---
        // 注意：为了使用 Warp Primitives，即使部分线程没有任务，也要让整个 Warp 进入循环体以参与同步
        for (int i = tid; i < ((q_curr_sz + 31) & ~31); i += blockDim.x) {
            
            int num_valid_neighbors = 0;
            int valid_nodes[4]; // 本地暂存合法的邻居节点
            
            if (i < q_curr_sz) {
                int node_idx = read_queue[i];
                int nx = node_idx / MAP_H;
                int ny = node_idx % MAP_H;

                int dx[4] = {0, 0, -1, 1};
                int dy[4] = {-1, 1, 0, 0};

                for (int d = 0; d < 4; ++d) {
                    int nnx = nx + dx[d];
                    int nny = ny + dy[d];

                    if (nnx >= 0 && nnx < MAP_W && nny >= 0 && nny < MAP_H) {
                        int n_idx = nnx * MAP_H + nny;
                        
                        // 1. 检查障碍物 (直接读取位图)
                        int word_idx = n_idx / 32;
                        int bit_offset = n_idx % 32;
                        uint32_t mask = 1u << bit_offset;
                        
                        bool is_static_obs = (ctx.d_maps[map_word_offset + word_idx] & mask) != 0;

                        if (!is_static_obs) {
                            // 2. 原子的 Bitmask 访问标记 (极大减少 shared memory 的写入冲突)
                            uint32_t old_visited = atomicOr(&s_visited_bits[word_idx], mask);
                            
                            // 只有当这一位原来是 0，说明当前线程抢到了这个节点的首次访问权
                            if ((old_visited & mask) == 0) {
                                s_dist[n_idx] = current_dist + 1;
                                valid_nodes[num_valid_neighbors++] = n_idx;
                            }
                        }
                    }
                }
            }

            // ==========================================================
            // 技术 3: Warp 级原语优化 (Warp-Level Queue Allocation)
            // 不使用频繁的 atomicAdd，而是在 Warp 级别汇聚要入队的数量，仅做一次 atomicAdd
            // ==========================================================
            uint32_t warp_mask = 0xFFFFFFFF; // 保证全部32个线程参与
            int val = num_valid_neighbors;
            
            // Warp 内部的 Inclusive Prefix Sum (使用 __shfl_up_sync)
            #pragma unroll
            for (int offset = 1; offset < 32; offset *= 2) {
                int n = __shfl_up_sync(warp_mask, val, offset);
                if (lane_id >= offset) val += n;
            }
            
            int warp_offset = val - num_valid_neighbors;     // 线程内的排他性偏移
            int warp_total = __shfl_sync(warp_mask, val, 31); // Warp 总共需要入队的元素个数
            
            int global_queue_offset = 0;
            // Lane 31 作为 Leader 代表整个 Warp 申请 Queue 空间
            if (lane_id == 31 && warp_total > 0) {
                global_queue_offset = atomicAdd(&q_next_sz, warp_total);
            }
            // 将申请到的基准地址广播给 Warp 内的所有线程
            global_queue_offset = __shfl_sync(warp_mask, global_queue_offset, 31);

            // 批量将合法的节点写入 Next Queue
            for (int k = 0; k < num_valid_neighbors; ++k) {
                int write_pos = global_queue_offset + warp_offset + k;
                if (write_pos < MAX_QUEUE_SIZE) { // 防止极端情况溢出
                    write_queue[write_pos] = valid_nodes[k];
                }
            }
        } // 结束当前 Level 的循环

        __syncthreads();

        // --- 队列指针交换 (Double Buffering) ---
        if (tid == 0) {
            current_dist++;
            q_curr_sz = (q_next_sz > MAX_QUEUE_SIZE) ? MAX_QUEUE_SIZE : q_next_sz;
            q_next_sz = 0;
            curr_queue_id ^= 1;
        }
        __syncthreads();
    }

    // --- 将能量图写回 Global Memory ---
    // 能量公式: E = max(0, 255 - d). 若不可达(d=255)，能量为0
    uint8_t* my_global_energy_map = ctx.d_global_energy_maps + global_agent_idx * GRID_SIZE;
    
    for (int i = tid; i < GRID_SIZE; i += blockDim.x) {
        uint8_t d = s_dist[i];
        my_global_energy_map[i] = (d == 255) ? 0 : (255 - d);
    }
}

void launch_energy_map_kernel(
    EnergyMapContext ctx,
    int num_agents,
    int length,
    const torch::Tensor& states_tensor,
    bool incremental_only)
{
    int threads_per_block = 128; // 可根据实际情况调整
    int shared_mem_size = compute_energy_map_shared_mem_size();
    configure_energy_map_dynamic_smem_if_needed(shared_mem_size);

    // Each block processes ONE agent (blockIdx.x == agent index in d_states)
    int blocks = length;
    auto d_states = reinterpret_cast<const AgentState*>((uint16_t*)states_tensor.data_ptr<int16_t>());

    generate_energy_map_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        ctx, num_agents, length, d_states, incremental_only);
}
