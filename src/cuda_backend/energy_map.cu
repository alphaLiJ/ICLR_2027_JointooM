#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <stdexcept>
#include <string>
#include "grid_world_cuda.h"
#include <torch/extension.h>
// Size for the maximum expected frontier; a 256x256 map typically requires fewer than 4000 entries
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
 * BFS energy-map generation kernel
 * Launch configuration: <<< n_agents, 128 or 256, dynamic_smem_size >>>
 */
 // Inputs: map and goal positions; output: an energy map for each agent
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
    
    // 1. Distance/energy array
    uint8_t* s_dist = smem; 
    
    // 2. Visited bitmap: bitmask compression avoids duplicate enqueues and reduces memory contention
    // 256x256 requires 8 KB of shared memory; 128x128 requires 2 KB
    uint32_t* s_visited_bits = (uint32_t*)(s_dist + GRID_SIZE); 
    int num_bit_words = (GRID_SIZE + 31) / 32;

    // 3. Technique 1: two-queue frontier expansion
    uint16_t* s_q_curr = (uint16_t*)(s_visited_bits + num_bit_words);
    uint16_t* s_q_next = s_q_curr + MAX_QUEUE_SIZE;

    __shared__ int q_curr_sz;
    __shared__ int q_next_sz;
    __shared__ int current_dist;
    __shared__ int curr_queue_id;

    int tid = threadIdx.x;
    int lane_id = tid % 32; // Lane ID within the warp

    // --- Initialize shared memory ---
    for (int i = tid; i < GRID_SIZE; i += blockDim.x) {
        s_dist[i] = 255; // 255 indicates an unreachable cell
    }
    for (int i = tid; i < num_bit_words; i += blockDim.x) {
        s_visited_bits[i] = 0;
    }
    __syncthreads();

    // --- Thread 0 initializes the starting point (the BFS starts at the goal) ---
    if (tid == 0) {
        int gx = d_states[block_id].target_x;
        int gy = d_states[block_id].target_y;

        // Project convention: x is the row index and y is the column index
        int start_idx = gx * MAP_H + gy;
        
        s_dist[start_idx] = 0;
        // Mark the starting point as visited
        s_visited_bits[start_idx / 32] |= (1u << (start_idx % 32));
        
        s_q_curr[0] = start_idx;
        q_curr_sz = 1;
        q_next_sz = 0;
        current_dist = 0;
        curr_queue_id = 0;
    }
    __syncthreads();

    // ==========================================================
    // Technique 4: adapt to frontier size with a grid-stride loop
    // When q_curr_sz is small, only a few warps are active, avoiding premature divergence
    // ==========================================================
    while (q_curr_sz > 0 && current_dist < 254) {
        uint16_t* read_queue = (curr_queue_id == 0) ? s_q_curr : s_q_next;
        uint16_t* write_queue = (curr_queue_id == 0) ? s_q_next : s_q_curr;
        
        // --- Main BFS expansion loop ---
        // All lanes must enter the loop for warp-primitive synchronization, including lanes without work
        for (int i = tid; i < ((q_curr_sz + 31) & ~31); i += blockDim.x) {
            
            int num_valid_neighbors = 0;
            int valid_nodes[4]; // Temporarily store valid neighbors locally
            
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
                        
                        // 1. Check obstacles by reading the bitmap directly
                        int word_idx = n_idx / 32;
                        int bit_offset = n_idx % 32;
                        uint32_t mask = 1u << bit_offset;
                        
                        bool is_static_obs = (ctx.d_maps[map_word_offset + word_idx] & mask) != 0;

                        if (!is_static_obs) {
                            // 2. Mark visits with atomic bitmask operations to reduce shared-memory write contention
                            uint32_t old_visited = atomicOr(&s_visited_bits[word_idx], mask);
                            
                            // A previously unset bit means this thread obtained the first visit to this node
                            if ((old_visited & mask) == 0) {
                                s_dist[n_idx] = current_dist + 1;
                                valid_nodes[num_valid_neighbors++] = n_idx;
                            }
                        }
                    }
                }
            }

            // ==========================================================
            // Technique 3: warp-level queue allocation
            // Aggregate enqueue counts within each warp and use one atomicAdd instead of frequent atomics
            // ==========================================================
            uint32_t warp_mask = 0xFFFFFFFF; // Ensure that all 32 lanes participate
            int val = num_valid_neighbors;
            
            // Inclusive prefix sum within the warp using __shfl_up_sync
            #pragma unroll
            for (int offset = 1; offset < 32; offset *= 2) {
                int n = __shfl_up_sync(warp_mask, val, offset);
                if (lane_id >= offset) val += n;
            }
            
            int warp_offset = val - num_valid_neighbors;     // Exclusive offset for this lane
            int warp_total = __shfl_sync(warp_mask, val, 31); // Total number of elements to enqueue for this warp
            
            int global_queue_offset = 0;
            // Lane 31 reserves queue space on behalf of the warp
            if (lane_id == 31 && warp_total > 0) {
                global_queue_offset = atomicAdd(&q_next_sz, warp_total);
            }
            // Broadcast the reserved base address to all lanes
            global_queue_offset = __shfl_sync(warp_mask, global_queue_offset, 31);

            // Write valid nodes to the next queue in a batch
            for (int k = 0; k < num_valid_neighbors; ++k) {
                int write_pos = global_queue_offset + warp_offset + k;
                if (write_pos < MAX_QUEUE_SIZE) { // Prevent overflow in extreme cases
                    write_queue[write_pos] = valid_nodes[k];
                }
            }
        } // Finish the loop for the current level

        __syncthreads();

        // --- Swap queue pointers (double buffering) ---
        if (tid == 0) {
            current_dist++;
            q_curr_sz = (q_next_sz > MAX_QUEUE_SIZE) ? MAX_QUEUE_SIZE : q_next_sz;
            q_next_sz = 0;
            curr_queue_id ^= 1;
        }
        __syncthreads();
    }

    // --- Write the energy map back to global memory ---
    // Energy: E = max(0, 255 - d). Unreachable cells (d=255) have zero energy
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
    int threads_per_block = 128; // Adjust as needed for the workload
    int shared_mem_size = compute_energy_map_shared_mem_size();
    configure_energy_map_dynamic_smem_if_needed(shared_mem_size);

    // Each block processes ONE agent (blockIdx.x == agent index in d_states)
    int blocks = length;
    auto d_states = reinterpret_cast<const AgentState*>((uint16_t*)states_tensor.data_ptr<int16_t>());

    generate_energy_map_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        ctx, num_agents, length, d_states, incremental_only);
}
