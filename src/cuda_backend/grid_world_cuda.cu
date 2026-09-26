#pragma once
#include <array>
#include <cub/cub.cuh>
#include <torch/extension.h>
#include "grid_world_cuda.h"
#include "utils_cuda.h"
// Note: keep this TU rebuilt when GridWorldContext layout changes (layout sync guard).


__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__global__ void finalize_edge_prefix_kernel(
    const int* __restrict__ d_edge_counts,
    int64_t* __restrict__ d_edge_prefix,
    int64_t* __restrict__ d_num_edges,
    int total_agents)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (total_agents <= 0) {
        d_edge_prefix[0] = 0;
        d_num_edges[0] = 0;
        return;
    }

    int64_t total_edges = d_edge_prefix[total_agents - 1] + d_edge_counts[total_agents - 1];
    d_edge_prefix[total_agents] = total_edges;
    d_num_edges[0] = total_edges;
}

__global__ void cast_edge_counts_i32_to_i64_kernel(
    const int* __restrict__ d_edge_counts,
    int64_t* __restrict__ d_edge_prefix,
    int total_agents)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_agents) return;
    d_edge_prefix[idx] = static_cast<int64_t>(d_edge_counts[idx]);
}

size_t query_exclusive_scan_edge_counts_temp_storage_bytes(
    const torch::Tensor& edge_counts,
    const torch::Tensor& edge_prefix)
{
    TORCH_CHECK(edge_counts.is_cuda(), "edge_counts must be CUDA");
    TORCH_CHECK(edge_prefix.is_cuda(), "edge_prefix must be CUDA");
    TORCH_CHECK(edge_counts.dtype() == torch::kInt32, "edge_counts must be int32");
    TORCH_CHECK(edge_prefix.dtype() == torch::kInt64, "edge_prefix must be int64");

    int total_agents = static_cast<int>(edge_counts.numel());
    if (total_agents == 0) return 0;

    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        temp_storage_bytes,
        edge_prefix.data_ptr<int64_t>(),
        edge_prefix.data_ptr<int64_t>(),
        total_agents);
    return temp_storage_bytes;
}

void launch_exclusive_scan_edge_counts(
    const torch::Tensor& edge_counts,
    torch::Tensor& edge_prefix,
    torch::Tensor& num_edges,
    const torch::Tensor& temp_storage)
{
    TORCH_CHECK(edge_counts.is_cuda(), "edge_counts must be CUDA");
    TORCH_CHECK(edge_prefix.is_cuda(), "edge_prefix must be CUDA");
    TORCH_CHECK(num_edges.is_cuda(), "num_edges must be CUDA");
    TORCH_CHECK(temp_storage.is_cuda(), "temp_storage must be CUDA");
    TORCH_CHECK(edge_counts.dtype() == torch::kInt32, "edge_counts must be int32");
    TORCH_CHECK(edge_prefix.dtype() == torch::kInt64, "edge_prefix must be int64");
    TORCH_CHECK(num_edges.dtype() == torch::kInt64, "num_edges must be int64");

    int total_agents = static_cast<int>(edge_counts.numel());
    if (total_agents == 0) {
        edge_prefix.zero_();
        num_edges.zero_();
        return;
    }

    size_t temp_storage_bytes = static_cast<size_t>(temp_storage.numel());
    void* d_temp_storage = static_cast<void*>(temp_storage.data_ptr<uint8_t>());
    int threads = 256;
    int blocks = (total_agents + threads - 1) / threads;

    cast_edge_counts_i32_to_i64_kernel<<<blocks, threads>>>(
        edge_counts.data_ptr<int>(),
        edge_prefix.data_ptr<int64_t>(),
        total_agents);

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        edge_prefix.data_ptr<int64_t>(),
        edge_prefix.data_ptr<int64_t>(),
        total_agents);

    finalize_edge_prefix_kernel<<<1, 1>>>(
        edge_counts.data_ptr<int>(),
        edge_prefix.data_ptr<int64_t>(),
        num_edges.data_ptr<int64_t>(),
        total_agents);
}

__global__ void encode_full_map_kernel(
    const uint32_t* __restrict__ input_map,    
    uint32_t* __restrict__ compressed_map,
    int total_output_elements, 
    int batch_size    
) {
    const uint32_t SMEM_INT_COUNT = blockDim.x * 32;
    const uint32_t INPUT_TOTAL = GRID_SIZE * batch_size;

    extern __shared__ uint32_t smem_buffer[];
    int block_input_start = blockIdx.x * SMEM_INT_COUNT;
    for (int i = threadIdx.x; i < SMEM_INT_COUNT; i += blockDim.x) {
        int global_idx = block_input_start + i;
        if (global_idx < INPUT_TOTAL) { 
             smem_buffer[i] = input_map[global_idx];
        } else {
             smem_buffer[i] = 0;
        }
    }

    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_output_elements) return;
    int smem_start_idx = threadIdx.x * 32;
    uint32_t packed_val = 0;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        int32_t pixel = smem_buffer[smem_start_idx + i];
        if (pixel != 0) {
            packed_val |= (1u << i);
        }
    }
    compressed_map[tid] = packed_val;
}

void launch_encode_full_map_kernel_on_stream(
    const uint32_t* input_map,
    uint32_t* compressed_map,
    int batch_size,
    int blockSize,
    cudaStream_t stream) {
    int total_output_elements = batch_size * MAP_OFFSET;
    int numBlocks = (total_output_elements + blockSize - 1) / blockSize;
    if (blockSize > 256) {
        printf("Error: blockSize too large for shared memory limit.\n");
        return;
    }
    
    // 可以考虑减少 numBlocks 数量 (让一个 block 承担更多的计算, 在 block 内部加一个与 batch_size 有关的循环)
    encode_full_map_kernel<<<
        numBlocks,
        blockSize,
        blockSize * 32 * sizeof(uint32_t),
        stream>>>(
        input_map,
        compressed_map,
        total_output_elements,
        batch_size
    );
}

// 根据 block smem 使用大小分析 blockSize 保证并行度不下降
void launch_encode_full_map_kernel(const uint32_t* input_map,
                                   uint32_t* compressed_map,
                                   int batch_size,
                                   int blockSize = 128)
{
    launch_encode_full_map_kernel_on_stream(
        input_map, compressed_map, batch_size, blockSize, nullptr);
    cudaDeviceSynchronize();
}

// 运用 uint16_t 进行压缩：行数为 diameter (obs 的每行压缩为一个 uint16_t 的数)
__global__ void decode_to_uint8_kernel(
    const uint16_t* __restrict__ compressed_obs, // 输入: [Batch, 2*Diameter] (uint16)
    uint8_t* __restrict__ output,                // 输出: [Batch, 2, Diameter, Diameter] (uint8)
    int total_pixels,                            // 总输出像素数
    int diameter                                 // 2 * RADIUS + 1
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_pixels) return;
    
    int col = tid % diameter;
    int temp = tid / diameter;
    
    int row = temp % diameter;
    temp = temp / diameter;
    
    int channel = temp % 2;
    int agent_global_idx = temp / 2;

    int input_idx = (agent_global_idx * 2 * diameter) + (channel * diameter) + row;

    uint16_t packed_row = compressed_obs[input_idx];
    uint8_t pixel_val = (packed_row >> col) & 1;

    output[tid] = pixel_val;
}

__device__ __forceinline__ uint32_t real_extent_mask_for_word(int word_idx, int real_map_h) {
    int bit_base = word_idx * 32;
    if (bit_base >= real_map_h) {
        return 0u;
    }

    int remaining = real_map_h - bit_base;
    if (remaining >= 32) {
        return 0xFFFFFFFFu;
    }
    return (1u << remaining) - 1u;
}

__global__ void cache_map_free_cells_kernel(GridWorldContext ctx) {
    int env_id = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % 32;

    __shared__ int local_count;
    __shared__ int global_start_offset;
    __shared__ int local_running_sum;
    __shared__ int real_map_w;
    __shared__ int real_map_h;

    if (tid == 0) {
        local_count = 0;
        global_start_offset = 0;
        local_running_sum = 0;
        real_map_w = ctx.d_real_map_extents[env_id * 2];
        real_map_h = ctx.d_real_map_extents[env_id * 2 + 1];
    }
    __syncthreads();

    int total_found_in_env = 0;
    for (int i = tid; i < MAP_OFFSET; i += blockDim.x) {
        int chunk_row = i / COL_OFFSET;
        if (chunk_row >= real_map_w) {
            continue;
        }
        uint32_t row_bits = ctx.d_maps[env_id * MAP_OFFSET + i];
        uint32_t valid_mask = real_extent_mask_for_word(i % COL_OFFSET, real_map_h);
        total_found_in_env += __popc((~row_bits) & valid_mask);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        total_found_in_env += __shfl_down_sync(0xffffffff, total_found_in_env, offset);
    }
    if (lane_id == 0) {
        atomicAdd(&local_count, total_found_in_env);
    }
    
    __syncthreads();
     
    if (tid == 0) {
        int count = local_count;
        ctx.d_env_counts[env_id] = count;
        if (count > 0) {
            global_start_offset = atomicAdd(ctx.d_global_pool_ptr, count);
            atomicMax(ctx.max_free_cell_count, count);
            ctx.d_env_offsets[env_id] = global_start_offset;
        } else {
            ctx.d_env_offsets[env_id] = -1;
        }
    }
    __syncthreads();

    int start_off = global_start_offset;
    int count_total = local_count;
    if (count_total <= 0 || start_off >= ctx.pool_capacity) return;

    for (int i = tid; i < MAP_OFFSET; i += blockDim.x) {
        int chunk_row = i / COL_OFFSET;
        if (chunk_row >= real_map_w) {
            continue;
        }
        uint32_t row_bits = ctx.d_maps[env_id * MAP_OFFSET + i];
        uint32_t empty_mask = (~row_bits) & real_extent_mask_for_word(i % COL_OFFSET, real_map_h);
        
        int my_popc = __popc(empty_mask);
        unsigned mask = __activemask();
        int warp_prefix = my_popc;
        for (int offset = 1; offset < 32; offset <<= 1) {
            int n = __shfl_up_sync(mask, warp_prefix, offset); // 注意这里传的是 warp_prefix
            if (lane_id >= offset) warp_prefix += n;
        }
        int warp_total = __shfl_sync(mask, warp_prefix, 31);
        int warp_write_base = 0;
        if (lane_id == 0) {
            warp_write_base = atomicAdd(&local_running_sum, warp_total);
        }
        warp_write_base = __shfl_sync(mask, warp_write_base, 0);
        int thread_write_idx = start_off + warp_write_base + (warp_prefix - my_popc);

        // 解码坐标并写入
        if (my_popc > 0) {
            int chunk_col_base = (i % COL_OFFSET) * 32;
            uint32_t coord_base = (chunk_row << 16) | chunk_col_base; 

            while (empty_mask != 0) {
                int x = __ffs(empty_mask) - 1;
                empty_mask &= ~(1u << x);
                // 直接写入全局显存 (Coalesced Access 效果很好)
                if (thread_write_idx < ctx.pool_capacity) {
                    ctx.d_free_cell_list[thread_write_idx++] = (coord_base | x);
                }
            }
        }
    }
}

__global__ void cache_map_smem_free_cells_kernel(GridWorldContext ctx) {
    int env_id = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % 32;

    extern __shared__ uint32_t local_coords[]; 
    __shared__ int local_count;
    __shared__ int global_start_offset;
    __shared__ int real_map_w;
    __shared__ int real_map_h;

    if (tid == 0) {
        local_count = 0;
        real_map_w = ctx.d_real_map_extents[env_id * 2];
        real_map_h = ctx.d_real_map_extents[env_id * 2 + 1];
    }
    __syncthreads();

    for (int i = tid; i < MAP_OFFSET; i += blockDim.x)
    {
        int chunk_row = i / COL_OFFSET;
        if (chunk_row >= real_map_w) {
            continue;
        }
        uint32_t row_bits = ctx.d_maps[env_id * MAP_OFFSET + i];
        uint32_t empty_mask = (~row_bits) & real_extent_mask_for_word(i % COL_OFFSET, real_map_h);
        
        int chunk_col_base = (i % COL_OFFSET) * 32;
        uint32_t coord_base = (chunk_row << 16) | chunk_col_base; 
        
        // Warp Scan 逻辑
        int my_count = __popc(empty_mask);
        unsigned mask = __activemask();
        int val = my_count;

        for (int offset = 1; offset < 32; offset <<= 1) {
            int n = __shfl_up_sync(mask, val, offset);
            if (lane_id >= offset) val += n;
        }
        int lane_prefix_sum = val;
        // 目的: 防止 MAP_OFFSET 不为 32 的倍数 (若为 32 倍数, 直接使用 31即可)
        int leader_lane = 31 - __clz(mask);
        int warp_total = __shfl_sync(mask, val, leader_lane); 
        int base_offset = 0;
        // 只有 Leader 执行原子加
        if (lane_id == leader_lane) { 
            base_offset = atomicAdd(&local_count, warp_total);
        }
        // 广播基地址
        base_offset = __shfl_sync(mask, base_offset, leader_lane);
        
        int my_write_offset = base_offset + (lane_prefix_sum - my_count);

        while (empty_mask != 0) {
            int x = __ffs(empty_mask) - 1; // x 是 0-31 的位偏移
            empty_mask &= ~(1u << x);
            local_coords[my_write_offset++] = coord_base | x; 
        }
    }
    __syncthreads();
    if (tid == 0) {
        int count = local_count;
        ctx.d_env_counts[env_id] = count;
        if (count > 0) {
            global_start_offset = atomicAdd(ctx.d_global_pool_ptr, count);
            atomicMax(ctx.max_free_cell_count, count);
            ctx.d_env_offsets[env_id] = global_start_offset;
        } else {
            ctx.d_env_offsets[env_id] = -1;
        }
    }
    __syncthreads();

    int start_off = global_start_offset;
    int count = local_count;
    if (count > 0 && start_off + count <= ctx.pool_capacity) {
        for (int i = tid; i < count; i += blockDim.x) {
            ctx.d_free_cell_list[start_off + i] = local_coords[i];
        }
    }
}


__global__ void init_agents_kernel(GridWorldContext ctx) {
    int env_id = blockIdx.x;
    int tid = threadIdx.x;

    // 动态共享内存大小需要在启动 kernel 时指定: sizeof(uint32_t) * ((total_free + 31) / 32)
    extern __shared__ uint32_t occupancy_bitmap[];

    int total_free = ctx.d_env_counts[env_id];
    int n_agents = ctx.n_agents;

    // 计算位图需要的 uint32 数量
    int bitmap_len = (total_free + 31) >> 5; 
    for (int i = tid; i < bitmap_len; i += blockDim.x) {
        occupancy_bitmap[i] = 0;
    }
    __syncthreads(); 

    for (int i = tid; i < n_agents; i += blockDim.x) {
        int global_agent_idx = env_id * n_agents + i;
        curandState local_state = ctx.rng_states[global_agent_idx];

        int start_slot_idx = find_unique_slot(occupancy_bitmap, total_free, &local_state);
        int goal_slot_idx = find_unique_slot(occupancy_bitmap, total_free, &local_state);

        ctx.rng_states[global_agent_idx] = local_state;
        int env_offset = ctx.d_env_offsets[env_id];
        
        uint32_t raw_start = ctx.d_free_cell_list[env_offset + start_slot_idx]; // 起始坐标(压缩过)
        uint32_t raw_goal  = ctx.d_free_cell_list[env_offset + goal_slot_idx];  // 终点坐标(压缩过)
        ctx.cur_x[global_agent_idx]   = (uint16_t)(raw_start >> 16);
        ctx.cur_y[global_agent_idx]   = (uint16_t)(raw_start & 0xFFFF);
        ctx.goals_x[global_agent_idx] = (uint16_t)(raw_goal >> 16);
        ctx.goals_y[global_agent_idx] = (uint16_t)(raw_goal & 0xFFFF);
    }

}

void launch_init_agents_kernel(GridWorldContext ctx, dim3 blocks, dim3 threads) {
    uint32_t max_count_shared;

    cudaMemcpy(
        &max_count_shared, 
        ctx.max_free_cell_count,  // 设备指针
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy failed: ") + cudaGetErrorString(err));
    }
    else std::cout<< max_count_shared<<std::endl;
    size_t smem_init = sizeof(uint32_t) * ((max_count_shared + 31) / 32);
    init_agents_kernel<<<blocks, threads, smem_init>>>(ctx);
    cudaDeviceSynchronize();
}

__global__ void setup_rng_kernel(
    curandState* rng_states, 
    unsigned long seed, 
    int total_agents
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < total_agents) {
        curand_init(seed, id, 0, &rng_states[id]);
    }
}

void launch_setup_rng_kernel(curandState* states, unsigned long seed, int total_agents) {
    int threads = 256;
    // printf("value = %d\n", total_agents);
    int blocks = (total_agents + threads - 1) / threads;
    setup_rng_kernel<<<blocks, threads>>>(states, seed, total_agents);
    cudaDeviceSynchronize();
}

void launch_cache_map_free_cells_kernel(GridWorldContext ctx, dim3 blocks, dim3 threads){
    // size_t smem_bake = GRID_SIZE * sizeof(uint32_t);
    cache_map_free_cells_kernel<<<blocks, threads>>>(ctx);
    cudaDeviceSynchronize();

}

void launch_decode_to_uint8_kernel(uint16_t* compressed_obs, int blocks, int threads,
                                    uint8_t* output, int total_pixels, int diameter)
{
    decode_to_uint8_kernel<<<blocks, threads>>>(
        compressed_obs,            // 源数据 (uint32*)
        output, // 目标数据 (uint8*)
        total_pixels,
        diameter
    );
    cudaDeviceSynchronize();
}

    
