#include <curand_kernel.h>
#include <stdint.h>

#define MAP_W 32


__device__ uint32_t generate_obstacle_row(curandState_t* local_state, float obstacle_prob) {
    const float EPSILON = 1e-5f;
    if (fabsf(obstacle_prob - 0.5f) < EPSILON) {
        return curand(local_state);
    }

    if (fabsf(obstacle_prob - 0.25f) < EPSILON) {
        return curand(local_state) & curand(local_state);
    }

    if (fabsf(obstacle_prob - 0.75f) < EPSILON) {
        return curand(local_state) | curand(local_state);
    }

    if (fabsf(obstacle_prob - 0.125f) < EPSILON) {
        return curand(local_state) & curand(local_state) & curand(local_state);
    }

    if (fabsf(obstacle_prob - 0.875f) < EPSILON) {
        return curand(local_state) | curand(local_state) | curand(local_state);
    }

    uint32_t threshold = (uint32_t)(obstacle_prob * 4294967295.0f);
    uint32_t row_bits = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t r0 = curand(local_state);
        uint32_t r1 = curand(local_state);
        uint32_t r2 = curand(local_state);
        uint32_t r3 = curand(local_state);

        int shift = i * 4;
        row_bits |= (r0 < threshold ? 1u : 0u) << (shift + 0);
        row_bits |= (r1 < threshold ? 1u : 0u) << (shift + 1);
        row_bits |= (r2 < threshold ? 1u : 0u) << (shift + 2);
        row_bits |= (r3 < threshold ? 1u : 0u) << (shift + 3);
    }

    return row_bits;
}

// 待优化空间: 1. 利用 obstacle_prob 取特殊值时的逻辑
//            2. 利用 自定义高效 rand 替代有状态的 curand
__global__ void generate_random_map_kernel(
    uint32_t* d_grids,      
    curandState* states, 
    float obstacle_prob    
) {
    int env_id = blockIdx.x;
    int lane_id = threadIdx.x;
    curandState local_state = states[env_id * 32 + lane_id];

    uint32_t row_bits = generate_obstacle_row(&local_state, obstacle_prob);

        if (lane_id == 16) {
        row_bits &= ~( (1u << 15) | (1u << 16) | (1u << 17) );
    }
    if (lane_id == 15 || lane_id == 17) {
        row_bits &= ~(1u << 16);
    }

    uint32_t row_empty = ~row_bits;
    uint32_t seed_mask = (lane_id == 16) ? (1u << 16) : 0;
    uint32_t visited = 0;
    uint32_t frontier = seed_mask;
    
    // 每个线程负责一行的扩展
    while (__any_sync(0xffffffff, frontier != 0)) {
        visited |= frontier;
        uint32_t spread_h = (frontier << 1) | (frontier >> 1);
        uint32_t from_up = __shfl_up_sync(0xffffffff, frontier, 1);
        if (lane_id == 0) from_up = 0;
        uint32_t from_down = __shfl_down_sync(0xffffffff, frontier, 1);
        if (lane_id == 31) from_down = 0;
        frontier = (spread_h | from_up | from_down) & row_empty & ~visited;
    }

    int my_pop = __popc(visited);
    for (int offset = 16; offset > 0; offset /= 2) {
        my_pop += __shfl_down_sync(0xffffffff, my_pop, offset);
    }
    int connected_area = __shfl_sync(0xffffffff, my_pop, 0);

    if (connected_area > 350) {
        row_bits = ~visited; 
    } else {
        if (lane_id == 16) row_bits = 0; 
        row_bits &= ~(1u << 16);
    }

    d_grids[env_id * 32 + lane_id] = row_bits;
    states[env_id * 32 + lane_id] = local_state;
}

