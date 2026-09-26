#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio> 
#include <stdexcept>  
#include <string>
#include <iostream>

#define CUDA_CHECK_KERNEL_ERROR()                                                      \
    do {                                                                               \
        cudaError_t __err = cudaGetLastError();                                        \
        if (__err != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA Kernel Launch Error: %s at %s:%d\n",                 \
                    cudaGetErrorString(__err), __FILE__, __LINE__);                    \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
        __err = cudaDeviceSynchronize();                                               \
        if (__err != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA Kernel Execution Error: %s at %s:%d\n",              \
                    cudaGetErrorString(__err), __FILE__, __LINE__);                    \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    } while (0)

// 后续采用 Fisher-Yates 洗牌算法针对极大规模的场景 (当前逻辑仅在负载较小时可靠且高效)
// 潜在字占据优化

__device__ __forceinline__ uint32_t quick_rand(uint32_t* seed) {
    *seed = *seed * 1664525u + 1013904223u;
    return *seed;
}

__device__ __forceinline__ uint32_t quick_rand_range(uint32_t* seed, uint32_t range) {
    uint32_t x = quick_rand(seed);
    return (uint32_t)(((uint64_t)x * range) >> 32);
}

__device__ inline int find_unique_slot(
    uint32_t* bitmap, 
    int total_free, 
    curandState* state
) {
    int start_idx = curand(state) % total_free;
    int curr_idx = start_idx;

    while (true) {
        int word_idx = curr_idx >> 5;      
        int bit_offset = curr_idx & 31;    
        uint32_t mask = (1u << bit_offset);
        uint32_t old_val = atomicOr(&bitmap[word_idx], mask);

        if ((old_val & mask) == 0) {
            return curr_idx; // 成功找到并占位
        }
        curr_idx++;
        if (curr_idx >= total_free) {
            curr_idx = 0;
        }
    }
}

