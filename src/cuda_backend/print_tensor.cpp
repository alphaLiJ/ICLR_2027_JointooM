#include <torch/torch.h>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include "grid_world_simulator.h"

// Type mapping helper for PyTorch 2.1.x compatibility
template<typename T>
struct TorchActualType { using type = T; };
// uint16_t and uint32_t need to use signed versions for PyTorch data_ptr
template<> struct TorchActualType<uint16_t> { using type = int16_t; };
template<> struct TorchActualType<uint32_t> { using type = int32_t; };

template<typename T>
struct TorchScalarType;

template<> struct TorchScalarType<uint8_t>  { static constexpr auto value = torch::kUInt8; };
template<> struct TorchScalarType<int16_t>  { static constexpr auto value = torch::kInt16; };
template<> struct TorchScalarType<int32_t>  { static constexpr auto value = torch::kInt32; };
template<> struct TorchScalarType<float>    { static constexpr auto value = torch::kFloat32; };
// uint16_t and uint32_t map to signed counterparts (same byte width)
template<> struct TorchScalarType<uint16_t> { static constexpr auto value = torch::kInt16; };
template<> struct TorchScalarType<uint32_t> { static constexpr auto value = torch::kInt32; };

// Safe print template: auto device migration (PyTorch 2.1.x compatible)
template<typename T>
void print_tensor(const torch::Tensor& tensor, const char* name) {
    try {
        // 1. Move tensor to CPU
        torch::Tensor cpu_tensor = tensor.device().is_cuda() ?
            tensor.to(torch::kCPU).contiguous() :
            tensor.contiguous();

        // 2. Type check using our mapping
        auto expected_type = TorchScalarType<T>::value;
        TORCH_CHECK(cpu_tensor.scalar_type() == expected_type,
                   "Type mismatch for ", name,
                   ". Expected ", c10::toString(expected_type),
                   ", got ", c10::toString(cpu_tensor.scalar_type()));

        // 3. Safe data access using PyTorch-compatible type
        using ActualType = typename TorchActualType<T>::type;
        auto data = reinterpret_cast<T*>(cpu_tensor.data_ptr<ActualType>());
        int64_t numel = cpu_tensor.numel();
        int64_t max_elements = std::min(numel, static_cast<int64_t>(100));

        // 4. Print metadata
        std::cout << "\n[PRINT] " << name
                  << "\nDevice: " << (tensor.device().is_cuda() ? "CUDA" : "CPU")
                  << "\nDtype: " << c10::toString(cpu_tensor.scalar_type())
                  << "\nShape: " << cpu_tensor.sizes()
                  << "\nTotal elements: " << numel
                  << "\nFirst " << max_elements << " elements:" << std::endl;

        // 5. Formatted output
        for (int64_t i = 0; i < max_elements; ++i) {
            if (i % 10 == 0 && i > 0) std::cout << "\n";
            if constexpr (std::is_floating_point_v<T>) {
                std::cout << static_cast<double>(data[i]) << " ";
            } else {
                std::cout << static_cast<int64_t>(data[i]) << " ";
            }
        }
        if (numel > max_elements) {
            std::cout << "\n... (remaining " << (numel - max_elements)
                      << " elements not shown)";
        }
        std::cout << "\n" << std::string(50, '-') << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to print tensor '" << name
                  << "': " << e.what() << std::endl;
    }
}

// 保持所有成员函数声明不变（与问题中完全一致）
void GridWorldSimulator::print_t_grid_compressed() { print_tensor<uint32_t>(t_grid_compressed, "t_grid_compressed"); }
void GridWorldSimulator::print_t_free_cell_list() { print_tensor<uint32_t>(t_free_cell_list, "t_free_cell_list"); }
void GridWorldSimulator::print_t_pool_ptr() { print_tensor<int32_t>(t_pool_ptr, "t_pool_ptr"); }
void GridWorldSimulator::print_t_offsets() { print_tensor<int32_t>(t_offsets, "t_offsets"); }
void GridWorldSimulator::print_t_counts() { print_tensor<int32_t>(t_counts, "t_counts"); }
void GridWorldSimulator::print_t_grid_ocp() { print_tensor<int32_t>(t_grid_ocp, "t_grid_ocp"); }
void GridWorldSimulator::print_t_cur_x() { print_tensor<int16_t>(t_cur_x, "t_cur_x"); }
void GridWorldSimulator::print_t_cur_y() { print_tensor<int16_t>(t_cur_y, "t_cur_y"); }
void GridWorldSimulator::print_t_goals_x() { print_tensor<int16_t>(t_goals_x, "t_goals_x"); }
void GridWorldSimulator::print_t_goals_y() { print_tensor<int16_t>(t_goals_y, "t_goals_y"); }
void GridWorldSimulator::print_t_agents_obs() { print_tensor<uint16_t>(t_agents_obs, "t_agents_obs"); }
void GridWorldSimulator::print_t_rewards() { print_tensor<float>(t_rewards, "t_rewards"); }
void GridWorldSimulator::print_t_actions() { print_tensor<uint8_t>(t_actions, "t_actions"); }
void GridWorldSimulator::print_t_max_free_cell_count() { print_tensor<uint32_t>(t_max_free_cell_count, "t_max_free_cell_count"); }