#pragma once
#include "grid_world_simulator.h"
#include "mapf_gpt_builder.h"
void costtracer_reward_goals(GridWorldSimulator&, MapfGPTObservationBuilder&,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void costtracer_observe(GridWorldSimulator&, MapfGPTObservationBuilder&, torch::Tensor, torch::Tensor);
void costtracer_reset(GridWorldSimulator&, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, int);
