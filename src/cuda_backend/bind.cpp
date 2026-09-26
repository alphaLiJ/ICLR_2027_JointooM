// bind.cpp
#include <torch/extension.h>
#include "grid_world_simulator.h"
#include "mapf_gpt_builder.h"

namespace py = pybind11;

namespace {

py::dict mapf_gpt_memory_tensors(MapfGPTObservationBuilder& self) {
    py::dict tensors;
    tensors["grid_compressed"] = self.t_grid_compressed;
    tensors["state_packed"] = self.t_state_packed;
    tensors["histories"] = self.t_histories;
    tensors["labels"] = self.t_labels;
    tensors["tokens"] = self.t_tokens;
    tensors["active_mask"] = self.t_active_mask;
    tensors["cost_to_go"] = self.t_cost_to_go;
    tensors["agent_at_cell"] = self.t_agent_at_cell;
    tensors["goal_cache"] = self.t_goal_cache;
    tensors["diagnostics"] = self.t_diagnostics;
    tensors["pyg_ptr"] = self.t_pyg_ptr;
    return tensors;
}

py::dict stateful_memory_tensors(GridWorldSimulator& self) {
    py::dict tensors;
    tensors["grid_compressed"] = self.t_grid_compressed;
    tensors["free_cell_list"] = self.t_free_cell_list;
    tensors["pool_ptr"] = self.t_pool_ptr;
    tensors["offsets"] = self.t_offsets;
    tensors["counts"] = self.t_counts;
    tensors["grid_ocp"] = self.t_grid_ocp;
    tensors["real_map_extents"] = self.t_real_map_extents;
    tensors["cur_x"] = self.t_cur_x;
    tensors["cur_y"] = self.t_cur_y;
    tensors["goal_x"] = self.t_goals_x;
    tensors["goal_y"] = self.t_goals_y;
    tensors["agents_obs"] = self.t_agents_obs;
    tensors["rewards"] = self.t_rewards;
    tensors["actions"] = self.t_actions;
    tensors["rng_states"] = self.t_rng_states;
    tensors["max_free_cell_count"] = self.t_max_free_cell_count;
    tensors["goal_changed_flags"] = self.t_goal_changed_flags;
    tensors["arrived"] = self.t_arrived;
    tensors["terminated"] = self.t_terminated;
    tensors["truncated"] = self.t_truncated;
    tensors["step_counts"] = self.t_step_counts;
    tensors["goal_changed_prefix"] = self.t_goal_changed_prefix;
    tensors["changed_state_packed"] = self.t_changed_state_packed;
    tensors["goal_changed_count"] = self.t_goal_changed_count;
    tensors["energy_maps"] = self.t_energy_maps;
    tensors["state_packed"] = self.t_state_packed;
    tensors["pyg_x"] = self.t_pyg_x;
    tensors["pyg_pos"] = self.t_pyg_pos;
    tensors["pyg_edge_index"] = self.t_pyg_edge_index;
    tensors["pyg_edge_attr"] = self.t_pyg_edge_attr;
    tensors["pyg_batch"] = self.t_pyg_batch;
    tensors["pyg_ptr"] = self.t_pyg_ptr;
    tensors["pyg_num_edges"] = self.t_pyg_num_edges;
    tensors["edge_counts"] = self.t_edge_counts;
    tensors["pyg_edge_index_storage"] = self.t_pyg_edge_index_storage;
    tensors["pyg_edge_attr_storage"] = self.t_pyg_edge_attr_storage;
    tensors["pyg_edge_prefix"] = self.t_pyg_edge_prefix;
    tensors["scan_temp_storage"] = self.t_scan_temp_storage;
    return tensors;
}

py::dict stateless_memory_tensors(StatelessGridWorldSimulator& self) {
    py::dict tensors;
    tensors["grid_compressed"] = self.t_grid_compressed;
    tensors["grid_ocp"] = self.t_grid_ocp;
    tensors["state_packed"] = self.t_state_packed;
    tensors["goal_x"] = self.t_goals_x;
    tensors["goal_y"] = self.t_goals_y;
    tensors["energy_maps"] = self.t_energy_maps;
    tensors["pyg_x"] = self.t_pyg_x;
    tensors["pyg_pos"] = self.t_pyg_pos;
    tensors["pyg_edge_index"] = self.t_pyg_edge_index;
    tensors["pyg_edge_attr"] = self.t_pyg_edge_attr;
    tensors["pyg_batch"] = self.t_pyg_batch;
    tensors["pyg_ptr"] = self.t_pyg_ptr;
    tensors["pyg_num_edges"] = self.t_pyg_num_edges;
    tensors["edge_counts"] = self.t_edge_counts;
    tensors["pyg_edge_index_storage"] = self.t_pyg_edge_index_storage;
    tensors["pyg_edge_attr_storage"] = self.t_pyg_edge_attr_storage;
    tensors["pyg_edge_prefix"] = self.t_pyg_edge_prefix;
    tensors["scan_temp_storage"] = self.t_scan_temp_storage;
    return tensors;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("COMPILED_MAP_W") = py::int_(MAP_W);
    m.attr("COMPILED_MAP_H") = py::int_(MAP_H);

    py::class_<MapfGPTObservationBuilder>(m, "MapfGPTObservationBuilder")
        .def(py::init<torch::Tensor, int>())
        .def("build_tokens", &MapfGPTObservationBuilder::build_tokens)
        .def(
            "build_tokens_from_state",
            &MapfGPTObservationBuilder::build_tokens_from_state)
        .def("append_actions", &MapfGPTObservationBuilder::append_actions)
        .def("reset_histories", &MapfGPTObservationBuilder::reset_histories)
        .def_readonly("tokens", &MapfGPTObservationBuilder::t_tokens)
        .def_readonly("active_mask", &MapfGPTObservationBuilder::t_active_mask)
        .def_readonly("labels", &MapfGPTObservationBuilder::t_labels)
        .def_readonly("cost_to_go", &MapfGPTObservationBuilder::t_cost_to_go)
        .def_readonly("state_packed", &MapfGPTObservationBuilder::t_state_packed)
        .def_readonly("histories", &MapfGPTObservationBuilder::t_histories)
        .def_readonly("agent_at_cell", &MapfGPTObservationBuilder::t_agent_at_cell)
        .def_readonly("diagnostics", &MapfGPTObservationBuilder::t_diagnostics)
        .def_readonly("pyg_ptr", &MapfGPTObservationBuilder::t_pyg_ptr)
        .def_readonly("goal_cache", &MapfGPTObservationBuilder::t_goal_cache)
        .def_property_readonly("memory_tensors", &mapf_gpt_memory_tensors);

    py::class_<GridWorldSimulator>(m, "GridWorldSimulator")
        .def(
            py::init<torch::Tensor, int, int, int, unsigned long, const std::string&, int>(),
            py::arg("grids"),
            py::arg("n_agents"),
            py::arg("num_features"),
            py::arg("pool_capacity"),
            py::arg("seed"),
            py::arg("task_mode") = "lifelong",
            py::arg("max_episode_steps") = 256)

        .def("init_rng", &GridWorldSimulator::init_rng)
        .def("run_initialization", &GridWorldSimulator::run_initialization)
        .def("step_sim_only", &GridWorldSimulator::step_sim_only)
        .def("step_compact_only", &GridWorldSimulator::step_compact_only)
        .def("update_derived_state", &GridWorldSimulator::update_derived_state)
        .def("build_magat_plus_nodes", &GridWorldSimulator::build_magat_plus_nodes)
        .def("finalize_magat_plus_graph", &GridWorldSimulator::finalize_magat_plus_graph)
        .def("build_magat_plus_inputs", &GridWorldSimulator::build_magat_plus_inputs)
        .def("materialize_pyg_inputs", &GridWorldSimulator::materialize_pyg_inputs)
        .def("step_and_build_pyg", &GridWorldSimulator::step_and_build_pyg)
        .def("step", &GridWorldSimulator::step)
        .def("update_actions", &GridWorldSimulator::update_actions)
        .def("set_real_map_extents", &GridWorldSimulator::set_real_map_extents)
        .def("load_state", &GridWorldSimulator::load_state)
        .def("decode_to_uint8", &GridWorldSimulator::decode_to_uint8)
        .def_property_readonly("task_mode", &GridWorldSimulator::get_task_mode)
        .def_property_readonly("max_episode_steps", &GridWorldSimulator::get_max_episode_steps)
        .def_property("pyg_builder_mode", &GridWorldSimulator::get_pyg_builder_mode, &GridWorldSimulator::set_pyg_builder_mode)
        .def_property("pyg_local_gather_impl", &GridWorldSimulator::get_pyg_local_gather_impl, &GridWorldSimulator::set_pyg_local_gather_impl)
        .def_property_readonly("resolved_pyg_builder_impl", &GridWorldSimulator::get_resolved_pyg_builder_impl)
        .def_property_readonly("memory_tensors", &stateful_memory_tensors)

        .def("print_t_grid_compressed", &GridWorldSimulator::print_t_grid_compressed)
        .def("print_t_free_cell_list", &GridWorldSimulator::print_t_free_cell_list)
        .def("print_t_pool_ptr", &GridWorldSimulator::print_t_pool_ptr)
        .def("print_t_offsets", &GridWorldSimulator::print_t_offsets)
        .def("print_t_counts", &GridWorldSimulator::print_t_counts)
        .def("print_t_grid_ocp", &GridWorldSimulator::print_t_grid_ocp)
        .def("print_t_cur_x", &GridWorldSimulator::print_t_cur_x)
        .def("print_t_cur_y", &GridWorldSimulator::print_t_cur_y)
        .def("print_t_goals_x", &GridWorldSimulator::print_t_goals_x)
        .def("print_t_goals_y", &GridWorldSimulator::print_t_goals_y)
        .def("print_t_agents_obs", &GridWorldSimulator::print_t_agents_obs)
        .def("print_t_rewards", &GridWorldSimulator::print_t_rewards)
        .def("print_t_actions", &GridWorldSimulator::print_t_actions)
        .def("print_t_max_free_cell_count", &GridWorldSimulator::print_t_max_free_cell_count)

        .def_readonly("agents_obs", &GridWorldSimulator::t_agents_obs)
        .def_readonly("rewards", &GridWorldSimulator::t_rewards)
        .def_readonly("actions", &GridWorldSimulator::t_actions)
        .def_readonly("cur_x", &GridWorldSimulator::t_cur_x)
        .def_readonly("cur_y", &GridWorldSimulator::t_cur_y)
        .def_readonly("goal_x", &GridWorldSimulator::t_goals_x)
        .def_readonly("goal_y", &GridWorldSimulator::t_goals_y)
        
        .def_readonly("grid_ocp", &GridWorldSimulator::t_grid_ocp)
        .def_readonly("state_packed", &GridWorldSimulator::t_state_packed)
        .def_readonly("counts", &GridWorldSimulator::t_counts)
        .def_readonly("pyg_x", &GridWorldSimulator::t_pyg_x)
        .def_readonly("pyg_pos", &GridWorldSimulator::t_pyg_pos)
        .def_property_readonly("pyg_edge_index", [](GridWorldSimulator& self) {
            int64_t num_edges = self.t_pyg_num_edges.cpu().item<int64_t>();
            return self.t_pyg_edge_index_storage.narrow(1, 0, num_edges);
        })
        .def_property_readonly("pyg_edge_attr", [](GridWorldSimulator& self) {
            int64_t num_edges = self.t_pyg_num_edges.cpu().item<int64_t>();
            return self.t_pyg_edge_attr_storage.narrow(0, 0, num_edges);
        })
        .def_readonly("pyg_edge_index_storage", &GridWorldSimulator::t_pyg_edge_index_storage)
        .def_readonly("pyg_edge_attr_storage", &GridWorldSimulator::t_pyg_edge_attr_storage)
        .def_readonly("pyg_batch", &GridWorldSimulator::t_pyg_batch)
        .def_readonly("pyg_ptr", &GridWorldSimulator::t_pyg_ptr)
        .def_readonly("pyg_num_edges", &GridWorldSimulator::t_pyg_num_edges)
        .def_readonly("energy_maps", &GridWorldSimulator::t_energy_maps)
        .def_readonly("goal_changed_flags", &GridWorldSimulator::t_goal_changed_flags)
        .def_readonly("arrived", &GridWorldSimulator::t_arrived)
        .def_readonly("terminated", &GridWorldSimulator::t_terminated)
        .def_readonly("truncated", &GridWorldSimulator::t_truncated)
        .def_readonly("step_counts", &GridWorldSimulator::t_step_counts)
        .def_readonly("goal_changed_prefix", &GridWorldSimulator::t_goal_changed_prefix)
        .def_readonly("changed_state_packed", &GridWorldSimulator::t_changed_state_packed)
        .def_readonly("goal_changed_count", &GridWorldSimulator::t_goal_changed_count);
    
    py::class_<StatelessGridWorldSimulator>(m, "StatelessGridWorldSimulator")
        .def(py::init<torch::Tensor, int, int>())

        .def("refresh_compact_state_from_raw_batch", &StatelessGridWorldSimulator::refresh_compact_state_from_raw_batch)
        .def("materialize_pyg_inputs", &StatelessGridWorldSimulator::materialize_pyg_inputs)
        .def("update_derived_state", &StatelessGridWorldSimulator::update_derived_state)
        .def("build_magat_plus_nodes", &StatelessGridWorldSimulator::build_magat_plus_nodes)
        .def("finalize_magat_plus_graph", &StatelessGridWorldSimulator::finalize_magat_plus_graph)
        .def("build_magat_plus_inputs", &StatelessGridWorldSimulator::build_magat_plus_inputs)
        .def("setup_imitation_obs", &StatelessGridWorldSimulator::setup_imitation_obs)
        .def("update_energy_maps", &StatelessGridWorldSimulator::update_energy_maps)
        .def_property("pyg_builder_mode", &StatelessGridWorldSimulator::get_pyg_builder_mode, &StatelessGridWorldSimulator::set_pyg_builder_mode)
        .def_property("pyg_local_gather_impl", &StatelessGridWorldSimulator::get_pyg_local_gather_impl, &StatelessGridWorldSimulator::set_pyg_local_gather_impl)
        .def_property_readonly("resolved_pyg_builder_impl", &StatelessGridWorldSimulator::get_resolved_pyg_builder_impl)
        .def_property_readonly("memory_tensors", &stateless_memory_tensors)

        // Output tensors (read-only from Python)
        .def_readonly("grid_ocp", &StatelessGridWorldSimulator::t_grid_ocp)
        .def_readonly("state_packed", &StatelessGridWorldSimulator::t_state_packed)
        .def_readonly("energy_maps", &StatelessGridWorldSimulator::t_energy_maps)
        .def_readonly("pyg_x", &StatelessGridWorldSimulator::t_pyg_x)
        .def_readonly("pyg_pos", &StatelessGridWorldSimulator::t_pyg_pos)
        .def_property_readonly("pyg_edge_index", [](StatelessGridWorldSimulator& self) {
            int64_t num_edges = self.t_pyg_num_edges.cpu().item<int64_t>();
            return self.t_pyg_edge_index_storage.narrow(1, 0, num_edges);
        })
        .def_property_readonly("pyg_edge_attr", [](StatelessGridWorldSimulator& self) {
            int64_t num_edges = self.t_pyg_num_edges.cpu().item<int64_t>();
            return self.t_pyg_edge_attr_storage.narrow(0, 0, num_edges);
        })
        .def_readonly("pyg_edge_index_storage", &StatelessGridWorldSimulator::t_pyg_edge_index_storage)
        .def_readonly("pyg_edge_attr_storage", &StatelessGridWorldSimulator::t_pyg_edge_attr_storage)
        .def_readonly("pyg_batch", &StatelessGridWorldSimulator::t_pyg_batch)
        .def_readonly("pyg_ptr", &StatelessGridWorldSimulator::t_pyg_ptr)
        .def_readonly("pyg_num_edges", &StatelessGridWorldSimulator::t_pyg_num_edges);
}
