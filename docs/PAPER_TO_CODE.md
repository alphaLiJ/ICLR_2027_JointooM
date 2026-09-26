# Paper-to-code map

| Paper narrative | Implementation | Evidence-producing / checking path |
| --- | --- | --- |
| Background: complete joint configuration and model view | `src/cuda_backend/grid_world_simulator.{h,cpp}`, `src/expert/fixed_magat_plus_runtime.py`, `src/expert/mapf_gpt_online.py` | `tests/expert/test_compact_state_api.py`, `test_mapf_gpt_schema.py` |
| Resident execution: state, view, policy, joint transition | `src/cuda_backend/128_agents_map_step.cu`, `grid_world_cuda.cu`, `energy_map.cu`; `src/expert/a4_runtime.py` | `src/expert/minimal_p0_mapf_gpt_resident_runner.py`, `minimal_scaling_runner.py`; `tests/expert/test_stateful_standard_mapf.py` |
| MAGAT graph builder and goal cache | `src/cuda_backend/grid_world_simulator.cpp`, `energy_map.cu`; `src/expert/fixed_magat_plus_runtime.py` | `tests/expert/test_compact_state_api.py`, `test_obs_consistency.py` |
| MAPF-GPT token and history builder | `src/cuda_backend/mapf_gpt_builder.cpp`, `mapf_gpt_cuda.cu`; `src/expert/mapf_gpt_runtime.py`, `mapf_gpt_online.py` | `tests/expert/test_mapf_gpt_cuda.py`, `test_mapf_gpt_online.py` |
| Expert state replay and complete-stage publication | `src/expert/expert_running.py` (`RingBuffer`, `DMAWorker`, `ExtremeMAPFPipeline`); `src/mapf_cuda/training/topology_async.py` | `tests/expert/test_ring_fail_stop.py`, `test_pipeline.py`, `test_training_health.py` |
| CPU-expanded, cached-prefetch and compact comparison paths | `src/expert/training_baseline_engines.py`; `src/baselines/jax_sim_baseline.py` | `src/expert/minimal_e5_magat_runner.py`, `minimal_e5_mapf_gpt_runner.py`; `tests/expert/test_training_wall_budget.py` |
| Ring depth, batch granularity and supply diagnostics | `src/expert/minimal_f2_producer_scaling_runner.py`, `minimal_f3_ring_depth_runner.py`, `minimal_f4_batch_granularity_runner.py` | corresponding `tests/expert/test_minimal_f*_runner.py` |
| Training progress and validation | `src/expert/minimal_e5_magat_runner.py`, `src/expert/checkpoint_manager.py`; `experiments/runners/run_s2_multiseed_magat_training.py` | `tests/expert/test_checkpoint_manager.py`, `test_training_wall_budget.py` |
| Protocol fidelity | `src/expert/benchmark_contract.py`, `transition_adapters.py`, `mapf_gpt_validation.py` | `tests/expert/test_cuda_kernels.py`, `test_mapf_gpt_validation.py` |

Paths identify implementation and checking surfaces, not a claim that all
large formal inputs or output registries are present in this code ZIP. The
`artifacts/` defaults in selected runners are relative placeholders for
separately supplied input assets.
