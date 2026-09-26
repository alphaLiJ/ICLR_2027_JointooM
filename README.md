# JointLoom: reconstructible input delivery for learned MAPF

This is an anonymous-review **code candidate** for the JointLoom paper. It
contains the GPU simulator and input builders, compact CPU-expert streaming,
MAGAT and MAPF-GPT integrations, comparison paths, experiment runners, and
focused tests. The implementation is organized around the paper's input
boundary: complete joint state and policy context enter a backend builder;
the resulting view is consumed by either a resident policy decision or an
expert-supervised learner.

The candidate has not been published as a repository. See
[`docs/RELEASE_STATUS.md`](docs/RELEASE_STATUS.md) before distributing it.

## Find the implementation

| Paper concept | Code location | Short check |
| --- | --- | --- |
| Complete joint state, transition, occupancy | `src/cuda_backend/grid_world_simulator.cpp`, `grid_world_cuda.cu`, `128_agents_map_step.cu` | `tests/expert/test_compact_state_api.py` |
| Goal-derived cache and MAGAT graph view | `src/cuda_backend/energy_map.cu`, `grid_world_simulator.cpp`; `src/expert/fixed_magat_plus_runtime.py` | `tests/expert/test_compact_state_api.py` |
| MAPF-GPT token/history view | `src/cuda_backend/mapf_gpt_builder.cpp`, `mapf_gpt_cuda.cu`; `src/expert/mapf_gpt_online.py` | `tests/expert/test_mapf_gpt_cuda.py` |
| Resident complete decision | `src/expert/a4_runtime.py`; `src/mapf_cuda/evaluation/closed_loop.py` | `tests/expert/test_stateful_standard_mapf.py` |
| Expert publication, bounded ring, transfer and replay | `src/expert/expert_running.py`; `src/mapf_cuda/training/topology_async.py` | `tests/expert/test_ring_fail_stop.py` |
| CPU and GPU comparison paths | `src/expert/training_baseline_engines.py`; `src/baselines/jax_sim_baseline.py` | `tests/expert/test_training_wall_budget.py` |

The complete section-to-code and experiment-to-code map is in
[`docs/PAPER_TO_CODE.md`](docs/PAPER_TO_CODE.md). Existing `expert.*` imports
are retained because the tested experiment runners use them; this release
adds a navigation layer rather than changing experimental behavior.

## Requirements and quick check

The evaluated system uses Linux, a CUDA-capable NVIDIA GPU, Python 3.10,
CUDA 12.8, PyTorch, and JAX. The paper's measurements are for an RTX 5080;
performance on another GPU should be measured anew. `environment.yml` lists
the Python and CUDA dependencies. The LaCAM source is built locally.

From the repository root:

```bash
conda env create -f environment.yml
conda run -n mapf-cuda bash scripts/build_extension.sh
conda run -n mapf-cuda bash scripts/build_lacam.sh
conda run -n mapf-cuda bash scripts/review_smoke.sh
```

`scripts/review_smoke.sh` runs focused compact-state, token-builder, and ring
lifecycle tests in a fresh process. The broader suite is
`conda run -n mapf-cuda bash scripts/run_tests.sh`; it can take considerably
longer. The build script detects the visible GPU architecture by default;
`MAPF_CUDA_ARCH` can override it.

## Reproduction map

`experiments/runners/` contains experiment orchestration and
`src/expert/minimal_*_runner.py` contains the retained formal command
implementations. `experiments/renderers/` and `src/expert/render_*_report.py`
consume experiment outputs. `experiments/configs/` and
`experiments/manifests/` define input and protocol surfaces.

The full training registries, large frozen inputs, and checkpoints are not in
this code ZIP. Paths in the release-specific runner defaults point under
`artifacts/`; callers may supply their own artifact paths through the CLI.
See [`evidence/README.md`](evidence/README.md) for the intended separation of
code, compact summaries, and large optional rerun inputs. No timing claim in
the paper should be treated as rerun from this code ZIP alone.

## Repository layout

```text
src/cuda_backend/           CUDA simulation and backend builders
src/mapf_cuda/              reusable training, simulation and evaluation modules
src/expert/                 expert supply, policy adapters and formal runners
src/baselines/              same-workflow comparison implementation
src/pogema/                 MAPF reference integration
src/real_expert_alg/        LaCAM integration
third_party/                retained upstream integration source
experiments/                configs, manifests, runners and renderers
tests/                      semantic, lifecycle and experiment checks
docs/                       architecture, paper mapping and release audit
evidence/                   evidence acquisition and provenance notes
```

Third-party components retain their own terms. Project-owned code terms and
third-party provenance are still under review; see the release status before
putting this candidate on a public host.
