# Rebuttal experiments: setup and entry points

[Data index](data/README.md) | [Implementation index](../docs/REBUTTAL_IMPLEMENTATION_INDEX.md)

Each comparison states its workload, timer, seed and checkpoint rule in the
revised paper's Appendix A and the recorded configurations under `data/config/`.
The code below supplies the corresponding execution paths.

## Environments and native builds

Use the base Python 3.10 / PyTorch / CUDA environment for the CPU builder and
online IL. Install the GPU reference dependencies from
`gpu-control/requirements-adaptation.txt` in its experiment environment.

MAGAT+ official adaptation uses **Pogema 1.3.0**; native Sample Factory / RL
uses **Pogema 1.2.2 and Sample Factory 2.1.1**. These are separate experiment
environments. The small requirements files record their additional dependencies:

```bash
python -m pip install -r rebuttal/official-finetuning/requirements.txt
# In the separate RL environment:
python -m pip install -r rebuttal/online-rl/requirements.txt
```

Start the official/RL commands in the corresponding environment without
preloading the base `src/pogema` package. Their bootstrap binds the installed
experiment version before adding repository source paths, including in spawned
worker processes.

Build the native source snapshot for the workflow being evaluated:

```bash
bash rebuttal/build_native.sh online-il
bash rebuttal/build_native.sh gpu-control
bash rebuttal/build_native.sh online-rl
```

Each command builds an independent `grid_world_cpp` extension under that
workflow's `native/src` or `native_optimized/src` directory. Each workflow runs
in its own process. `CUDA_HOME` and `MAPF_CUDA_ARCH` use the base build
conventions. `JOINTLOOM_NATIVE_DIR` can select another built extension directory.
Build LaCAM with `bash scripts/build_lacam.sh` for the online expert paths.

## External inputs and output locations

Repository-owned Python/CUDA sources are resolved relative to this checkout.
The following variables select data and build inputs:

| Variable | Default / meaning |
| --- | --- |
| `JOINTLOOM_ARTIFACTS` | `artifacts/rebuttal/`: generated experiment inputs and outputs |
| `JOINTLOOM_PYTHON` | Current Python executable, including subprocesses |
| `JOINTLOOM_REPOSITORY` | Checkout root, inherited automatically by captured-source subprocesses |
| `JOINTLOOM_RAW_CORPUS` | Raw MAGAT records used to warm the online IL comparisons |
| `JOINTLOOM_VALIDATION` | Frozen validation trajectories for the original learning study |
| `JOINTLOOM_DECISION_INPUT` / `JOINTLOOM_DECISION_CHECKPOINT` | Frozen state input and MAGAT checkpoint for the GPU decision comparison |
| `JOINTLOOM_ORIGINAL_NATIVE_DIR` / `JOINTLOOM_OPTIMIZED_NATIVE_DIR` | Original/optimized CUDA builds used by the GPU comparison subprocesses |
| `JOINTLOOM_LACAM_LIBRARY` | Built `src/real_expert_alg/lacam/liblacam.so` |
| `JOINTLOOM_MAGAT_ARTIFACTS` | MAGAT+ dataset, maps and adaptation output root |
| `JOINTLOOM_MAGAT_CHECKPOINT` | Released pretrained `195.pt` |
| `JOINTLOOM_MAGAT_UPSTREAM` | Bundled `official-finetuning/upstream-learning/` Python sources, or a full upstream checkout |
| `JOINTLOOM_MAGAT_REFERENCE` | Complete JointLoom adaptation output used to replay the shared correction corpus |
| `JOINTLOOM_DAGGER_ARTIFACTS` | Controlled-learning input/output root |
| `JOINTLOOM_ORDER_ARTIFACTS` | Sampling-order output root |
| `JOINTLOOM_WAREHOUSE128` | Shared 128-agent evaluation map directory |
| `JOINTLOOM_RL_ARTIFACTS` | RL task-bank, runs and checkpoint root |
| `JOINTLOOM_COSTTRACER_REFERENCE` | Optional released CostTracer ONNX/C++ reference used by oracle checks |
| `JOINTLOOM_LEGACY_ARTIFACTS` | Original training/validation artifact root used by retained diagnostic scripts |
| `JOINTLOOM_ONLINE_ARTIFACTS` / `JOINTLOOM_GPU_ARTIFACTS` | Online IL / GPU diagnostic output roots |
| `JOINTLOOM_MAGAT_CPU_ARTIFACTS` | CPU online-input verification output root |
| `JOINTLOOM_REPORT_DIR` | Experiment status/report output directory |

Large corpora and checkpoints are supplied separately. The recorded files
under `data/` contain the reported measurements; the commands use actual
local input locations chosen above.

## Entry points

Run from the repository root. `--help` shows the CLI for each runner.

**Cached CPU / compact expert streaming (S1).**

```bash
PYTHONPATH=src:. python experiments/runners/run_training_revision.py --help
```

`CachedHostBuilder` caches goal-derived CPU inputs; `HostPrefetch` prepares
complete configurations in a bounded process pool. The runner supports
`cpu_cached_sync`, `cpu_cached_prefetch` and `compact_stream` and accepts
recorded JSON settings through `--config`. Configuration search records are
under `data/config/configuration-selection/`.

**Raw-trajectory consumption and online sampling (S3/S1).**

```bash
python rebuttal/online-il/run_il_order.py --help
python rebuttal/online-il/run_live_il.py --help
python rebuttal/online-il/run_live_ready_il.py --help
```

Fixed-record consumption accepts `--corpus`; live runs use
`JOINTLOOM_RAW_CORPUS` for their recorded warmup input. Windowed and ready-source
selection retain complete joint configurations.

**Optimized GPU execution (S2).**

```bash
python rebuttal/gpu-control/benchmark_input_optimization.py --help
```

This author adaptation implements discrete MAPF and MAGAT input logic within
CAMAR's JAX/Flax framework. Warmup covers every runtime JIT variant including
post-DLPack state variants; complete decisions end at the completed successor
state. `native_optimized/` is the corresponding JointLoom input-builder source.

**Complete MAGAT+ adaptation (S8).**

```bash
python rebuttal/official-finetuning/cpu-online/train_official.py --help
python rebuttal/official-finetuning/gpu/train_official.py --help
```

Supply `195.pt`, the initial raw expert corpus and warehouse maps under the
configured MAGAT input root. These runners retain the released network,
learning settings and policy-dependent LaCAM correction. Each minibatch has
64 joint graphs; the executed settings are in `official-configuration.json`.

**Identical-learning execution and same-corpus sampling (S9/S10).**

```bash
python rebuttal/controlled-learning/train_dagger.py --help
python rebuttal/fine-tuning-order/train_order.py --help
python rebuttal/fine-tuning-order/evaluate_order.py --help
```

S9 uses a fixed-iteration expert library, built by
`python rebuttal/controlled-learning/build_work_expert.py` after building
LaCAM. S10 replays the correction corpus from the full adaptation run using
`JOINTLOOM_MAGAT_REFERENCE`; the official per-task evaluator and recorded
sampling schedule match the data index. Initial/corrective corpora and their
run records are experiment inputs.

**Native SF and resident PPO (S5/S6).**

```bash
python rebuttal/online-rl/implementation/run_one_hour.py --help
python rebuttal/online-rl/implementation/train_pipeline.py --help
python rebuttal/online-rl/implementation/sf_native.py --env=jointloom_costtracer_pogema --help
```

The one-hour runner uses the current experiment Python for both subprocesses.
Task assignments are generated into `JOINTLOOM_RL_ARTIFACTS`; observations,
actions, rewards and policy updates are computed during interaction. The
2048-episode task bank and 3600-second launch-to-deadline protocol are recorded
in `formal-one-hour-arguments.json` and `data/config/online-rl/`.

The other retained report/diagnostic scripts identify their experiment-specific
inputs in their recorded paths or CLI. Model-free replays in S7 and complete
native-SF training in S5 use distinct timing scopes.
