# JointLoom: reconstructible input delivery for learned MAPF

JointLoom accelerates simulation, policy-input construction and delivery around
neural computation. Complete joint configurations and required policy context
feed GPU builders for graph, token and CNN inputs. The implementation supports
resident execution, buffered expert learning and on-policy PPO.

## Implementation and experiment navigation

| Component | Implementation |
| --- | --- |
| CUDA MAPF simulation and input builders | [src/cuda_backend/](src/cuda_backend/) |
| MAGAT graph integration | [src/expert/fixed_magat_plus_runtime.py](src/expert/fixed_magat_plus_runtime.py) |
| MAPF-GPT tokens and history | [src/expert/mapf_gpt_online.py](src/expert/mapf_gpt_online.py), [mapf_gpt_schema.py](src/expert/mapf_gpt_schema.py) |
| Cached CPU builder and bounded prefetch | [src/mapf_cuda/training/host_inputs.py](src/mapf_cuda/training/host_inputs.py) |
| Frozen records and ring source | [src/mapf_cuda/training/frozen_stage_source.py](src/mapf_cuda/training/frozen_stage_source.py) |
| Strong CPU / compact-stream runner | [experiments/runners/run_training_revision.py](experiments/runners/run_training_revision.py) |
| Online selection and raw-trajectory consumption | [rebuttal/online-il/](rebuttal/online-il/) |
| Optimized CAMAR-framework MAPF adaptation | [rebuttal/gpu-control/](rebuttal/gpu-control/) |
| Full MAGAT+ adaptation, CPU online inputs / JointLoom | [rebuttal/official-finetuning/](rebuttal/official-finetuning/) |
| Identical-learning execution and sampling order | [rebuttal/controlled-learning/](rebuttal/controlled-learning/), [fine-tuning-order/](rebuttal/fine-tuning-order/) |
| Native Sample Factory and resident PPO | [rebuttal/online-rl/](rebuttal/online-rl/) |
| Measurements, protocols, configurations and learning curves | [rebuttal/data/](rebuttal/data/README.md) |

The [implementation index](docs/REBUTTAL_IMPLEMENTATION_INDEX.md) maps
reviewer questions to source files. The [paper-to-code map](docs/PAPER_TO_CODE.md)
covers the original simulator, builders, lifecycle and scaling experiments.
[Rebuttal setup and entry points](rebuttal/README.md) describe the added workflows.

## Build and check the base system

The evaluated system uses Linux, Python 3.10, CUDA 12.8, PyTorch and an RTX 5080.
`environment.yml` records the base environment. From the repository root:

```bash
conda env create -f environment.yml
conda run -n mapf-cuda bash scripts/build_extension.sh
conda run -n mapf-cuda bash scripts/build_lacam.sh
conda run -n mapf-cuda bash scripts/review_smoke.sh
```

The base build script detects the visible GPU architecture; `MAPF_CUDA_ARCH`
can override it. Added experiments retain separate native source snapshots
and dependency profiles, documented in [rebuttal/README.md](rebuttal/README.md).

## Data and experiment inputs

[rebuttal/data/README.md](rebuttal/data/README.md) indexes Tables S1--S3,
S4b, S5--S10 and Figures S1/S2. The same records are available as
[JointLoom_Rebuttal_Data.zip](rebuttal/data/JointLoom_Rebuttal_Data.zip),
including per-seed/per-task measurements, recorded configurations and curves.

Large raw expert corpora, pretrained checkpoints and generated task banks
are separate experiment inputs. Supply these through the documented CLI
arguments and `JOINTLOOM_*` variables; outputs default to `artifacts/rebuttal/`.
Recorded JSON paths identify experiment inputs and are retained as provenance;
execution uses the configured local input locations. The full adaptation
experiments start from released MAGAT+ checkpoint `195.pt`.

Bundled third-party components retain their own notices and license files.
