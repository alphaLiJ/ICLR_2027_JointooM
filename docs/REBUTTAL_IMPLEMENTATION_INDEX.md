# Implementation entry points

Anonymous code: https://anonymous.4open.science/r/JointooM

The repository supplies implementations. This data ZIP supplies measurements,
configuration records and protocols rather than a second copy of the code.

| Component | Repository location |
| --- | --- |
| Executed cached CPU builder and pinned prefetch | src/mapf_cuda/training/host_inputs.py (CachedHostBuilder, HostPrefetch) |
| Frozen expert records and ring stage source | src/mapf_cuda/training/frozen_stage_source.py |
| Strong CPU training runner | experiments/runners/run_training_revision.py |
| Online window shuffle and random-ready selection | rebuttal/online-il/run_live_il.py; run_live_ready_il.py |
| Raw trajectory consumption | rebuttal/online-il/run_il_order.py |
| Optimized GPU adaptation and measurement | rebuttal/gpu-control/benchmark_input_optimization.py; input_optimization.py; discrete_mapf.py |
| CPU online-input full MAGAT+ adaptation | rebuttal/official-finetuning/cpu-online/ |
| JointLoom full MAGAT+ adaptation | rebuttal/official-finetuning/gpu/ |
| Identical-learning execution | rebuttal/controlled-learning/ |
| Same-corpus sampling/evaluation | rebuttal/fine-tuning-order/ |
| Native SF and resident PPO | rebuttal/online-rl/implementation/sf_native.py; train_pipeline.py; run_one_hour.py |
| Experimental data and configurations | rebuttal/data/ (extracted files and JointLoom_Rebuttal_Data.zip) |

The accompanying recorded protocols identify large initial corpora,
pretrained checkpoints and runtime dependencies as experiment inputs.

[Workflow setup and input locations](../rebuttal/README.md) describe the
repository-local import bindings, separate native builds and per-workflow
Pogema versions. The cached token path also uses
`src/expert/mapf_gpt_schema.py::tokenize_stage_reference(cost_maps_by_env=...)`,
ported from the executed cached-prefetch source.
