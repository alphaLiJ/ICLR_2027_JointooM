# Release contents and verification

The repository includes the original JointLoom CUDA simulation, input builders,
expert streaming and model integrations, together with the discussion-period
CPU baseline, optimized GPU adaptation, full MAGAT+ adaptation, identical-learning
control, sampling-order study and native-SF/resident-PPO paths.

[Implementation index](REBUTTAL_IMPLEMENTATION_INDEX.md) |
[Experiment setup](../rebuttal/README.md) |
[Measurements and configurations](../rebuttal/data/README.md)

The integration checks exercise the cached MAGAT and MAPF-GPT CPU builders,
unchanged-goal cache reuse, single-goal refresh and reference token agreement.
The MAPF-GPT schema suite has six passing tests. Source and configuration
files parse, and runner help/import checks cover the added entry points using
their respective experiment dependencies. The new cached-token argument is
ported from the executed cached-prefetch source. The official MAGAT+ parser
reproduces all 118 non-path fields of its recorded configuration, and the native
SF parser matches 16 checked rollout/PPO constants. Spawned workers preserve
their workflow-specific installed Pogema version.

The base CUDA build and GPU semantic checks remain available through
`scripts/build_extension.sh` and `scripts/review_smoke.sh`. Added workflows
build their separate native snapshots with `rebuttal/build_native.sh`.
Large corpora, released pretrained checkpoints, generated task banks and full
training logs are supplied as experiment inputs; compact measured records and
configurations are included under `rebuttal/data/`.

Build outputs, checkpoints and generated runs remain outside tracked source.
Third-party source retains its original notices and license files.

All repository documentation, comments and generated-report text use English.
Integration includes default online-IL YAML settings and the auxiliary experiment
plans referenced by controllers. Captured controlled-learning source resolves
its base modules through the inherited checkout path. Native source translation
preserves executable text across all 77 bundled C++/CUDA/header files. The 58
extracted measurement records match the accompanying archive byte for byte.

The release manifest lists the current tracked source, configurations and data,
excluding the manifest itself. GPU input diagnostics use the workflow-local native
build selected by their bootstrap, including when spawning child processes.
