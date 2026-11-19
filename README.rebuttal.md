# Discussion-period implementation and data

The additional experiment implementations are integrated in this repository.

- [Setup and executable entry points](rebuttal/README.md)
- [Implementation index](docs/REBUTTAL_IMPLEMENTATION_INDEX.md)
- [Data and configuration index](rebuttal/data/README.md)
- [Data ZIP](rebuttal/data/JointLoom_Rebuttal_Data.zip)

The original simulator, models and expert modules remain the shared base.
`src/mapf_cuda/training/host_inputs.py` and
`experiments/runners/run_training_revision.py` provide the cached CPU baseline
identified in the reviewer responses. Added source paths resolve from this
repository; external corpora and checkpoints use the documented input variables.
