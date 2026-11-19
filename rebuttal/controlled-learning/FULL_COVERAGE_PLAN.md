# Full-horizon controlled online correction

Compare CPU input construction and JointLoom while preserving the same learning
computation. Each run uses 16 environments with 128 agents, the official 32x32
warehouse (416 traversable cells), released MAGAT+ checkpoint `195.pt`, and
32 rounds of 32 Adam updates, for 1024 updates per method.

Each round collects 512 steps in eight 64-step segments. Query every environment
once per segment, giving 128 joint-state queries per round. Track actual episode
steps across resets. Stratify replay by eight actual episode phases, taking two
online graphs per phase when labels are available and filling shortfalls from
other valid labeled states. Each batch contains 48 historical 64-agent graphs
and 16 online 128-agent graphs; the historical buffer contains 512 graphs.

Keep learning rate 1e-6, original Adam/weight decay, frozen BN running statistics,
deterministic FP32 and independent action/query/replay/dropout random streams.
Use one RTX5080, CPU0-15, eight environment/input workers and four live LaCAM
expert processes, with 2000 first-solution search iterations per query and ten
PIBT candidate threads per process. The CPU path retains caching, pinned
transfers, parallel preparation and reuse of constructed GPU replay inputs.

Per complete run: 33,554,432 agent transitions, 4096 expert queries and 5,242,880
learner agent samples. First run a separate four-round/128-update pilot and
compare model, BN, Adam, actions, states, labels and update records directly.
Then execute seeds 0/1/2, alternating CPU/JointLoom order, and report online and
full-process times separately.

Evaluate fixed final checkpoints on the same 128 new tasks with seeds
2037000000-2037000127 and a 512-step horizon. Include the pretrained reference
and previous 256-step models. Report success, arrival rate, SoC, makespan,
complete actions and actual sampling coverage. Tests do not select checkpoints.
The coverage comparison measures the combined longer-horizon and stratification
changes; it does not isolate an individual sampling component.

Inputs and outputs live under `JOINTLOOM_DAGGER_ARTIFACTS`. The controller uses
the prior `controlled-parity-agents128-u1024/maps/warehouse32` input and writes
`coverage512-agents128-u1024/`. Set `JOINTLOOM_REPORT_DIR` for status and reports.
See `../data/` for the recorded protocol and measurements.
