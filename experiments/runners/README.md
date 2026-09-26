# Runner wrappers

Stable CLI wrappers live here; reusable logic belongs under `src/mapf_cuda`.
Temporary compatibility CLIs remain in `src/expert` only for retained formal
commands.

- `run_s1_closed_loop_audit.py`: isolated standard-MAPF policy-quality rows and
  the retained-checkpoint pilot suite. Every pilot row is launched in a fresh
  process; timing is diagnostic only.
- `run_s3_memory_decomposition.py`: isolated allocator/tensor memory census
  rows.
