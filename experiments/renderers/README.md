# Report wrappers

Report wrappers must read immutable result files and write derived tables or
figures without executing simulation or training.

- `render_s1_closed_loop_audit.py`: seed-clustered paired policy-progress
  analysis from an accepted S1 aggregate; never reruns simulation.
- `render_s3_memory_decomposition.py`: allocator/tensor memory report.
