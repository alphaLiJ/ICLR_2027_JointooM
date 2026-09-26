# Paper experiment surface

This directory owns only versioned experiment intent and evidence references.
Large outputs stay outside Git and are supplied through explicit
`--artifact-root` arguments or the `MAPF_CUDA_ARTIFACT_ROOT` environment
variable.

Current direct runners remain importable from the compatibility namespace
`expert.minimal_*_runner`; report conversion lives in
`expert.render_*_report`. Their reusable simulator/training logic has already
been extracted into `mapf_cuda`. Moving command names is deferred until after
the migration commit so the retained formal commands remain reproducible.

- `configs/`: explicit future run configurations;
- `manifests/`: hashes and paths for frozen inputs/evidence;
- `runners/`: reserved for stable CLI wrappers;
- `renderers/`: reserved for pure report entry points.
