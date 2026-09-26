# Release status

This is a **local review candidate**, not a published anonymous repository.

Completed: source-only staging; exclusion of compiled files, logs, worktrees
and Git metadata from the archive inputs; replacement of six local-path
defaults in the review copy; Python syntax validation; and a text anonymity
scan. The staged source and a fresh extraction both built the CUDA extension
and LaCAM on Linux with CUDA 12.8/PyTorch 2.11. The focused smoke had 32
passed and 1 skipped tests on both; 21 additional experiment-entry and
wall-budget tests passed in staging. Generated build outputs are excluded
from the ZIP. Original/release hashes for changed files are in
`release_transforms.json`.

Before public upload, the authors need to settle project-owned code terms,
verify notices and redistribution rights for forked/bundled dependencies and
maps, add the accepted evidence bundle if paper-number reproduction is
promised, and rerun the anonymization and build checks if the code ZIP changes.

The code's source snapshot contained uncommitted changes when staged.
Re-export and rerun the checks if the implementation or manuscript changes.
Do not attach a development repository's Git history to the anonymous host.
