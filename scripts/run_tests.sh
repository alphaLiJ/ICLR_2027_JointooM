#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export PYTHONPATH="${repo_root}/src:${repo_root}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Keep PyTorch/CUDA and JAX allocations in separate process lifetimes.
pytest -q tests/expert --ignore=tests/expert/test_transition_adapters.py
pytest -q tests/test_jax_sim_baseline.py
pytest -q tests/expert/test_transition_adapters.py
