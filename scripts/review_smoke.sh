#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/src:${repo_root}"

python -c 'import torch, grid_world_cpp; print("CUDA extension import: OK")'
pytest -q \
  tests/expert/test_compact_state_api.py \
  tests/expert/test_mapf_gpt_cuda.py \
  tests/expert/test_ring_fail_stop.py
