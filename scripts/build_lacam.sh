#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

PYTHONPATH="${repo_root}/src:${repo_root}" python -c \
  'from real_expert_alg.lacam import build_lacam_library; print(build_lacam_library())'
