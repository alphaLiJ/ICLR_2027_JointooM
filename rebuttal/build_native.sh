#!/usr/bin/env bash
set -euo pipefail

jointloom_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
  online-il) jointloom_native="${jointloom_repo}/rebuttal/online-il/native" ;;
  gpu-control) jointloom_native="${jointloom_repo}/rebuttal/gpu-control/native_optimized" ;;
  online-rl) jointloom_native="${jointloom_repo}/rebuttal/online-rl/native" ;;
  *) echo 'Usage: bash rebuttal/build_native.sh {online-il|gpu-control|online-rl}' >&2; exit 2 ;;
esac
export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX:?Activate the CUDA build environment or set CUDA_HOME}}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export MAX_JOBS="${MAX_JOBS:-4}"
cd "${jointloom_native}"
python setup.py build_ext --inplace
