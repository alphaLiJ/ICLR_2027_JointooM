#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the mapf-cuda conda environment first." >&2
  exit 2
fi

export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX}}"
export PATH="${CUDA_HOME}/bin:${PATH}"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "No nvcc found at ${CUDA_HOME}/bin/nvcc." >&2
  exit 2
fi

python - <<'PY'
import re
import subprocess
import torch

nvcc = subprocess.run(
    ["nvcc", "--version"], check=True, capture_output=True, text=True
).stdout
match = re.search(r"release\s+(\d+\.\d+)", nvcc)
if match is None:
    raise SystemExit("Could not parse nvcc version")
nvcc_version = match.group(1)
torch_version = str(torch.version.cuda)
if nvcc_version != torch_version:
    raise SystemExit(
        f"CUDA toolchain mismatch: nvcc={nvcc_version}, torch={torch_version}"
    )
print(f"Building with CUDA {nvcc_version} for PyTorch {torch.__version__}")
PY

python setup.py build_ext --inplace "$@"

# ``package_dir={'': 'src'}`` makes recent setuptools releases place this
# top-level extension under src/, while Python started from the repository
# root will prefer an older root-level artifact.  Keep the documented
# root-level import target synchronized so a rebuild cannot silently test a
# stale binary.
extension_path="$(find src -maxdepth 1 -type f -name 'grid_world_cpp*.so' -print -quit)"
if [[ -z "${extension_path}" ]]; then
  echo "Build completed without producing src/grid_world_cpp*.so" >&2
  exit 2
fi
cp -f -- "${extension_path}" "./${extension_path##*/}"
