#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
  echo "Missing virtual environment at ${PROJECT_ROOT}/.venv" >&2
  exit 1
fi

export VIRTUAL_ENV="${PROJECT_ROOT}/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export LIBTORCH_USE_PYTORCH=1
TORCH_LIB_DIR="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

printf 'Activated Azul project environment\n'
printf 'Python: %s\n' "$(command -v python)"
printf 'Torch: %s\n' "$(python -c 'import torch; print(torch.__version__)')"
printf 'Torch libraries: %s\n' "${TORCH_LIB_DIR}"
