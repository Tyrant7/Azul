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

printf 'Activated Azul project environment\n'
printf 'Python: %s\n' "$(command -v python)"
printf 'Torch: %s\n' "$(python -c 'import torch; print(torch.__version__)')"
