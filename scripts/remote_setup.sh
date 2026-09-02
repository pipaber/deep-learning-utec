#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' '== Host =='
hostname
uname -a

printf '%s\n' '== GPU =='
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  printf '%s\n' 'nvidia-smi not found'
fi

printf '%s\n' '== Disk =='
df -h .

if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' 'uv is required: https://docs.astral.sh/uv/getting-started/installation/'
  exit 1
fi

if ! command -v 7z >/dev/null 2>&1 \
  && ! command -v 7zz >/dev/null 2>&1 \
  && ! command -v 7za >/dev/null 2>&1; then
  printf '%s\n' 'Warning: install p7zip/7zip before using prepare --extract'
fi

printf '%s\n' '== Python environment =='
uv sync --frozen
uv run python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

printf '%s\n' 'Setup complete. No training was started.'
printf '%s\n' 'Next safe command: uv run animal-audio inspect-model --config configs/nddr_pcen.yaml --device cuda'
