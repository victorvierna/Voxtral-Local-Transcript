#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install -U pip
pip install -e '.[ui,dev]'

echo "Installed voxtray in .venv"
echo "If tray mode fails with Qt xcb errors, install system dependency:"
echo "  sudo apt-get install -y libxcb-cursor0"
echo "Install vLLM (CUDA 12.9 wheels) with:"
echo "  .venv/bin/pip install -U vllm --extra-index-url https://download.pytorch.org/whl/cu129"
echo "If you prefer nightly and use uv, run:"
echo "  uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly"
