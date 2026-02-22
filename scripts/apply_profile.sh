#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/apply_profile.sh <balanced|vram-saver|latency>

Applies a Voxtray profile to ~/.config/voxtray/config.toml,
creates a timestamped backup, and restarts local vLLM.
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

PROFILE_NAME="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$PROFILE_NAME" in
  balanced)
    PROFILE_PATH="$ROOT_DIR/profiles/voxtray-balanced.toml"
    ;;
  vram-saver)
    PROFILE_PATH="$ROOT_DIR/profiles/voxtray-vram-saver.toml"
    ;;
  latency)
    PROFILE_PATH="$ROOT_DIR/profiles/voxtray-latency.toml"
    ;;
  *)
    usage
    exit 1
    ;;
esac

CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/voxtray"
CONFIG_PATH="$CONFIG_DIR/config.toml"
mkdir -p "$CONFIG_DIR"

if [[ -f "$CONFIG_PATH" ]]; then
  cp "$CONFIG_PATH" "$CONFIG_PATH.bak.$(date +%Y%m%d-%H%M%S)"
fi

cp "$PROFILE_PATH" "$CONFIG_PATH"

# Restart local vLLM so new flags are applied immediately.
pkill -f "vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602" >/dev/null 2>&1 || true

echo "Applied profile: $PROFILE_NAME"
echo "Config: $CONFIG_PATH"
