#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/apply_profile.sh <local-balanced|local-vram-saver|local-latency|online-mistral|online-openai>

Applies a Voxtray profile to ~/.config/voxtray/config.toml,
creates a timestamped backup, and updates the local vLLM process only
when the selected provider uses the local engine.

Legacy aliases still work: balanced, vram-saver, latency.
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

PROFILE_NAME="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$PROFILE_NAME" in
  local-balanced|balanced)
    PROFILE_PATH="$ROOT_DIR/profiles/voxtray-balanced.toml"
    ;;
  local-vram-saver|vram-saver)
    PROFILE_PATH="$ROOT_DIR/profiles/voxtray-vram-saver.toml"
    ;;
  local-latency|latency)
    PROFILE_PATH="$ROOT_DIR/profiles/voxtray-latency.toml"
    ;;
  online-mistral)
    PROFILE_PATH="$ROOT_DIR/profiles/voxtray-online-mistral.toml"
    ;;
  online-openai)
    PROFILE_PATH="$ROOT_DIR/profiles/voxtray-online-openai.toml"
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

PROVIDER="$(awk -F= '
  /^\[transcription\]/ { in_section = 1; next }
  /^\[/ { in_section = 0 }
  in_section && $1 ~ /^[[:space:]]*provider[[:space:]]*$/ {
    gsub(/[[:space:]"]/, "", $2)
    print $2
    exit
  }
' "$CONFIG_PATH")"

if [[ "$PROVIDER" == "local_voxtral" || -z "$PROVIDER" ]]; then
  # Restart local vLLM so local profile flags are applied immediately.
  pkill -f "vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602" >/dev/null 2>&1 || true
  echo "Local vLLM will be restarted on next recording/warm command."
else
  # Cloud profiles must not keep consuming local VRAM.
  pkill -f "vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602" >/dev/null 2>&1 || true
  VOXTRAY_BIN="${VOXTRAY_BIN:-$ROOT_DIR/.venv/bin/voxtray}"
  if [[ -x "$VOXTRAY_BIN" ]]; then
    "$VOXTRAY_BIN" warm off >/dev/null 2>&1 || true
  fi
  echo "Stopped local vLLM if it was running; cloud provider does not use local VRAM."
fi

echo "Applied profile: $PROFILE_NAME"
echo "Provider: ${PROVIDER:-local_voxtral}"
echo "Config: $CONFIG_PATH"
