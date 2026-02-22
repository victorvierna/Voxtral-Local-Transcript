#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_DIR="${HOME}/.voxtray"
VLLM_INDEX_URL="https://download.pytorch.org/whl/cu129"
SKIP_APT=0
SKIP_VLLM=0
WITH_UI=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/install_wsl2.sh [options]

Options:
  --install-dir <path>        Install directory (default: ~/.voxtray)
  --vllm-index-url <url>      Extra index URL for vLLM wheels
  --skip-apt                  Skip apt-get system dependencies
  --skip-vllm                 Skip vLLM installation
  --with-ui                   Install voxtray with UI extras (PySide6)
  -h, --help                  Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    --vllm-index-url)
      VLLM_INDEX_URL="$2"
      shift 2
      ;;
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    --skip-vllm)
      SKIP_VLLM=1
      shift
      ;;
    --with-ui)
      WITH_UI=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "${ROOT_DIR}/pyproject.toml" ]]; then
  echo "Could not find pyproject.toml in ${ROOT_DIR}" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

if ! grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
  echo "Warning: this installer is intended for WSL2 (Microsoft kernel)." >&2
fi

if [[ "$SKIP_APT" -eq 0 ]]; then
  SUDO=()
  if command -v sudo >/dev/null 2>&1; then
    SUDO=(sudo)
  fi

  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y \
    ffmpeg \
    libportaudio2 \
    python3-venv \
    python3-pip \
    xclip
fi

INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"
TARGET_SRC="${INSTALL_DIR}/src/voxtray"
VENV_DIR="${INSTALL_DIR}/.venv"
WRAPPER_DIR="${INSTALL_DIR}/bin"

mkdir -p "$TARGET_SRC" "$WRAPPER_DIR"
rm -rf "$TARGET_SRC"
mkdir -p "$TARGET_SRC"

tar \
  --exclude=".git" \
  --exclude=".venv" \
  --exclude=".pytest_cache" \
  --exclude="dist" \
  --exclude="build" \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  -cf - \
  -C "$ROOT_DIR" . | tar -xf - -C "$TARGET_SRC"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install -U pip

if [[ "$WITH_UI" -eq 1 ]]; then
  "$VENV_DIR/bin/pip" install "${TARGET_SRC}[ui]"
else
  "$VENV_DIR/bin/pip" install "$TARGET_SRC"
fi

if [[ "$SKIP_VLLM" -eq 0 ]]; then
  "$VENV_DIR/bin/pip" install -U vllm --extra-index-url "$VLLM_INDEX_URL"
fi

"$VENV_DIR/bin/voxtray" init

cat > "${WRAPPER_DIR}/voxtray" <<EOF
#!/usr/bin/env bash
exec "${VENV_DIR}/bin/voxtray" "\$@"
EOF
chmod +x "${WRAPPER_DIR}/voxtray"

echo
echo "WSL2 installation completed."
echo "Install dir: ${INSTALL_DIR}"
echo "Command: ${VENV_DIR}/bin/voxtray"
echo
echo "Try:"
echo "  ${VENV_DIR}/bin/voxtray status"
echo "  ${VENV_DIR}/bin/voxtray record --toggle"
