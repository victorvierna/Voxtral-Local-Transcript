#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/dist}"

mkdir -p "$OUTPUT_DIR"

VERSION="$(
  python3 - "$ROOT_DIR/pyproject.toml" <<'PY'
import pathlib
import sys
import tomllib

pyproject = pathlib.Path(sys.argv[1])
with pyproject.open("rb") as handle:
    data = tomllib.load(handle)
print(data["project"]["version"])
PY
)"

DATE="$(date +%Y%m%d)"
BUNDLE_NAME="voxtray-wsl2-${VERSION}-${DATE}"
STAGE_DIR="${OUTPUT_DIR}/${BUNDLE_NAME}"
ZIP_PATH="${OUTPUT_DIR}/${BUNDLE_NAME}.zip"

rm -rf "$STAGE_DIR" "$ZIP_PATH"
mkdir -p "$STAGE_DIR"

tar \
  --exclude=".git" \
  --exclude=".venv" \
  --exclude=".pytest_cache" \
  --exclude="dist" \
  --exclude="build" \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  -cf - \
  -C "$ROOT_DIR" . | tar -xf - -C "$STAGE_DIR"

python3 - "$OUTPUT_DIR" "$BUNDLE_NAME" <<'PY'
from pathlib import Path
import sys
import zipfile

output_dir = Path(sys.argv[1])
bundle_name = sys.argv[2]
source_dir = output_dir / bundle_name
zip_path = output_dir / f"{bundle_name}.zip"

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(source_dir.rglob("*")):
        if path.is_file():
            zf.write(path, arcname=path.relative_to(output_dir))

print(zip_path)
PY

echo
echo "WSL2 bundle created:"
echo "  ${ZIP_PATH}"
echo
echo "Share this .zip with students."
