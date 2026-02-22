#!/usr/bin/env bash
set -euo pipefail

BINDING="${1:-<Super>F9}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/voxtray" ]]; then
  VOXTRAY_CMD="$ROOT_DIR/.venv/bin/voxtray"
else
  VOXTRAY_CMD="voxtray"
fi

APP_DIR="$HOME/.local/share/applications"
AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_FILE="$APP_DIR/voxtray.desktop"

mkdir -p "$APP_DIR" "$AUTOSTART_DIR"

cat > "$DESKTOP_FILE" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Voxtray
Comment=Local Voxtral realtime transcription tray app
Exec=$VOXTRAY_CMD tray
Icon=audio-input-microphone
Terminal=false
Categories=Utility;AudioVideo;
X-GNOME-Autostart-enabled=true
DESKTOP

cp "$DESKTOP_FILE" "$AUTOSTART_DIR/voxtray.desktop"

echo "Installed desktop entry: $DESKTOP_FILE"
echo "Installed autostart entry: $AUTOSTART_DIR/voxtray.desktop"

"$VOXTRAY_CMD" install-shortcut --binding "$BINDING" --command "$VOXTRAY_CMD record --toggle"

echo "GNOME shortcut installed: $BINDING"
