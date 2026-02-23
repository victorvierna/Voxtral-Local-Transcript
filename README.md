# Voxtray

Voxtray is a local real-time transcription utility (CLI + system tray) built around `mistralai/Voxtral-Mini-4B-Realtime-2602`.

It is designed for daily Ubuntu/WSL2 usage with quick toggle activation, automatic clipboard copy, and recent-history tracking.

Spanish documentation: `LEEME.md`.

## Features

- `start/stop/toggle` recording from terminal.
- Global GNOME shortcut (`Super+F9` by default).
- Automatic copy of final transcript to clipboard.
- Persistent history (last 5 transcripts by default).
- Tray mode (`tray`) with quick actions and engine status.
- Audio file transcription (`transcribe-file`).
- Windows + WSL2 distribution flow (included scripts).

## Requirements

### Ubuntu 24.04 / WSL2

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg xclip libportaudio2 libxcb-cursor0
```

### Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```

If you want Qt tray mode:

```bash
pip install -e '.[ui]'
```

Install `vLLM` (NVIDIA GPU):

```bash
pip install -U vllm --extra-index-url https://download.pytorch.org/whl/cu129
```

Hugging Face access (optional):
- For public models, a token is usually not required.
- You only need a token if the model is gated/restricted or private.

```bash
export HF_TOKEN=...
```

## Quick Start

Initialize config:

```bash
voxtray init
```

Check status and logs:

```bash
voxtray status
voxtray logs --target all --lines 200
```

Record from terminal:

```bash
voxtray record --start
voxtray record --stop
# recommended for global shortcut usage
voxtray record --toggle
```

Engine/model controls:

```bash
voxtray warm on
voxtray warm off
voxtray warm status
voxtray model load
voxtray model unload
voxtray model status
```

History:

```bash
voxtray history list
voxtray history copy 1
```

Transcribe audio file:

```bash
voxtray transcribe-file /path/audio.m4a --copy
```

## Configuration and Profiles

Main file:

- `~/.config/voxtray/config.toml`

Included memory profiles:

- `profiles/voxtray-balanced.toml`
- `profiles/voxtray-vram-saver.toml`
- `profiles/voxtray-latency.toml`

Apply profile:

```bash
scripts/apply_profile.sh balanced
```

Available values:

- `balanced`
- `vram-saver`
- `latency`

## GNOME Integration

Install desktop entry + autostart + default shortcut (`Super+F9`):

```bash
scripts/install_ubuntu_integration.sh
```

Install only shortcut:

```bash
voxtray install-shortcut --binding '<Super>F9'
```

## Windows + WSL2

Build bundle to share:

```bash
scripts/build_wsl2_bundle.sh
```

Initial install on Windows (PowerShell), inside the extracted folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_windows_shortcuts.ps1 -Distro Ubuntu
```

What this installer does:
- Installs dependencies inside WSL2 (Ubuntu).
- Creates the Python environment and Voxtray in `~/.voxtray`.
- Generates Windows shortcuts to run Voxtray without opening a Linux terminal.

Daily activation and usage on Windows:
1. Use `Voxtray Toggle` shortcut (desktop or start menu) to start/stop recording.
2. Use `Voxtray Warm On/Off` to keep or release the engine in memory.
3. Use `Voxtray Status` for quick status checks.
4. Use `Voxtray Logs` for diagnostics.

Global shortcut on Windows (optional):
1. Right-click `Voxtray Toggle` and open `Properties`.
2. In the `Shortcut` tab, focus `Shortcut key`.
3. Press your desired key combination (recommended: `Ctrl + Alt + F9`).
4. Click `Apply` and `OK`.

Notes:
- The shortcut also exists in `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Voxtray WSL`.
- If you choose a simple key, Windows typically converts it to `Ctrl + Alt + <key>`.
- Avoid combinations already used by other apps or the OS.

Quick check from PowerShell:

```powershell
wsl -d Ubuntu -- bash -lc "~/.voxtray/.venv/bin/voxtray status"
wsl -d Ubuntu -- bash -lc "~/.voxtray/.venv/bin/voxtray warm on"
```

## Development

Run tests:

```bash
pytest
```

Project structure:

- `src/voxtray/`: main implementation (CLI, controller, realtime, tray).
- `tests/`: unit tests.
- `scripts/`: Ubuntu/WSL2 integration and packaging utilities.
- `profiles/`: performance/memory configuration templates.

## Security and Local Data

- Do not commit credentials or tokens.
- Use local environment variables (example in `.env.example`).
- State, logs, and history are stored in your home directory:
  - `~/.local/state/voxtray/`
  - `~/.local/share/voxtray/`

## License

Apache-2.0. See `LICENSE`.
