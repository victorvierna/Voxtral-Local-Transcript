from __future__ import annotations

import os
from shutil import which
import subprocess


class ClipboardError(RuntimeError):
    pass


def _copy_with_qt(text: str) -> bool:
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:
        return False
    app = QApplication.instance()
    if app is None:
        return False
    app.clipboard().setText(text)
    app.processEvents()
    return True


def _copy_with_cmd(command: list[str], text: str) -> None:
    proc = subprocess.run(
        command,
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise ClipboardError(proc.stderr.decode("utf-8", errors="replace").strip())


def _running_in_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/sys/kernel/osrelease", "r", encoding="utf-8") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


def copy_to_clipboard(text: str, backend: str = "auto") -> str:
    selected = backend.lower().strip()
    if selected in {"auto", "qt"} and _copy_with_qt(text):
        return "qt"

    prefer_windows_clipboard = selected in {"clip", "clip.exe"} or (
        selected == "auto" and _running_in_wsl()
    )
    if prefer_windows_clipboard and which("clip.exe"):
        _copy_with_cmd(["clip.exe"], text)
        return "clip.exe"

    if selected in {"auto", "clip", "clip.exe"} and which("clip.exe"):
        _copy_with_cmd(["clip.exe"], text)
        return "clip.exe"

    if selected in {"auto", "wl-copy"} and which("wl-copy"):
        _copy_with_cmd(["wl-copy"], text)
        return "wl-copy"
    if selected in {"auto", "xclip"} and which("xclip"):
        _copy_with_cmd(["xclip", "-selection", "clipboard"], text)
        return "xclip"
    if selected in {"auto", "xsel"} and which("xsel"):
        _copy_with_cmd(["xsel", "--clipboard", "--input"], text)
        return "xsel"

    if selected not in {"auto", "qt", "clip", "clip.exe", "wl-copy", "xclip", "xsel"}:
        raise ClipboardError(f"unsupported clipboard backend: {backend}")

    raise ClipboardError(
        "no clipboard backend found. Install clip.exe (WSL), xclip or wl-copy, or select an available backend."
    )
