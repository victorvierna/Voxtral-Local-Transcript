from __future__ import annotations

import os
from shutil import which
import subprocess
import time


class ClipboardError(RuntimeError):
    pass


CLIPBOARD_COMMAND_TIMEOUT_SECONDS = 5.0
XCLIP_START_TIMEOUT_SECONDS = 0.75
CLIPBOARD_VERIFY_TIMEOUT_SECONDS = 1.0
CLIPBOARD_VERIFY_ATTEMPTS = 4
CLIPBOARD_VERIFY_DELAY_SECONDS = 0.1


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


def _read_with_qt() -> str | None:
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:
        return None
    app = QApplication.instance()
    if app is None:
        return None
    return app.clipboard().text()


def _copy_with_cmd(
    command: list[str],
    text: str,
    *,
    timeout_seconds: float = CLIPBOARD_COMMAND_TIMEOUT_SECONDS,
) -> None:
    try:
        proc = subprocess.run(
            command,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClipboardError(
            f"clipboard command timed out after {timeout_seconds:.1f}s: {command[0]}"
        ) from exc
    if proc.returncode != 0:
        raise ClipboardError(proc.stderr.decode("utf-8", errors="replace").strip())


def _read_with_cmd(
    command: list[str],
    *,
    timeout_seconds: float = CLIPBOARD_VERIFY_TIMEOUT_SECONDS,
) -> str:
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClipboardError(
            f"clipboard read timed out after {timeout_seconds:.1f}s: {command[0]}"
        ) from exc
    if proc.returncode != 0:
        raise ClipboardError(proc.stderr.decode("utf-8", errors="replace").strip())
    return proc.stdout.decode("utf-8", errors="replace")


def _copy_with_xclip(text: str) -> None:
    # xclip owns the selection until another app replaces it. Do not wait for that
    # long-lived owner process from the record worker.
    try:
        proc = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise ClipboardError(f"xclip failed to start: {exc}") from exc
    try:
        if proc.stdin is None:
            raise ClipboardError("xclip stdin is unavailable")
        proc.stdin.write(text.encode("utf-8"))
        proc.stdin.close()
        try:
            returncode = proc.wait(timeout=XCLIP_START_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return
        if returncode != 0:
            stderr = ""
            if proc.stderr is not None:
                stderr = proc.stderr.read().decode("utf-8", errors="replace").strip()
            detail = f": {stderr}" if stderr else ""
            raise ClipboardError(f"xclip exited with status {returncode}{detail}")
    except ClipboardError:
        if proc.poll() is None:
            proc.kill()
        raise
    except OSError as exc:
        if proc.poll() is None:
            proc.kill()
        detail = str(exc).strip()
        if not detail and proc.stderr is not None:
            detail = proc.stderr.read().decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise ClipboardError(f"xclip pipe failed{suffix}") from exc
    except Exception:
        if proc.poll() is None:
            proc.kill()
        raise


def _read_with_backend(backend: str) -> str | None:
    selected = backend.lower().strip()
    if selected == "qt":
        return _read_with_qt()
    if selected == "xclip":
        if not which("xclip"):
            return None
        return _read_with_cmd(["xclip", "-selection", "clipboard", "-out"])
    if selected == "wl-copy":
        if not which("wl-paste"):
            return None
        return _read_with_cmd(["wl-paste", "--no-newline"])
    if selected == "xsel":
        if not which("xsel"):
            return None
        return _read_with_cmd(["xsel", "--clipboard", "--output"])
    return None


def verify_clipboard_text(text: str, backend: str) -> bool | None:
    """Return True/False when readable, or None when the backend cannot verify."""
    selected = backend.lower().strip()
    if not selected:
        return None

    supported = selected in {"qt", "xclip", "wl-copy", "xsel"}
    if not supported:
        return None

    for attempt in range(CLIPBOARD_VERIFY_ATTEMPTS):
        try:
            current = _read_with_backend(selected)
        except ClipboardError:
            return False
        if current is None:
            return None
        if current == text:
            return True
        if attempt < CLIPBOARD_VERIFY_ATTEMPTS - 1:
            time.sleep(CLIPBOARD_VERIFY_DELAY_SECONDS)
    return False


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
    supported = {"auto", "qt", "clip", "clip.exe", "wl-copy", "xclip", "xsel"}
    if selected not in supported:
        raise ClipboardError(f"unsupported clipboard backend: {backend}")

    if selected == "qt":
        if _copy_with_qt(text):
            return "qt"
        raise ClipboardError("qt clipboard backend is unavailable")

    if selected in {"clip", "clip.exe"}:
        if not which("clip.exe"):
            raise ClipboardError("clipboard backend not found: clip.exe")
        _copy_with_cmd(["clip.exe"], text)
        return "clip.exe"

    if selected == "wl-copy":
        if not which("wl-copy"):
            raise ClipboardError("clipboard backend not found: wl-copy")
        _copy_with_cmd(["wl-copy"], text)
        return "wl-copy"

    if selected == "xclip":
        if not which("xclip"):
            raise ClipboardError("clipboard backend not found: xclip")
        _copy_with_xclip(text)
        return "xclip"

    if selected == "xsel":
        if not which("xsel"):
            raise ClipboardError("clipboard backend not found: xsel")
        _copy_with_cmd(["xsel", "--clipboard", "--input"], text)
        return "xsel"

    if _copy_with_qt(text):
        return "qt"

    candidates: list[str] = []
    if _running_in_wsl() and which("clip.exe"):
        candidates.append("clip.exe")
    for candidate in ("clip.exe", "wl-copy", "xclip", "xsel"):
        if candidate not in candidates and which(candidate):
            candidates.append(candidate)

    if not candidates:
        raise ClipboardError(
            "no clipboard backend found. Install clip.exe (WSL), xclip or wl-copy, or select an available backend."
        )

    errors: list[str] = []
    for candidate in candidates:
        try:
            if candidate == "clip.exe":
                _copy_with_cmd(["clip.exe"], text)
            elif candidate == "wl-copy":
                _copy_with_cmd(["wl-copy"], text)
            elif candidate == "xclip":
                _copy_with_xclip(text)
            elif candidate == "xsel":
                _copy_with_cmd(["xsel", "--clipboard", "--input"], text)
        except ClipboardError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        return candidate

    raise ClipboardError("clipboard backends failed: " + "; ".join(errors))
