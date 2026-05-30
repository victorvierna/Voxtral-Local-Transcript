from __future__ import annotations

import logging
from shutil import which
import subprocess


def notify(title: str, body: str, urgency: str = "normal") -> bool:
    logger = logging.getLogger("voxtray.notify")
    if not which("notify-send"):
        logger.warning("notify-send is unavailable")
        return False
    try:
        proc = subprocess.run(
            ["notify-send", "-u", urgency, "-t", "8000", title, body],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=3.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("notify-send failed: %s", exc)
        return False
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        logger.warning("notify-send exited with %s: %s", proc.returncode, stderr)
        return False
    return True


def speak(text: str, *, language: str = "es") -> None:
    exe = which("spd-say")
    if not exe:
        return
    value = " ".join(str(text or "").split())
    if not value:
        return
    subprocess.Popen(
        [exe, "-l", language, value[:260]],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
