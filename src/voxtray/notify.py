from __future__ import annotations

from shutil import which
import subprocess


def notify(title: str, body: str, urgency: str = "normal") -> None:
    if not which("notify-send"):
        return
    subprocess.run(
        ["notify-send", "-u", urgency, title, body],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
