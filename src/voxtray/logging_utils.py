from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .paths import APP_LOG_FILE, ensure_app_dirs


def configure_logging(verbose: bool = False) -> None:
    ensure_app_dirs()
    root = logging.getLogger()
    if root.handlers:
        return

    root_level = logging.DEBUG if verbose else logging.INFO
    root.setLevel(root_level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(APP_LOG_FILE, maxBytes=5_000_000, backupCount=2)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(root_level)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    root.addHandler(stream_handler)

    # Polling /v1/models for tray status can flood logs with httpx INFO entries.
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
