from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir, user_state_dir

APP_NAME = "voxtray"

CONFIG_DIR = Path(user_config_dir(APP_NAME))
DATA_DIR = Path(user_data_dir(APP_NAME))
STATE_DIR = Path(user_state_dir(APP_NAME))
RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", STATE_DIR)) / APP_NAME

CONFIG_FILE = CONFIG_DIR / "config.toml"
STATE_FILE = STATE_DIR / "state.json"
STATE_LOCK_FILE = STATE_DIR / "state.lock"
HISTORY_FILE = DATA_DIR / "history.json"
RECORDINGS_DIR = DATA_DIR / "recordings"
VLLM_LOG_FILE = STATE_DIR / "vllm.log"
APP_LOG_FILE = STATE_DIR / "voxtray.log"


def ensure_app_dirs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def ensure_recordings_dir() -> Path:
    ensure_app_dirs()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return RECORDINGS_DIR
