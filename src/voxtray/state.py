from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Callable

from .paths import STATE_FILE, STATE_LOCK_FILE, ensure_app_dirs


def now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def pid_is_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False

    # Zombie processes still respond to kill(0) but are not actually running.
    status_path = Path(f"/proc/{pid}/status")
    if status_path.exists():
        try:
            with status_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("State:"):
                        if "\tZ" in line or " zombie" in line:
                            return False
                        break
        except OSError:
            pass

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class StateStore:
    def __init__(self, path: Path | None = None) -> None:
        ensure_app_dirs()
        self.path = path or STATE_FILE
        self.lock_path = STATE_LOCK_FILE if path is None else self.path.with_suffix(".lock")
        self._ensure_exists()

    def _default_state(self) -> dict[str, Any]:
        return {
            "recording_pid": None,
            "engine_pid": None,
            "warm_enabled": True,
            "last_toggle_epoch": 0.0,
            "last_error": "",
            "updated_at": now_utc_iso(),
        }

    def _ensure_exists(self) -> None:
        if not self.path.exists():
            self._write_unlocked(self._default_state())

    @contextmanager
    def _locked(self):
        ensure_app_dirs()
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default_state()
        with self.path.open("r", encoding="utf-8") as handle:
            try:
                loaded = json.load(handle)
            except json.JSONDecodeError:
                return self._default_state()
            if not isinstance(loaded, dict):
                return self._default_state()
            base = self._default_state()
            base.update(loaded)
            return base

    def _write_unlocked(self, state: dict[str, Any]) -> None:
        state["updated_at"] = now_utc_iso()
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=True, indent=2)
        tmp_path.replace(self.path)

    def read(self) -> dict[str, Any]:
        with self._locked():
            state = self._read_unlocked()
            changed = False
            if state.get("recording_pid") and not pid_is_alive(state["recording_pid"]):
                state["recording_pid"] = None
                changed = True
            if state.get("engine_pid") and not pid_is_alive(state["engine_pid"]):
                state["engine_pid"] = None
                changed = True
            if changed:
                self._write_unlocked(state)
            return state

    def write(self, state: dict[str, Any]) -> dict[str, Any]:
        with self._locked():
            self._write_unlocked(state)
            return state

    def update(self, mutator: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        with self._locked():
            state = self._read_unlocked()
            updated = mutator(state)
            self._write_unlocked(updated)
            return updated

    def set_values(self, **values: Any) -> dict[str, Any]:
        def _mutate(state: dict[str, Any]) -> dict[str, Any]:
            state.update(values)
            return state

        return self.update(_mutate)
