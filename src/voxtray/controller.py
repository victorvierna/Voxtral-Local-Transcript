from __future__ import annotations

import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .clipboard import ClipboardError, copy_to_clipboard
from .config import VoxtrayConfig, load_config, write_default_config
from .engine import EngineError, EngineManager
from .history import HistoryStore
from .postprocess import normalize_transcript
from .realtime import RealtimeTranscriber
from .state import StateStore, pid_is_alive


class Controller:
    _ENGINE_READY_CACHE_TTL_SECONDS = 3.0
    _TOGGLE_DEBOUNCE_SECONDS = 0.45

    def __init__(self, config_path: Path | None = None) -> None:
        write_default_config(config_path)
        self.config: VoxtrayConfig = load_config(config_path)
        self.state_store = StateStore()
        self.history = HistoryStore(max_items=self.config.history.max_items)
        self.engine = EngineManager(self.config, self.state_store)
        self.logger = logging.getLogger("voxtray.controller")
        self._cached_engine_ready: bool | None = None
        self._cached_engine_ready_at: float = 0.0

    def _get_engine_ready(self, use_cache: bool = True) -> bool:
        now = time.monotonic()
        if (
            use_cache
            and self._cached_engine_ready is not None
            and now - self._cached_engine_ready_at <= self._ENGINE_READY_CACHE_TTL_SECONDS
        ):
            return self._cached_engine_ready

        ready = self.engine.is_ready(timeout_seconds=1.5)
        self._cached_engine_ready = ready
        self._cached_engine_ready_at = now
        return ready

    def _invalidate_engine_ready_cache(self) -> None:
        self._cached_engine_ready = None
        self._cached_engine_ready_at = 0.0

    def status(self) -> dict[str, object]:
        state = self.state_store.read()
        engine_ready = self._get_engine_ready(use_cache=True)
        return {
            "recording": bool(state.get("recording_pid")),
            "recording_pid": state.get("recording_pid"),
            "warm_enabled": bool(state.get("warm_enabled")),
            "engine_pid": state.get("engine_pid"),
            "engine_ready": engine_ready,
            "model_loaded": engine_ready,
            "server_url": self.config.server_base_url,
            "model_id": self.config.model_id,
            "last_error": state.get("last_error", ""),
        }

    def _spawn_record_worker(self) -> int:
        command = [sys.executable, "-m", "voxtray", "_record-worker"]
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return proc.pid

    def start_recording(self) -> str:
        state = self.state_store.read()
        pid = state.get("recording_pid")
        if pid and pid_is_alive(pid):
            return f"already recording (pid={pid})"

        self._spawn_record_worker()

        deadline = time.time() + 3.0
        while time.time() < deadline:
            new_state = self.state_store.read()
            if new_state.get("recording_pid"):
                return "recording started"
            time.sleep(0.1)
        return "recording worker started (initializing)"

    def stop_recording(self, timeout_seconds: float = 10.0) -> str:
        state = self.state_store.read()
        pid = state.get("recording_pid")
        if not pid:
            return "not recording"
        if not pid_is_alive(pid):
            self.state_store.set_values(recording_pid=None)
            return "recording was stale and is now cleared"

        os.kill(pid, signal.SIGUSR1)
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if not pid_is_alive(pid):
                self.state_store.set_values(recording_pid=None)
                return "recording stopped"
            time.sleep(0.1)

        return f"stop signal sent to pid={pid}, still shutting down"

    def toggle_recording(self) -> str:
        state = self.state_store.read()
        now = time.time()
        last_toggle = float(state.get("last_toggle_epoch") or 0.0)
        if now - last_toggle < self._TOGGLE_DEBOUNCE_SECONDS:
            return "toggle ignored (debounce)"
        self.state_store.set_values(last_toggle_epoch=now)
        pid = state.get("recording_pid")
        if pid and pid_is_alive(pid):
            return self.stop_recording()
        return self.start_recording()

    def warm_on(self) -> str:
        self.state_store.set_values(last_error="")
        try:
            self.engine.ensure_running()
        except Exception:
            self._invalidate_engine_ready_cache()
            self.state_store.set_values(warm_enabled=False)
            raise
        self._cached_engine_ready = True
        self._cached_engine_ready_at = time.monotonic()
        self.state_store.set_values(warm_enabled=True)
        return "warm mode enabled"

    def warm_off(self) -> str:
        self.state_store.set_values(warm_enabled=False)
        state = self.state_store.read()
        if not state.get("recording_pid"):
            self.engine.stop_if_running()
        self._invalidate_engine_ready_cache()
        return "warm mode disabled"

    def warm_status(self) -> dict[str, object]:
        state = self.state_store.read()
        engine_ready = self._get_engine_ready(use_cache=True)
        return {
            "warm_enabled": bool(state.get("warm_enabled")),
            "engine_pid": state.get("engine_pid"),
            "engine_ready": engine_ready,
            "model_loaded": engine_ready,
        }

    def load_model(self) -> str:
        self.state_store.set_values(last_error="")
        self.engine.ensure_running()
        self._cached_engine_ready = True
        self._cached_engine_ready_at = time.monotonic()
        return "model loaded"

    def unload_model(self) -> str:
        state = self.state_store.read()
        pid = state.get("recording_pid")
        if pid and pid_is_alive(pid):
            raise RuntimeError("cannot unload model while recording is active")
        self.engine.stop_if_running()
        self._cached_engine_ready = False
        self._cached_engine_ready_at = time.monotonic()
        return "model unloaded"

    def model_status(self) -> dict[str, object]:
        state = self.state_store.read()
        engine_ready = self._get_engine_ready(use_cache=True)
        return {
            "model_loaded": engine_ready,
            "engine_ready": engine_ready,
            "engine_pid": state.get("engine_pid"),
            "warm_enabled": bool(state.get("warm_enabled")),
            "last_error": state.get("last_error", ""),
        }

    def list_history(self) -> list[dict[str, object]]:
        return self.history.list_entries()

    def copy_history_item(self, index: int) -> tuple[dict[str, object], str]:
        entry = self.history.get_by_index(index)
        backend = copy_to_clipboard(entry["text"], backend=self.config.clipboard.backend)
        return entry, backend

    def transcribe_file(self, audio_path: Path, copy_result: bool = True) -> dict[str, object]:
        state = self.state_store.read()
        warm_enabled = bool(state.get("warm_enabled"))

        self.engine.ensure_running()
        transcriber = RealtimeTranscriber(self.config)
        text = transcriber.transcribe_file_blocking(audio_path)

        if self.config.postprocess.clean_text:
            text = normalize_transcript(text)

        entry = self.history.add_entry(text)
        clipboard_backend = ""
        if copy_result and text:
            try:
                clipboard_backend = copy_to_clipboard(
                    text, backend=self.config.clipboard.backend
                )
            except ClipboardError as exc:
                self.logger.warning("clipboard copy failed: %s", exc)

        if not warm_enabled:
            self.engine.stop_if_running()

        return {
            "text": text,
            "history_entry": entry,
            "clipboard_backend": clipboard_backend,
        }

    def apply_warm_preference(self, enabled: bool) -> str:
        if enabled:
            return self.warm_on()
        return self.warm_off()

    def preload_if_warm_enabled(self) -> str:
        state = self.state_store.read()
        if not state.get("warm_enabled"):
            return "warm mode disabled; skipping preload"
        self.load_model()
        return "model preloaded"

    def shutdown_for_exit(self) -> str:
        messages: list[str] = []
        state = self.state_store.read()
        pid = state.get("recording_pid")
        if pid and pid_is_alive(pid):
            messages.append(self.stop_recording(timeout_seconds=5.0))
        self.engine.stop_if_running(timeout_seconds=8.0)
        self._cached_engine_ready = False
        self._cached_engine_ready_at = time.monotonic()
        messages.append("model unloaded")
        return "; ".join(messages)

    def clear_last_error(self) -> None:
        self.state_store.set_values(last_error="")


def handle_engine_error(exc: Exception) -> str:
    return f"engine error: {exc}"
