from __future__ import annotations

import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .backends import create_transcription_backend, provider_api_key_env
from .clipboard import ClipboardError, copy_to_clipboard, verify_clipboard_text
from .config import VoxtrayConfig, load_config, write_default_config
from .engine import EngineError, EngineManager
from .history import HistoryStore
from .postprocess import normalize_transcript
from .state import StateStore, UNEXPECTED_RECORDING_EXIT_MESSAGE, pid_is_alive


class Controller:
    _ENGINE_READY_CACHE_TTL_SECONDS = 3.0
    _TOGGLE_DEBOUNCE_SECONDS = 0.45

    def __init__(self, config_path: Path | None = None) -> None:
        write_default_config(config_path)
        self.config: VoxtrayConfig = load_config(config_path)
        self.state_store = StateStore()
        self.history = HistoryStore(max_items=self.config.history.max_items)
        self.backend = create_transcription_backend(self.config)
        self.engine = EngineManager(self.config, self.state_store)
        self.logger = logging.getLogger("voxtray.controller")
        self._cached_engine_ready: bool | None = None
        self._cached_engine_ready_at: float = 0.0

    def _requires_local_engine(self) -> bool:
        return bool(self.backend.capabilities.local_engine_required)

    def _get_engine_ready(self, use_cache: bool = True) -> bool:
        if not self._requires_local_engine():
            self._cached_engine_ready = False
            self._cached_engine_ready_at = time.monotonic()
            return False
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
        api_key_env = provider_api_key_env(self.config)
        api_key_env_present = False
        if api_key_env:
            backend_key_present = getattr(self.backend, "api_key_env_present", None)
            if callable(backend_key_present):
                api_key_env_present = bool(backend_key_present())
            else:
                api_key_env_present = bool(os.environ.get(api_key_env))
        warm_enabled = bool(state.get("warm_enabled")) and self.backend.capabilities.warm_supported
        if self._requires_local_engine():
            provider_ready = engine_ready
        else:
            try:
                self.backend.check_ready_blocking()
            except Exception:
                provider_ready = False
            else:
                provider_ready = True
        recording_pid = state.get("recording_pid")
        activity_state = str(state.get("activity_state") or "idle")
        stop_requested = bool(recording_pid and state.get("recording_stop_requested"))
        is_recording = bool(
            recording_pid and not stop_requested and activity_state == "recording"
        )
        is_processing = bool(
            recording_pid
            and (
                stop_requested
                or activity_state in {"transcribing", "copying"}
            )
        )
        return {
            "recording": is_recording,
            "processing": is_processing,
            "activity_state": activity_state,
            "recording_pid": recording_pid,
            "recording_stop_requested": stop_requested,
            "warm_enabled": warm_enabled,
            "engine_pid": state.get("engine_pid"),
            "engine_ready": engine_ready,
            "local_engine_ready": engine_ready,
            "provider": self.backend.provider_id,
            "provider_ready": provider_ready,
            "warm_supported": self.backend.capabilities.warm_supported,
            "api_key_env_present": api_key_env_present,
            "model_loaded": engine_ready,
            "server_url": self.config.server_base_url,
            "model_id": self.backend.provider_model,
            "last_error": state.get("last_error", ""),
            "last_notice_id": state.get("last_notice_id", ""),
            "last_notice_title": state.get("last_notice_title", ""),
            "last_notice_body": state.get("last_notice_body", ""),
            "last_notice_level": state.get("last_notice_level", "info"),
            "last_artifact_path": state.get("last_artifact_path", ""),
            "last_history_id": state.get("last_history_id", ""),
            "last_history_index": state.get("last_history_index", 0),
            "last_clipboard_backend": state.get("last_clipboard_backend", ""),
            "last_clipboard_verified": state.get("last_clipboard_verified", False),
            "last_clipboard_verification_supported": state.get(
                "last_clipboard_verification_supported",
                False,
            ),
            "last_clipboard_error": state.get("last_clipboard_error", ""),
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
            if state.get("recording_stop_requested"):
                return f"recording is stopping and transcribing (pid={pid})"
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
            self.state_store.set_values(
                recording_pid=None,
                recording_stop_requested=False,
                activity_state="idle",
            )
            return "recording was stale and is now cleared"
        if state.get("recording_stop_requested"):
            message = f"recording is already stopping and transcribing (pid={pid})"
        else:
            os.kill(pid, signal.SIGUSR1)
            self.state_store.set_values(
                recording_stop_requested=True,
                activity_state="transcribing",
            )
            message = f"stop signal sent to pid={pid}, still shutting down"
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if not pid_is_alive(pid):
                self.state_store.set_values(
                    recording_pid=None,
                    recording_stop_requested=False,
                    activity_state="idle",
                )
                return "recording stopped"
            time.sleep(0.1)

        return message

    def toggle_recording(self) -> str:
        state = self.state_store.read()
        if (
            not state.get("recording_pid")
            and state.get("activity_state") == "error"
            and state.get("last_error") == UNEXPECTED_RECORDING_EXIT_MESSAGE
        ):
            self.state_store.set_values(activity_state="idle")
            return UNEXPECTED_RECORDING_EXIT_MESSAGE

        now = time.time()
        last_toggle = float(state.get("last_toggle_epoch") or 0.0)
        if now - last_toggle < self._TOGGLE_DEBOUNCE_SECONDS:
            return "toggle ignored (debounce)"
        self.state_store.set_values(last_toggle_epoch=now)
        pid = state.get("recording_pid")
        if pid and pid_is_alive(pid):
            if state.get("recording_stop_requested"):
                return f"recording is already stopping and transcribing (pid={pid})"
            return self.stop_recording()
        return self.start_recording()

    def warm_on(self) -> str:
        self.state_store.set_values(last_error="")
        if not self.backend.capabilities.warm_supported:
            self.state_store.set_values(warm_enabled=False)
            return f"warm mode unsupported for {self.backend.provider_id}; cloud provider has no local engine"
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
        if not self._requires_local_engine():
            self._invalidate_engine_ready_cache()
            return f"warm mode disabled; {self.backend.provider_id} has no local engine"
        state = self.state_store.read()
        if not state.get("recording_pid"):
            self.engine.stop_if_running()
        self._invalidate_engine_ready_cache()
        return "warm mode disabled"

    def warm_status(self) -> dict[str, object]:
        state = self.state_store.read()
        engine_ready = self._get_engine_ready(use_cache=True)
        warm_enabled = bool(state.get("warm_enabled")) and self.backend.capabilities.warm_supported
        return {
            "warm_enabled": warm_enabled,
            "engine_pid": state.get("engine_pid"),
            "engine_ready": engine_ready,
            "local_engine_ready": engine_ready,
            "provider": self.backend.provider_id,
            "warm_supported": self.backend.capabilities.warm_supported,
            "model_loaded": engine_ready,
        }

    def load_model(self) -> str:
        self.state_store.set_values(last_error="")
        if not self.backend.capabilities.model_control_supported:
            return f"model load unsupported for {self.backend.provider_id}; cloud provider has no local engine"
        self.engine.ensure_running()
        self._cached_engine_ready = True
        self._cached_engine_ready_at = time.monotonic()
        return "model loaded"

    def unload_model(self) -> str:
        if not self.backend.capabilities.model_control_supported:
            self._cached_engine_ready = False
            self._cached_engine_ready_at = time.monotonic()
            return f"model unload unsupported for {self.backend.provider_id}; cloud provider has no local engine"
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
        warm_enabled = bool(state.get("warm_enabled")) and self.backend.capabilities.warm_supported
        return {
            "model_loaded": engine_ready,
            "engine_ready": engine_ready,
            "local_engine_ready": engine_ready,
            "provider": self.backend.provider_id,
            "model_id": self.backend.provider_model,
            "model_control_supported": self.backend.capabilities.model_control_supported,
            "engine_pid": state.get("engine_pid"),
            "warm_enabled": warm_enabled,
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

        if self._requires_local_engine():
            self.engine.ensure_running()
        transcriber = create_transcription_backend(self.config)
        transcriber.check_ready_blocking()
        text = transcriber.transcribe_file_blocking(audio_path)

        if self.config.postprocess.clean_text:
            text = normalize_transcript(text)

        entry = self.history.add_entry(text)
        clipboard_backend = ""
        clipboard_verified: bool | None = None
        if copy_result and text:
            try:
                clipboard_backend = copy_to_clipboard(
                    text, backend=self.config.clipboard.backend
                )
                clipboard_verified = verify_clipboard_text(text, clipboard_backend)
            except ClipboardError as exc:
                self.logger.warning("clipboard copy failed: %s", exc)

        if self._requires_local_engine() and not warm_enabled:
            self.engine.stop_if_running()

        return {
            "text": text,
            "history_entry": entry,
            "clipboard_backend": clipboard_backend,
            "clipboard_verified": clipboard_verified,
            "provider": transcriber.provider_id,
            "provider_model": transcriber.provider_model,
        }

    def apply_warm_preference(self, enabled: bool) -> str:
        if enabled:
            return self.warm_on()
        return self.warm_off()

    def preload_if_warm_enabled(self) -> str:
        state = self.state_store.read()
        if not state.get("warm_enabled"):
            return "warm mode disabled; skipping preload"
        if not self.backend.capabilities.warm_supported:
            self.state_store.set_values(warm_enabled=False)
            return f"warm mode disabled; {self.backend.provider_id} has no local engine"
        self.load_model()
        return "model preloaded"

    def shutdown_for_exit(self) -> str:
        messages: list[str] = []
        state = self.state_store.read()
        pid = state.get("recording_pid")
        if pid and pid_is_alive(pid):
            messages.append(self.stop_recording(timeout_seconds=5.0))
        if self._requires_local_engine():
            self.engine.stop_if_running(timeout_seconds=8.0)
            messages.append("model unloaded")
        else:
            messages.append(f"{self.backend.provider_id} has no local engine")
        self._cached_engine_ready = False
        self._cached_engine_ready_at = time.monotonic()
        return "; ".join(messages)

    def clear_last_error(self) -> None:
        self.state_store.set_values(last_error="")


def handle_engine_error(exc: Exception) -> str:
    return f"engine error: {exc}"
