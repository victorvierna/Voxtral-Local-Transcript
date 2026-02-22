from __future__ import annotations

import logging
import os
import signal
import threading

from .clipboard import ClipboardError, copy_to_clipboard
from .config import load_config
from .engine import EngineError, EngineManager
from .history import HistoryStore
from .notify import notify
from .postprocess import normalize_transcript
from .realtime import RealtimeError, RealtimeTranscriber
from .state import StateStore


def _is_recoverable_runtime_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "enginecore encountered an issue",
            "enginedeaderror",
            "connection closed",
            "internal error",
        )
    )


def run_record_worker() -> int:
    config = load_config()
    state_store = StateStore()
    history = HistoryStore(max_items=config.history.max_items)
    engine = EngineManager(config, state_store)
    transcriber = RealtimeTranscriber(config)
    logger = logging.getLogger("voxtray.worker")

    stop_event = threading.Event()

    def _signal_handler(signum, frame) -> None:  # type: ignore[no-untyped-def]
        del signum, frame
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGUSR1, _signal_handler)

    state_store.set_values(recording_pid=os.getpid(), last_error="")

    notify("Voxtray", "Recording started")

    try:
        text = ""
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                engine.ensure_running()
                text = transcriber.transcribe_microphone_blocking(stop_event=stop_event)
                break
            except (EngineError, RealtimeError) as exc:
                can_retry = (
                    attempt < max_attempts
                    and not stop_event.is_set()
                    and _is_recoverable_runtime_failure(exc)
                )
                if not can_retry:
                    raise
                logger.warning(
                    "realtime attempt %s/%s failed (%s); restarting engine and retrying",
                    attempt,
                    max_attempts,
                    exc,
                )
                state_store.set_values(last_error=str(exc))
                engine.stop_if_running(timeout_seconds=5.0)

        if config.postprocess.clean_text:
            text = normalize_transcript(text)

        if text:
            state_store.set_values(last_error="")
            history.add_entry(text)
            try:
                copy_to_clipboard(text, backend=config.clipboard.backend)
                notify("Voxtray", "Transcription copied to clipboard")
            except ClipboardError as exc:
                logger.warning("clipboard copy failed: %s", exc)
                notify("Voxtray", "Transcription done (clipboard backend missing)")
        else:
            logger.warning("recording finished without transcript text")
            state_store.set_values(
                last_error="No text detected. Verify mic input and hold recording slightly longer."
            )
            notify(
                "Voxtray",
                "Recording stopped (no text). Try speaking 1-2 seconds before releasing.",
            )

    except (EngineError, RealtimeError, OSError, RuntimeError) as exc:
        logger.exception("record worker failed: %s", exc)
        state_store.set_values(last_error=str(exc))
        notify("Voxtray Error", str(exc), urgency="critical")
        return 1
    finally:
        state = state_store.read()
        state_store.set_values(recording_pid=None)
        if not state.get("warm_enabled"):
            try:
                engine.stop_if_running()
            except EngineError:
                logger.exception("failed stopping engine after recording")

    return 0
