from __future__ import annotations

import logging
import os
import signal
import threading
from uuid import uuid4

from .audio import MicrophoneStream
from .clipboard import ClipboardError, copy_to_clipboard
from .config import load_config
from .engine import EngineError, EngineManager
from .history import HistoryStore
from .notify import notify
from .postprocess import normalize_transcript
from .recordings import RecordingArtifactStore
from .realtime import RealtimeError, RealtimeTranscriber
from .state import StateStore


def _publish_notice(
    state_store: StateStore,
    title: str,
    body: str,
    *,
    level: str = "info",
    urgency: str = "normal",
) -> None:
    state_store.set_values(
        last_notice_id=uuid4().hex,
        last_notice_title=title,
        last_notice_body=body,
        last_notice_level=level,
    )
    notify(title, body, urgency=urgency)


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
    recording_store = RecordingArtifactStore()
    logger = logging.getLogger("voxtray.worker")

    stop_event = threading.Event()
    mic: MicrophoneStream | None = None

    def _signal_handler(signum, frame) -> None:  # type: ignore[no-untyped-def]
        del signum, frame
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGUSR1, _signal_handler)

    state_store.set_values(recording_pid=os.getpid(), last_error="")

    try:
        mic = MicrophoneStream(
            sample_rate=config.audio.sample_rate,
            chunk_ms=config.audio.chunk_ms,
            device=config.audio.device,
            max_queue_chunks=config.realtime.mic_queue_chunks,
        )
        mic.start()
        _publish_notice(
            state_store,
            "Voxtray",
            "Recording started",
        )

        text = ""
        normalized_text = ""
        max_attempts = 2
        saved_artifact_path = ""
        completed_attempt = 1
        for attempt in range(1, max_attempts + 1):
            try:
                engine.ensure_running()
                text = transcriber.transcribe_microphone_blocking(
                    stop_event=stop_event,
                    mic=mic,
                    close_mic=False,
                )
                completed_attempt = attempt
                break
            except (EngineError, RealtimeError) as exc:
                capture = transcriber.last_capture
                if capture is not None:
                    artifact = recording_store.save(
                        source=capture.source,
                        model_id=config.model_id,
                        sample_rate=capture.sample_rate,
                        chunk_ms=capture.chunk_ms,
                        pcm16_audio=capture.audio_bytes(),
                        raw_text=capture.raw_text,
                        normalized_text="",
                        status="error",
                        error=str(exc),
                        error_payload=getattr(exc, "payload", None) or capture.error_payload,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        requested_segment_max_seconds=capture.requested_segment_max_seconds,
                        effective_segment_max_seconds=capture.effective_segment_max_seconds,
                        segment_texts=capture.segment_texts,
                        input_path=capture.input_path,
                    )
                    saved_artifact_path = str(artifact.directory)
                    logger.info("saved failed recording artifact: %s", artifact.directory)
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
            normalized_text = normalize_transcript(text)
        else:
            normalized_text = text

        capture = transcriber.last_capture
        if capture is not None:
            capture.raw_text = text
            artifact_status = "success" if normalized_text else "empty"
            artifact = recording_store.save(
                source=capture.source,
                model_id=config.model_id,
                sample_rate=capture.sample_rate,
                chunk_ms=capture.chunk_ms,
                pcm16_audio=capture.audio_bytes(),
                raw_text=text,
                normalized_text=normalized_text,
                status=artifact_status,
                attempt=completed_attempt,
                max_attempts=max_attempts,
                requested_segment_max_seconds=capture.requested_segment_max_seconds,
                effective_segment_max_seconds=capture.effective_segment_max_seconds,
                segment_texts=capture.segment_texts,
                error=capture.error_message,
                error_payload=capture.error_payload,
                input_path=capture.input_path,
            )
            saved_artifact_path = str(artifact.directory)
            logger.info("saved recording artifact: %s", artifact.directory)

        if normalized_text:
            state_store.set_values(last_error="")
            history.add_entry(normalized_text)
            try:
                copy_to_clipboard(normalized_text, backend=config.clipboard.backend)
                _publish_notice(
                    state_store,
                    "Voxtray",
                    "Transcription copied to clipboard",
                    level="info",
                )
            except ClipboardError as exc:
                logger.warning("clipboard copy failed: %s", exc)
                _publish_notice(
                    state_store,
                    "Voxtray",
                    "Transcription done (clipboard backend missing)",
                    level="warning",
                    urgency="normal",
                )
        else:
            logger.warning("recording finished without transcript text")
            state_store.set_values(
                last_error="No text detected. Verify mic input and hold recording slightly longer."
            )
            _publish_notice(
                state_store,
                "Voxtray",
                "Recording stopped (no text). Try speaking 1-2 seconds before releasing.",
                level="warning",
                urgency="normal",
            )

    except (EngineError, RealtimeError, OSError, RuntimeError) as exc:
        logger.exception("record worker failed: %s", exc)
        message = str(exc)
        if saved_artifact_path:
            message = f"{message} [artifact: {saved_artifact_path}]"
        state_store.set_values(last_error=message)
        _publish_notice(
            state_store,
            "Voxtray Error",
            str(exc),
            level="error",
            urgency="critical",
        )
        return 1
    finally:
        if mic is not None:
            mic.stop()
        state = state_store.read()
        state_store.set_values(recording_pid=None)
        if not state.get("warm_enabled"):
            try:
                engine.stop_if_running()
            except EngineError:
                logger.exception("failed stopping engine after recording")

    return 0
