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
from . import realtime as realtime_module
from .realtime import RealtimeError, RealtimeTranscriber, TranscriptionCapture
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

    def _save_capture_artifact(
        capture: TranscriptionCapture,
        *,
        status: str,
        raw_text: str,
        normalized_text: str,
        error: str = "",
        error_payload: object | None = None,
        attempt: int = 1,
        max_attempts: int = 1,
    ):
        return recording_store.save(
            source=capture.source,
            model_id=config.model_id,
            sample_rate=capture.sample_rate,
            chunk_ms=capture.chunk_ms,
            pcm16_audio=capture.audio_bytes(),
            raw_text=raw_text,
            normalized_text=normalized_text,
            status=status,
            error=error,
            error_payload=error_payload or capture.error_payload,
            attempt=attempt,
            max_attempts=max_attempts,
            requested_segment_max_seconds=capture.requested_segment_max_seconds,
            effective_segment_max_seconds=capture.effective_segment_max_seconds,
            segment_texts=capture.segment_texts,
            diagnostics=capture.diagnostics(),
            input_path=capture.input_path,
        )

    def _stop_mic_and_append_queued_audio(capture: TranscriptionCapture | None) -> None:
        nonlocal mic
        if mic is None:
            return
        mic.stop()
        drain = getattr(mic, "drain", None)
        queued_chunks = drain() if callable(drain) else []
        if capture is not None:
            for chunk in queued_chunks or []:
                capture.append_audio_chunk(chunk)
        mic = None

    def _engine_ready_for_recording() -> bool:
        return engine.is_ready(timeout_seconds=1.5)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGUSR1, _signal_handler)

    engine_ready = _engine_ready_for_recording()
    state_store.set_values(
        recording_pid=os.getpid(),
        recording_stop_requested=False,
        activity_state="recording" if engine_ready else "loading_model",
        last_error="",
    )

    try:
        if not engine_ready:
            _publish_notice(
                state_store,
                "Voxtray",
                "Loading model before recording",
            )

        text = ""
        normalized_text = ""
        max_attempts = 2
        saved_artifact_path = ""
        completed_attempt = 1
        for attempt in range(1, max_attempts + 1):
            try:
                if attempt > 1:
                    engine_ready = _engine_ready_for_recording()
                state_store.set_values(
                    activity_state="recording" if engine_ready else "loading_model"
                )
                engine.ensure_running()
                if stop_event.is_set():
                    return 0
                transcriber.check_realtime_session_blocking()
                if stop_event.is_set():
                    return 0
                if mic is None:
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
                state_store.set_values(activity_state="recording")
                text = transcriber.transcribe_microphone_blocking(
                    stop_event=stop_event,
                    mic=mic,
                    close_mic=False,
                )
                if stop_event.is_set():
                    state_store.set_values(
                        recording_stop_requested=True,
                        activity_state="transcribing",
                    )
                completed_attempt = attempt
                break
            except (EngineError, RealtimeError) as exc:
                failure_exc: Exception = exc
                capture = transcriber.last_capture
                artifact = None
                if isinstance(exc, RealtimeError) and stop_event.is_set():
                    _stop_mic_and_append_queued_audio(capture)
                if capture is not None:
                    artifact = _save_capture_artifact(
                        capture,
                        status="error",
                        raw_text=capture.raw_text,
                        normalized_text="",
                        error=str(exc),
                        error_payload=getattr(exc, "payload", None) or capture.error_payload,
                        attempt=attempt,
                        max_attempts=max_attempts,
                    )
                    saved_artifact_path = str(artifact.directory)
                    logger.info("saved failed recording artifact: %s", artifact.directory)

                if (
                    isinstance(exc, RealtimeError)
                    and stop_event.is_set()
                    and capture is not None
                    and capture.audio_bytes()
                    and artifact is not None
                ):
                    logger.warning(
                        "live transcription failed after stop (%s); retrying from saved WAV %s",
                        exc,
                        artifact.audio_path,
                    )
                    capture.events.append(
                        {
                            "event": "offline_fallback_started",
                            "source_audio_path": str(artifact.audio_path),
                            "reason": str(exc),
                        }
                    )
                    state_store.set_values(activity_state="transcribing")
                    fallback_transcriber = RealtimeTranscriber(config)
                    try:
                        fallback_text = fallback_transcriber.transcribe_file_blocking(
                            artifact.audio_path
                        )
                    except RealtimeError as fallback_exc:
                        capture.events.append(
                            {
                                "event": "offline_fallback_failed",
                                "error": str(fallback_exc),
                            }
                        )
                        failure_exc = RealtimeError(
                            f"{exc}; offline fallback failed: {fallback_exc}",
                            payload=getattr(fallback_exc, "payload", None)
                            or getattr(exc, "payload", None),
                        )
                    else:
                        fallback_capture = fallback_transcriber.last_capture
                        if fallback_capture is not None:
                            fallback_capture.fallback_used = True
                            fallback_capture.fallback_source = str(artifact.audio_path)
                            fallback_capture.events.append(
                                {
                                    "event": "offline_fallback_succeeded",
                                    "source_audio_path": str(artifact.audio_path),
                                    "live_error": str(exc),
                                }
                            )
                        if fallback_text.strip():
                            text = fallback_text
                            transcriber = fallback_transcriber
                            completed_attempt = attempt
                            break
                        capture.events.append(
                            {
                                "event": "offline_fallback_empty",
                                "source_audio_path": str(artifact.audio_path),
                                "live_error": str(exc),
                            }
                        )
                        failure_exc = RealtimeError(
                            f"{exc}; offline fallback produced no transcript text",
                            payload=getattr(exc, "payload", None),
                        )
                    artifact = _save_capture_artifact(
                        capture,
                        status="error",
                        raw_text=capture.raw_text,
                        normalized_text="",
                        error=str(failure_exc),
                        error_payload=getattr(failure_exc, "payload", None)
                        or capture.error_payload,
                        attempt=attempt,
                        max_attempts=max_attempts,
                    )
                    saved_artifact_path = str(artifact.directory)
                    logger.info(
                        "saved failed recording artifact with fallback diagnostics: %s",
                        artifact.directory,
                    )

                can_retry = (
                    attempt < max_attempts
                    and not stop_event.is_set()
                    and _is_recoverable_runtime_failure(failure_exc)
                )
                if not can_retry:
                    raise failure_exc
                logger.warning(
                    "realtime attempt %s/%s failed (%s); restarting engine and retrying",
                    attempt,
                    max_attempts,
                    failure_exc,
                )
                state_store.set_values(last_error=str(failure_exc))
                engine.stop_if_running(timeout_seconds=5.0)

        if config.postprocess.clean_text:
            normalized_text = normalize_transcript(text)
        else:
            normalized_text = text

        capture = transcriber.last_capture
        if capture is not None:
            capture.raw_text = text
            completion_problem = realtime_module.RealtimeTranscriber.completion_problem(
                capture,
                normalized_text,
            )
            if completion_problem:
                capture.completion_status = "incomplete"
                capture.completion_reason = completion_problem
                capture.error_message = completion_problem
            artifact_status = (
                "error"
                if completion_problem
                else ("success" if normalized_text else "empty")
            )
            artifact = _save_capture_artifact(
                capture,
                status=artifact_status,
                raw_text=text,
                normalized_text="" if completion_problem else normalized_text,
                attempt=completed_attempt,
                max_attempts=max_attempts,
                error=capture.error_message,
                error_payload=capture.error_payload,
            )
            saved_artifact_path = str(artifact.directory)
            logger.info("saved recording artifact: %s", artifact.directory)
            if completion_problem:
                raise RealtimeError(completion_problem, payload=capture.error_payload)

        if normalized_text:
            state_store.set_values(activity_state="copying")
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
        state_store.set_values(
            recording_pid=None,
            recording_stop_requested=False,
            activity_state="idle",
        )
        if not state.get("warm_enabled"):
            try:
                engine.stop_if_running()
            except EngineError:
                logger.exception("failed stopping engine after recording")

    return 0
