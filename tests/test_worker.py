from pathlib import Path
from copy import deepcopy
import signal
from types import SimpleNamespace

from voxtray.realtime import RealtimeError, TranscriptionCapture
from voxtray.worker import run_record_worker


class DummyStateStore:
    def __init__(self) -> None:
        self.updates: list[dict[str, object]] = []
        self.values = {
            "recording_pid": None,
            "recording_stop_requested": False,
            "activity_state": "idle",
            "warm_enabled": True,
            "last_error": "",
            "last_notice_id": "",
            "last_notice_title": "",
            "last_notice_body": "",
            "last_notice_level": "info",
        }

    def set_values(self, **values):
        self.updates.append(dict(values))
        self.values.update(values)
        return self.values

    def read(self):
        return dict(self.values)


class DummyHistoryStore:
    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self.entries: list[str] = []

    def add_entry(self, text: str):
        self.entries.append(text)
        return {"text": text}


class DummyEngineManager:
    def __init__(self, config, state_store) -> None:
        self.config = config
        self.state_store = state_store

    def is_ready(self, timeout_seconds: float = 1.5) -> bool:
        return False

    def ensure_running(self) -> None:
        return None

    def stop_if_running(self, timeout_seconds: float = 5.0) -> None:
        return None


class DummyMicrophoneStream:
    instances = []

    def __init__(
        self,
        sample_rate: int,
        chunk_ms: int,
        device: str | None,
        max_queue_chunks: int,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.device = device
        self.max_queue_chunks = max_queue_chunks
        self.started = False
        self.stopped = False
        self.queued_chunks: list[bytes] = []
        DummyMicrophoneStream.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def drain(self) -> list[bytes]:
        chunks = self.queued_chunks
        self.queued_chunks = []
        return chunks


class DummyRecordingArtifactStore:
    def __init__(self) -> None:
        self.saved: list[dict[str, object]] = []

    def save(self, **kwargs):
        self.saved.append(deepcopy(kwargs))
        directory = Path("/tmp/voxtray-artifact")
        return SimpleNamespace(
            directory=directory,
            audio_path=directory / "audio.wav",
            metadata_path=directory / "result.json",
        )


def _build_config():
    return SimpleNamespace(
        history=SimpleNamespace(max_items=5),
        audio=SimpleNamespace(sample_rate=16000, chunk_ms=40, device="default"),
        realtime=SimpleNamespace(mic_queue_chunks=4096),
        postprocess=SimpleNamespace(clean_text=True),
        clipboard=SimpleNamespace(backend="auto"),
        model_id="mistralai/Voxtral-Mini-4B-Realtime-2602",
    )


def test_run_record_worker_saves_success_artifact(monkeypatch):
    state_store = DummyStateStore()
    history_store = DummyHistoryStore(max_items=5)
    artifact_store = DummyRecordingArtifactStore()
    DummyMicrophoneStream.instances = []
    seen = {}

    capture = TranscriptionCapture(source="microphone", sample_rate=16000, chunk_ms=40)
    capture.append_audio_chunk(b"\x01\x00" * 160)
    capture.segment_texts.append(" hola mundo ")

    class SuccessTranscriber:
        def __init__(self, config) -> None:
            self.config = config
            self.last_capture = capture

        def transcribe_microphone_blocking(
            self,
            stop_event,
            mic=None,
            close_mic=True,
        ):
            del stop_event
            seen["mic"] = mic
            seen["close_mic"] = close_mic
            return " hola mundo "

        def check_realtime_session_blocking(self) -> None:
            return None

    monkeypatch.setattr("voxtray.worker.load_config", _build_config)
    monkeypatch.setattr("voxtray.worker.StateStore", lambda: state_store)
    monkeypatch.setattr("voxtray.worker.HistoryStore", lambda max_items: history_store)
    monkeypatch.setattr("voxtray.worker.EngineManager", DummyEngineManager)
    monkeypatch.setattr("voxtray.worker.RealtimeTranscriber", SuccessTranscriber)
    monkeypatch.setattr("voxtray.worker.MicrophoneStream", DummyMicrophoneStream)
    monkeypatch.setattr("voxtray.worker.RecordingArtifactStore", lambda: artifact_store)
    monkeypatch.setattr("voxtray.worker.notify", lambda *args, **kwargs: None)
    monkeypatch.setattr("voxtray.worker.copy_to_clipboard", lambda text, backend: backend)
    monkeypatch.setattr("voxtray.worker.normalize_transcript", lambda text: text.strip().upper())
    monkeypatch.setattr("voxtray.worker.signal.signal", lambda *args, **kwargs: None)

    result = run_record_worker()

    assert result == 0
    assert history_store.entries == ["HOLA MUNDO"]
    assert artifact_store.saved[0]["status"] == "success"
    assert artifact_store.saved[0]["normalized_text"] == "HOLA MUNDO"
    assert state_store.values["last_error"] == ""
    assert state_store.values["activity_state"] == "idle"
    assert state_store.values["last_notice_title"] == "Voxtray"
    assert state_store.values["last_notice_body"] == "Transcription copied to clipboard"
    assert state_store.values["last_notice_level"] == "info"
    assert seen["mic"] is DummyMicrophoneStream.instances[0]
    assert seen["close_mic"] is False
    assert DummyMicrophoneStream.instances[0].started is True
    assert DummyMicrophoneStream.instances[0].stopped is True


def test_run_record_worker_loads_engine_before_starting_microphone(monkeypatch):
    state_store = DummyStateStore()
    history_store = DummyHistoryStore(max_items=5)
    artifact_store = DummyRecordingArtifactStore()
    DummyMicrophoneStream.instances = []
    events: list[str] = []

    capture = TranscriptionCapture(source="microphone", sample_rate=16000, chunk_ms=40)

    class OrderedEngineManager(DummyEngineManager):
        def ensure_running(self) -> None:
            events.append("engine")

    class OrderedMicrophoneStream(DummyMicrophoneStream):
        def start(self) -> None:
            events.append("mic")
            super().start()

    class SuccessTranscriber:
        def __init__(self, config) -> None:
            self.config = config
            self.last_capture = capture

        def transcribe_microphone_blocking(
            self,
            stop_event,
            mic=None,
            close_mic=True,
        ):
            del stop_event, mic, close_mic
            events.append("transcribe")
            return "hola"

        def check_realtime_session_blocking(self) -> None:
            events.append("canary")

    monkeypatch.setattr("voxtray.worker.load_config", _build_config)
    monkeypatch.setattr("voxtray.worker.StateStore", lambda: state_store)
    monkeypatch.setattr("voxtray.worker.HistoryStore", lambda max_items: history_store)
    monkeypatch.setattr("voxtray.worker.EngineManager", OrderedEngineManager)
    monkeypatch.setattr("voxtray.worker.RealtimeTranscriber", SuccessTranscriber)
    monkeypatch.setattr("voxtray.worker.MicrophoneStream", OrderedMicrophoneStream)
    monkeypatch.setattr("voxtray.worker.RecordingArtifactStore", lambda: artifact_store)
    monkeypatch.setattr("voxtray.worker.notify", lambda *args, **kwargs: None)
    monkeypatch.setattr("voxtray.worker.copy_to_clipboard", lambda text, backend: backend)
    monkeypatch.setattr("voxtray.worker.normalize_transcript", lambda text: text)
    monkeypatch.setattr("voxtray.worker.signal.signal", lambda *args, **kwargs: None)

    result = run_record_worker()

    assert result == 0
    assert events == ["engine", "canary", "mic", "transcribe"]


def test_run_record_worker_skips_loading_notice_when_engine_is_ready(monkeypatch):
    state_store = DummyStateStore()
    history_store = DummyHistoryStore(max_items=5)
    artifact_store = DummyRecordingArtifactStore()
    DummyMicrophoneStream.instances = []
    notices: list[tuple[str, str, str]] = []

    capture = TranscriptionCapture(source="microphone", sample_rate=16000, chunk_ms=40)
    capture.append_audio_chunk(b"\x01\x00" * 160)

    class ReadyEngineManager(DummyEngineManager):
        def is_ready(self, timeout_seconds: float = 1.5) -> bool:
            return True

    class SuccessTranscriber:
        def __init__(self, config) -> None:
            self.config = config
            self.last_capture = capture

        def transcribe_microphone_blocking(
            self,
            stop_event,
            mic=None,
            close_mic=True,
        ):
            del stop_event, mic, close_mic
            return "hola"

        def check_realtime_session_blocking(self) -> None:
            return None

    monkeypatch.setattr("voxtray.worker.load_config", _build_config)
    monkeypatch.setattr("voxtray.worker.StateStore", lambda: state_store)
    monkeypatch.setattr("voxtray.worker.HistoryStore", lambda max_items: history_store)
    monkeypatch.setattr("voxtray.worker.EngineManager", ReadyEngineManager)
    monkeypatch.setattr("voxtray.worker.RealtimeTranscriber", SuccessTranscriber)
    monkeypatch.setattr("voxtray.worker.MicrophoneStream", DummyMicrophoneStream)
    monkeypatch.setattr("voxtray.worker.RecordingArtifactStore", lambda: artifact_store)
    monkeypatch.setattr(
        "voxtray.worker.notify",
        lambda title, body, urgency="normal": notices.append((title, body, urgency)),
    )
    monkeypatch.setattr("voxtray.worker.copy_to_clipboard", lambda text, backend: backend)
    monkeypatch.setattr("voxtray.worker.normalize_transcript", lambda text: text)
    monkeypatch.setattr("voxtray.worker.signal.signal", lambda *args, **kwargs: None)

    result = run_record_worker()

    assert result == 0
    assert "Loading model before recording" not in [body for _, body, _ in notices]
    assert "Recording started" in [body for _, body, _ in notices]
    loading_updates = [
        update for update in state_store.updates if update.get("activity_state") == "loading_model"
    ]
    assert loading_updates == []


def test_run_record_worker_cancel_during_engine_load_skips_canary(monkeypatch):
    state_store = DummyStateStore()
    history_store = DummyHistoryStore(max_items=5)
    artifact_store = DummyRecordingArtifactStore()
    DummyMicrophoneStream.instances = []
    handlers = {}
    events: list[str] = []
    notices: list[tuple[str, str, str]] = []

    class CancelingEngineManager(DummyEngineManager):
        def ensure_running(self) -> None:
            events.append("engine")
            handlers[signal.SIGUSR1](signal.SIGUSR1, None)

    class CanaryShouldNotRun:
        def __init__(self, config) -> None:
            self.config = config
            self.last_capture = None

        def check_realtime_session_blocking(self) -> None:
            raise AssertionError("canary should not run after cancellation")

        def transcribe_microphone_blocking(
            self,
            stop_event,
            mic=None,
            close_mic=True,
        ):
            raise AssertionError("microphone transcription should not start")

    monkeypatch.setattr("voxtray.worker.load_config", _build_config)
    monkeypatch.setattr("voxtray.worker.StateStore", lambda: state_store)
    monkeypatch.setattr("voxtray.worker.HistoryStore", lambda max_items: history_store)
    monkeypatch.setattr("voxtray.worker.EngineManager", CancelingEngineManager)
    monkeypatch.setattr("voxtray.worker.RealtimeTranscriber", CanaryShouldNotRun)
    monkeypatch.setattr("voxtray.worker.MicrophoneStream", DummyMicrophoneStream)
    monkeypatch.setattr("voxtray.worker.RecordingArtifactStore", lambda: artifact_store)
    monkeypatch.setattr(
        "voxtray.worker.notify",
        lambda title, body, urgency="normal": notices.append((title, body, urgency)),
    )
    monkeypatch.setattr("voxtray.worker.copy_to_clipboard", lambda text, backend: backend)
    monkeypatch.setattr("voxtray.worker.normalize_transcript", lambda text: text)
    monkeypatch.setattr(
        "voxtray.worker.signal.signal",
        lambda sig, handler: handlers.setdefault(sig, handler),
    )

    result = run_record_worker()

    assert result == 0
    assert events == ["engine"]
    assert DummyMicrophoneStream.instances == []
    assert artifact_store.saved == []
    assert history_store.entries == []
    assert state_store.values["last_error"] == ""
    assert state_store.values["activity_state"] == "idle"
    assert not any("No text detected" in str(update) for update in state_store.updates)
    assert all("no text" not in body.casefold() for _, body, _ in notices)


def test_run_record_worker_cancel_during_canary_skips_microphone(monkeypatch):
    state_store = DummyStateStore()
    history_store = DummyHistoryStore(max_items=5)
    artifact_store = DummyRecordingArtifactStore()
    DummyMicrophoneStream.instances = []
    handlers = {}
    events: list[str] = []
    notices: list[tuple[str, str, str]] = []

    class CancelingCanaryTranscriber:
        def __init__(self, config) -> None:
            self.config = config
            self.last_capture = None

        def check_realtime_session_blocking(self) -> None:
            events.append("canary")
            handlers[signal.SIGUSR1](signal.SIGUSR1, None)

        def transcribe_microphone_blocking(
            self,
            stop_event,
            mic=None,
            close_mic=True,
        ):
            raise AssertionError("microphone transcription should not start")

    monkeypatch.setattr("voxtray.worker.load_config", _build_config)
    monkeypatch.setattr("voxtray.worker.StateStore", lambda: state_store)
    monkeypatch.setattr("voxtray.worker.HistoryStore", lambda max_items: history_store)
    monkeypatch.setattr("voxtray.worker.EngineManager", DummyEngineManager)
    monkeypatch.setattr("voxtray.worker.RealtimeTranscriber", CancelingCanaryTranscriber)
    monkeypatch.setattr("voxtray.worker.MicrophoneStream", DummyMicrophoneStream)
    monkeypatch.setattr("voxtray.worker.RecordingArtifactStore", lambda: artifact_store)
    monkeypatch.setattr(
        "voxtray.worker.notify",
        lambda title, body, urgency="normal": notices.append((title, body, urgency)),
    )
    monkeypatch.setattr("voxtray.worker.copy_to_clipboard", lambda text, backend: backend)
    monkeypatch.setattr("voxtray.worker.normalize_transcript", lambda text: text)
    monkeypatch.setattr(
        "voxtray.worker.signal.signal",
        lambda sig, handler: handlers.setdefault(sig, handler),
    )

    result = run_record_worker()

    assert result == 0
    assert events == ["canary"]
    assert DummyMicrophoneStream.instances == []
    assert artifact_store.saved == []
    assert history_store.entries == []
    assert state_store.values["last_error"] == ""
    assert state_store.values["activity_state"] == "idle"
    assert not any("No text detected" in str(update) for update in state_store.updates)
    assert all("no text" not in body.casefold() for _, body, _ in notices)


def test_run_record_worker_marks_recording_when_reusing_mic_after_retry(monkeypatch):
    state_store = DummyStateStore()
    history_store = DummyHistoryStore(max_items=5)
    artifact_store = DummyRecordingArtifactStore()
    DummyMicrophoneStream.instances = []
    states_during_transcribe: list[str] = []

    failed_capture = TranscriptionCapture(source="microphone", sample_rate=16000, chunk_ms=40)
    failed_capture.append_audio_chunk(b"\x01\x00" * 160)
    success_capture = TranscriptionCapture(source="microphone", sample_rate=16000, chunk_ms=40)
    success_capture.append_audio_chunk(b"\x01\x00" * 160)
    success_capture.segment_texts.append("hola")

    class RetryTranscriber:
        def __init__(self, config) -> None:
            self.config = config
            self.calls = 0
            self.last_capture = failed_capture

        def check_realtime_session_blocking(self) -> None:
            return None

        def transcribe_microphone_blocking(
            self,
            stop_event,
            mic=None,
            close_mic=True,
        ):
            del stop_event, close_mic
            assert mic is DummyMicrophoneStream.instances[0]
            self.calls += 1
            states_during_transcribe.append(state_store.values["activity_state"])
            if self.calls == 1:
                self.last_capture = failed_capture
                raise RealtimeError("EngineCore encountered an issue.")
            self.last_capture = success_capture
            return "hola"

    monkeypatch.setattr("voxtray.worker.load_config", _build_config)
    monkeypatch.setattr("voxtray.worker.StateStore", lambda: state_store)
    monkeypatch.setattr("voxtray.worker.HistoryStore", lambda max_items: history_store)
    monkeypatch.setattr("voxtray.worker.EngineManager", DummyEngineManager)
    monkeypatch.setattr("voxtray.worker.RealtimeTranscriber", RetryTranscriber)
    monkeypatch.setattr("voxtray.worker.MicrophoneStream", DummyMicrophoneStream)
    monkeypatch.setattr("voxtray.worker.RecordingArtifactStore", lambda: artifact_store)
    monkeypatch.setattr("voxtray.worker.notify", lambda *args, **kwargs: None)
    monkeypatch.setattr("voxtray.worker.copy_to_clipboard", lambda text, backend: backend)
    monkeypatch.setattr("voxtray.worker.normalize_transcript", lambda text: text)
    monkeypatch.setattr("voxtray.worker.signal.signal", lambda *args, **kwargs: None)

    result = run_record_worker()

    assert result == 0
    assert states_during_transcribe == ["recording", "recording"]
    assert len(DummyMicrophoneStream.instances) == 1
    assert history_store.entries == ["hola"]


def test_run_record_worker_saves_error_artifact(monkeypatch):
    state_store = DummyStateStore()
    history_store = DummyHistoryStore(max_items=5)
    artifact_store = DummyRecordingArtifactStore()
    DummyMicrophoneStream.instances = []

    capture = TranscriptionCapture(source="microphone", sample_rate=16000, chunk_ms=40)
    capture.append_audio_chunk(b"\x01\x00" * 160)

    class FailingTranscriber:
        def __init__(self, config) -> None:
            self.config = config
            self.last_capture = capture

        def transcribe_microphone_blocking(
            self,
            stop_event,
            mic=None,
            close_mic=True,
        ):
            del stop_event
            del mic, close_mic
            raise RealtimeError("backend timeout", payload={"type": "error"})

        def check_realtime_session_blocking(self) -> None:
            return None

    monkeypatch.setattr("voxtray.worker.load_config", _build_config)
    monkeypatch.setattr("voxtray.worker.StateStore", lambda: state_store)
    monkeypatch.setattr("voxtray.worker.HistoryStore", lambda max_items: history_store)
    monkeypatch.setattr("voxtray.worker.EngineManager", DummyEngineManager)
    monkeypatch.setattr("voxtray.worker.RealtimeTranscriber", FailingTranscriber)
    monkeypatch.setattr("voxtray.worker.MicrophoneStream", DummyMicrophoneStream)
    monkeypatch.setattr("voxtray.worker.RecordingArtifactStore", lambda: artifact_store)
    monkeypatch.setattr("voxtray.worker.notify", lambda *args, **kwargs: None)
    monkeypatch.setattr("voxtray.worker.copy_to_clipboard", lambda text, backend: backend)
    monkeypatch.setattr("voxtray.worker.normalize_transcript", lambda text: text)
    monkeypatch.setattr("voxtray.worker.signal.signal", lambda *args, **kwargs: None)

    result = run_record_worker()

    assert result == 1
    assert history_store.entries == []
    assert artifact_store.saved[0]["status"] == "error"
    assert artifact_store.saved[0]["error"] == "backend timeout"
    assert "[artifact: /tmp/voxtray-artifact]" in state_store.values["last_error"]
    assert state_store.values["last_notice_title"] == "Voxtray Error"
    assert state_store.values["last_notice_body"] == "backend timeout"
    assert state_store.values["last_notice_level"] == "error"
    assert DummyMicrophoneStream.instances[0].stopped is True


def test_run_record_worker_falls_back_to_saved_audio_after_stop_failure(monkeypatch):
    state_store = DummyStateStore()
    history_store = DummyHistoryStore(max_items=5)
    artifact_store = DummyRecordingArtifactStore()
    DummyMicrophoneStream.instances = []

    queued_tail = b"\x02\x00" * 160
    live_capture = TranscriptionCapture(source="microphone", sample_rate=16000, chunk_ms=40)
    live_capture.append_audio_chunk(b"\x01\x00" * 160)
    fallback_capture = TranscriptionCapture(
        source="file",
        sample_rate=16000,
        chunk_ms=40,
        input_path=Path("/tmp/voxtray-artifact/audio.wav"),
    )
    fallback_capture.append_audio_chunk(b"\x01\x00" * 160)
    fallback_capture.segment_texts.append("fallback completo")
    fallback_capture.segments.append(
        {
            "index": 1,
            "source": "file",
            "status": "success",
            "audio_seconds": 0.02,
            "text_chars": len("fallback completo"),
        }
    )
    fallback_capture.completion_status = "complete"

    class FallbackTranscriber:
        instances = 0

        def __init__(self, config) -> None:
            del config
            type(self).instances += 1
            self.instance = type(self).instances
            self.last_capture = live_capture if self.instance == 1 else fallback_capture

        def check_realtime_session_blocking(self) -> None:
            return None

        def transcribe_microphone_blocking(
            self,
            stop_event,
            mic=None,
            close_mic=True,
        ):
            del close_mic
            mic.queued_chunks.append(queued_tail)
            stop_event.set()
            raise RealtimeError("timed out waiting for transcription.done")

        def transcribe_file_blocking(self, audio_path: Path):
            assert audio_path == Path("/tmp/voxtray-artifact/audio.wav")
            assert DummyMicrophoneStream.instances[0].stopped is True
            assert artifact_store.saved[0]["pcm16_audio"].endswith(queued_tail)
            return "fallback completo"

    monkeypatch.setattr("voxtray.worker.load_config", _build_config)
    monkeypatch.setattr("voxtray.worker.StateStore", lambda: state_store)
    monkeypatch.setattr("voxtray.worker.HistoryStore", lambda max_items: history_store)
    monkeypatch.setattr("voxtray.worker.EngineManager", DummyEngineManager)
    monkeypatch.setattr("voxtray.worker.RealtimeTranscriber", FallbackTranscriber)
    monkeypatch.setattr("voxtray.worker.MicrophoneStream", DummyMicrophoneStream)
    monkeypatch.setattr("voxtray.worker.RecordingArtifactStore", lambda: artifact_store)
    monkeypatch.setattr("voxtray.worker.notify", lambda *args, **kwargs: None)
    monkeypatch.setattr("voxtray.worker.copy_to_clipboard", lambda text, backend: backend)
    monkeypatch.setattr("voxtray.worker.normalize_transcript", lambda text: text)
    monkeypatch.setattr("voxtray.worker.signal.signal", lambda *args, **kwargs: None)

    result = run_record_worker()

    assert result == 0
    assert history_store.entries == ["fallback completo"]
    assert artifact_store.saved[0]["status"] == "error"
    assert artifact_store.saved[0]["pcm16_audio"] == (b"\x01\x00" * 160) + queued_tail
    assert artifact_store.saved[1]["status"] == "success"
    assert artifact_store.saved[1]["source"] == "file"
    assert artifact_store.saved[1]["diagnostics"]["fallback_used"] is True
    assert artifact_store.saved[1]["diagnostics"]["fallback_source"] == (
        "/tmp/voxtray-artifact/audio.wav"
    )


def test_run_record_worker_resaves_artifact_after_fallback_failure(monkeypatch):
    state_store = DummyStateStore()
    history_store = DummyHistoryStore(max_items=5)
    artifact_store = DummyRecordingArtifactStore()
    DummyMicrophoneStream.instances = []

    live_capture = TranscriptionCapture(source="microphone", sample_rate=16000, chunk_ms=40)
    live_capture.append_audio_chunk(b"\x01\x00" * 160)

    class FailingFallbackTranscriber:
        instances = 0

        def __init__(self, config) -> None:
            del config
            type(self).instances += 1
            self.last_capture = live_capture

        def check_realtime_session_blocking(self) -> None:
            return None

        def transcribe_microphone_blocking(
            self,
            stop_event,
            mic=None,
            close_mic=True,
        ):
            del mic, close_mic
            stop_event.set()
            raise RealtimeError("timed out waiting for transcription.done")

        def transcribe_file_blocking(self, audio_path: Path):
            assert audio_path == Path("/tmp/voxtray-artifact/audio.wav")
            raise RealtimeError("fallback exploded", payload={"type": "fallback_error"})

    monkeypatch.setattr("voxtray.worker.load_config", _build_config)
    monkeypatch.setattr("voxtray.worker.StateStore", lambda: state_store)
    monkeypatch.setattr("voxtray.worker.HistoryStore", lambda max_items: history_store)
    monkeypatch.setattr("voxtray.worker.EngineManager", DummyEngineManager)
    monkeypatch.setattr("voxtray.worker.RealtimeTranscriber", FailingFallbackTranscriber)
    monkeypatch.setattr("voxtray.worker.MicrophoneStream", DummyMicrophoneStream)
    monkeypatch.setattr("voxtray.worker.RecordingArtifactStore", lambda: artifact_store)
    monkeypatch.setattr("voxtray.worker.notify", lambda *args, **kwargs: None)
    monkeypatch.setattr("voxtray.worker.copy_to_clipboard", lambda text, backend: backend)
    monkeypatch.setattr("voxtray.worker.normalize_transcript", lambda text: text)
    monkeypatch.setattr("voxtray.worker.signal.signal", lambda *args, **kwargs: None)

    result = run_record_worker()

    assert result == 1
    assert len(artifact_store.saved) == 2
    assert artifact_store.saved[0]["diagnostics"]["events"] == []
    assert artifact_store.saved[1]["status"] == "error"
    assert "offline fallback failed: fallback exploded" in artifact_store.saved[1]["error"]
    assert artifact_store.saved[1]["diagnostics"]["events"] == [
        {
            "event": "offline_fallback_started",
            "source_audio_path": "/tmp/voxtray-artifact/audio.wav",
            "reason": "timed out waiting for transcription.done",
        },
        {
            "event": "offline_fallback_failed",
            "error": "fallback exploded",
        },
    ]
    assert "[artifact: /tmp/voxtray-artifact]" in state_store.values["last_error"]


def test_record_worker_signal_handler_only_sets_memory_flag(monkeypatch):
    class GuardedStateStore(DummyStateStore):
        def __init__(self) -> None:
            super().__init__()
            self.in_signal_handler = False

        def set_values(self, **values):
            if self.in_signal_handler:
                raise AssertionError("signal handler must not write state")
            return super().set_values(**values)

    state_store = GuardedStateStore()
    history_store = DummyHistoryStore(max_items=5)
    artifact_store = DummyRecordingArtifactStore()
    DummyMicrophoneStream.instances = []
    handlers = {}

    capture = TranscriptionCapture(source="microphone", sample_rate=16000, chunk_ms=40)
    capture.append_audio_chunk(b"\x01\x00" * 160)

    class StoppedTranscriber:
        def __init__(self, config) -> None:
            self.config = config
            self.last_capture = capture

        def check_realtime_session_blocking(self) -> None:
            return None

        def transcribe_microphone_blocking(
            self,
            stop_event,
            mic=None,
            close_mic=True,
        ):
            del mic, close_mic
            state_store.in_signal_handler = True
            handlers[signal.SIGUSR1](signal.SIGUSR1, None)
            state_store.in_signal_handler = False
            assert stop_event.is_set()
            return ""

    monkeypatch.setattr("voxtray.worker.load_config", _build_config)
    monkeypatch.setattr("voxtray.worker.StateStore", lambda: state_store)
    monkeypatch.setattr("voxtray.worker.HistoryStore", lambda max_items: history_store)
    monkeypatch.setattr("voxtray.worker.EngineManager", DummyEngineManager)
    monkeypatch.setattr("voxtray.worker.RealtimeTranscriber", StoppedTranscriber)
    monkeypatch.setattr("voxtray.worker.MicrophoneStream", DummyMicrophoneStream)
    monkeypatch.setattr("voxtray.worker.RecordingArtifactStore", lambda: artifact_store)
    monkeypatch.setattr("voxtray.worker.notify", lambda *args, **kwargs: None)
    monkeypatch.setattr("voxtray.worker.copy_to_clipboard", lambda text, backend: backend)
    monkeypatch.setattr("voxtray.worker.normalize_transcript", lambda text: text)
    monkeypatch.setattr(
        "voxtray.worker.signal.signal",
        lambda sig, handler: handlers.setdefault(sig, handler),
    )

    result = run_record_worker()

    assert result == 0
    assert artifact_store.saved[0]["status"] == "empty"
    assert state_store.values["activity_state"] == "idle"
