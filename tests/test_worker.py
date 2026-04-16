from pathlib import Path
from types import SimpleNamespace

from voxtray.realtime import RealtimeError, TranscriptionCapture
from voxtray.worker import run_record_worker


class DummyStateStore:
    def __init__(self) -> None:
        self.values = {
            "recording_pid": None,
            "warm_enabled": True,
            "last_error": "",
            "last_notice_id": "",
            "last_notice_title": "",
            "last_notice_body": "",
            "last_notice_level": "info",
        }

    def set_values(self, **values):
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
        DummyMicrophoneStream.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class DummyRecordingArtifactStore:
    def __init__(self) -> None:
        self.saved: list[dict[str, object]] = []

    def save(self, **kwargs):
        self.saved.append(kwargs)
        return SimpleNamespace(directory=Path("/tmp/voxtray-artifact"))


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
    assert state_store.values["last_notice_title"] == "Voxtray"
    assert state_store.values["last_notice_body"] == "Transcription copied to clipboard"
    assert state_store.values["last_notice_level"] == "info"
    assert seen["mic"] is DummyMicrophoneStream.instances[0]
    assert seen["close_mic"] is False
    assert DummyMicrophoneStream.instances[0].started is True
    assert DummyMicrophoneStream.instances[0].stopped is True


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
