from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Callable, Protocol


DeltaCallback = Callable[[str], None]
RecordingStoppedCallback = Callable[[], None]


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    realtime_microphone: bool = True
    transcribe_file: bool = True
    local_engine_required: bool = False
    warm_supported: bool = False
    model_control_supported: bool = False


class TranscriptionBackend(Protocol):
    provider_id: str
    provider_model: str
    sample_rate: int
    chunk_ms: int
    capabilities: BackendCapabilities

    @property
    def last_capture(self):
        ...

    def check_ready_blocking(self) -> None:
        ...

    def transcribe_microphone_blocking(
        self,
        stop_event: threading.Event,
        on_delta: DeltaCallback | None = None,
        final_timeout_seconds: float | None = None,
        mic=None,
        close_mic: bool = True,
        on_recording_stopped: RecordingStoppedCallback | None = None,
    ) -> str:
        ...

    def transcribe_file_blocking(
        self,
        audio_path: Path,
        on_delta: DeltaCallback | None = None,
    ) -> str:
        ...
