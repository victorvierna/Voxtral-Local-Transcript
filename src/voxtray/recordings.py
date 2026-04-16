from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4
import wave

from .paths import RECORDINGS_DIR, ensure_recordings_dir


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


@dataclass(slots=True)
class SavedRecordingArtifact:
    directory: Path
    audio_path: Path
    metadata_path: Path


class RecordingArtifactStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or ensure_recordings_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_json(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): RecordingArtifactStore._sanitize_json(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [RecordingArtifactStore._sanitize_json(item) for item in value]
        return repr(value)

    @staticmethod
    def _write_wav(audio_path: Path, pcm16_audio: bytes, sample_rate: int) -> None:
        with wave.open(str(audio_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm16_audio)

    def _new_artifact_dir(self) -> Path:
        stamp = _now_utc()
        day_dir = self.base_dir / stamp.strftime("%Y") / stamp.strftime("%m") / stamp.strftime(
            "%d"
        )
        day_dir.mkdir(parents=True, exist_ok=True)
        slug = f"{stamp.strftime('%H%M%S')}-{uuid4().hex[:8]}"
        artifact_dir = day_dir / slug
        artifact_dir.mkdir(parents=True, exist_ok=False)
        return artifact_dir

    def save(
        self,
        *,
        source: str,
        model_id: str,
        sample_rate: int,
        chunk_ms: int,
        pcm16_audio: bytes,
        raw_text: str,
        normalized_text: str,
        status: str,
        error: str = "",
        error_payload: Any = None,
        attempt: int = 1,
        max_attempts: int = 1,
        requested_segment_max_seconds: float = 0.0,
        effective_segment_max_seconds: float = 0.0,
        segment_texts: list[str] | None = None,
        input_path: Path | None = None,
    ) -> SavedRecordingArtifact:
        artifact_dir = self._new_artifact_dir()
        audio_path = artifact_dir / "audio.wav"
        metadata_path = artifact_dir / "result.json"

        self._write_wav(audio_path, pcm16_audio, sample_rate)

        duration_seconds = 0.0
        if sample_rate > 0:
            duration_seconds = len(pcm16_audio) / float(sample_rate * 2)

        metadata = {
            "created_at": _now_iso(),
            "source": source,
            "status": status,
            "model_id": model_id,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "audio_path": str(audio_path),
            "audio_bytes": len(pcm16_audio),
            "audio_duration_seconds": round(duration_seconds, 3),
            "sample_rate": sample_rate,
            "chunk_ms": chunk_ms,
            "raw_text": raw_text,
            "normalized_text": normalized_text,
            "error": error,
            "error_payload": self._sanitize_json(error_payload),
            "requested_segment_max_seconds": requested_segment_max_seconds,
            "effective_segment_max_seconds": effective_segment_max_seconds,
            "segment_texts": list(segment_texts or []),
            "input_path": str(input_path) if input_path else "",
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return SavedRecordingArtifact(
            directory=artifact_dir,
            audio_path=audio_path,
            metadata_path=metadata_path,
        )
