from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from .paths import RECORDINGS_DIR


_WAKE_WORD_RE = re.compile(
    r"^(harvis|jarvis|harbis|hardisk|agente|ordenador)\b",
    re.IGNORECASE,
)
_COMMAND_PREFIX_RE = re.compile(
    r"^(harvis|jarvis|harbis|hardisk|agente|ordenador|manda|envia|envía|"
    r"añade|anade|añades|anades|añada|anada|añadas|anadas|agrega|crea|pon|"
    r"programa|redacta|escribe|dile|llama|revisa|busca|abre|haz|prepara)\b",
    re.IGNORECASE,
)
_ASSISTANT_FRAGMENT_RE = re.compile(
    r"\b(google\s+calendar|whatsapp|guasap|wasap|gmail|correo|email)\b",
    re.IGNORECASE,
)
_CALENDAR_FRAGMENT_RE = re.compile(
    r"\b(?:\d{1,2}|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|"
    r"once|doce|trece|catorce|quince|dieci\w+|veinti\w+|treinta)\s+"
    r"(?:a|al)\s+"
    r"(?:\d{1,2}|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|"
    r"once|doce|trece|catorce|quince|dieci\w+|veinti\w+|treinta)\s+"
    r"de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    r"septiembre|setiembre|octubre|noviembre|diciembre)\b.*"
    r"\b(concepto|congreso|evento|cita)\b",
    re.IGNORECASE,
)
_TERMINAL_PUNCTUATION = (".", "!", "?", "¿", "¡", "…")
_TRAILING_CONNECTORS = {
    "a",
    "al",
    "con",
    "de",
    "del",
    "en",
    "para",
    "por",
    "que",
    "y",
    "o",
}


@dataclass(slots=True)
class RecordingQualityIssue:
    code: str
    severity: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass(slots=True)
class RecordingQualityResult:
    path: Path
    created_at: str = ""
    status: str = ""
    provider_id: str = ""
    provider_model: str = ""
    duration_seconds: float = 0.0
    text_chars: int = 0
    word_count: int = 0
    fallback_used: bool = False
    issues: list[RecordingQualityIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "created_at": self.created_at,
            "status": self.status,
            "provider_id": self.provider_id,
            "provider_model": self.provider_model,
            "duration_seconds": self.duration_seconds,
            "text_chars": self.text_chars,
            "word_count": self.word_count,
            "fallback_used": self.fallback_used,
            "passed": self.passed,
            "issues": [issue.as_dict() for issue in self.issues],
        }


def _words(text: str) -> list[str]:
    return [word for word in text.strip().split() if word.strip()]


def _duration_from_metadata(metadata: dict[str, Any]) -> float:
    duration = metadata.get("audio_duration_seconds")
    if duration in (None, ""):
        diagnostics = metadata.get("diagnostics")
        if isinstance(diagnostics, dict):
            audio_signal = diagnostics.get("audio_signal")
            if isinstance(audio_signal, dict):
                duration = audio_signal.get("duration_seconds")
    try:
        return max(0.0, float(duration or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _diagnostics(metadata: dict[str, Any]) -> dict[str, Any]:
    diagnostics = metadata.get("diagnostics")
    return diagnostics if isinstance(diagnostics, dict) else {}


def _audio_has_signal(metadata: dict[str, Any]) -> bool | None:
    audio_signal = _diagnostics(metadata).get("audio_signal")
    if not isinstance(audio_signal, dict):
        return None
    if "has_signal" in audio_signal:
        return bool(audio_signal.get("has_signal"))
    try:
        peak = float(audio_signal.get("peak") or 0.0)
        rms = float(audio_signal.get("rms") or 0.0)
        nonzero = int(audio_signal.get("nonzero_samples") or 0)
    except (TypeError, ValueError):
        return None
    return peak > 0 and rms > 0 and nonzero > 0


def _events(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    events = _diagnostics(metadata).get("events")
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _segments(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    segments = _diagnostics(metadata).get("segments")
    if not isinstance(segments, list):
        return []
    return [segment for segment in segments if isinstance(segment, dict)]


def looks_like_truncated_transcript(text: str, duration_seconds: float) -> bool:
    stripped = text.strip()
    duration = max(0.0, float(duration_seconds or 0.0))
    if duration < 1.5:
        return False
    if not stripped:
        return True
    words = _words(stripped)
    if (
        duration >= 6.0
        and not _WAKE_WORD_RE.search(stripped)
        and (
            _ASSISTANT_FRAGMENT_RE.search(stripped)
            or _CALENDAR_FRAGMENT_RE.search(stripped)
        )
    ):
        return True
    if duration < 20.0:
        lower_words = [
            word.strip(".,;:!?¿¡()[]{}\"'").lower()
            for word in words
            if word.strip(".,;:!?¿¡()[]{}\"'")
        ]
        last_word = lower_words[-1] if lower_words else ""
        has_terminal_punctuation = stripped.endswith(_TERMINAL_PUNCTUATION)
        command_like = bool(_COMMAND_PREFIX_RE.search(stripped))
        short_command_floor = max(12, int(duration * 1.2))
        if (
            duration >= 8.0
            and command_like
            and not has_terminal_punctuation
            and len(words) < short_command_floor
        ):
            return True
        if duration >= 4.0 and last_word in _TRAILING_CONNECTORS:
            return True
        return False
    has_terminal_punctuation = stripped.endswith(_TERMINAL_PUNCTUATION)
    sparse_word_floor = max(12, int(duration * 0.75))
    sparse_char_floor = int(duration * 4.0)
    if (
        not has_terminal_punctuation
        and len(words) <= sparse_word_floor
        and len(stripped) < sparse_char_floor
    ):
        return True
    min_words = max(12, int(duration / 4.0))
    min_chars = int(duration * 2.5)
    return len(words) < min_words or len(stripped) < min_chars


def looks_like_sparse_final_segment(text: str, duration_seconds: float) -> bool:
    stripped = text.strip()
    duration = max(0.0, float(duration_seconds or 0.0))
    if duration < 8.0:
        return False
    if stripped.endswith(_TERMINAL_PUNCTUATION):
        return False

    words = _words(stripped)
    return len(stripped) < int(duration * 5.0) or len(words) < int(duration * 0.85)


def _fallback_recovered(metadata: dict[str, Any]) -> bool:
    diagnostics = _diagnostics(metadata)
    if bool(diagnostics.get("fallback_used")):
        return True
    for event in _events(metadata):
        if event.get("event") != "openai_audio_api_fallback_completed":
            continue
        return True
    return False


def _fallback_failed_without_recovery(metadata: dict[str, Any]) -> bool:
    failed = False
    recovered_after_failure = False
    for event in _events(metadata):
        event_name = str(event.get("event") or "")
        if event_name == "openai_audio_api_fallback_failed":
            failed = True
            recovered_after_failure = False
        elif failed and event_name == "openai_audio_api_fallback_completed":
            recovered_after_failure = True
    return failed and not recovered_after_failure


def evaluate_recording_metadata(
    metadata: dict[str, Any],
    *,
    path: Path | None = None,
) -> RecordingQualityResult:
    text = str(metadata.get("normalized_text") or metadata.get("raw_text") or "").strip()
    duration = _duration_from_metadata(metadata)
    diagnostics = _diagnostics(metadata)
    status = str(metadata.get("status") or "").strip()
    completion_status = str(diagnostics.get("completion_status") or status).strip()
    segments = _segments(metadata)
    raw_segment_texts = metadata.get("segment_texts")
    segment_texts = (
        raw_segment_texts
        if isinstance(raw_segment_texts, list) and len(raw_segment_texts) == len(segments)
        else []
    )
    issues: list[RecordingQualityIssue] = []
    result = RecordingQualityResult(
        path=path or Path(str(metadata.get("audio_path") or "")),
        created_at=str(metadata.get("created_at") or ""),
        status=status,
        provider_id=str(metadata.get("provider_id") or ""),
        provider_model=str(
            metadata.get("provider_model") or metadata.get("model_id") or ""
        ),
        duration_seconds=round(duration, 3),
        text_chars=len(text),
        word_count=len(_words(text)),
        fallback_used=_fallback_recovered(metadata),
        issues=issues,
    )

    has_signal = _audio_has_signal(metadata)
    if status == "error":
        issues.append(
            RecordingQualityIssue(
                code="status_error",
                severity="error",
                message=str(metadata.get("error") or "recording finished with error"),
            )
        )
    if has_signal is False:
        code = "no_signal_text" if text else "no_signal"
        message = (
            "audio signal is missing even though transcript text exists"
            if text
            else "audio signal is missing"
        )
        issues.append(
            RecordingQualityIssue(code=code, severity="warning", message=message)
        )
        return result

    if duration >= 1.5 and not text and status != "error":
        issues.append(
            RecordingQualityIssue(
                code="missing_transcript",
                severity="error",
                message="audio has signal and duration but transcript text is empty",
            )
        )
    if text and looks_like_truncated_transcript(text, duration) and not result.fallback_used:
        issues.append(
            RecordingQualityIssue(
                code="truncated_transcript",
                severity="error",
                message="transcript is too sparse for the recorded audio duration",
            )
        )
    if completion_status in {"incomplete", "partial"} and duration >= 2.0:
        issues.append(
            RecordingQualityIssue(
                code=f"completion_{completion_status}",
                severity="error",
                message=f"capture completion status is {completion_status}",
            )
        )
    if _fallback_failed_without_recovery(metadata):
        issues.append(
            RecordingQualityIssue(
                code="fallback_failed",
                severity="error",
                message="audio API fallback failed and no later fallback recovery is recorded",
            )
        )
    for index, segment in enumerate(segments):
        if bool(segment.get("recovered")):
            continue
        segment_status = str(segment.get("status") or "")
        segment_duration = float(segment.get("audio_seconds") or 0.0)
        segment_text = str(
            segment.get("text") or (segment_texts[index] if segment_texts else "") or ""
        )
        if bool(segment.get("final_segment")) and looks_like_sparse_final_segment(
            segment_text,
            segment_duration,
        ):
            issues.append(
                RecordingQualityIssue(
                    code="sparse_final_segment",
                    severity="error",
                    message=(
                        f"final segment {segment.get('index', '?')} is too sparse "
                        "for its audio duration and was not recovered"
                    ),
                )
            )
        if segment_status not in {"error", "timeout"}:
            continue
        issues.append(
            RecordingQualityIssue(
                code="unrecovered_segment",
                severity="error",
                message=f"segment {segment.get('index', '?')} ended as {segment_status}",
            )
        )
    return result


def load_recording_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"recording metadata must be an object: {path}")
    return loaded


def iter_recording_result_paths(root: Path | None = None) -> list[Path]:
    base_dir = root or RECORDINGS_DIR
    if not base_dir.exists():
        return []
    return sorted(
        base_dir.rglob("result.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def audit_recording_results(
    root: Path | None = None,
    *,
    limit: int | None = None,
) -> list[RecordingQualityResult]:
    paths = iter_recording_result_paths(root)
    if limit is not None and limit > 0:
        paths = paths[:limit]
    results: list[RecordingQualityResult] = []
    for path in paths:
        try:
            metadata = load_recording_metadata(path)
        except Exception as exc:
            results.append(
                RecordingQualityResult(
                    path=path,
                    status="invalid",
                    issues=[
                        RecordingQualityIssue(
                            code="invalid_result_json",
                            severity="error",
                            message=str(exc),
                        )
                    ],
                )
            )
            continue
        results.append(evaluate_recording_metadata(metadata, path=path))
    return results


def summarize_quality_results(
    results: list[RecordingQualityResult],
) -> dict[str, object]:
    issue_counts = Counter(
        issue.code for result in results for issue in result.issues
    )
    severity_counts = Counter(
        issue.severity for result in results for issue in result.issues
    )
    passed = sum(1 for result in results if result.passed)
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "issue_counts": dict(sorted(issue_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
    }
