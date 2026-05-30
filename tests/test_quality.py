import json
from pathlib import Path

from voxtray.quality import (
    audit_recording_results,
    evaluate_recording_metadata,
    looks_like_sparse_final_segment,
    looks_like_truncated_transcript,
    summarize_quality_results,
)


def _metadata(
    text: str,
    *,
    duration_seconds: float = 22.0,
    status: str = "success",
    has_signal: bool = True,
    fallback_used: bool = False,
    events: list[dict[str, object]] | None = None,
    segments: list[dict[str, object]] | None = None,
    segment_texts: list[str] | None = None,
) -> dict[str, object]:
    return {
        "created_at": "2026-05-22T11:16:51+00:00",
        "source": "microphone",
        "status": status,
        "model_id": "gpt-realtime-whisper",
        "provider_id": "openai_realtime",
        "provider_model": "gpt-realtime-whisper",
        "audio_path": "/tmp/audio.wav",
        "audio_duration_seconds": duration_seconds,
        "raw_text": text,
        "normalized_text": text,
        "segment_texts": segment_texts or [],
        "diagnostics": {
            "completion_status": "complete" if status == "success" else status,
            "fallback_used": fallback_used,
            "audio_signal": {
                "duration_seconds": duration_seconds,
                "nonzero_samples": 1000 if has_signal else 0,
                "peak": 32000 if has_signal else 0,
                "rms": 3000.0 if has_signal else 0.0,
                "has_signal": has_signal,
            },
            "events": events or [],
            "segments": segments or [],
        },
    }


def test_quality_heuristic_flags_latest_sparse_fragment() -> None:
    assert looks_like_truncated_transcript(
        "Es decir, todo lo que sea seis llevarlo a nueve, los siete",
        22.48,
    )


def test_quality_heuristic_allows_dense_long_text_without_period() -> None:
    text = (
        "Vale, te mando nota de voz para esto porque es larguísimo y necesito "
        "explicarte varias pruebas seguidas sin que el sistema lo recorte al "
        "final de la frase"
    )

    assert not looks_like_truncated_transcript(text, 22.0)


def test_quality_heuristic_flags_sparse_final_segment() -> None:
    assert looks_like_sparse_final_segment(
        "todo cerrado. Y que en vez de poner usando elementos cerrados",
        18.4,
    )


def test_evaluate_recording_metadata_flags_truncated_realtime() -> None:
    result = evaluate_recording_metadata(
        _metadata("Es decir, todo lo que sea seis llevarlo a nueve, los siete")
    )

    assert not result.passed
    assert [issue.code for issue in result.issues] == ["truncated_transcript"]


def test_evaluate_recording_metadata_flags_unrecovered_sparse_final_segment() -> None:
    result = evaluate_recording_metadata(
        _metadata(
            (
                "Vale, hay cosas que no me acaban de convencer. El botón de guardar "
                "y verificar datos debería seguir en la tarjeta primera de Payment "
                "Configuration y lo de métodos de pago quiero que hagas los siguientes "
                "cambios. Solo tiene que tener un botón guardar que se active solo "
                "cuando haya cambios. Y quiero que sea un servidor más bonito. Es "
                "decir, no me gusta que tengamos el global y luego tengamos los "
                "distintos elementos podríamos intentar dibujar algún tipo de tarjeta "
                "o dentro de la tarjeta todo cerrado. Y que en vez de poner usando "
                "elementos cerrados"
            ),
            duration_seconds=78.4,
            fallback_used=True,
            events=[
                {
                    "event": "openai_audio_api_fallback_completed",
                    "scope": "segment",
                    "segment_index": 1,
                    "text_chars": 289,
                }
            ],
            segments=[
                {
                    "index": 1,
                    "audio_seconds": 30.0,
                    "final_segment": False,
                    "status": "recovered",
                    "recovered": True,
                },
                {
                    "index": 2,
                    "audio_seconds": 30.0,
                    "final_segment": False,
                    "status": "success",
                },
                {
                    "index": 3,
                    "audio_seconds": 18.4,
                    "final_segment": True,
                    "status": "success",
                },
            ],
            segment_texts=[
                (
                    "Vale, hay cosas que no me acaban de convencer. El botón de "
                    "guardar y verificar datos debería seguir en la tarjeta primera "
                    "de Payment Configuration y lo de métodos de pago quiero que "
                    "hagas los siguientes cambios. Solo tiene que tener un botón "
                    "guardar que se active solo cuando haya cambios."
                ),
                (
                    "Y quiero que sea un servidor más bonito. Es decir, no me gusta "
                    "que tengamos el global y luego tengamos los distintos elementos "
                    "podríamos intentar dibujar algún tipo de tarjeta o dentro de la "
                    "tarjeta"
                ),
                "todo cerrado. Y que en vez de poner usando elementos cerrados",
            ],
        )
    )

    assert result.fallback_used is True
    assert not result.passed
    assert "sparse_final_segment" in {issue.code for issue in result.issues}


def test_evaluate_recording_metadata_accepts_recovered_fallback() -> None:
    result = evaluate_recording_metadata(
        _metadata(
            "Es decir, todo lo que sea seis llevarlo a nueve, los siete",
            fallback_used=True,
            events=[
                {
                    "event": "openai_audio_api_fallback_completed",
                    "scope": "full",
                    "text_chars": 120,
                }
            ],
        )
    )

    assert result.passed
    assert result.fallback_used is True


def test_evaluate_recording_metadata_flags_failed_fallback() -> None:
    result = evaluate_recording_metadata(
        _metadata(
            "texto parcial",
            events=[
                {
                    "event": "openai_audio_api_fallback_failed",
                    "scope": "full",
                    "error": "network unreachable",
                }
            ],
        )
    )

    assert not result.passed
    assert "fallback_failed" in {issue.code for issue in result.issues}


def test_evaluate_recording_metadata_flags_text_from_silent_audio() -> None:
    result = evaluate_recording_metadata(
        _metadata("Añádelo en Google Calendar.", has_signal=False)
    )

    assert not result.passed
    assert [issue.code for issue in result.issues] == ["no_signal_text"]


def test_audit_recording_results_summarizes_local_artifact_tree(tmp_path: Path) -> None:
    good_dir = tmp_path / "2026" / "05" / "22" / "good"
    bad_dir = tmp_path / "2026" / "05" / "22" / "bad"
    good_dir.mkdir(parents=True)
    bad_dir.mkdir(parents=True)
    (good_dir / "result.json").write_text(
        json.dumps(
            _metadata(
                "Quiero que cambies el número que hay antes de estos fotos. "
                "Por ejemplo, ahora mismo empezamos por el 6, pero deberíamos "
                "empezar por el 9, es decir, todo lo que sea 6, llevarlo a 9, "
                "los 7 al 10, etc."
            )
        ),
        encoding="utf-8",
    )
    (bad_dir / "result.json").write_text(
        json.dumps(
            _metadata("Es decir, todo lo que sea seis llevarlo a nueve, los siete")
        ),
        encoding="utf-8",
    )

    results = audit_recording_results(tmp_path)
    summary = summarize_quality_results(results)

    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["issue_counts"] == {"truncated_transcript": 1}
