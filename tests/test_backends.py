import asyncio
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from voxtray.backends import create_transcription_backend
from voxtray import cloud_backends
from voxtray.cloud_backends import MistralRealtimeBackend, OpenAIRealtimeBackend
from voxtray.config import load_config
from voxtray.realtime import RealtimeError, TranscriptionCapture


def _openai_http_status_error(
    status_code: int,
    text: str = "",
) -> httpx.HTTPStatusError:
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/audio/transcriptions",
    )
    response = httpx.Response(status_code, text=text, request=request)
    return httpx.HTTPStatusError(
        text or str(status_code),
        request=request,
        response=response,
    )


def _openai_capture(
    backend: OpenAIRealtimeBackend,
    source: str = "file",
) -> TranscriptionCapture:
    capture = TranscriptionCapture(
        source=source,
        sample_rate=backend.sample_rate,
        chunk_ms=backend.chunk_ms,
        provider_id=backend.provider_id,
        provider_model=backend.provider_model,
    )
    backend._configure_capture_segmentation(capture)
    return capture


async def _collect_stream_bytes(audio_stream):
    data = bytearray()
    async for chunk in audio_stream:
        data.extend(chunk)
    return bytes(data)


def test_factory_defaults_legacy_config_to_local_voxtral(tmp_path: Path):
    cfg_path = tmp_path / "legacy.toml"
    cfg_path.write_text(
        """
model_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"

[audio]
sample_rate = 16000
chunk_ms = 40
""".strip(),
        encoding="utf-8",
    )

    config = load_config(cfg_path)
    backend = create_transcription_backend(config)

    assert config.transcription.provider == "local_voxtral"
    assert backend.provider_id == "local_voxtral"
    assert backend.provider_model == "mistralai/Voxtral-Mini-4B-Realtime-2602"


def test_factory_selects_cloud_provider(tmp_path: Path):
    cfg_path = tmp_path / "openai.toml"
    cfg_path.write_text(
        """
[transcription]
provider = "openai_realtime"

[openai_realtime]
api_key_env = "OPENAI_API_KEY"
model = "gpt-4o-transcribe"
sample_rate = 24000
turn_detection = "manual"
language = ""
prompt = ""
""".strip(),
        encoding="utf-8",
    )

    backend = create_transcription_backend(load_config(cfg_path))

    assert isinstance(backend, OpenAIRealtimeBackend)
    assert backend.provider_model == "gpt-4o-transcribe"
    assert backend.sample_rate == 24000


def test_cloud_provider_missing_api_key_fails_before_microphone(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cloud_backends, "_dotenv_value", lambda env_name: "")
    cfg_path = tmp_path / "openai.toml"
    cfg_path.write_text(
        """
[transcription]
provider = "openai_realtime"
""".strip(),
        encoding="utf-8",
    )
    backend = create_transcription_backend(load_config(cfg_path))

    with pytest.raises(RealtimeError, match="OPENAI_API_KEY"):
        backend.check_ready_blocking()


def test_cloud_api_key_can_resolve_from_local_dotenv(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("export OPENAI_API_KEY='test-key'\n", encoding="utf-8")
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    backend = OpenAIRealtimeBackend(config)

    assert backend.api_key_env_present() is True
    assert backend._api_key("OPENAI_API_KEY") == "test-key"


def test_cloud_microphone_stops_on_user_stop_even_when_worker_owns_cleanup(
    tmp_path: Path,
):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    config.realtime.stop_tail_ms = 0
    backend = OpenAIRealtimeBackend(config)
    capture = TranscriptionCapture(
        source="microphone",
        sample_rate=24000,
        chunk_ms=40,
    )
    stop_event = threading.Event()
    stop_event.set()
    events: list[str] = []

    class FakeMic:
        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

        def get_chunk(self, timeout):
            events.append(f"get:{timeout}")
            return b""

        def drain(self):
            events.append("drain")
            return [b"\x01\x00" * 4]

    async def collect_chunks():
        return [
            chunk
            async for chunk in backend._iter_microphone_chunks(
                stop_event=stop_event,
                capture=capture,
                mic=FakeMic(),
                close_mic=False,
                on_recording_stopped=lambda: events.append("notify"),
            )
        ]

    chunks = asyncio.run(collect_chunks())

    assert chunks == [b"\x01\x00" * 4]
    assert events == ["start", "stop", "notify", "drain"]
    assert capture.audio_bytes() == b"\x01\x00" * 4


def test_mistral_imports_support_v1_sdk_layout(monkeypatch, tmp_path: Path):
    class FakeMistral:
        pass

    class FakeAudioFormat:
        pass

    mistralai_module = ModuleType("mistralai")
    mistralai_module.Mistral = FakeMistral
    models_module = ModuleType("mistralai.models")
    models_module.AudioFormat = FakeAudioFormat

    monkeypatch.setitem(sys.modules, "mistralai", mistralai_module)
    monkeypatch.setitem(sys.modules, "mistralai.models", models_module)
    monkeypatch.delitem(sys.modules, "mistralai.client", raising=False)
    monkeypatch.delitem(sys.modules, "mistralai.client.models", raising=False)

    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "mistral_realtime"
    backend = MistralRealtimeBackend(config)

    imports = backend._imports()

    assert imports["Mistral"] is FakeMistral
    assert imports["AudioFormat"] is FakeAudioFormat


def test_openai_mapper_handles_delta_completed_error_and_empty_completed(tmp_path: Path):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    backend = OpenAIRealtimeBackend(config)
    capture = TranscriptionCapture(
        source="microphone",
        sample_rate=24000,
        chunk_ms=40,
    )
    state = SimpleNamespace(deltas=[], final_text=None, done_seen=False, error=None)
    deltas: list[str] = []

    backend._handle_event(
        {"type": "conversation.item.input_audio_transcription.delta", "delta": "hola "},
        state,
        deltas.append,
        capture,
    )
    assert deltas == ["hola "]
    assert state.deltas == ["hola "]

    backend._handle_event(
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": ""},
        state,
        None,
        capture,
    )
    assert state.done_seen is True
    assert state.final_text == "hola "

    state = SimpleNamespace(deltas=[], final_text=None, done_seen=False, error=None)
    backend._handle_event(
        {"type": "error", "error": {"message": "bad request"}},
        state,
        None,
        capture,
    )
    assert isinstance(state.error, RealtimeError)
    assert str(state.error) == "bad request"


def test_openai_mapper_handles_transcription_failed_event(tmp_path: Path):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    backend = OpenAIRealtimeBackend(config)
    capture = TranscriptionCapture(
        source="microphone",
        sample_rate=24000,
        chunk_ms=40,
    )
    state = SimpleNamespace(
        deltas=["texto parcial"],
        final_text=None,
        done_seen=False,
        error=None,
    )

    payload = {
        "type": "conversation.item.input_audio_transcription.failed",
        "error": {"message": "transcription failed"},
    }
    backend._handle_event(payload, state, None, capture)

    assert isinstance(state.error, RealtimeError)
    assert str(state.error) == "transcription failed"
    assert state.error.partial_text == "texto parcial"
    assert capture.error_message == "transcription failed"
    assert capture.error_payload == payload


def test_openai_session_update_uses_current_transcription_shape(tmp_path: Path):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    config.openai_realtime.model = "gpt-realtime-whisper"
    config.openai_realtime.sample_rate = 24000
    config.openai_realtime.turn_detection = "manual"
    config.openai_realtime.delay = "high"
    backend = OpenAIRealtimeBackend(config)

    payload = backend._session_update_payload()

    assert payload == {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {
                        "model": "gpt-realtime-whisper",
                        "delay": "high",
                    },
                    "turn_detection": None,
                    "noise_reduction": {"type": "near_field"},
                }
            },
        },
    }


def test_openai_session_update_includes_prompt_for_whisper_1(tmp_path: Path):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    config.openai_realtime.model = "whisper-1"
    config.openai_realtime.language = "es"
    config.openai_realtime.prompt = "Vocabulario probable: Harvis"
    backend = OpenAIRealtimeBackend(config)

    payload = backend._session_update_payload()

    transcription = payload["session"]["audio"]["input"]["transcription"]
    assert transcription == {
        "model": "whisper-1",
        "language": "es",
        "prompt": "Vocabulario probable: Harvis",
    }


def test_openai_session_update_omits_prompt_for_gpt_realtime_whisper(tmp_path: Path):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    config.openai_realtime.model = "gpt-realtime-whisper"
    config.openai_realtime.language = "es"
    config.openai_realtime.prompt = "Vocabulario probable: Harvis"
    backend = OpenAIRealtimeBackend(config)

    payload = backend._session_update_payload()

    transcription = payload["session"]["audio"]["input"]["transcription"]
    assert transcription == {
        "model": "gpt-realtime-whisper",
        "delay": "high",
        "language": "es",
    }


def test_openai_audio_api_fallback_recovers_truncated_long_realtime(
    monkeypatch,
    tmp_path: Path,
):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    config.openai_realtime.fallback_model = "gpt-4o-transcribe"
    backend = OpenAIRealtimeBackend(config)
    capture = TranscriptionCapture(
        source="microphone",
        sample_rate=24000,
        chunk_ms=40,
        provider_id="openai_realtime",
        provider_model="gpt-realtime-whisper",
    )
    capture.append_audio_chunk(b"\x01\x00" * 24000 * 60)

    def fake_audio_api(raw_audio, sample_rate, *, timeout_seconds):
        assert raw_audio
        assert sample_rate == 24000
        assert timeout_seconds >= 120
        return "texto completo recuperado desde batch"

    monkeypatch.setattr(backend, "_transcribe_audio_api_blocking", fake_audio_api)

    text = asyncio.run(
        backend._recover_with_audio_api_if_needed(capture, "demasiado corto")
    )

    assert text == "texto completo recuperado desde batch"
    assert capture.fallback_used is True
    assert capture.fallback_source == "openai_audio_transcriptions:gpt-4o-transcribe"
    assert capture.events[-1]["event"] == "openai_audio_api_fallback_completed"


def test_openai_audio_api_fallback_skips_short_complete_realtime(tmp_path: Path):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    backend = OpenAIRealtimeBackend(config)
    capture = TranscriptionCapture(
        source="microphone",
        sample_rate=24000,
        chunk_ms=40,
        provider_id="openai_realtime",
        provider_model="gpt-realtime-whisper",
    )
    capture.append_audio_chunk(b"\x01\x00" * 24000 * 4)

    text = asyncio.run(
        backend._recover_with_audio_api_if_needed(
            capture,
            "Hola, hola, hola, esto es una prueba.",
        )
    )

    assert text == "Hola, hola, hola, esto es una prueba."
    assert capture.fallback_used is False
    assert capture.events == []


def test_openai_audio_api_fallback_recovers_suspicious_short_command(
    monkeypatch,
    tmp_path: Path,
):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    config.openai_realtime.fallback_model = "gpt-4o-transcribe"
    backend = OpenAIRealtimeBackend(config)
    capture = TranscriptionCapture(
        source="microphone",
        sample_rate=24000,
        chunk_ms=40,
        provider_id="openai_realtime",
        provider_model="gpt-realtime-whisper",
    )
    capture.append_audio_chunk(b"\x01\x00" * 24000 * 11)

    def fake_audio_api(raw_audio, sample_rate, *, timeout_seconds):
        assert raw_audio
        assert sample_rate == 24000
        assert timeout_seconds >= 71
        return "manda un WhatsApp al teléfono treinta y nueve cincuenta y cuatro diecinueve con el texto hola"

    monkeypatch.setattr(backend, "_transcribe_audio_api_blocking", fake_audio_api)

    text = asyncio.run(
        backend._recover_with_audio_api_if_needed(
            capture,
            "manda al teléfono treinta y nueve cincuenta y cuatro diecinueve",
        )
    )

    assert text == (
        "manda un WhatsApp al teléfono treinta y nueve cincuenta y cuatro "
        "diecinueve con el texto hola"
    )
    assert capture.fallback_used is True
    assert capture.events[0]["event"] == "openai_audio_api_fallback_started"
    assert capture.events[-1]["event"] == "openai_audio_api_fallback_completed"


def test_openai_audio_api_fallback_recovers_sparse_long_fragment(
    monkeypatch,
    tmp_path: Path,
):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    config.openai_realtime.fallback_model = "gpt-4o-transcribe"
    backend = OpenAIRealtimeBackend(config)
    capture = TranscriptionCapture(
        source="microphone",
        sample_rate=24000,
        chunk_ms=40,
        provider_id="openai_realtime",
        provider_model="gpt-realtime-whisper",
    )
    capture.append_audio_chunk(b"\x01\x00" * 24000 * 22)

    def fake_audio_api(raw_audio, sample_rate, *, timeout_seconds):
        assert raw_audio
        assert sample_rate == 24000
        assert timeout_seconds >= 82
        return (
            "Quiero que cambies el número que hay antes de estos fotos. "
            "Por ejemplo, ahora mismo empezamos por el 6, pero deberíamos "
            "empezar por el 9, es decir, todo lo que sea 6, llevarlo a 9, "
            "los 7 al 10, etc."
        )

    monkeypatch.setattr(backend, "_transcribe_audio_api_blocking", fake_audio_api)

    text = asyncio.run(
        backend._recover_with_audio_api_if_needed(
            capture,
            "Es decir, todo lo que sea seis llevarlo a nueve, los siete",
        )
    )

    assert text.startswith("Quiero que cambies el número")
    assert "los 7 al 10" in text
    assert capture.fallback_used is True
    assert capture.events[0]["reason"] == "realtime_empty_or_truncated"


def test_openai_audio_api_fallback_skips_dense_long_realtime_without_period(
    tmp_path: Path,
):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    backend = OpenAIRealtimeBackend(config)
    capture = TranscriptionCapture(
        source="microphone",
        sample_rate=24000,
        chunk_ms=40,
        provider_id="openai_realtime",
        provider_model="gpt-realtime-whisper",
    )
    capture.append_audio_chunk(b"\x01\x00" * 24000 * 22)
    text = (
        "Vale, te mando nota de voz para esto porque es larguísimo y necesito "
        "explicarte varias pruebas seguidas sin que el sistema lo recorte al "
        "final de la frase"
    )

    recovered = asyncio.run(backend._recover_with_audio_api_if_needed(capture, text))

    assert recovered == text
    assert capture.fallback_used is False
    assert capture.events == []


def test_openai_audio_api_fallback_recovers_missing_wake_calendar_fragment(
    monkeypatch,
    tmp_path: Path,
):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    backend = OpenAIRealtimeBackend(config)
    capture = TranscriptionCapture(
        source="microphone",
        sample_rate=24000,
        chunk_ms=40,
        provider_id="openai_realtime",
        provider_model="gpt-realtime-whisper",
    )
    capture.append_audio_chunk(b"\x01\x00" * 24000 * 17)

    def fake_audio_api(raw_audio, sample_rate, *, timeout_seconds):
        assert raw_audio
        assert sample_rate == 24000
        return "Harbis, añade al Google Calendar un evento para los días 5 a 10 de mayo de 2027 con el concepto Congreso Proyecto."

    monkeypatch.setattr(backend, "_transcribe_audio_api_blocking", fake_audio_api)

    text = asyncio.run(
        backend._recover_with_audio_api_if_needed(
            capture,
            "cinco a diez de mayo de dos mil veintisiete, con el concepto Congreso Basatas",
        )
    )

    assert text.startswith("Harbis, añade al Google Calendar")
    assert capture.fallback_used is True


def test_openai_audio_api_fallback_retries_network_errors(
    monkeypatch,
    tmp_path: Path,
):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    backend = OpenAIRealtimeBackend(config)
    capture = _openai_capture(backend)
    attempts: list[float] = []
    sleep_calls: list[float] = []

    def fake_audio_api(raw_audio, sample_rate, *, timeout_seconds):
        assert raw_audio == b"\x01\x00"
        assert sample_rate == 24000
        attempts.append(timeout_seconds)
        if len(attempts) < 3:
            raise httpx.ConnectError("network down")
        return "recuperado"

    monkeypatch.setattr(backend, "_transcribe_audio_api_blocking", fake_audio_api)
    monkeypatch.setattr(backend, "_sleep_audio_api_retry", sleep_calls.append)

    text = backend._transcribe_audio_api_fallback_blocking(
        b"\x01\x00",
        24000,
        timeout_seconds=12.0,
        capture=capture,
    )

    assert text == "recuperado"
    assert len(attempts) == 3
    assert sleep_calls == [0.0, 2.0, 6.0]
    assert [
        event["event"] for event in capture.events
    ] == [
        "openai_audio_api_fallback_attempt_failed",
        "openai_audio_api_fallback_attempt_failed",
    ]
    assert [event["attempt"] for event in capture.events] == [1, 2]
    assert all(event["retryable"] is True for event in capture.events)
    assert [event["next_backoff_seconds"] for event in capture.events] == [2.0, 6.0]


def test_openai_audio_api_fallback_does_not_retry_nonrecoverable_status(
    monkeypatch,
    tmp_path: Path,
):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    backend = OpenAIRealtimeBackend(config)
    capture = _openai_capture(backend)
    capture.append_audio_chunk(b"\x01\x00" * 24000 * 60)
    attempts = 0
    sleep_calls: list[float] = []

    def fake_audio_api(raw_audio, sample_rate, *, timeout_seconds):
        nonlocal attempts
        del raw_audio, sample_rate, timeout_seconds
        attempts += 1
        raise _openai_http_status_error(401, "unauthorized")

    monkeypatch.setattr(backend, "_transcribe_audio_api_blocking", fake_audio_api)
    monkeypatch.setattr(backend, "_sleep_audio_api_retry", sleep_calls.append)

    text = asyncio.run(
        backend._recover_with_audio_api_if_needed(capture, "demasiado corto")
    )

    assert text == "demasiado corto"
    assert attempts == 1
    assert sleep_calls == [0.0]
    assert capture.events[-2]["event"] == "openai_audio_api_fallback_attempt_failed"
    assert capture.events[-2]["retryable"] is False
    assert capture.events[-2]["status_code"] == 401
    assert capture.events[-1]["event"] == "openai_audio_api_fallback_failed"


@pytest.mark.parametrize("status_code", [429, 500])
def test_openai_audio_api_fallback_retries_retryable_status(
    monkeypatch,
    tmp_path: Path,
    status_code: int,
):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    backend = OpenAIRealtimeBackend(config)
    capture = _openai_capture(backend)
    attempts = 0
    sleep_calls: list[float] = []

    def fake_audio_api(raw_audio, sample_rate, *, timeout_seconds):
        nonlocal attempts
        del raw_audio, sample_rate, timeout_seconds
        attempts += 1
        if attempts == 1:
            raise _openai_http_status_error(status_code, "retry later")
        return "recuperado"

    monkeypatch.setattr(backend, "_transcribe_audio_api_blocking", fake_audio_api)
    monkeypatch.setattr(backend, "_sleep_audio_api_retry", sleep_calls.append)

    text = backend._transcribe_audio_api_fallback_blocking(
        b"\x01\x00",
        24000,
        timeout_seconds=12.0,
        capture=capture,
    )

    assert text == "recuperado"
    assert attempts == 2
    assert sleep_calls == [0.0, 2.0]
    assert capture.events[0]["event"] == "openai_audio_api_fallback_attempt_failed"
    assert capture.events[0]["retryable"] is True
    assert capture.events[0]["status_code"] == status_code


def test_openai_segmented_realtime_splits_long_pcm_and_records_segments(
    monkeypatch,
    tmp_path: Path,
):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    config.realtime.segment_max_seconds = 120
    backend = OpenAIRealtimeBackend(config)
    capture = _openai_capture(backend)
    raw_audio = b"\x01\x00" * backend.sample_rate * 86
    sent_segments: list[bytes] = []

    async def fake_send(audio_stream, *, capture, on_delta, timeout_seconds):
        del capture, on_delta, timeout_seconds
        sent_segments.append(await _collect_stream_bytes(audio_stream))
        return f"segmento {len(sent_segments)} " + ("palabra " * 20)

    monkeypatch.setattr(backend, "_send_stream_to_openai", fake_send)

    text = asyncio.run(
        backend._transcribe_pcm16_audio(
            raw_audio,
            capture=capture,
            on_delta=None,
            timeout_seconds=12.0,
            source="file",
        )
    )

    assert capture.requested_segment_max_seconds == 120
    assert capture.effective_segment_max_seconds == 30.0
    assert [
        round(len(segment) / float(backend.sample_rate * 2), 3)
        for segment in sent_segments
    ] == [30.0, 30.0, 26.0]
    assert text.splitlines()[0].startswith("segmento 1")
    assert text.splitlines()[1].startswith("segmento 2")
    assert text.splitlines()[2].startswith("segmento 3")
    assert [segment["status"] for segment in capture.segments] == [
        "success",
        "success",
        "success",
    ]
    assert [segment["audio_seconds"] for segment in capture.segments] == [
        30.0,
        30.0,
        26.0,
    ]
    assert capture.segment_texts == text.splitlines()


def test_openai_segment_max_zero_keeps_unsegmented_flow(monkeypatch, tmp_path: Path):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    config.realtime.segment_max_seconds = 0
    backend = OpenAIRealtimeBackend(config)
    capture = _openai_capture(backend)
    raw_audio = b"\x01\x00" * backend.sample_rate * 86
    sent_streams: list[bytes] = []

    async def fake_send_with_recovery(
        audio_stream,
        *,
        capture,
        on_delta,
        timeout_seconds,
    ):
        del capture, on_delta, timeout_seconds
        sent_streams.append(await _collect_stream_bytes(audio_stream))
        return "sin segmentar " + ("palabra " * 20)

    monkeypatch.setattr(backend, "_send_stream_with_recovery", fake_send_with_recovery)

    text = asyncio.run(
        backend._transcribe_pcm16_audio(
            raw_audio,
            capture=capture,
            on_delta=None,
            timeout_seconds=12.0,
            source="file",
        )
    )

    assert text.startswith("sin segmentar")
    assert capture.requested_segment_max_seconds == 0
    assert capture.effective_segment_max_seconds == 0.0
    assert capture.segments == []
    assert sent_streams == [raw_audio]


def test_openai_segmented_file_uses_per_segment_timeout(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    config.realtime.segment_max_seconds = 120
    backend = OpenAIRealtimeBackend(config)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"placeholder")
    raw_audio = b"\x01\x00" * backend.sample_rate * 300
    timeouts: list[float] = []

    async def fake_send(audio_stream, *, capture, on_delta, timeout_seconds):
        del capture, on_delta
        await _collect_stream_bytes(audio_stream)
        timeouts.append(timeout_seconds)
        return f"segmento {len(timeouts)} " + ("palabra " * 20) + "."

    monkeypatch.setattr(
        cloud_backends,
        "_audio_file_to_pcm16_bytes",
        lambda path, sample_rate: raw_audio,
    )
    monkeypatch.setattr(backend, "_send_stream_to_openai", fake_send)

    text = asyncio.run(backend.transcribe_file(audio_path))

    assert len(text.splitlines()) == 10
    assert timeouts == [50.0] * 10
    assert [segment["wait_seconds"] for segment in backend.last_capture.segments] == [
        50.0
    ] * 10


def test_openai_truncated_segment_uses_audio_api_only_for_that_segment(
    monkeypatch,
    tmp_path: Path,
):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    config.realtime.segment_max_seconds = 30
    backend = OpenAIRealtimeBackend(config)
    capture = _openai_capture(backend)
    raw_audio = b"\x01\x00" * backend.sample_rate * 60
    realtime_texts = [
        "primer segmento " + ("palabra " * 20),
        "corto",
    ]
    fallback_calls: list[bytes] = []

    async def fake_send(audio_stream, *, capture, on_delta, timeout_seconds):
        del capture, on_delta, timeout_seconds
        await _collect_stream_bytes(audio_stream)
        return realtime_texts.pop(0)

    def fake_audio_api(raw_audio, sample_rate, *, timeout_seconds):
        del sample_rate, timeout_seconds
        fallback_calls.append(raw_audio)
        return "segundo segmento recuperado " + ("palabra " * 20)

    monkeypatch.setattr(backend, "_send_stream_to_openai", fake_send)
    monkeypatch.setattr(backend, "_transcribe_audio_api_blocking", fake_audio_api)
    monkeypatch.setattr(backend, "_sleep_audio_api_retry", lambda seconds: None)

    text = asyncio.run(
        backend._transcribe_pcm16_audio(
            raw_audio,
            capture=capture,
            on_delta=None,
            timeout_seconds=12.0,
            source="file",
        )
    )

    assert len(fallback_calls) == 1
    assert len(fallback_calls[0]) == backend.sample_rate * 30 * 2
    assert text.splitlines()[0].startswith("primer segmento")
    assert text.splitlines()[1].startswith("segundo segmento recuperado")
    assert capture.fallback_used is True
    assert capture.segments[0]["fallback_used"] is False
    assert capture.segments[1]["fallback_used"] is True
    assert capture.segments[1]["status"] == "recovered"


def test_openai_sparse_final_segment_uses_audio_api_for_final_segment(
    monkeypatch,
    tmp_path: Path,
):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    config.realtime.segment_max_seconds = 30
    backend = OpenAIRealtimeBackend(config)
    capture = _openai_capture(backend)
    raw_audio = b"\x01\x00" * backend.sample_rate * 78
    realtime_texts = [
        "primer segmento " + ("palabra " * 28),
        "segundo segmento " + ("palabra " * 28),
        "todo cerrado. Y que en vez de poner usando elementos cerrados",
    ]
    fallback_calls: list[bytes] = []

    async def fake_send(audio_stream, *, capture, on_delta, timeout_seconds):
        del capture, on_delta, timeout_seconds
        await _collect_stream_bytes(audio_stream)
        return realtime_texts.pop(0)

    def fake_audio_api(raw_audio, sample_rate, *, timeout_seconds):
        del sample_rate, timeout_seconds
        fallback_calls.append(raw_audio)
        return "segmento final recuperado " + ("palabra " * 20) + "."

    monkeypatch.setattr(backend, "_send_stream_to_openai", fake_send)
    monkeypatch.setattr(backend, "_transcribe_audio_api_blocking", fake_audio_api)
    monkeypatch.setattr(backend, "_sleep_audio_api_retry", lambda seconds: None)

    text = asyncio.run(
        backend._transcribe_pcm16_audio(
            raw_audio,
            capture=capture,
            on_delta=None,
            timeout_seconds=12.0,
            source="file",
        )
    )

    assert len(fallback_calls) == 1
    assert len(fallback_calls[0]) == backend.sample_rate * 18 * 2
    assert text.splitlines()[2].startswith("segmento final recuperado")
    assert capture.segments[2]["status"] == "recovered"
    assert capture.events[-2]["reason"] == "sparse_final_segment"


def test_openai_sparse_final_segment_raises_when_fallback_does_not_recover(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    config.realtime.segment_max_seconds = 30
    backend = OpenAIRealtimeBackend(config)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"placeholder")
    raw_audio = b"\x01\x00" * backend.sample_rate * 78
    realtime_texts = [
        "primer segmento " + ("palabra " * 28),
        "segundo segmento " + ("palabra " * 28),
        "todo cerrado. Y que en vez de poner usando elementos cerrados",
    ]

    async def fake_send(audio_stream, *, capture, on_delta, timeout_seconds):
        del capture, on_delta, timeout_seconds
        await _collect_stream_bytes(audio_stream)
        return realtime_texts.pop(0)

    def empty_audio_api(raw_audio, sample_rate, *, timeout_seconds):
        del raw_audio, sample_rate, timeout_seconds
        return ""

    monkeypatch.setattr(
        cloud_backends,
        "_audio_file_to_pcm16_bytes",
        lambda path, sample_rate: raw_audio,
    )
    monkeypatch.setattr(backend, "_send_stream_to_openai", fake_send)
    monkeypatch.setattr(backend, "_transcribe_audio_api_blocking", empty_audio_api)
    monkeypatch.setattr(backend, "_sleep_audio_api_retry", lambda seconds: None)

    with pytest.raises(RealtimeError, match="failed segment\\(s\\) 3") as exc:
        asyncio.run(backend.transcribe_file(audio_path))

    capture = backend.last_capture
    assert capture is not None
    assert capture.completion_status == "incomplete"
    assert capture.segments[2]["status"] == "incomplete"
    assert capture.segments[2]["error"].startswith("audio API fallback returned empty")
    assert exc.value.partial_text.endswith(
        "todo cerrado. Y que en vez de poner usando elementos cerrados"
    )


def test_openai_stream_stops_uploading_after_receive_error(
    monkeypatch,
    tmp_path: Path,
):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    backend = OpenAIRealtimeBackend(config)
    capture = TranscriptionCapture(
        source="file",
        sample_rate=24000,
        chunk_ms=40,
        provider_id="openai_realtime",
        provider_model="gpt-realtime-whisper",
    )
    sent_chunks: list[bytes] = []
    sent_messages: list[dict[str, object]] = []

    class FakeWebSocket:
        async def send(self, payload):
            sent_messages.append(json.loads(payload))

    class FakeConnection:
        async def __aenter__(self):
            return FakeWebSocket()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    async def fake_connect():
        return FakeConnection()

    async def fake_receive(ws, state, on_delta, capture):
        del ws, on_delta
        await asyncio.sleep(0)
        state.error = RealtimeError("bad session")
        capture.error_message = str(state.error)

    async def fake_send_audio_chunk(ws, chunk):
        del ws
        sent_chunks.append(chunk)
        await asyncio.sleep(0)

    async def many_chunks():
        for _index in range(5):
            yield b"\x01\x00" * 120
            await asyncio.sleep(0)

    monkeypatch.setattr(backend, "_connect", fake_connect)
    monkeypatch.setattr(backend, "_receive_events", fake_receive)
    monkeypatch.setattr(
        OpenAIRealtimeBackend,
        "_send_audio_chunk",
        staticmethod(fake_send_audio_chunk),
    )

    with pytest.raises(RealtimeError, match="bad session"):
        asyncio.run(
            backend._send_stream_to_openai(
                many_chunks(),
                capture=capture,
                on_delta=None,
                timeout_seconds=12.0,
            )
        )

    assert len(sent_chunks) == 1
    assert [message["type"] for message in sent_messages] == ["session.update"]


def test_openai_segmented_file_raises_when_segment_remains_failed(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    config.realtime.segment_max_seconds = 30
    backend = OpenAIRealtimeBackend(config)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"placeholder")
    raw_audio = b"\x01\x00" * backend.sample_rate * 60
    send_calls = 0

    async def fake_send(audio_stream, *, capture, on_delta, timeout_seconds):
        nonlocal send_calls
        del capture, on_delta, timeout_seconds
        await _collect_stream_bytes(audio_stream)
        send_calls += 1
        if send_calls == 1:
            return "segmento uno " + ("palabra " * 25)
        raise RealtimeError(
            "timed out",
            partial_text="segmento dos parcial " + ("palabra " * 25),
        )

    def empty_audio_api(raw_audio, sample_rate, *, timeout_seconds):
        del raw_audio, sample_rate, timeout_seconds
        return ""

    monkeypatch.setattr(
        cloud_backends,
        "_audio_file_to_pcm16_bytes",
        lambda path, sample_rate: raw_audio,
    )
    monkeypatch.setattr(backend, "_send_stream_to_openai", fake_send)
    monkeypatch.setattr(backend, "_transcribe_audio_api_blocking", empty_audio_api)
    monkeypatch.setattr(backend, "_sleep_audio_api_retry", lambda seconds: None)

    with pytest.raises(RealtimeError, match="failed segment\\(s\\) 2") as exc:
        asyncio.run(backend.transcribe_file(audio_path))

    capture = backend.last_capture
    assert capture is not None
    assert capture.completion_status == "incomplete"
    assert capture.completion_reason == str(exc.value)
    assert capture.segments[1]["status"] == "error"
    assert capture.segments[1]["fallback_used"] is True
    assert capture.raw_text.startswith("segmento uno")
    assert exc.value.partial_text == capture.raw_text


def test_openai_segmented_microphone_raises_when_segment_remains_failed(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    config.realtime.segment_max_seconds = 30
    config.realtime.stop_tail_ms = 0
    backend = OpenAIRealtimeBackend(config)
    raw_audio = b"\x01\x00" * backend.sample_rate * 60
    send_calls = 0

    class FakeMic:
        def start(self):
            pass

        def stop(self):
            pass

        def get_chunk(self, timeout):
            del timeout
            return b""

        def drain(self):
            return [raw_audio]

    async def fake_send(audio_stream, *, capture, on_delta, timeout_seconds):
        nonlocal send_calls
        del capture, on_delta, timeout_seconds
        await _collect_stream_bytes(audio_stream)
        send_calls += 1
        if send_calls == 1:
            return "segmento uno " + ("palabra " * 25)
        raise RealtimeError(
            "timed out",
            partial_text="segmento dos parcial " + ("palabra " * 25),
        )

    def empty_audio_api(raw_audio, sample_rate, *, timeout_seconds):
        del raw_audio, sample_rate, timeout_seconds
        return ""

    monkeypatch.setattr(backend, "_send_stream_to_openai", fake_send)
    monkeypatch.setattr(backend, "_transcribe_audio_api_blocking", empty_audio_api)
    monkeypatch.setattr(backend, "_sleep_audio_api_retry", lambda seconds: None)

    stop_event = threading.Event()
    stop_event.set()

    with pytest.raises(RealtimeError, match="failed segment\\(s\\) 2") as exc:
        asyncio.run(
            backend.transcribe_microphone(
                stop_event,
                mic=FakeMic(),
                close_mic=False,
            )
        )

    capture = backend.last_capture
    assert capture is not None
    assert capture.completion_status == "incomplete"
    assert capture.segments[1]["status"] == "error"
    assert capture.raw_text.startswith("segmento uno")
    assert exc.value.partial_text == capture.raw_text


def test_openai_realtime_error_can_recover_from_audio_api(monkeypatch, tmp_path: Path):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    backend = OpenAIRealtimeBackend(config)
    capture = TranscriptionCapture(
        source="microphone",
        sample_rate=24000,
        chunk_ms=40,
        provider_id="openai_realtime",
        provider_model="gpt-realtime-whisper",
    )
    capture.append_audio_chunk(b"\x01\x00" * 24000 * 45)

    async def fake_send(*args, **kwargs):
        raise RealtimeError("timed out", partial_text="texto parcial")

    def fake_audio_api(*args, **kwargs):
        return "texto completo recuperado"

    monkeypatch.setattr(backend, "_send_stream_to_openai", fake_send)
    monkeypatch.setattr(backend, "_transcribe_audio_api_blocking", fake_audio_api)

    async def empty_stream():
        if False:
            yield b""

    text = asyncio.run(
        backend._send_stream_with_recovery(
            empty_stream(),
            capture=capture,
            on_delta=None,
            timeout_seconds=12.0,
        )
    )

    assert text == "texto completo recuperado"
    assert capture.fallback_used is True
    assert capture.events[-1]["event"] == "openai_realtime_error_recovered"


def test_openai_stream_send_exception_is_reported_as_realtime_error(
    monkeypatch,
    tmp_path: Path,
):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "openai_realtime"
    backend = OpenAIRealtimeBackend(config)
    capture = TranscriptionCapture(
        source="microphone",
        sample_rate=24000,
        chunk_ms=40,
        provider_id="openai_realtime",
        provider_model="gpt-realtime-whisper",
    )

    class FakeWebSocket:
        async def send(self, payload):
            del payload

    class FakeConnection:
        async def __aenter__(self):
            return FakeWebSocket()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    async def fake_connect():
        return FakeConnection()

    async def fake_receive(*args, **kwargs):
        del args, kwargs
        await asyncio.sleep(60)

    async def fake_send_audio_chunk(ws, chunk):
        del ws, chunk
        raise RuntimeError("websocket closed without close frame")

    async def one_chunk():
        yield b"\x01\x00" * 120

    monkeypatch.setattr(backend, "_connect", fake_connect)
    monkeypatch.setattr(backend, "_receive_events", fake_receive)
    monkeypatch.setattr(
        OpenAIRealtimeBackend,
        "_send_audio_chunk",
        staticmethod(fake_send_audio_chunk),
    )

    with pytest.raises(RealtimeError, match="openai realtime stream failed"):
        asyncio.run(
            backend._send_stream_to_openai(
                one_chunk(),
                capture=capture,
                on_delta=None,
                timeout_seconds=12.0,
            )
        )

    assert capture.error_message == (
        "openai realtime stream failed: websocket closed without close frame"
    )


def test_mistral_mapper_handles_session_delta_done_and_error(tmp_path: Path):
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "mistral_realtime"
    backend = MistralRealtimeBackend(config)
    capture = TranscriptionCapture(
        source="microphone",
        sample_rate=16000,
        chunk_ms=40,
    )
    state = SimpleNamespace(deltas=[], final_text=None, done_seen=False, error=None)
    deltas: list[str] = []

    class RealtimeTranscriptionSessionCreated:
        pass

    class TranscriptionStreamTextDelta:
        text = "hola "

    class TranscriptionStreamDone:
        pass

    class RealtimeTranscriptionError:
        message = "mistral failed"

    backend._handle_event(RealtimeTranscriptionSessionCreated(), state, None, capture)
    backend._handle_event(TranscriptionStreamTextDelta(), state, deltas.append, capture)
    backend._handle_event(TranscriptionStreamDone(), state, None, capture)

    assert capture.events == [{"event": "mistral_session_created"}]
    assert deltas == ["hola "]
    assert state.final_text == "hola "
    assert state.done_seen is True

    state = SimpleNamespace(deltas=[], final_text=None, done_seen=False, error=None)
    backend._handle_event(RealtimeTranscriptionError(), state, None, capture)
    assert isinstance(state.error, RealtimeError)
    assert str(state.error) == "mistral failed"


def test_mistral_stream_enforces_timeout_after_audio_finishes(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "mistral_realtime"
    backend = MistralRealtimeBackend(config)
    capture = TranscriptionCapture(
        source="microphone",
        sample_rate=16000,
        chunk_ms=40,
    )
    deltas: list[str] = []

    class FakeAudioFormat:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class TranscriptionStreamTextDelta:
        text = "hola "

    class FakeRealtime:
        def transcribe_stream(self, audio_stream, **kwargs):
            del kwargs

            async def events():
                async for _chunk in audio_stream:
                    pass
                yield TranscriptionStreamTextDelta()
                await asyncio.sleep(60)

            return events()

    class FakeAudio:
        realtime = FakeRealtime()

    class FakeMistral:
        def __init__(self, api_key):
            self.api_key = api_key
            self.audio = FakeAudio()

    monkeypatch.setattr(
        backend,
        "_imports",
        lambda: {"Mistral": FakeMistral, "AudioFormat": FakeAudioFormat},
    )

    async def one_chunk():
        yield b"\x01\x00" * 120

    with pytest.raises(RealtimeError, match="Mistral transcription completion") as exc:
        asyncio.run(
            backend._transcribe_stream(
                one_chunk(),
                capture=capture,
                on_delta=deltas.append,
                timeout_seconds=0.01,
            )
        )

    assert deltas == ["hola "]
    assert exc.value.partial_text == "hola"
    assert capture.error_message.startswith(
        "timed out waiting for Mistral transcription completion"
    )


def test_mistral_stream_timeout_uses_stop_event_when_sdk_stalls(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    config = load_config(tmp_path / "missing.toml")
    config.transcription.provider = "mistral_realtime"
    backend = MistralRealtimeBackend(config)
    capture = TranscriptionCapture(
        source="microphone",
        sample_rate=16000,
        chunk_ms=40,
    )

    class FakeAudioFormat:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRealtime:
        def transcribe_stream(self, audio_stream, **kwargs):
            del audio_stream, kwargs

            async def events():
                await asyncio.sleep(60)
                if False:
                    yield object()

            return events()

    class FakeAudio:
        realtime = FakeRealtime()

    class FakeMistral:
        def __init__(self, api_key):
            self.api_key = api_key
            self.audio = FakeAudio()

    monkeypatch.setattr(
        backend,
        "_imports",
        lambda: {"Mistral": FakeMistral, "AudioFormat": FakeAudioFormat},
    )

    async def one_chunk():
        yield b"\x01\x00" * 120

    stop_event = threading.Event()
    stop_event.set()

    with pytest.raises(RealtimeError, match="Mistral transcription completion"):
        asyncio.run(
            backend._transcribe_stream(
                one_chunk(),
                capture=capture,
                on_delta=None,
                timeout_seconds=0.01,
                completion_event=stop_event,
            )
        )

    assert capture.error_message.startswith(
        "timed out waiting for Mistral transcription completion"
    )
