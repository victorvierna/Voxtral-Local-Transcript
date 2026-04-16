import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

from voxtray.realtime import RealtimeError, RealtimeTranscriber


class FakeWebSocket:
    def __init__(self, first_event: dict[str, object]) -> None:
        self._first_event = first_event
        self.sent: list[str] = []

    async def recv(self) -> str:
        return json.dumps(self._first_event)

    async def send(self, payload: str) -> None:
        self.sent.append(payload)


def _make_transcriber() -> RealtimeTranscriber:
    config = SimpleNamespace(
        model_id="mistralai/Voxtral-Mini-4B-Realtime-2602",
        websocket_url="ws://127.0.0.1:8000/v1/realtime",
        audio=SimpleNamespace(sample_rate=16000, chunk_ms=40),
        realtime=SimpleNamespace(final_timeout_seconds=12.0, segment_max_seconds=90),
        engine=SimpleNamespace(extra_args=["--max-model-len", "1536"]),
    )
    return RealtimeTranscriber(config=config)


def test_init_session_only_sends_model_update() -> None:
    transcriber = _make_transcriber()
    ws = FakeWebSocket({"type": "session.created"})

    asyncio.run(transcriber._init_session(ws))

    assert ws.sent == [
        json.dumps(
            {
                "type": "session.update",
                "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
            }
        )
    ]


def test_init_session_rejects_unexpected_first_event() -> None:
    transcriber = _make_transcriber()
    ws = FakeWebSocket({"type": "oops"})

    with pytest.raises(RealtimeError, match="unexpected first event"):
        asyncio.run(transcriber._init_session(ws))


def test_start_generation_uses_non_final_commit() -> None:
    transcriber = _make_transcriber()
    ws = FakeWebSocket({"type": "session.created"})

    asyncio.run(transcriber._start_generation(ws))

    assert ws.sent == [json.dumps({"type": "input_audio_buffer.commit", "final": False})]


def test_append_chunk_starts_generation_after_first_audio() -> None:
    transcriber = _make_transcriber()
    ws = FakeWebSocket({"type": "session.created"})

    asyncio.run(
        transcriber._append_chunk_and_maybe_start_generation(
            ws,
            b"\x01\x00\x02\x00",
            generation_started=False,
        )
    )

    assert ws.sent == [
        json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(b"\x01\x00\x02\x00").decode("utf-8"),
            }
        ),
        json.dumps({"type": "input_audio_buffer.commit", "final": False}),
    ]


def test_append_chunk_does_not_restart_generation_once_running() -> None:
    transcriber = _make_transcriber()
    ws = FakeWebSocket({"type": "session.created"})

    asyncio.run(
        transcriber._append_chunk_and_maybe_start_generation(
            ws,
            b"\x03\x00\x04\x00",
            generation_started=True,
        )
    )

    assert ws.sent == [
        json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(b"\x03\x00\x04\x00").decode("utf-8"),
            }
        )
    ]


class FakeQueuedWebSocket:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self._events = [json.dumps(event) for event in events]
        self.sent: list[str] = []

    async def recv(self) -> str:
        return self._events.pop(0)

    async def send(self, payload: str) -> None:
        self.sent.append(payload)


class NeverRespondingWebSocket:
    async def recv(self) -> str:
        await asyncio.sleep(3600)


def test_effective_segment_max_seconds_clamps_to_model_budget() -> None:
    transcriber = _make_transcriber()

    effective = transcriber._effective_segment_max_seconds(90)

    assert effective == pytest.approx(24.576, rel=1e-3)


def test_live_finalize_wait_seconds_scales_with_audio_length() -> None:
    transcriber = _make_transcriber()

    wait_seconds = transcriber._live_finalize_wait_seconds(
        audio_seconds=24.576,
        configured_seconds=6.0,
        final_segment=False,
    )

    assert wait_seconds == pytest.approx(16.288, rel=1e-3)


def test_collect_done_text_can_treat_timeout_as_empty_segment() -> None:
    transcriber = _make_transcriber()
    ws = NeverRespondingWebSocket()

    text = asyncio.run(
        transcriber._collect_done_text(
            ws,
            [],
            None,
            timeout_seconds=0.01,
            allow_empty_timeout=True,
        )
    )

    assert text == ""


def test_merge_tail_text_appends_only_missing_words() -> None:
    merged = RealtimeTranscriber._merge_tail_text(
        "Por lo demás, perfecto. Haces esos ajustes y súbelo. Y verifica que haces bien",
        "Perfecto. Haces esos ajustes y súbelo y verifica que haces bien los saltos de línea.",
    )

    assert merged.endswith("verifica que haces bien los saltos de línea.")
    assert merged.casefold().count("perfecto") == 1


def test_tail_pcm16_audio_uses_recent_context_window() -> None:
    transcriber = _make_transcriber()
    raw_audio = b"".join(bytes([i % 256, 0]) for i in range(16000 * 10))

    tail = transcriber._tail_pcm16_audio(raw_audio, seconds=2.0)

    assert len(tail) == 16000 * 2 * 2
    assert tail == raw_audio[-len(tail) :]


def test_recv_once_error_uses_serialized_payload_when_message_is_empty() -> None:
    transcriber = _make_transcriber()
    ws = FakeQueuedWebSocket([{"type": "error", "error": {}}])
    capture = transcriber._new_capture(source="microphone")

    with pytest.raises(RealtimeError) as excinfo:
        asyncio.run(
            transcriber._recv_once(
                ws,
                [],
                None,
                timeout=0.1,
                capture=capture,
            )
        )

    assert '"type": "error"' in str(excinfo.value)
    assert capture.error_payload == {"type": "error", "error": {}}
