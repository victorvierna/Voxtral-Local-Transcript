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
