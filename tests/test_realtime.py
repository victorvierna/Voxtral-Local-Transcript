import asyncio
import base64
import json
import threading
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
        realtime=SimpleNamespace(
            final_timeout_seconds=12.0,
            segment_max_seconds=90,
            segment_finalize_timeout_seconds=12.0,
            stop_tail_ms=240,
            first_chunk_grace_ms=220,
        ),
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


class FakeRealtimeWebSocket:
    def __init__(self, done_text: str, on_final_commit=None) -> None:
        self.done_text = done_text
        self.on_final_commit = on_final_commit
        self.sent: list[str] = []
        self._session_sent = False
        self._final_committed = False
        self._done_sent = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def recv(self) -> str:
        if not self._session_sent:
            self._session_sent = True
            return json.dumps({"type": "session.created"})
        if self._final_committed and not self._done_sent:
            self._done_sent = True
            return json.dumps({"type": "transcription.done", "text": self.done_text})
        await asyncio.sleep(3600)
        return ""

    async def send(self, payload: str) -> None:
        self.sent.append(payload)
        msg = json.loads(payload)
        if (
            msg.get("type") == "input_audio_buffer.commit"
            and msg.get("final") is True
            and not self._final_committed
        ):
            self._final_committed = True
            if self.on_final_commit:
                self.on_final_commit()


class FakeErrorRealtimeWebSocket(FakeRealtimeWebSocket):
    def __init__(self, error_message: str = "EngineCore encountered an issue.") -> None:
        super().__init__("")
        self.error_message = error_message

    async def recv(self) -> str:
        if not self._session_sent:
            self._session_sent = True
            return json.dumps({"type": "session.created"})
        if self._final_committed and not self._done_sent:
            self._done_sent = True
            return json.dumps(
                {
                    "type": "error",
                    "error": self.error_message,
                    "code": "processing_error",
                }
            )
        await asyncio.sleep(3600)
        return ""


class FakeMicrophone:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def get_chunk(self, timeout: float = 0.05) -> bytes | None:
        del timeout
        if self.chunks:
            return self.chunks.pop(0)
        return None

    def drain(self) -> list[bytes]:
        chunks = self.chunks
        self.chunks = []
        return chunks


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


def test_stop_after_segment_rollover_transcribes_queued_tail(monkeypatch) -> None:
    transcriber = _make_transcriber()
    transcriber.config.realtime.segment_max_seconds = 0.08
    transcriber.config.realtime.stop_tail_ms = 0
    transcriber.config.realtime.segment_finalize_timeout_seconds = 0.01
    transcriber.config.realtime.final_timeout_seconds = 0.01
    stop_event = threading.Event()
    mic = FakeMicrophone([b"\x01\x00" * 320, b"\x02\x00" * 320])

    def mark_stopped_with_tail() -> None:
        stop_event.set()
        mic.chunks.append(b"\x03\x00" * 320)

    sockets = [
        FakeRealtimeWebSocket("uno dos ponemos", on_final_commit=mark_stopped_with_tail),
        FakeRealtimeWebSocket("ponemos el resto final"),
    ]

    async def fake_connect():
        return sockets.pop(0)

    monkeypatch.setattr(transcriber, "_connect", fake_connect)

    text = asyncio.run(
        transcriber.transcribe_microphone(
            stop_event=stop_event,
            mic=mic,
            close_mic=False,
        )
    )

    assert text == "uno dos ponemos el resto final"
    assert transcriber.last_capture is not None
    assert len(transcriber.last_capture.audio_bytes()) == 320 * 2 * 3
    assert transcriber.last_capture.segment_texts == [
        "uno dos ponemos",
        "ponemos el resto final",
    ]
    assert mic.started is True
    assert mic.stopped is False


def test_stop_with_text_recovers_final_tail(monkeypatch) -> None:
    transcriber = _make_transcriber()
    transcriber.config.realtime.stop_tail_ms = 0
    stop_event = threading.Event()
    stop_event.set()
    mic = FakeMicrophone([b"\x01\x00" * 320])

    sockets = [
        FakeRealtimeWebSocket("principio"),
        FakeRealtimeWebSocket("principio final completo"),
    ]

    async def fake_connect():
        return sockets.pop(0)

    monkeypatch.setattr(transcriber, "_connect", fake_connect)

    text = asyncio.run(
        transcriber.transcribe_microphone(
            stop_event=stop_event,
            mic=mic,
            close_mic=False,
        )
    )

    assert text == "principio final completo"
    assert transcriber.last_capture is not None
    assert transcriber.last_capture.segment_texts == [
        "principio",
        "principio final completo",
    ]


def test_post_rollover_tail_backend_error_is_not_returned_as_success(
    monkeypatch,
) -> None:
    transcriber = _make_transcriber()
    transcriber.config.realtime.segment_max_seconds = 0.08
    transcriber.config.realtime.stop_tail_ms = 0
    transcriber.config.realtime.segment_finalize_timeout_seconds = 0.01
    transcriber.config.realtime.final_timeout_seconds = 0.01
    stop_event = threading.Event()
    mic = FakeMicrophone([b"\x01\x00" * 320, b"\x02\x00" * 320])

    def mark_stopped_with_tail() -> None:
        stop_event.set()
        mic.chunks.append(b"\x03\x00" * 320)

    sockets = [
        FakeRealtimeWebSocket("uno dos", on_final_commit=mark_stopped_with_tail),
        FakeErrorRealtimeWebSocket(),
    ]

    async def fake_connect():
        return sockets.pop(0)

    monkeypatch.setattr(transcriber, "_connect", fake_connect)

    with pytest.raises(RealtimeError, match="transcription incomplete"):
        asyncio.run(
            transcriber.transcribe_microphone(
                stop_event=stop_event,
                mic=mic,
                close_mic=False,
            )
        )

    assert transcriber.last_capture is not None
    assert transcriber.last_capture.raw_text == "uno dos"
    assert transcriber.last_capture.error_message == "EngineCore encountered an issue."
