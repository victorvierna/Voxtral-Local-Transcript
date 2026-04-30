import asyncio
import base64
import json
import threading
from types import SimpleNamespace

import pytest

from voxtray.realtime import RealtimeError, RealtimeTranscriber, TranscriptionCapture


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


def test_check_realtime_session_uses_realtime_websocket(monkeypatch) -> None:
    transcriber = _make_transcriber()
    ws = FakeRealtimeWebSocket("")

    async def fake_connect():
        return ws

    monkeypatch.setattr(transcriber, "_connect", fake_connect)

    transcriber.check_realtime_session_blocking()

    assert ws.sent == [
        json.dumps(
            {
                "type": "session.update",
                "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
            }
        )
    ]


def test_capture_diagnostics_include_audio_signal_stats() -> None:
    capture = TranscriptionCapture(source="microphone", sample_rate=16000, chunk_ms=40)
    capture.append_audio_chunk(b"\x00\x00" * 16000)

    stats = capture.diagnostics()["audio_signal"]

    assert stats["duration_seconds"] == 1.0
    assert stats["sample_count"] == 16000
    assert stats["nonzero_samples"] == 0
    assert stats["peak"] == 0
    assert stats["rms"] == 0.0
    assert stats["has_signal"] is False


def test_capture_signal_stats_detect_nonzero_audio() -> None:
    capture = TranscriptionCapture(source="microphone", sample_rate=16000, chunk_ms=40)
    capture.append_audio_chunk(b"\x00\x00" * 8000)
    capture.append_audio_chunk((1024).to_bytes(2, "little", signed=True) * 8000)

    stats = capture.audio_signal_stats()

    assert stats["duration_seconds"] == 1.0
    assert stats["nonzero_samples"] == 8000
    assert stats["peak"] == 1024
    assert stats["has_signal"] is True
    assert capture.lacks_input_signal() is False


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


class PartialThenNeverWebSocket:
    def __init__(self) -> None:
        self._sent_delta = False

    async def recv(self) -> str:
        if not self._sent_delta:
            self._sent_delta = True
            return json.dumps({"type": "transcription.delta", "delta": "texto parcial"})
        await asyncio.sleep(3600)
        return ""


class NeverDoneRealtimeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self._session_sent = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def recv(self) -> str:
        if not self._session_sent:
            self._session_sent = True
            return json.dumps({"type": "session.created"})
        await asyncio.sleep(3600)
        return ""

    async def send(self, payload: str) -> None:
        self.sent.append(payload)


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
        while True:
            if self._final_committed and not self._done_sent:
                self._done_sent = True
                return json.dumps({"type": "transcription.done", "text": self.done_text})
            await asyncio.sleep(0.001)

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
        while True:
            if self._final_committed and not self._done_sent:
                self._done_sent = True
                return json.dumps(
                    {
                        "type": "error",
                        "error": self.error_message,
                        "code": "processing_error",
                    }
                )
            await asyncio.sleep(0.001)


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


def test_effective_segment_max_seconds_uses_voxtral_realtime_token_budget() -> None:
    transcriber = _make_transcriber()

    effective = transcriber._effective_segment_max_seconds(90)

    assert effective == pytest.approx(73.728, rel=1e-3)
    assert transcriber._effective_segment_max_seconds(120) == pytest.approx(
        73.728,
        rel=1e-3,
    )
    transcriber.config.engine.extra_args.extend(["--max-num-batched-tokens", "512"])
    assert transcriber._effective_segment_max_seconds(90) == pytest.approx(
        24.576,
        rel=1e-3,
    )


def test_stream_finalize_wait_seconds_extends_rollover_budget() -> None:
    wait_seconds = RealtimeTranscriber._stream_finalize_wait_seconds(
        audio_seconds=90.0,
        configured_seconds=6.0,
        final_segment=False,
    )

    assert wait_seconds == pytest.approx(45.0)


def test_stream_finalize_wait_seconds_extends_spoken_final_budget() -> None:
    wait_seconds = RealtimeTranscriber._stream_finalize_wait_seconds(
        audio_seconds=2.0,
        configured_seconds=8.0,
        final_segment=True,
    )

    assert wait_seconds == pytest.approx(16.0)


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


def test_collect_done_text_rejects_partial_timeout() -> None:
    transcriber = _make_transcriber()
    ws = PartialThenNeverWebSocket()
    deltas: list[str] = []
    capture = transcriber._new_capture(source="microphone")

    with pytest.raises(RealtimeError, match="partial transcript"):
        asyncio.run(
            transcriber._collect_done_text(
                ws,
                deltas,
                None,
                timeout_seconds=0.01,
                capture=capture,
            )
        )

    assert deltas == ["texto parcial"]
    assert capture.error_message.endswith("texto parcial")


def test_collect_done_text_uses_deltas_when_done_omits_text() -> None:
    transcriber = _make_transcriber()
    ws = FakeQueuedWebSocket(
        [
            {"type": "transcription.delta", "delta": "texto "},
            {"type": "transcription.delta", "delta": "por deltas"},
            {"type": "transcription.done"},
        ]
    )
    deltas: list[str] = []

    text = asyncio.run(
        transcriber._collect_done_text(
            ws,
            deltas,
            None,
            timeout_seconds=0.5,
        )
    )

    assert text == "texto por deltas"
    assert deltas == ["texto ", "por deltas"]


def test_merge_tail_text_appends_only_missing_words() -> None:
    merged = RealtimeTranscriber._merge_tail_text(
        "Por lo demás, perfecto. Haces esos ajustes y súbelo. Y verifica que haces bien",
        "Perfecto. Haces esos ajustes y súbelo y verifica que haces bien los saltos de línea.",
    )

    assert merged.endswith("verifica que haces bien los saltos de línea.")
    assert merged.casefold().count("perfecto") == 1


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


def test_transcribe_microphone_skips_final_commit_when_audio_has_no_signal(
    monkeypatch,
) -> None:
    transcriber = _make_transcriber()
    transcriber.config.realtime.stop_tail_ms = 0
    stop_event = threading.Event()
    stop_event.set()
    mic = FakeMicrophone([b"\x00\x00" * 16000])
    ws = FakeRealtimeWebSocket("should not be used")

    async def fake_connect():
        return ws

    monkeypatch.setattr(transcriber, "_connect", fake_connect)

    text = asyncio.run(
        transcriber.transcribe_microphone(
            stop_event=stop_event,
            mic=mic,
        )
    )

    final_commits = [
        payload
        for payload in ws.sent
        if json.loads(payload).get("type") == "input_audio_buffer.commit"
        and json.loads(payload).get("final") is True
    ]
    assert text == ""
    assert final_commits == []
    assert mic.stopped is True
    assert transcriber.last_capture is not None
    assert transcriber.last_capture.lacks_input_signal() is True
    assert transcriber.last_capture.events == [
        {
            "event": "microphone_signal_missing",
            "audio_signal": transcriber.last_capture.audio_signal_stats(),
        }
    ]


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
    assert [segment["source"] for segment in transcriber.last_capture.segments] == [
        "microphone-live",
        "microphone-live",
    ]
    assert transcriber.last_capture.segments[1]["break_reason"] == "stop"
    assert transcriber.last_capture.segments[1]["status"] == "success"
    assert mic.started is True
    assert mic.stopped is True


def test_empty_retry_after_partial_stop_failure_keeps_segment_incomplete(monkeypatch) -> None:
    transcriber = _make_transcriber()
    transcriber.config.realtime.stop_tail_ms = 0
    stop_event = threading.Event()
    stop_event.set()
    mic = FakeMicrophone([b"\x01\x00" * 320])
    ws = FakeRealtimeWebSocket("")

    async def fake_connect():
        return ws

    async def partial_timeout(
        receive_task,
        receive_state,
        timeout_seconds,
        capture=None,
        allow_empty_timeout=False,
    ):
        del receive_task, timeout_seconds, capture, allow_empty_timeout
        receive_state.deltas.append("texto parcial")
        raise RealtimeError("timed out waiting for transcription.done after partial")

    async def empty_retry_audio(raw_audio, *, timeout_seconds, on_delta=None):
        del raw_audio, timeout_seconds, on_delta
        return ""

    monkeypatch.setattr(transcriber, "_connect", fake_connect)
    monkeypatch.setattr(transcriber, "_collect_stream_done_text", partial_timeout)
    monkeypatch.setattr(transcriber, "_transcribe_pcm16_audio", empty_retry_audio)

    with pytest.raises(RealtimeError, match="empty transcript"):
        asyncio.run(
            transcriber.transcribe_microphone(
                stop_event=stop_event,
                mic=mic,
                close_mic=False,
            )
        )

    assert transcriber.last_capture is not None
    assert transcriber.last_capture.segment_texts == []
    assert transcriber.last_capture.raw_text == ""
    assert transcriber.last_capture.completion_status == "incomplete"
    assert transcriber.last_capture.segments[0]["status"] == "error"
    attempt_statuses = [
        attempt["status"] for attempt in transcriber.last_capture.segments[0]["attempts"]
    ]
    assert attempt_statuses == ["error", "empty"]


def test_segment_rollover_timeout_is_not_returned_as_success(monkeypatch) -> None:
    transcriber = _make_transcriber()
    transcriber.config.realtime.segment_max_seconds = 0.08
    transcriber.config.realtime.segment_finalize_timeout_seconds = 0.01
    stop_event = threading.Event()
    mic = FakeMicrophone([b"\x01\x00" * 320, b"\x02\x00" * 320])

    sockets = [NeverDoneRealtimeWebSocket()]

    async def fake_connect():
        return sockets.pop(0)

    async def fail_retry(
        raw_audio, *, timeout_seconds, segment, on_delta=None, allow_empty_retry=False
    ):
        del raw_audio, timeout_seconds, segment, on_delta, allow_empty_retry
        raise RealtimeError("timed out waiting for transcription.done after retry")

    monkeypatch.setattr(transcriber, "_connect", fake_connect)
    monkeypatch.setattr(transcriber, "_retry_segment_audio", fail_retry)

    with pytest.raises(RealtimeError, match="timed out waiting"):
        asyncio.run(
            transcriber.transcribe_microphone(
                stop_event=stop_event,
                mic=mic,
                close_mic=False,
            )
        )

    assert transcriber.last_capture is not None
    assert transcriber.last_capture.raw_text == ""
    assert transcriber.last_capture.error_message.startswith(
        "timed out waiting for transcription.done"
    )
    assert mic.stopped is False


def test_stop_with_text_uses_single_streaming_session(monkeypatch) -> None:
    transcriber = _make_transcriber()
    transcriber.config.realtime.stop_tail_ms = 0
    stop_event = threading.Event()
    stop_event.set()
    mic = FakeMicrophone([b"\x01\x00" * 320])

    sockets = [FakeRealtimeWebSocket("principio final completo")]

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
    assert sockets == []
    assert transcriber.last_capture is not None
    assert transcriber.last_capture.segment_texts == [
        "principio final completo",
    ]
    assert len(transcriber.last_capture.segments) == 1
    assert mic.stopped is True


def test_transcribe_microphone_forwards_streaming_deltas(monkeypatch) -> None:
    transcriber = _make_transcriber()
    transcriber.config.realtime.stop_tail_ms = 0
    stop_event = threading.Event()
    stop_event.set()
    mic = FakeMicrophone([b"\x01\x00" * 320])

    class DeltaThenDoneRealtimeWebSocket(FakeRealtimeWebSocket):
        def __init__(self) -> None:
            super().__init__("hola mundo")
            self._delta_sent = False

        async def recv(self) -> str:
            if not self._session_sent:
                self._session_sent = True
                return json.dumps({"type": "session.created"})
            while True:
                if self._final_committed and not self._delta_sent:
                    self._delta_sent = True
                    return json.dumps({"type": "transcription.delta", "delta": "hola "})
                if self._final_committed and not self._done_sent:
                    self._done_sent = True
                    return json.dumps(
                        {"type": "transcription.done", "text": "hola mundo"}
                    )
                await asyncio.sleep(0.001)

    async def fake_connect():
        return DeltaThenDoneRealtimeWebSocket()

    deltas: list[str] = []
    monkeypatch.setattr(transcriber, "_connect", fake_connect)

    text = asyncio.run(
        transcriber.transcribe_microphone(
            stop_event=stop_event,
            mic=mic,
            close_mic=False,
            on_delta=deltas.append,
        )
    )

    assert text == "hola mundo"
    assert deltas == ["hola "]


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

    with pytest.raises(RealtimeError, match="EngineCore"):
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


def test_post_rollover_empty_stop_tail_timeout_keeps_existing_text(monkeypatch) -> None:
    transcriber = _make_transcriber()
    transcriber.config.realtime.segment_max_seconds = 0.08
    transcriber.config.realtime.stop_tail_ms = 0
    transcriber.config.realtime.segment_finalize_timeout_seconds = 0.01
    transcriber.config.realtime.final_timeout_seconds = 0.01
    stop_event = threading.Event()
    mic = FakeMicrophone([b"\x01\x00" * 320, b"\x02\x00" * 320])

    def mark_stopped_with_silent_tail() -> None:
        stop_event.set()
        mic.chunks.append(b"\x00\x00" * 320)

    sockets = [
        FakeRealtimeWebSocket("uno dos", on_final_commit=mark_stopped_with_silent_tail),
        NeverDoneRealtimeWebSocket(),
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

    assert text == "uno dos"
    assert transcriber.last_capture is not None
    assert transcriber.last_capture.completion_status == "complete"
    assert [segment["status"] for segment in transcriber.last_capture.segments] == [
        "success",
        "empty",
    ]
    assert transcriber.last_capture.segments[1]["source"] == "microphone-live"
    assert transcriber.last_capture.segments[1]["break_reason"] == "stop"
    assert transcriber.last_capture.segments[1]["known_silence"] is True


def test_post_rollover_non_silent_empty_tail_timeout_is_incomplete(monkeypatch) -> None:
    transcriber = _make_transcriber()
    transcriber.config.realtime.segment_max_seconds = 0.08
    transcriber.config.realtime.stop_tail_ms = 0
    transcriber.config.realtime.segment_finalize_timeout_seconds = 0.01
    transcriber.config.realtime.final_timeout_seconds = 0.01
    stop_event = threading.Event()
    mic = FakeMicrophone([b"\x01\x00" * 320, b"\x02\x00" * 320])

    def mark_stopped_with_voiced_tail() -> None:
        stop_event.set()
        mic.chunks.append(b"\x03\x00" * 320)

    sockets = [
        FakeRealtimeWebSocket("uno dos", on_final_commit=mark_stopped_with_voiced_tail),
        NeverDoneRealtimeWebSocket(),
    ]

    async def fake_connect():
        return sockets.pop(0)

    monkeypatch.setattr(transcriber, "_connect", fake_connect)

    with pytest.raises(RealtimeError, match="final stop segment returned empty transcript"):
        asyncio.run(
            transcriber.transcribe_microphone(
                stop_event=stop_event,
                mic=mic,
                close_mic=False,
            )
        )

    assert transcriber.last_capture is not None
    assert transcriber.last_capture.raw_text == "uno dos"
    assert transcriber.last_capture.completion_status == "incomplete"
    assert transcriber.last_capture.segments[1]["source"] == "microphone-live"
    assert transcriber.last_capture.segments[1]["status"] == "error"
    assert transcriber.last_capture.segments[1]["known_silence"] is False
    assert transcriber.last_capture.segment_texts == [
        "uno dos",
    ]


def test_transcribe_file_retries_failed_segment(monkeypatch, tmp_path) -> None:
    transcriber = _make_transcriber()
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"not a real wav")
    monkeypatch.setattr(
        transcriber,
        "_audio_file_to_pcm16_bytes",
        lambda path: b"\x01\x00" * 320,
    )

    sockets = [
        FakeErrorRealtimeWebSocket("temporary segment failure"),
        FakeRealtimeWebSocket("texto recuperado"),
    ]

    async def fake_connect():
        return sockets.pop(0)

    monkeypatch.setattr(transcriber, "_connect", fake_connect)

    text = asyncio.run(transcriber.transcribe_file(audio_path))

    assert text == "texto recuperado"
    assert transcriber.last_capture is not None
    assert transcriber.last_capture.completion_status == "complete"
    assert transcriber.last_capture.segments[0]["status"] == "recovered"
    assert transcriber.last_capture.segments[0]["recovered"] is True
    assert len(transcriber.last_capture.segments[0]["attempts"]) == 2


def test_transcribe_file_empty_retry_after_error_is_incomplete(monkeypatch, tmp_path) -> None:
    transcriber = _make_transcriber()
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"not a real wav")
    monkeypatch.setattr(
        transcriber,
        "_audio_file_to_pcm16_bytes",
        lambda path: b"\x01\x00" * 320,
    )

    sockets = [
        FakeErrorRealtimeWebSocket("temporary segment failure"),
        FakeRealtimeWebSocket(""),
    ]

    async def fake_connect():
        return sockets.pop(0)

    monkeypatch.setattr(transcriber, "_connect", fake_connect)

    with pytest.raises(RealtimeError, match="empty transcript"):
        asyncio.run(transcriber.transcribe_file(audio_path))

    assert transcriber.last_capture is not None
    assert transcriber.last_capture.completion_status == "incomplete"
    assert transcriber.last_capture.raw_text == ""
    assert transcriber.last_capture.segments[0]["status"] == "error"
    assert "recovered" not in transcriber.last_capture.segments[0]
    attempt_statuses = [
        attempt["status"] for attempt in transcriber.last_capture.segments[0]["attempts"]
    ]
    assert attempt_statuses == [
        "error",
        "empty",
    ]


def test_transcribe_file_forwards_delta_callbacks(monkeypatch, tmp_path) -> None:
    transcriber = _make_transcriber()
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"not a real wav")
    monkeypatch.setattr(
        transcriber,
        "_audio_file_to_pcm16_bytes",
        lambda path: b"\x01\x00" * 320,
    )

    class DeltaThenDoneWebSocket(FakeRealtimeWebSocket):
        async def recv(self) -> str:
            if not self._session_sent:
                self._session_sent = True
                return json.dumps({"type": "session.created"})
            if self._final_committed and not self._done_sent:
                self._done_sent = True
                return json.dumps({"type": "transcription.delta", "delta": "hola "})
            if self._final_committed:
                return json.dumps({"type": "transcription.done", "text": "hola mundo"})
            await asyncio.sleep(3600)
            return ""

    async def fake_connect():
        return DeltaThenDoneWebSocket("hola mundo")

    deltas: list[str] = []
    monkeypatch.setattr(transcriber, "_connect", fake_connect)

    text = asyncio.run(transcriber.transcribe_file(audio_path, on_delta=deltas.append))

    assert text == "hola mundo"
    assert deltas == ["hola "]


def test_completion_problem_detects_missing_long_segments() -> None:
    transcriber = _make_transcriber()
    capture = transcriber._new_capture(source="microphone")
    capture.effective_segment_max_seconds = 10.0
    capture.append_audio_chunk(b"\x01\x00" * 16000 * 25)
    capture.segments.append(
        {
            "index": 1,
            "status": "success",
            "audio_seconds": 10.0,
            "text_chars": 12,
        }
    )

    problem = RealtimeTranscriber.completion_problem(capture, "texto parcial")

    assert "expected about" in problem


def test_completion_problem_accepts_recorded_stop_tail_segment() -> None:
    transcriber = _make_transcriber()
    capture = transcriber._new_capture(source="microphone")
    capture.effective_segment_max_seconds = 120.0
    capture.append_audio_chunk(b"\x01\x00" * 16000 * 242)
    capture.segments.extend(
        [
            {"index": 1, "source": "microphone-live", "status": "success"},
            {"index": 2, "source": "microphone-live", "status": "success"},
            {"index": 3, "source": "microphone-stop-tail", "status": "success"},
        ]
    )

    problem = RealtimeTranscriber.completion_problem(capture, "texto completo")

    assert problem == ""


def test_completion_problem_allows_chunk_boundary_rollover_slack() -> None:
    transcriber = _make_transcriber()
    capture = transcriber._new_capture(source="microphone")
    capture.effective_segment_max_seconds = 120.0
    capture.append_audio_chunk(b"\x01\x00" * int(16000 * 240.04))
    capture.segments.extend(
        [
            {
                "index": 1,
                "source": "microphone-live",
                "status": "success",
                "audio_seconds": 120.04,
            },
            {
                "index": 2,
                "source": "microphone-live",
                "status": "success",
                "audio_seconds": 120.0,
            },
        ]
    )

    problem = RealtimeTranscriber.completion_problem(capture, "texto completo")

    assert problem == ""


def test_completion_problem_still_detects_missing_segment_past_rollover_slack() -> None:
    transcriber = _make_transcriber()
    capture = transcriber._new_capture(source="microphone")
    capture.effective_segment_max_seconds = 120.0
    capture.append_audio_chunk(b"\x01\x00" * int(16000 * 240.5))
    capture.segments.extend(
        [
            {"index": 1, "source": "microphone-live", "status": "success"},
            {"index": 2, "source": "microphone-live", "status": "success"},
        ]
    )

    problem = RealtimeTranscriber.completion_problem(capture, "texto parcial")

    assert "expected about 3" in problem


def test_completion_problem_allows_stop_tail_to_cover_multiple_nominal_segments() -> None:
    transcriber = _make_transcriber()
    capture = transcriber._new_capture(source="microphone")
    capture.effective_segment_max_seconds = 10.0
    voiced_seconds = 32.02
    silent_tail_seconds = 6.98
    capture.append_audio_chunk(
        (b"\x01\x00" * int(16000 * voiced_seconds))
        + (b"\x00\x00" * int(16000 * silent_tail_seconds))
    )
    capture.segments.extend(
        [
            {
                "index": 1,
                "source": "microphone-live",
                "status": "success",
                "audio_seconds": 10.02,
                "audio_end_seconds": 10.02,
            },
            {
                "index": 2,
                "source": "microphone-stop-tail",
                "status": "success",
                "audio_seconds": 22.0,
                "audio_end_seconds": 32.02,
            },
        ]
    )

    problem = RealtimeTranscriber.completion_problem(capture, "texto completo")

    assert problem == ""


def test_completion_problem_keeps_non_silent_uncovered_tail_fatal() -> None:
    transcriber = _make_transcriber()
    capture = transcriber._new_capture(source="microphone")
    capture.effective_segment_max_seconds = 10.0
    capture.append_audio_chunk(b"\x01\x00" * int(16000 * 39.0))
    capture.segments.extend(
        [
            {
                "index": 1,
                "source": "microphone-live",
                "status": "success",
                "audio_seconds": 10.02,
                "audio_end_seconds": 10.02,
            },
            {
                "index": 2,
                "source": "microphone-stop-tail",
                "status": "success",
                "audio_seconds": 22.0,
                "audio_end_seconds": 32.02,
            },
        ]
    )

    problem = RealtimeTranscriber.completion_problem(capture, "texto parcial")

    assert "untranscribed trailing audio" in problem
