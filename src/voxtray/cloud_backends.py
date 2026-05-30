from __future__ import annotations

import asyncio
import base64
from contextlib import suppress
from dataclasses import dataclass, field
from importlib import import_module
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, AsyncIterator
import wave

import httpx
import websockets

from .audio import MicrophoneStream
from .backend_contract import (
    BackendCapabilities,
    DeltaCallback,
    RecordingStoppedCallback,
)
from .config import VoxtrayConfig
from .quality import looks_like_sparse_final_segment, looks_like_truncated_transcript
from .realtime import RealtimeError, TranscriptionCapture


def _dotenv_value(env_name: str) -> str:
    candidates = [Path.cwd() / ".env"]
    try:
        candidates.append(Path(__file__).resolve().parents[2] / ".env")
    except IndexError:
        pass

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, separator, value = line.partition("=")
            if separator != "=" or key.strip() != env_name:
                continue
            value = value.strip()
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1]
            return value
    return ""


def _first_attr(module_names: list[str], attr_name: str) -> Any:
    errors: list[ImportError] = []
    for module_name in module_names:
        try:
            module = import_module(module_name)
        except ImportError as exc:
            errors.append(exc)
            continue
        try:
            return getattr(module, attr_name)
        except AttributeError as exc:
            errors.append(ImportError(f"{module_name}.{attr_name} is unavailable"))
            continue
    if errors:
        raise errors[-1]
    raise ImportError(f"{attr_name} is unavailable")


@dataclass(slots=True)
class _CloudReceiveState:
    deltas: list[str] = field(default_factory=list)
    final_text: str | None = None
    done_seen: bool = False
    error: RealtimeError | None = None


def _audio_file_to_pcm16_bytes(audio_path: Path, sample_rate: int) -> bytes:
    if not audio_path.exists():
        raise RealtimeError(f"audio file not found: {audio_path}")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RealtimeError(
            f"ffmpeg conversion failed for {audio_path}: {stderr or proc.returncode}"
        )
    return proc.stdout


class _CloudRealtimeBackend:
    capabilities = BackendCapabilities(
        realtime_microphone=True,
        transcribe_file=True,
        local_engine_required=False,
        warm_supported=False,
        model_control_supported=False,
    )

    def __init__(self, config: VoxtrayConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(f"voxtray.{self.provider_id}")
        self._last_capture: TranscriptionCapture | None = None

    @property
    def last_capture(self) -> TranscriptionCapture | None:
        return self._last_capture

    @property
    def chunk_ms(self) -> int:
        return int(getattr(getattr(self.config, "audio", None), "chunk_ms", 40))

    @property
    def mic_queue_chunks(self) -> int:
        return int(
            getattr(getattr(self.config, "realtime", None), "mic_queue_chunks", 4096)
        )

    @property
    def stop_tail_ms(self) -> int:
        return int(getattr(getattr(self.config, "realtime", None), "stop_tail_ms", 240))

    @property
    def final_timeout_seconds(self) -> float:
        return float(
            getattr(getattr(self.config, "realtime", None), "final_timeout_seconds", 12.0)
        )

    def _api_key(self, env_name: str) -> str:
        value = os.environ.get(env_name, "") or _dotenv_value(env_name)
        if not value:
            raise RealtimeError(f"{self.provider_id} requires ${env_name}")
        return value

    def api_key_env_present(self) -> bool:
        env_name = self.api_key_env
        return bool(env_name and (os.environ.get(env_name) or _dotenv_value(env_name)))

    def _new_capture(
        self,
        source: str,
        input_path: Path | None = None,
    ) -> TranscriptionCapture:
        capture = TranscriptionCapture(
            source=source,
            sample_rate=self.sample_rate,
            chunk_ms=self.chunk_ms,
            provider_id=self.provider_id,
            provider_model=self.provider_model,
            input_path=input_path,
        )
        self._last_capture = capture
        return capture

    def _open_microphone(self, mic: MicrophoneStream | None) -> MicrophoneStream:
        if mic is not None:
            return mic
        return MicrophoneStream(
            sample_rate=self.sample_rate,
            chunk_ms=self.chunk_ms,
            device=self.config.audio.device,
            max_queue_chunks=self.mic_queue_chunks,
        )

    async def _iter_microphone_chunks(
        self,
        *,
        stop_event: threading.Event,
        capture: TranscriptionCapture,
        mic: MicrophoneStream,
        close_mic: bool,
        on_recording_stopped: RecordingStoppedCallback | None,
    ) -> AsyncIterator[bytes]:
        mic.start()
        notified = False
        mic_stopped = False

        def notify_stopped() -> None:
            nonlocal notified
            if notified:
                return
            notified = True
            if on_recording_stopped is not None:
                on_recording_stopped()

        def stop_mic_after_user_stop() -> None:
            nonlocal mic_stopped
            if mic_stopped:
                return
            mic.stop()
            mic_stopped = True

        try:
            while not stop_event.is_set():
                chunk = mic.get_chunk(timeout=0.02)
                if not chunk:
                    await asyncio.sleep(0)
                    continue
                capture.append_audio_chunk(chunk)
                yield chunk

            tail_deadline = asyncio.get_running_loop().time() + (
                max(0, self.stop_tail_ms) / 1000.0
            )
            while asyncio.get_running_loop().time() < tail_deadline:
                chunk = mic.get_chunk(timeout=0.02)
                if not chunk:
                    await asyncio.sleep(0)
                    continue
                capture.append_audio_chunk(chunk)
                yield chunk

            stop_mic_after_user_stop()
            notify_stopped()
            for chunk in mic.drain():
                capture.append_audio_chunk(chunk)
                yield chunk
        finally:
            if close_mic and not mic_stopped:
                mic.stop()
            notify_stopped()

    async def _iter_file_chunks(
        self,
        raw_audio: bytes,
        *,
        chunk_size: int = 4096,
    ) -> AsyncIterator[bytes]:
        for index in range(0, len(raw_audio), chunk_size):
            yield raw_audio[index : index + chunk_size]
            await asyncio.sleep(0)

    def _finish_capture(
        self,
        capture: TranscriptionCapture,
        text: str,
        *,
        status: str | None = None,
        reason: str = "",
    ) -> str:
        final_text = text.strip()
        capture.raw_text = final_text
        if final_text:
            if not capture.segment_texts:
                capture.segment_texts = [final_text]
        else:
            capture.segment_texts = []
        capture.completion_status = status or ("complete" if final_text else "empty")
        capture.completion_reason = reason
        capture.error_message = reason if status == "incomplete" else ""
        return final_text


class MistralRealtimeBackend(_CloudRealtimeBackend):
    provider_id = "mistral_realtime"

    @property
    def api_key_env(self) -> str:
        return self.config.mistral_realtime.api_key_env

    @property
    def provider_model(self) -> str:
        return self.config.mistral_realtime.model

    @property
    def sample_rate(self) -> int:
        return int(self.config.mistral_realtime.sample_rate)

    def _imports(self) -> dict[str, Any]:
        try:
            Mistral = _first_attr(
                [
                    "mistralai.client",
                    "mistralai",
                ],
                "Mistral",
            )
            AudioFormat = _first_attr(
                [
                    "mistralai.client.models",
                    "mistralai.models",
                ],
                "AudioFormat",
            )
        except ImportError as exc:
            raise RealtimeError(
                "mistral_realtime requires package mistralai[realtime]"
            ) from exc
        return {"Mistral": Mistral, "AudioFormat": AudioFormat}

    def check_ready_blocking(self) -> None:
        self._api_key(self.api_key_env)
        self._imports()

    def _handle_event(
        self,
        event: object,
        state: _CloudReceiveState,
        on_delta: DeltaCallback | None,
        capture: TranscriptionCapture,
    ) -> None:
        event_name = type(event).__name__
        if event_name == "RealtimeTranscriptionSessionCreated":
            capture.events.append({"event": "mistral_session_created"})
            return
        if event_name == "TranscriptionStreamTextDelta":
            delta = str(getattr(event, "text", "") or "")
            if delta:
                state.deltas.append(delta)
                if on_delta is not None:
                    on_delta(delta)
            return
        if event_name == "TranscriptionStreamDone":
            state.done_seen = True
            text = str(getattr(event, "text", "") or "")
            state.final_text = text if text else "".join(state.deltas)
            return
        if event_name == "RealtimeTranscriptionError":
            message = str(getattr(event, "message", "") or event).strip()
            state.error = RealtimeError(message or "mistral realtime error", payload=repr(event))
            capture.error_message = str(state.error)
            capture.error_payload = repr(event)

    async def _transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        *,
        capture: TranscriptionCapture,
        on_delta: DeltaCallback | None,
        timeout_seconds: float,
        completion_event: threading.Event | None = None,
    ) -> str:
        imports = self._imports()
        client = imports["Mistral"](api_key=self._api_key(self.api_key_env))
        audio_format = imports["AudioFormat"](
            encoding="pcm_s16le",
            sample_rate=self.sample_rate,
        )
        state = _CloudReceiveState()

        audio_stream_done = asyncio.Event()
        completion_deadline: float | None = None

        def remaining_completion_timeout() -> float:
            nonlocal completion_deadline
            loop = asyncio.get_running_loop()
            if completion_deadline is None:
                completion_deadline = loop.time() + max(0.0, timeout_seconds)
            return max(0.0, completion_deadline - loop.time())

        def completion_requested() -> bool:
            return audio_stream_done.is_set() or (
                completion_event is not None and completion_event.is_set()
            )

        async def wait_for_completion_request() -> None:
            if completion_event is None:
                await audio_stream_done.wait()
                return
            while not completion_requested():
                await asyncio.sleep(0.01)

        async def tracked_audio_stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in audio_stream:
                    yield chunk
            finally:
                audio_stream_done.set()

        async def next_event(iterator) -> object:
            if completion_requested():
                return await asyncio.wait_for(
                    iterator.__anext__(),
                    timeout=remaining_completion_timeout(),
                )

            event_task = asyncio.create_task(iterator.__anext__())
            done_task = asyncio.create_task(wait_for_completion_request())
            try:
                done, _pending = await asyncio.wait(
                    {event_task, done_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if event_task in done:
                    event = event_task.result()
                    if completion_requested():
                        remaining_completion_timeout()
                    return event
                return await asyncio.wait_for(
                    event_task,
                    timeout=remaining_completion_timeout(),
                )
            finally:
                if not done_task.done():
                    done_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await done_task
                if not event_task.done():
                    event_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await event_task

        try:
            event_stream = client.audio.realtime.transcribe_stream(
                audio_stream=tracked_audio_stream(),
                model=self.provider_model,
                audio_format=audio_format,
                target_streaming_delay_ms=self.config.mistral_realtime.target_delay_ms,
            ).__aiter__()
            while True:
                try:
                    event = await next_event(event_stream)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    partial_text = "".join(state.deltas).strip()
                    message = (
                        "timed out waiting for Mistral transcription completion "
                        f"after {timeout_seconds:.1f}s"
                    )
                    if partial_text:
                        message = f"{message}: {partial_text}"
                    capture.error_message = message
                    raise RealtimeError(message, partial_text=partial_text) from exc
                self._handle_event(event, state, on_delta, capture)
                if state.error is not None:
                    raise state.error
                if state.done_seen:
                    break
        except RealtimeError:
            raise
        except Exception as exc:
            raise RealtimeError(f"mistral realtime failed: {exc}") from exc
        return (
            state.final_text
            if state.final_text is not None
            else "".join(state.deltas)
        ).strip()

    async def transcribe_microphone(
        self,
        stop_event: threading.Event,
        on_delta: DeltaCallback | None = None,
        final_timeout_seconds: float | None = None,
        mic: MicrophoneStream | None = None,
        close_mic: bool = True,
        on_recording_stopped: RecordingStoppedCallback | None = None,
    ) -> str:
        self.check_ready_blocking()
        capture = self._new_capture(source="microphone")
        mic = self._open_microphone(mic)
        stream = self._iter_microphone_chunks(
            stop_event=stop_event,
            capture=capture,
            mic=mic,
            close_mic=close_mic,
            on_recording_stopped=on_recording_stopped,
        )
        text = await self._transcribe_stream(
            stream,
            capture=capture,
            on_delta=on_delta,
            timeout_seconds=final_timeout_seconds or self.final_timeout_seconds,
            completion_event=stop_event,
        )
        return self._finish_capture(capture, text)

    def transcribe_microphone_blocking(
        self,
        stop_event: threading.Event,
        on_delta: DeltaCallback | None = None,
        final_timeout_seconds: float | None = None,
        mic: MicrophoneStream | None = None,
        close_mic: bool = True,
        on_recording_stopped: RecordingStoppedCallback | None = None,
    ) -> str:
        return asyncio.run(
            self.transcribe_microphone(
                stop_event=stop_event,
                on_delta=on_delta,
                final_timeout_seconds=final_timeout_seconds,
                mic=mic,
                close_mic=close_mic,
                on_recording_stopped=on_recording_stopped,
            )
        )

    async def transcribe_file(
        self,
        audio_path: Path,
        on_delta: DeltaCallback | None = None,
    ) -> str:
        self.check_ready_blocking()
        raw_audio = _audio_file_to_pcm16_bytes(audio_path, self.sample_rate)
        capture = self._new_capture(source="file", input_path=audio_path)
        capture.append_audio_chunk(raw_audio)
        text = await self._transcribe_stream(
            self._iter_file_chunks(raw_audio),
            capture=capture,
            on_delta=on_delta,
            timeout_seconds=max(
                self.final_timeout_seconds,
                len(raw_audio) / float(max(1, self.sample_rate * 2)) + 20.0,
            ),
        )
        return self._finish_capture(capture, text)

    def transcribe_file_blocking(
        self,
        audio_path: Path,
        on_delta: DeltaCallback | None = None,
    ) -> str:
        return asyncio.run(self.transcribe_file(audio_path, on_delta=on_delta))


class OpenAIRealtimeBackend(_CloudRealtimeBackend):
    provider_id = "openai_realtime"
    websocket_url = "wss://api.openai.com/v1/realtime?intent=transcription"
    audio_transcriptions_url = "https://api.openai.com/v1/audio/transcriptions"
    _AUDIO_API_FALLBACK_BACKOFF_SECONDS = (0.0, 2.0, 6.0)

    @property
    def api_key_env(self) -> str:
        return self.config.openai_realtime.api_key_env

    @property
    def provider_model(self) -> str:
        return self.config.openai_realtime.model

    @property
    def fallback_model(self) -> str:
        return self.config.openai_realtime.fallback_model

    @property
    def sample_rate(self) -> int:
        return int(self.config.openai_realtime.sample_rate)

    def check_ready_blocking(self) -> None:
        self._api_key(self.api_key_env)

    @staticmethod
    def _pcm16_wav_bytes(raw_audio: bytes, sample_rate: int) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(raw_audio)
        return output.getvalue()

    @staticmethod
    def _rounded_seconds(value: float) -> float:
        return round(max(0.0, value), 3)

    def _audio_seconds_for_bytes(self, raw_audio: bytes) -> float:
        bytes_per_second = max(1, self.sample_rate * 2)
        return len(raw_audio) / float(bytes_per_second)

    def _effective_segment_max_seconds(self, requested_seconds: float) -> float:
        if requested_seconds <= 0:
            return 0.0
        return min(float(requested_seconds), 30.0)

    def _configure_capture_segmentation(self, capture: TranscriptionCapture) -> None:
        requested_seconds = float(self.config.realtime.segment_max_seconds)
        capture.requested_segment_max_seconds = requested_seconds
        capture.effective_segment_max_seconds = self._effective_segment_max_seconds(
            requested_seconds
        )

    def _split_pcm16_audio(self, raw_audio: bytes, max_seconds: float) -> list[bytes]:
        if not raw_audio:
            return []
        if max_seconds <= 0:
            return [raw_audio]
        bytes_per_second = max(1, self.sample_rate * 2)
        segment_bytes = max(2, int(max_seconds * bytes_per_second))
        if segment_bytes % 2 != 0:
            segment_bytes -= 1
        segment_bytes = max(2, segment_bytes)
        return [
            raw_audio[offset : offset + segment_bytes]
            for offset in range(0, len(raw_audio), segment_bytes)
        ]

    def _new_segment_record(
        self,
        capture: TranscriptionCapture,
        *,
        source: str,
        audio_start_seconds: float,
        audio_seconds: float,
        chunk_count: int,
        final_segment: bool,
        wait_seconds: float,
    ) -> dict[str, object]:
        segment = {
            "index": len(capture.segments) + 1,
            "source": source,
            "audio_start_seconds": self._rounded_seconds(audio_start_seconds),
            "audio_end_seconds": self._rounded_seconds(
                audio_start_seconds + audio_seconds
            ),
            "audio_seconds": self._rounded_seconds(audio_seconds),
            "chunk_count": chunk_count,
            "final_segment": final_segment,
            "wait_seconds": self._rounded_seconds(wait_seconds),
            "status": "pending",
            "text_chars": 0,
            "fallback_used": False,
            "attempts": [],
        }
        capture.segments.append(segment)
        return segment

    @staticmethod
    def _record_segment_attempt(
        segment: dict[str, object],
        *,
        attempt: int,
        status: str,
        timeout_seconds: float,
        text: str = "",
        error: str = "",
        payload: object | None = None,
    ) -> None:
        attempts = segment.setdefault("attempts", [])
        if not isinstance(attempts, list):
            return
        item: dict[str, object] = {
            "attempt": attempt,
            "status": status,
            "timeout_seconds": round(max(0.0, timeout_seconds), 3),
            "text_chars": len(text.strip()),
        }
        if error:
            item["error"] = error
        if payload is not None:
            item["error_payload"] = payload
        attempts.append(item)

    @staticmethod
    def _looks_truncated(text: str, capture: TranscriptionCapture) -> bool:
        return looks_like_truncated_transcript(text, capture.audio_duration_seconds())

    @staticmethod
    def _segment_recovery_reason(
        text: str,
        capture: TranscriptionCapture,
        *,
        final_segment: bool,
        force: bool,
    ) -> str:
        if force:
            return "realtime_error"
        if looks_like_truncated_transcript(text, capture.audio_duration_seconds()):
            return "realtime_empty_or_truncated"
        if final_segment and looks_like_sparse_final_segment(
            text,
            capture.audio_duration_seconds(),
        ):
            return "sparse_final_segment"
        return ""

    @staticmethod
    def _mark_segment_incomplete(
        segment: dict[str, object],
        *,
        text: str,
        error: str,
    ) -> None:
        segment["status"] = "incomplete"
        segment["text_chars"] = len(text.strip())
        segment["error"] = error

    @staticmethod
    def _audio_api_status_code(exc: BaseException) -> int | None:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code
        return None

    @staticmethod
    def _audio_api_error_message(exc: BaseException) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            message = exc.response.text.strip()
            return message or str(exc.response.status_code)
        return str(exc)

    @classmethod
    def _is_retryable_audio_api_error(cls, exc: BaseException) -> bool:
        status_code = cls._audio_api_status_code(exc)
        if status_code is not None:
            return status_code == 429 or status_code >= 500
        return isinstance(exc, httpx.TransportError)

    def _sleep_audio_api_retry(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def _transcribe_audio_api_blocking(
        self,
        raw_audio: bytes,
        sample_rate: int,
        *,
        timeout_seconds: float,
    ) -> str:
        fallback_model = self.fallback_model.strip() or "whisper-1"
        wav_audio = self._pcm16_wav_bytes(raw_audio, sample_rate)
        data = {"model": fallback_model}
        language = self.config.openai_realtime.language.strip()
        prompt = self.config.openai_realtime.prompt.strip()
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt

        timeout = httpx.Timeout(
            connect=15.0,
            read=max(60.0, timeout_seconds),
            write=60.0,
            pool=15.0,
        )
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                self.audio_transcriptions_url,
                headers={"Authorization": f"Bearer {self._api_key(self.api_key_env)}"},
                data=data,
                files={"file": ("audio.wav", wav_audio, "audio/wav")},
            )
            response.raise_for_status()

        payload = response.json()
        return str(payload.get("text") or "").strip()

    def _transcribe_audio_api_fallback_blocking(
        self,
        raw_audio: bytes,
        sample_rate: int,
        *,
        timeout_seconds: float,
        capture: TranscriptionCapture,
        diagnostic_context: dict[str, object] | None = None,
    ) -> str:
        max_attempts = len(self._AUDIO_API_FALLBACK_BACKOFF_SECONDS)
        last_error = "unknown error"
        for attempt, backoff_seconds in enumerate(
            self._AUDIO_API_FALLBACK_BACKOFF_SECONDS,
            start=1,
        ):
            self._sleep_audio_api_retry(backoff_seconds)
            try:
                return self._transcribe_audio_api_blocking(
                    raw_audio,
                    sample_rate,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                retryable = self._is_retryable_audio_api_error(exc)
                last_error = self._audio_api_error_message(exc)
                event: dict[str, object] = {
                    "event": "openai_audio_api_fallback_attempt_failed",
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "retryable": retryable,
                    "backoff_seconds": backoff_seconds,
                    "error": last_error,
                }
                status_code = self._audio_api_status_code(exc)
                if status_code is not None:
                    event["status_code"] = status_code
                if retryable and attempt < max_attempts:
                    event["next_backoff_seconds"] = (
                        self._AUDIO_API_FALLBACK_BACKOFF_SECONDS[attempt]
                    )
                if diagnostic_context:
                    event.update(diagnostic_context)
                capture.events.append(event)
                if not retryable or attempt >= max_attempts:
                    raise RealtimeError(
                        f"openai audio transcription failed: {last_error}"
                    ) from exc
        raise RealtimeError(f"openai audio transcription failed: {last_error}")

    async def _recover_with_audio_api_if_needed(
        self,
        capture: TranscriptionCapture,
        text: str,
    ) -> str:
        if not self._looks_truncated(text, capture):
            return text
        raw_audio = capture.audio_bytes()
        if not raw_audio:
            return text

        duration = capture.audio_duration_seconds()
        capture.events.append(
            {
                "event": "openai_audio_api_fallback_started",
                "scope": "full",
                "reason": "realtime_empty_or_truncated",
                "realtime_text_chars": len(text.strip()),
                "audio_duration_seconds": round(duration, 3),
                "fallback_model": self.fallback_model,
            }
        )
        try:
            recovered = self._transcribe_audio_api_fallback_blocking(
                raw_audio,
                capture.sample_rate,
                timeout_seconds=max(self.final_timeout_seconds, duration + 60.0),
                capture=capture,
                diagnostic_context={
                    "scope": "full",
                    "reason": "realtime_empty_or_truncated",
                },
            )
        except RealtimeError as exc:
            capture.events.append(
                {
                    "event": "openai_audio_api_fallback_failed",
                    "scope": "full",
                    "error": str(exc),
                }
            )
            return text

        if not recovered:
            capture.events.append(
                {
                    "event": "openai_audio_api_fallback_empty",
                    "scope": "full",
                }
            )
            return text

        capture.fallback_used = True
        capture.fallback_source = f"openai_audio_transcriptions:{self.fallback_model}"
        capture.events.append(
            {
                "event": "openai_audio_api_fallback_completed",
                "scope": "full",
                "text_chars": len(recovered),
            }
        )
        return recovered

    def _segment_capture(
        self,
        parent: TranscriptionCapture,
        raw_audio: bytes,
    ) -> TranscriptionCapture:
        capture = TranscriptionCapture(
            source=parent.source,
            sample_rate=parent.sample_rate,
            chunk_ms=parent.chunk_ms,
            provider_id=parent.provider_id,
            provider_model=parent.provider_model,
            input_path=parent.input_path,
        )
        capture.append_audio_chunk(raw_audio)
        return capture

    async def _recover_segment_with_audio_api_if_needed(
        self,
        *,
        capture: TranscriptionCapture,
        segment: dict[str, object],
        raw_audio: bytes,
        text: str,
        force: bool = False,
    ) -> str:
        if not raw_audio:
            return text
        segment_capture = self._segment_capture(capture, raw_audio)
        reason = self._segment_recovery_reason(
            text,
            segment_capture,
            final_segment=bool(segment.get("final_segment")),
            force=force,
        )
        if not reason:
            return text

        segment_index = segment.get("index")
        segment["fallback_used"] = True
        segment["fallback_source"] = (
            f"openai_audio_transcriptions:{self.fallback_model}"
        )
        duration = segment_capture.audio_duration_seconds()
        capture.events.append(
            {
                "event": "openai_audio_api_fallback_started",
                "scope": "segment",
                "segment_index": segment_index,
                "reason": reason,
                "realtime_text_chars": len(text.strip()),
                "audio_duration_seconds": round(duration, 3),
                "fallback_model": self.fallback_model,
            }
        )
        try:
            recovered = self._transcribe_audio_api_fallback_blocking(
                raw_audio,
                capture.sample_rate,
                timeout_seconds=max(self.final_timeout_seconds, duration + 60.0),
                capture=capture,
                diagnostic_context={
                    "scope": "segment",
                    "segment_index": segment_index,
                    "reason": reason,
                },
            )
        except RealtimeError as exc:
            capture.events.append(
                {
                    "event": "openai_audio_api_fallback_failed",
                    "scope": "segment",
                    "segment_index": segment_index,
                    "error": str(exc),
                }
            )
            segment["error"] = str(exc)
            if reason == "sparse_final_segment":
                self._mark_segment_incomplete(
                    segment,
                    text=text,
                    error=str(exc),
                )
            if force and not text.strip():
                raise
            return text

        if not recovered:
            if reason == "sparse_final_segment":
                self._mark_segment_incomplete(
                    segment,
                    text=text,
                    error="audio API fallback returned empty text for sparse final segment",
                )
            capture.events.append(
                {
                    "event": "openai_audio_api_fallback_empty",
                    "scope": "segment",
                    "segment_index": segment_index,
                }
            )
            return text

        if reason == "sparse_final_segment" and looks_like_sparse_final_segment(
            recovered,
            duration,
        ):
            message = "audio API fallback did not resolve sparse final segment"
            self._mark_segment_incomplete(
                segment,
                text=recovered,
                error=message,
            )
            capture.events.append(
                {
                    "event": "openai_audio_api_fallback_still_sparse",
                    "scope": "segment",
                    "segment_index": segment_index,
                    "text_chars": len(recovered),
                }
            )
            return recovered

        capture.fallback_used = True
        capture.fallback_source = f"openai_audio_transcriptions:{self.fallback_model}"
        segment["status"] = "recovered"
        segment["text_chars"] = len(recovered)
        segment["recovered"] = True
        capture.events.append(
            {
                "event": "openai_audio_api_fallback_completed",
                "scope": "segment",
                "segment_index": segment_index,
                "text_chars": len(recovered),
            }
        )
        return recovered

    def _segment_timeout_seconds(
        self,
        segment_audio: bytes,
        requested_timeout_seconds: float,
    ) -> float:
        return max(
            self.final_timeout_seconds,
            requested_timeout_seconds,
            self._audio_seconds_for_bytes(segment_audio) + 20.0,
        )

    async def _transcribe_pcm16_segments(
        self,
        raw_audio: bytes,
        *,
        capture: TranscriptionCapture,
        on_delta: DeltaCallback | None,
        timeout_seconds: float,
        source: str,
    ) -> str:
        segment_texts: list[str] = []
        audio_start_seconds = 0.0
        raw_segments = self._split_pcm16_audio(
            raw_audio,
            capture.effective_segment_max_seconds,
        )
        for offset, segment_audio in enumerate(raw_segments):
            segment_seconds = self._audio_seconds_for_bytes(segment_audio)
            wait_seconds = self._segment_timeout_seconds(
                segment_audio,
                timeout_seconds,
            )
            segment_record = self._new_segment_record(
                capture,
                source=source,
                audio_start_seconds=audio_start_seconds,
                audio_seconds=segment_seconds,
                chunk_count=max(1, (len(segment_audio) + 4095) // 4096),
                final_segment=offset == len(raw_segments) - 1,
                wait_seconds=wait_seconds,
            )
            audio_start_seconds += segment_seconds

            try:
                segment_text = await self._send_stream_to_openai(
                    self._iter_file_chunks(segment_audio),
                    capture=capture,
                    on_delta=on_delta,
                    timeout_seconds=wait_seconds,
                )
            except RealtimeError as exc:
                partial_text = str(getattr(exc, "partial_text", "") or "").strip()
                self._record_segment_attempt(
                    segment_record,
                    attempt=1,
                    status="error",
                    timeout_seconds=wait_seconds,
                    text=partial_text,
                    error=str(exc),
                    payload=exc.payload,
                )
                segment_record["status"] = "error"
                segment_record["error"] = str(exc)
                segment_text = await self._recover_segment_with_audio_api_if_needed(
                    capture=capture,
                    segment=segment_record,
                    raw_audio=segment_audio,
                    text=partial_text,
                    force=True,
                )
            else:
                segment_text = segment_text.strip()
                segment_record["status"] = "success" if segment_text else "empty"
                segment_record["text_chars"] = len(segment_text)
                self._record_segment_attempt(
                    segment_record,
                    attempt=1,
                    status=str(segment_record["status"]),
                    timeout_seconds=wait_seconds,
                    text=segment_text,
                )
                segment_text = await self._recover_segment_with_audio_api_if_needed(
                    capture=capture,
                    segment=segment_record,
                    raw_audio=segment_audio,
                    text=segment_text,
                )

            segment_text = segment_text.strip()
            if segment_text:
                if segment_record.get("status") not in {
                    "recovered",
                    "error",
                    "incomplete",
                }:
                    segment_record["status"] = "success"
                    segment_record["text_chars"] = len(segment_text)
                segment_texts.append(segment_text)
                capture.segment_texts.append(segment_text)
            elif segment_record.get("status") == "pending":
                segment_record["status"] = "empty"
                segment_record["text_chars"] = 0

        return "\n".join(segment_texts)

    @staticmethod
    def _full_audio_fallback_completed(capture: TranscriptionCapture) -> bool:
        for event in capture.events:
            if event.get("event") != "openai_audio_api_fallback_completed":
                continue
            if str(event.get("scope") or "full") == "full":
                return True
        return False

    @staticmethod
    def _failed_segment_indexes(capture: TranscriptionCapture) -> list[str]:
        indexes: list[str] = []
        for segment in capture.segments:
            if bool(segment.get("recovered")):
                continue
            if str(segment.get("status") or "") not in {
                "error",
                "timeout",
                "incomplete",
            }:
                continue
            indexes.append(str(segment.get("index", "?")))
        return indexes

    def _raise_if_unrecovered_segments(
        self,
        capture: TranscriptionCapture,
        text: str,
    ) -> None:
        if self._full_audio_fallback_completed(capture):
            return
        failed_indexes = self._failed_segment_indexes(capture)
        if not failed_indexes:
            return
        indexes = ", ".join(failed_indexes)
        message = f"openai segmented transcription incomplete: failed segment(s) {indexes}"
        partial_text = text.strip()
        capture.raw_text = partial_text
        capture.completion_status = "incomplete"
        capture.completion_reason = message
        capture.error_message = message
        raise RealtimeError(
            message,
            payload=capture.error_payload,
            partial_text=partial_text,
        )

    async def _recover_and_finish_segmented_capture(
        self,
        capture: TranscriptionCapture,
        text: str,
    ) -> str:
        recovered = await self._recover_with_audio_api_if_needed(capture, text)
        self._raise_if_unrecovered_segments(capture, recovered)
        return self._finish_capture(capture, recovered)

    async def _transcribe_pcm16_audio(
        self,
        raw_audio: bytes,
        *,
        capture: TranscriptionCapture,
        on_delta: DeltaCallback | None,
        timeout_seconds: float,
        source: str,
    ) -> str:
        if not raw_audio:
            return ""
        if capture.effective_segment_max_seconds <= 0:
            return await self._send_stream_with_recovery(
                self._iter_file_chunks(raw_audio),
                capture=capture,
                on_delta=on_delta,
                timeout_seconds=timeout_seconds,
            )
        return await self._transcribe_pcm16_segments(
            raw_audio,
            capture=capture,
            on_delta=on_delta,
            timeout_seconds=timeout_seconds,
            source=source,
        )

    async def _send_stream_with_recovery(
        self,
        audio_stream: AsyncIterator[bytes],
        *,
        capture: TranscriptionCapture,
        on_delta: DeltaCallback | None,
        timeout_seconds: float,
    ) -> str:
        try:
            text = await self._send_stream_to_openai(
                audio_stream,
                capture=capture,
                on_delta=on_delta,
                timeout_seconds=timeout_seconds,
            )
        except RealtimeError as exc:
            partial_text = str(getattr(exc, "partial_text", "") or "").strip()
            recovered = await self._recover_with_audio_api_if_needed(
                capture,
                partial_text,
            )
            if recovered.strip() and recovered.strip() != partial_text:
                capture.events.append(
                    {
                        "event": "openai_realtime_error_recovered",
                        "error": str(exc),
                    }
                )
                return recovered
            raise
        return await self._recover_with_audio_api_if_needed(capture, text)

    async def _connect(self):
        headers = {"Authorization": f"Bearer {self._api_key(self.api_key_env)}"}
        try:
            return await websockets.connect(
                self.websocket_url,
                additional_headers=headers,
                max_size=None,
                ping_interval=20,
                ping_timeout=20,
            )
        except TypeError as exc:
            if "additional_headers" not in str(exc):
                raise
            return await websockets.connect(
                self.websocket_url,
                extra_headers=headers,
                max_size=None,
                ping_interval=20,
                ping_timeout=20,
            )

    def _session_update_payload(self) -> dict[str, Any]:
        turn_detection = self.config.openai_realtime.turn_detection.strip().lower()
        turn_detection_payload: dict[str, Any] | None
        if turn_detection in {"", "manual", "none", "null"}:
            turn_detection_payload = None
        else:
            turn_detection_payload = {"type": turn_detection}
        transcription: dict[str, Any] = {
            "model": self.provider_model,
        }
        delay = self.config.openai_realtime.delay.strip().lower()
        if delay and self.provider_model == "gpt-realtime-whisper":
            transcription["delay"] = delay
        language = self.config.openai_realtime.language.strip()
        prompt = self.config.openai_realtime.prompt.strip()
        if language:
            transcription["language"] = language
        if prompt and self.provider_model != "gpt-realtime-whisper":
            transcription["prompt"] = prompt
        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": self.sample_rate,
                        },
                        "transcription": transcription,
                        "turn_detection": turn_detection_payload,
                        "noise_reduction": {"type": "near_field"},
                    }
                },
            },
        }

    @staticmethod
    def _format_error_message(msg: dict[str, Any]) -> str:
        error = msg.get("error")
        if isinstance(error, dict):
            for key in ("message", "code", "type"):
                value = str(error.get(key) or "").strip()
                if value:
                    return value
        if isinstance(error, str) and error.strip():
            return error.strip()
        for key in ("message", "detail", "code"):
            value = str(msg.get(key) or "").strip()
            if value:
                return value
        return json.dumps(msg, ensure_ascii=False, sort_keys=True)

    def _set_event_error(
        self,
        msg: dict[str, Any],
        state: _CloudReceiveState,
        capture: TranscriptionCapture,
        *,
        preserve_partial: bool,
    ) -> None:
        message = self._format_error_message(msg)
        partial_text = "".join(state.deltas).strip() if preserve_partial else ""
        state.error = RealtimeError(
            message,
            payload=msg,
            partial_text=partial_text,
        )
        capture.error_message = message
        capture.error_payload = msg

    def _handle_event(
        self,
        msg: dict[str, Any],
        state: _CloudReceiveState,
        on_delta: DeltaCallback | None,
        capture: TranscriptionCapture,
    ) -> None:
        msg_type = msg.get("type")
        if msg_type in {"transcription_session.created", "session.created"}:
            capture.events.append({"event": str(msg_type)})
            return
        if msg_type in {"transcription_session.updated", "session.updated"}:
            capture.events.append({"event": str(msg_type)})
            return
        if msg_type == "conversation.item.input_audio_transcription.delta":
            delta = str(msg.get("delta") or "")
            if delta:
                state.deltas.append(delta)
                if on_delta is not None:
                    on_delta(delta)
            return
        if msg_type == "conversation.item.input_audio_transcription.completed":
            transcript = str(msg.get("transcript") or "")
            state.final_text = transcript if transcript else "".join(state.deltas)
            state.done_seen = True
            return
        if msg_type == "conversation.item.input_audio_transcription.failed":
            self._set_event_error(
                msg,
                state,
                capture,
                preserve_partial=True,
            )
            return
        if msg_type == "error":
            self._set_event_error(
                msg,
                state,
                capture,
                preserve_partial=False,
            )

    async def _receive_events(
        self,
        ws,
        state: _CloudReceiveState,
        on_delta: DeltaCallback | None,
        capture: TranscriptionCapture,
    ) -> None:
        while True:
            try:
                raw = await ws.recv()
                msg = json.loads(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state.error = RealtimeError(f"openai realtime receive failed: {exc}")
                capture.error_message = str(state.error)
                return
            self._handle_event(msg, state, on_delta, capture)
            if state.error is not None or state.done_seen:
                return

    async def _collect_done_text(
        self,
        receive_task: asyncio.Task[None],
        state: _CloudReceiveState,
        timeout_seconds: float,
        capture: TranscriptionCapture,
    ) -> str:
        if state.error is not None:
            raise state.error
        if not state.done_seen:
            try:
                await asyncio.wait_for(asyncio.shield(receive_task), timeout_seconds)
            except asyncio.TimeoutError:
                pass
        if state.error is not None:
            raise state.error
        if state.done_seen:
            return (
                state.final_text
                if state.final_text is not None
                else "".join(state.deltas)
            ).strip()
        partial_text = "".join(state.deltas).strip()
        message = (
            f"timed out waiting for OpenAI transcription completion after {timeout_seconds:.1f}s"
        )
        if partial_text:
            message = f"{message}: {partial_text}"
        capture.error_message = message
        raise RealtimeError(message, partial_text=partial_text)

    @staticmethod
    def _raise_if_openai_receive_stopped(
        state: _CloudReceiveState,
        capture: TranscriptionCapture,
    ) -> None:
        if state.error is not None:
            partial_text = "".join(state.deltas).strip()
            if partial_text and not getattr(state.error, "partial_text", ""):
                raise RealtimeError(
                    str(state.error),
                    payload=state.error.payload,
                    partial_text=partial_text,
                ) from state.error
            raise state.error
        if not state.done_seen:
            return
        partial_text = (
            state.final_text if state.final_text is not None else "".join(state.deltas)
        ).strip()
        message = "OpenAI transcription completed before audio upload finished"
        capture.error_message = message
        raise RealtimeError(message, partial_text=partial_text)

    @staticmethod
    async def _send_audio_chunk(ws, chunk: bytes) -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("utf-8"),
                }
            )
        )

    async def _send_stream_to_openai(
        self,
        audio_stream: AsyncIterator[bytes],
        *,
        capture: TranscriptionCapture,
        on_delta: DeltaCallback | None,
        timeout_seconds: float,
    ) -> str:
        state = _CloudReceiveState()
        receive_task: asyncio.Task[None] | None = None
        try:
            async with await self._connect() as ws:
                try:
                    receive_task = asyncio.create_task(
                        self._receive_events(ws, state, on_delta, capture)
                    )
                    await ws.send(json.dumps(self._session_update_payload()))
                    sent_audio = False
                    async for chunk in audio_stream:
                        if not chunk:
                            continue
                        self._raise_if_openai_receive_stopped(state, capture)
                        sent_audio = True
                        await self._send_audio_chunk(ws, chunk)
                        self._raise_if_openai_receive_stopped(state, capture)
                    if not sent_audio:
                        return ""
                    self._raise_if_openai_receive_stopped(state, capture)
                    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    return await self._collect_done_text(
                        receive_task,
                        state,
                        timeout_seconds,
                        capture,
                    )
                finally:
                    if receive_task is not None and not receive_task.done():
                        receive_task.cancel()
                        try:
                            await receive_task
                        except asyncio.CancelledError:
                            pass
        except RealtimeError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            partial_text = "".join(state.deltas).strip()
            message = f"openai realtime stream failed: {exc}"
            capture.error_message = message
            raise RealtimeError(message, partial_text=partial_text) from exc

    async def transcribe_microphone(
        self,
        stop_event: threading.Event,
        on_delta: DeltaCallback | None = None,
        final_timeout_seconds: float | None = None,
        mic: MicrophoneStream | None = None,
        close_mic: bool = True,
        on_recording_stopped: RecordingStoppedCallback | None = None,
    ) -> str:
        self.check_ready_blocking()
        capture = self._new_capture(source="microphone")
        self._configure_capture_segmentation(capture)
        if capture.effective_segment_max_seconds > 0:
            mic = self._open_microphone(mic)
            async for _chunk in self._iter_microphone_chunks(
                stop_event=stop_event,
                capture=capture,
                mic=mic,
                close_mic=close_mic,
                on_recording_stopped=on_recording_stopped,
            ):
                pass
            raw_audio = capture.audio_bytes()
            text = await self._transcribe_pcm16_audio(
                raw_audio,
                capture=capture,
                on_delta=on_delta,
                timeout_seconds=final_timeout_seconds or self.final_timeout_seconds,
                source="microphone",
            )
            return await self._recover_and_finish_segmented_capture(capture, text)

        mic = self._open_microphone(mic)
        stream = self._iter_microphone_chunks(
            stop_event=stop_event,
            capture=capture,
            mic=mic,
            close_mic=close_mic,
            on_recording_stopped=on_recording_stopped,
        )
        text = await self._send_stream_with_recovery(
            stream,
            capture=capture,
            on_delta=on_delta,
            timeout_seconds=final_timeout_seconds or self.final_timeout_seconds,
        )
        return self._finish_capture(capture, text)

    def transcribe_microphone_blocking(
        self,
        stop_event: threading.Event,
        on_delta: DeltaCallback | None = None,
        final_timeout_seconds: float | None = None,
        mic: MicrophoneStream | None = None,
        close_mic: bool = True,
        on_recording_stopped: RecordingStoppedCallback | None = None,
    ) -> str:
        return asyncio.run(
            self.transcribe_microphone(
                stop_event=stop_event,
                on_delta=on_delta,
                final_timeout_seconds=final_timeout_seconds,
                mic=mic,
                close_mic=close_mic,
                on_recording_stopped=on_recording_stopped,
            )
        )

    async def transcribe_file(
        self,
        audio_path: Path,
        on_delta: DeltaCallback | None = None,
    ) -> str:
        self.check_ready_blocking()
        raw_audio = _audio_file_to_pcm16_bytes(audio_path, self.sample_rate)
        capture = self._new_capture(source="file", input_path=audio_path)
        self._configure_capture_segmentation(capture)
        capture.append_audio_chunk(raw_audio)
        if capture.effective_segment_max_seconds > 0:
            timeout_seconds = self.final_timeout_seconds
        else:
            timeout_seconds = max(
                self.final_timeout_seconds,
                len(raw_audio) / float(max(1, self.sample_rate * 2)) + 20.0,
            )
        text = await self._transcribe_pcm16_audio(
            raw_audio,
            capture=capture,
            on_delta=on_delta,
            timeout_seconds=timeout_seconds,
            source="file",
        )
        if capture.effective_segment_max_seconds > 0:
            return await self._recover_and_finish_segmented_capture(capture, text)
        return self._finish_capture(capture, text)

    def transcribe_file_blocking(
        self,
        audio_path: Path,
        on_delta: DeltaCallback | None = None,
    ) -> str:
        return asyncio.run(self.transcribe_file(audio_path, on_delta=on_delta))
