from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
import shlex
import subprocess
import threading

import websockets

from .audio import MicrophoneStream
from .config import VoxtrayConfig

DeltaCallback = Callable[[str], None]


class RealtimeError(RuntimeError):
    def __init__(self, message: str, payload: object | None = None) -> None:
        super().__init__(message)
        self.payload = payload


@dataclass(slots=True)
class TranscriptionCapture:
    source: str
    sample_rate: int
    chunk_ms: int
    input_path: Path | None = None
    requested_segment_max_seconds: float = 0.0
    effective_segment_max_seconds: float = 0.0
    pcm16_audio: bytearray = field(default_factory=bytearray)
    segment_texts: list[str] = field(default_factory=list)
    raw_text: str = ""
    error_message: str = ""
    error_payload: object | None = None

    def append_audio_chunk(self, chunk: bytes) -> None:
        self.pcm16_audio.extend(chunk)

    def audio_bytes(self) -> bytes:
        return bytes(self.pcm16_audio)


class RealtimeTranscriber:
    def __init__(self, config: VoxtrayConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("voxtray.realtime")
        self._last_capture: TranscriptionCapture | None = None

    @property
    def last_capture(self) -> TranscriptionCapture | None:
        return self._last_capture

    def _audio_sample_rate(self) -> int:
        return int(getattr(getattr(self.config, "audio", None), "sample_rate", 16000))

    def _audio_chunk_ms(self) -> int:
        return int(getattr(getattr(self.config, "audio", None), "chunk_ms", 40))

    def _audio_chunk_seconds(self) -> float:
        return max(0.001, self._audio_chunk_ms() / 1000.0)

    def _new_capture(self, source: str, input_path: Path | None = None) -> TranscriptionCapture:
        capture = TranscriptionCapture(
            source=source,
            sample_rate=self._audio_sample_rate(),
            chunk_ms=self._audio_chunk_ms(),
            input_path=input_path,
        )
        self._last_capture = capture
        return capture

    def _engine_extra_args(self) -> list[str]:
        return list(getattr(getattr(self.config, "engine", None), "extra_args", []) or [])

    def _engine_arg_value(self, flag: str) -> str | None:
        extra_args = self._engine_extra_args()
        for index, arg in enumerate(extra_args):
            if arg == flag and index + 1 < len(extra_args):
                return extra_args[index + 1]
            if arg.startswith(flag + "="):
                return arg.split("=", 1)[1]
        return None

    def _safe_segment_max_seconds(self) -> float | None:
        raw_max_model_len = self._engine_arg_value("--max-model-len")
        if not raw_max_model_len:
            return None
        try:
            max_model_len = int(raw_max_model_len)
        except ValueError:
            return None
        if max_model_len <= 0:
            return None
        # Voxtral realtime consumes roughly one audio token per 20ms. Keep
        # headroom to avoid saturating KV cache on long recordings.
        return max(5.0, max_model_len * 0.02 * 0.8)

    def _effective_segment_max_seconds(self, requested_seconds: float) -> float:
        safe_seconds = self._safe_segment_max_seconds()
        if safe_seconds is None:
            return requested_seconds
        if requested_seconds <= 0:
            return safe_seconds
        return min(requested_seconds, safe_seconds)

    @staticmethod
    def _live_finalize_wait_seconds(
        *,
        audio_seconds: float,
        configured_seconds: float,
        final_segment: bool,
    ) -> float:
        base = max(0.5, configured_seconds)
        dynamic = 4.0 + (max(0.0, audio_seconds) * 0.5)
        if final_segment:
            dynamic += 2.0
        return max(base, min(30.0, dynamic))

    @staticmethod
    def _format_error_message(msg: dict[str, object]) -> str:
        error = msg.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "").strip()
            code = str(error.get("code") or "").strip()
            err_type = str(error.get("type") or "").strip()
            for candidate in (message, code, err_type):
                if candidate:
                    return candidate
        elif isinstance(error, str) and error.strip():
            return error.strip()

        for key in ("message", "detail", "code"):
            candidate = str(msg.get(key) or "").strip()
            if candidate:
                return candidate
        return json.dumps(msg, ensure_ascii=False, sort_keys=True)

    async def _connect(self):
        return await websockets.connect(
            self.config.websocket_url,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        )

    async def _init_session(self, ws) -> None:
        raw = await ws.recv()
        msg = json.loads(raw)
        if msg.get("type") != "session.created":
            raise RealtimeError(f"unexpected first event: {msg}")

        await ws.send(
            json.dumps({"type": "session.update", "model": self.config.model_id})
        )

    async def _recv_once(
        self,
        ws,
        deltas: list[str],
        on_delta: DeltaCallback | None,
        timeout: float,
        accept_done: bool = True,
        capture: TranscriptionCapture | None = None,
    ) -> tuple[bool, str | None]:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            return False, None

        msg = json.loads(raw)
        msg_type = msg.get("type")

        if msg_type == "transcription.delta":
            delta = msg.get("delta", "")
            if delta:
                deltas.append(delta)
                if on_delta:
                    on_delta(delta)
            return False, None

        if msg_type == "transcription.done":
            if not accept_done:
                return False, None
            text = msg.get("text")
            return True, text

        if msg_type == "error":
            message = self._format_error_message(msg)
            if capture is not None:
                capture.error_message = message
                capture.error_payload = msg
            raise RealtimeError(message, payload=msg)

        return False, None

    async def _append_audio_chunk(self, ws, chunk: bytes) -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("utf-8"),
                }
            )
        )

    async def _start_generation(self, ws) -> None:
        # vLLM realtime starts generation on non-final commit.
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))

    async def _append_audio_chunks(self, ws, chunks: list[bytes]) -> None:
        for chunk in chunks:
            await self._append_audio_chunk(ws, chunk)

    @staticmethod
    def _record_audio_chunk(capture: TranscriptionCapture | None, chunk: bytes | None) -> None:
        if capture is None or not chunk:
            return
        capture.append_audio_chunk(chunk)

    async def _append_chunk_and_maybe_start_generation(
        self,
        ws,
        chunk: bytes,
        generation_started: bool,
    ) -> bool:
        # vLLM realtime can crash if generation starts before the first audio
        # chunk has been appended and converted into multimodal embeddings.
        await self._append_audio_chunk(ws, chunk)
        if generation_started:
            return True
        await self._start_generation(ws)
        return True

    async def _capture_initial_chunk_before_commit(
        self,
        ws,
        mic: MicrophoneStream,
        grace_ms: int,
        generation_started: bool,
        capture: TranscriptionCapture | None = None,
    ) -> tuple[int, bool]:
        """Best effort: when user stops quickly, wait briefly for first mic frame."""
        if grace_ms <= 0:
            return 0, generation_started

        appended = 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (grace_ms / 1000.0)
        while loop.time() < deadline:
            chunk = mic.get_chunk(timeout=0.02)
            if not chunk:
                continue
            self._record_audio_chunk(capture, chunk)
            generation_started = await self._append_chunk_and_maybe_start_generation(
                ws,
                chunk,
                generation_started,
            )
            appended += 1
            break
        return appended, generation_started

    async def _collect_done_text(
        self,
        ws,
        deltas: list[str],
        on_delta: DeltaCallback | None,
        timeout_seconds: float,
        capture: TranscriptionCapture | None = None,
        allow_empty_timeout: bool = False,
    ) -> str | None:
        if timeout_seconds <= 0:
            return None

        final_text: str | None = None
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            done, text = await self._recv_once(
                ws,
                deltas,
                on_delta,
                timeout=0.5,
                accept_done=True,
                capture=capture,
            )
            if done:
                final_text = text
                break
        if final_text is None and not deltas:
            if allow_empty_timeout:
                self.logger.warning(
                    "timed out waiting for transcription.done after %.1fs; treating as empty segment",
                    timeout_seconds,
                )
                return ""
            message = (
                f"timed out waiting for transcription.done after {timeout_seconds:.1f}s"
            )
            if capture is not None:
                capture.error_message = message
            raise RealtimeError(message)
        return final_text

    async def _flush_stop_tail(
        self,
        ws,
        mic: MicrophoneStream,
        deltas: list[str],
        on_delta: DeltaCallback | None,
        tail_ms: int,
        generation_started: bool,
        capture: TranscriptionCapture | None = None,
    ) -> tuple[int, bool]:
        if tail_ms <= 0:
            return 0, generation_started

        loop = asyncio.get_running_loop()
        idle_extension_ms = max(tail_ms, self._audio_chunk_ms() * 3)
        hard_cap_ms = max(idle_extension_ms, tail_ms * 5, 1200)
        hard_deadline = loop.time() + (hard_cap_ms / 1000.0)
        quiet_deadline = loop.time() + (tail_ms / 1000.0)
        appended = 0
        while loop.time() < hard_deadline:
            chunk = mic.get_chunk(timeout=0.02)
            if chunk:
                self._record_audio_chunk(capture, chunk)
                generation_started = await self._append_chunk_and_maybe_start_generation(
                    ws,
                    chunk,
                    generation_started,
                )
                appended += 1
                quiet_deadline = min(
                    hard_deadline,
                    loop.time() + (idle_extension_ms / 1000.0),
                )

            await self._recv_once(
                ws,
                deltas,
                on_delta,
                timeout=0.002,
                accept_done=False,
                capture=capture,
            )
            if loop.time() >= quiet_deadline:
                break
        return appended, generation_started

    def _split_pcm16_audio(self, raw_audio: bytes, max_seconds: float) -> list[bytes]:
        if not raw_audio:
            return []
        if max_seconds <= 0:
            return [raw_audio]
        bytes_per_second = max(1, self._audio_sample_rate() * 2)
        segment_bytes = max(2, int(max_seconds * bytes_per_second))
        if segment_bytes % 2 != 0:
            segment_bytes -= 1
        segment_bytes = max(2, segment_bytes)
        return [
            raw_audio[offset : offset + segment_bytes]
            for offset in range(0, len(raw_audio), segment_bytes)
        ]

    def _tail_pcm16_audio(self, raw_audio: bytes, seconds: float) -> bytes:
        if not raw_audio or seconds <= 0:
            return b""
        bytes_per_second = max(1, self._audio_sample_rate() * 2)
        tail_bytes = max(2, int(seconds * bytes_per_second))
        if tail_bytes % 2 != 0:
            tail_bytes -= 1
        return raw_audio[-tail_bytes:]

    @staticmethod
    def _normalized_words(text: str) -> list[str]:
        return [
            re.sub(r"[^\w]+", "", token.casefold())
            for token in text.split()
            if re.sub(r"[^\w]+", "", token.casefold())
        ]

    @classmethod
    def _merge_tail_text(cls, existing_text: str, tail_text: str) -> str:
        existing = existing_text.strip()
        tail = tail_text.strip()
        if not existing:
            return tail
        if not tail:
            return existing
        if tail in existing:
            return existing

        existing_words = existing.split()
        tail_words = tail.split()
        existing_norm = cls._normalized_words(existing)
        tail_norm = cls._normalized_words(tail)
        max_overlap = min(len(existing_norm), len(tail_norm))

        for size in range(max_overlap, 0, -1):
            if existing_norm[-size:] != tail_norm[:size]:
                continue
            remainder = " ".join(tail_words[size:]).strip()
            if not remainder:
                return existing
            separator = "" if existing.endswith(("\n", " ", "-", "/")) else " "
            return f"{existing}{separator}{remainder}"

        return f"{existing}\n{tail}"

    async def _transcribe_pcm16_audio(
        self,
        raw_audio: bytes,
        *,
        timeout_seconds: float,
    ) -> str:
        if not raw_audio:
            return ""

        deltas: list[str] = []
        final_text: str | None = None
        generation_started = False

        async with await self._connect() as ws:
            await self._init_session(ws)

            chunk_size = 4096
            for i in range(0, len(raw_audio), chunk_size):
                chunk = raw_audio[i : i + chunk_size]
                generation_started = await self._append_chunk_and_maybe_start_generation(
                    ws,
                    chunk,
                    generation_started,
                )

            if not generation_started:
                return ""

            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
            final_text = await self._collect_done_text(
                ws,
                deltas,
                None,
                timeout_seconds=timeout_seconds,
            )

        return (final_text or "".join(deltas)).strip()

    async def _recover_final_tail_text(
        self,
        capture: TranscriptionCapture,
        existing_text: str,
    ) -> tuple[str, str]:
        tail_audio = self._tail_pcm16_audio(capture.audio_bytes(), seconds=8.0)
        if not tail_audio:
            return existing_text, ""

        tail_seconds = len(tail_audio) / max(1, self._audio_sample_rate() * 2)
        timeout_seconds = max(15.0, tail_seconds + 10.0)
        try:
            tail_text = await self._transcribe_pcm16_audio(
                tail_audio,
                timeout_seconds=timeout_seconds,
            )
        except RealtimeError as exc:
            self.logger.warning("final tail recovery failed: %s", exc)
            return existing_text, ""

        merged_text = self._merge_tail_text(existing_text, tail_text)
        if merged_text != existing_text:
            self.logger.info(
                "recovered final tail using %.1fs context: %s",
                tail_seconds,
                tail_text,
            )
        return merged_text, tail_text

    async def transcribe_microphone(
        self,
        stop_event: threading.Event,
        on_delta: DeltaCallback | None = None,
        final_timeout_seconds: float | None = None,
        mic: MicrophoneStream | None = None,
        close_mic: bool = True,
    ) -> str:
        final_timeout = (
            final_timeout_seconds
            if final_timeout_seconds is not None
            else self.config.realtime.final_timeout_seconds
        )
        requested_segment_max_seconds = self.config.realtime.segment_max_seconds
        segment_max_seconds = self._effective_segment_max_seconds(
            requested_segment_max_seconds
        )
        segment_finalize_timeout = self.config.realtime.segment_finalize_timeout_seconds
        stop_tail_ms = self.config.realtime.stop_tail_ms
        first_chunk_grace_ms = self.config.realtime.first_chunk_grace_ms

        all_segments: list[str] = []
        capture = self._new_capture(source="microphone")
        capture.requested_segment_max_seconds = requested_segment_max_seconds
        capture.effective_segment_max_seconds = segment_max_seconds
        if mic is None:
            mic = MicrophoneStream(
                sample_rate=self.config.audio.sample_rate,
                chunk_ms=self.config.audio.chunk_ms,
                device=self.config.audio.device,
                max_queue_chunks=self.config.realtime.mic_queue_chunks,
            )
        mic.start()
        try:
            while True:
                deltas: list[str] = []
                final_text: str | None = None
                sent_audio_chunks = 0
                generation_started = False

                async with await self._connect() as ws:
                    await self._init_session(ws)

                    while True:
                        chunk = mic.get_chunk(timeout=0.05)
                        if chunk:
                            self._record_audio_chunk(capture, chunk)
                            generation_started = await self._append_chunk_and_maybe_start_generation(
                                ws,
                                chunk,
                                generation_started,
                            )
                            sent_audio_chunks += 1

                        await self._recv_once(
                            ws,
                            deltas,
                            on_delta,
                            timeout=0.005,
                            accept_done=False,
                            capture=capture,
                        )

                        if stop_event.is_set():
                            if sent_audio_chunks == 0:
                                captured, generation_started = await self._capture_initial_chunk_before_commit(
                                    ws,
                                    mic,
                                    grace_ms=first_chunk_grace_ms,
                                    generation_started=generation_started,
                                    capture=capture,
                                )
                                sent_audio_chunks += captured
                            # Capture a short tail to avoid dropping the last syllable/word.
                            tail_appended, generation_started = await self._flush_stop_tail(
                                ws,
                                mic,
                                deltas,
                                on_delta,
                                tail_ms=stop_tail_ms,
                                generation_started=generation_started,
                                capture=capture,
                            )
                            sent_audio_chunks += tail_appended
                            # Flush any remaining tail audio captured right before stop.
                            tail_chunks = mic.drain()
                            if tail_chunks:
                                sent_audio_chunks += len(tail_chunks)
                                for index, tail_chunk in enumerate(tail_chunks):
                                    self._record_audio_chunk(capture, tail_chunk)
                                    generation_started = await self._append_chunk_and_maybe_start_generation(
                                        ws,
                                        tail_chunk,
                                        generation_started if index > 0 else generation_started,
                                    )
                            break

                        segment_audio_seconds = sent_audio_chunks * self._audio_chunk_seconds()
                        if (
                            generation_started
                            and segment_max_seconds > 0
                            and segment_audio_seconds >= segment_max_seconds
                        ):
                            self.logger.debug(
                                "segment rollover after %.1fs of audio (effective max %.1fs, requested %.1fs)",
                                segment_audio_seconds,
                                segment_max_seconds,
                                requested_segment_max_seconds,
                            )
                            break

                    if sent_audio_chunks == 0 or not generation_started:
                        self.logger.debug(
                            "skipping commit for empty segment (no captured audio)"
                        )
                        if stop_event.is_set():
                            break
                        continue

                    await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

                    segment_audio_seconds = sent_audio_chunks * self._audio_chunk_seconds()
                    wait_seconds = self._live_finalize_wait_seconds(
                        audio_seconds=segment_audio_seconds,
                        configured_seconds=(
                            final_timeout
                            if stop_event.is_set()
                            else segment_finalize_timeout
                        ),
                        final_segment=stop_event.is_set(),
                    )
                    allow_empty_timeout = bool(stop_event.is_set()) or (
                        segment_max_seconds > 0
                        and segment_audio_seconds >= (segment_max_seconds * 0.9)
                    )
                    final_text = await self._collect_done_text(
                        ws,
                        deltas,
                        on_delta,
                        timeout_seconds=wait_seconds,
                        capture=capture,
                        allow_empty_timeout=allow_empty_timeout,
                    )

                segment_text = (final_text or "".join(deltas)).strip()
                if segment_text:
                    all_segments.append(segment_text)
                    capture.segment_texts.append(segment_text)
                elif stop_event.is_set() and sent_audio_chunks > 0 and all_segments:
                    existing_text = "\n".join(all_segments)
                    recovered_text, tail_text = await self._recover_final_tail_text(
                        capture,
                        existing_text,
                    )
                    if recovered_text != existing_text:
                        all_segments = [recovered_text]
                        capture.segment_texts.append(tail_text)

                if stop_event.is_set():
                    break

            final_text_value = "\n".join(all_segments)
            capture.raw_text = final_text_value
            return final_text_value
        except RealtimeError as exc:
            capture.raw_text = "\n".join(all_segments)
            if not capture.error_message:
                capture.error_message = str(exc)
            if capture.error_payload is None and getattr(exc, "payload", None) is not None:
                capture.error_payload = exc.payload
            raise
        finally:
            if close_mic:
                mic.stop()

    def transcribe_microphone_blocking(
        self,
        stop_event: threading.Event,
        on_delta: DeltaCallback | None = None,
        final_timeout_seconds: float | None = None,
        mic: MicrophoneStream | None = None,
        close_mic: bool = True,
    ) -> str:
        return asyncio.run(
            self.transcribe_microphone(
                stop_event=stop_event,
                on_delta=on_delta,
                final_timeout_seconds=final_timeout_seconds,
                mic=mic,
                close_mic=close_mic,
            )
        )

    @staticmethod
    def _audio_file_to_pcm16_bytes(audio_path: Path) -> bytes:
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
            "16000",
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

    async def transcribe_file(
        self, audio_path: Path, on_delta: DeltaCallback | None = None
    ) -> str:
        raw_audio = self._audio_file_to_pcm16_bytes(audio_path)
        capture = self._new_capture(source="file", input_path=audio_path)
        capture.append_audio_chunk(raw_audio)
        requested_segment_max_seconds = self.config.realtime.segment_max_seconds
        effective_segment_max_seconds = self._effective_segment_max_seconds(
            requested_segment_max_seconds
        )
        capture.requested_segment_max_seconds = requested_segment_max_seconds
        capture.effective_segment_max_seconds = effective_segment_max_seconds

        if not raw_audio:
            capture.raw_text = ""
            return ""

        try:
            segment_texts: list[str] = []
            for segment_audio in self._split_pcm16_audio(
                raw_audio, effective_segment_max_seconds
            ):
                deltas: list[str] = []
                final_text: str | None = None
                generation_started = False

                async with await self._connect() as ws:
                    await self._init_session(ws)

                    chunk_size = 4096
                    for i in range(0, len(segment_audio), chunk_size):
                        chunk = segment_audio[i : i + chunk_size]
                        generation_started = await self._append_chunk_and_maybe_start_generation(
                            ws,
                            chunk,
                            generation_started,
                        )

                    if not generation_started:
                        continue

                    await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

                    max_wait_seconds = max(
                        15.0,
                        self.config.realtime.final_timeout_seconds,
                        (len(segment_audio) / (self._audio_sample_rate() * 2)) + 10.0,
                    )
                    final_text = await self._collect_done_text(
                        ws,
                        deltas,
                        on_delta,
                        timeout_seconds=max_wait_seconds,
                        capture=capture,
                    )

                segment_text = (final_text or "".join(deltas)).strip()
                if segment_text:
                    segment_texts.append(segment_text)
                    capture.segment_texts.append(segment_text)

            capture.raw_text = "\n".join(segment_texts)
            return capture.raw_text
        except RealtimeError as exc:
            if not capture.error_message:
                capture.error_message = str(exc)
            if capture.error_payload is None and getattr(exc, "payload", None) is not None:
                capture.error_payload = exc.payload
            raise

    def transcribe_file_blocking(
        self, audio_path: Path, on_delta: DeltaCallback | None = None
    ) -> str:
        self.logger.debug("transcribing file: %s", shlex.quote(str(audio_path)))
        return asyncio.run(self.transcribe_file(audio_path, on_delta=on_delta))
