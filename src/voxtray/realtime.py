from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
import json
import logging
from pathlib import Path
import shlex
import subprocess
import threading

import websockets

from .audio import MicrophoneStream
from .config import VoxtrayConfig

DeltaCallback = Callable[[str], None]


class RealtimeError(RuntimeError):
    pass


class RealtimeTranscriber:
    def __init__(self, config: VoxtrayConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("voxtray.realtime")

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
            raise RealtimeError(str(msg.get("error", msg)))

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

    async def _capture_initial_chunk_before_commit(
        self,
        ws,
        mic: MicrophoneStream,
        grace_ms: int,
        generation_started: bool,
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
            if not generation_started:
                await self._start_generation(ws)
                generation_started = True
            await self._append_audio_chunk(ws, chunk)
            appended += 1
            break
        return appended, generation_started

    async def _collect_done_text(
        self,
        ws,
        deltas: list[str],
        on_delta: DeltaCallback | None,
        timeout_seconds: float,
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
            )
            if done:
                final_text = text
                break
        return final_text

    async def _flush_stop_tail(
        self,
        ws,
        mic: MicrophoneStream,
        deltas: list[str],
        on_delta: DeltaCallback | None,
        tail_ms: int,
        generation_started: bool,
    ) -> tuple[int, bool]:
        if tail_ms <= 0:
            return 0, generation_started

        loop = asyncio.get_running_loop()
        deadline = loop.time() + (tail_ms / 1000.0)
        appended = 0
        while loop.time() < deadline:
            chunk = mic.get_chunk(timeout=0.02)
            if chunk:
                if not generation_started:
                    await self._start_generation(ws)
                    generation_started = True
                await self._append_audio_chunk(ws, chunk)
                appended += 1

            await self._recv_once(
                ws,
                deltas,
                on_delta,
                timeout=0.002,
                accept_done=False,
            )
        return appended, generation_started

    async def transcribe_microphone(
        self,
        stop_event: threading.Event,
        on_delta: DeltaCallback | None = None,
        final_timeout_seconds: float | None = None,
    ) -> str:
        final_timeout = (
            final_timeout_seconds
            if final_timeout_seconds is not None
            else self.config.realtime.final_timeout_seconds
        )
        segment_max_seconds = self.config.realtime.segment_max_seconds
        segment_finalize_timeout = self.config.realtime.segment_finalize_timeout_seconds
        stop_tail_ms = self.config.realtime.stop_tail_ms
        first_chunk_grace_ms = self.config.realtime.first_chunk_grace_ms

        all_segments: list[str] = []
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
                loop = asyncio.get_running_loop()
                segment_started_at = loop.time()

                async with await self._connect() as ws:
                    await self._init_session(ws)

                    while True:
                        chunk = mic.get_chunk(timeout=0.05)
                        if chunk:
                            if not generation_started:
                                await self._start_generation(ws)
                                generation_started = True
                            await self._append_audio_chunk(ws, chunk)
                            sent_audio_chunks += 1

                        await self._recv_once(
                            ws,
                            deltas,
                            on_delta,
                            timeout=0.005,
                            accept_done=False,
                        )

                        if stop_event.is_set():
                            if sent_audio_chunks == 0:
                                captured, generation_started = await self._capture_initial_chunk_before_commit(
                                    ws,
                                    mic,
                                    grace_ms=first_chunk_grace_ms,
                                    generation_started=generation_started,
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
                            )
                            sent_audio_chunks += tail_appended
                            # Flush any remaining tail audio captured right before stop.
                            tail_chunks = mic.drain()
                            if tail_chunks:
                                if not generation_started:
                                    await self._start_generation(ws)
                                    generation_started = True
                                sent_audio_chunks += len(tail_chunks)
                                await self._append_audio_chunks(ws, tail_chunks)
                            break

                        if generation_started and segment_max_seconds > 0 and (
                            loop.time() - segment_started_at >= segment_max_seconds
                        ):
                            self.logger.debug(
                                "segment rollover after %.1fs",
                                loop.time() - segment_started_at,
                            )
                            break

                    if sent_audio_chunks == 0 or not generation_started:
                        self.logger.debug(
                            "skipping commit for empty segment (no captured audio)"
                        )
                        continue

                    await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

                    wait_seconds = (
                        final_timeout
                        if stop_event.is_set()
                        else segment_finalize_timeout
                    )
                    final_text = await self._collect_done_text(
                        ws,
                        deltas,
                        on_delta,
                        timeout_seconds=wait_seconds,
                    )

                segment_text = (final_text or "".join(deltas)).strip()
                if segment_text:
                    all_segments.append(segment_text)

                if stop_event.is_set():
                    break

            return "\n".join(all_segments)
        finally:
            mic.stop()

    def transcribe_microphone_blocking(
        self,
        stop_event: threading.Event,
        on_delta: DeltaCallback | None = None,
        final_timeout_seconds: float | None = None,
    ) -> str:
        return asyncio.run(
            self.transcribe_microphone(
                stop_event=stop_event,
                on_delta=on_delta,
                final_timeout_seconds=final_timeout_seconds,
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
        deltas: list[str] = []
        final_text: str | None = None

        async with await self._connect() as ws:
            await self._init_session(ws)
            await self._start_generation(ws)

            chunk_size = 4096
            for i in range(0, len(raw_audio), chunk_size):
                chunk = raw_audio[i : i + chunk_size]
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("utf-8"),
                        }
                    )
                )

            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

            max_wait_seconds = max(
                15.0,
                self.config.realtime.final_timeout_seconds,
                (len(raw_audio) / (16000 * 2)) + 10.0,
            )
            deadline = asyncio.get_running_loop().time() + max_wait_seconds
            while asyncio.get_running_loop().time() < deadline:
                remaining = max(0.0, deadline - asyncio.get_running_loop().time())
                done, text = await self._recv_once(
                    ws,
                    deltas,
                    on_delta,
                    timeout=min(2.0, remaining),
                )
                if done:
                    final_text = text
                    break

        if final_text:
            return final_text
        return "".join(deltas)

    def transcribe_file_blocking(
        self, audio_path: Path, on_delta: DeltaCallback | None = None
    ) -> str:
        self.logger.debug("transcribing file: %s", shlex.quote(str(audio_path)))
        return asyncio.run(self.transcribe_file(audio_path, on_delta=on_delta))
