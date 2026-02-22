from __future__ import annotations

import queue
from typing import Any, Optional


class MicrophoneStream:
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_ms: int = 40,
        device: str | None = "default",
        max_queue_chunks: int = 256,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.device = None if device in {None, "", "default"} else device
        self.blocksize = int(self.sample_rate * self.chunk_ms / 1000)
        self.queue: queue.Queue[bytes] = queue.Queue(maxsize=max_queue_chunks)
        self._stream: Optional[Any] = None

    def _callback(self, indata, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
        del frames, time_info
        if status:
            # Drop status-only events, audio still usable in most cases.
            pass
        payload = bytes(indata)
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            # Keep the most recent audio by discarding oldest chunk.
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(payload)
            except queue.Full:
                pass

    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
        except OSError as exc:
            raise RuntimeError(
                "PortAudio is not available. Install libportaudio2 (Ubuntu) to record from microphone."
            ) from exc
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype="int16",
            channels=1,
            callback=self._callback,
            device=self.device,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def get_chunk(self, timeout: float = 0.05) -> bytes | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> list[bytes]:
        chunks: list[bytes] = []
        while True:
            try:
                chunks.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return chunks
