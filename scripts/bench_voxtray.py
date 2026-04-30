#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime
import difflib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import httpx

from voxtray.config import VoxtrayConfig, load_config
from voxtray.realtime import RealtimeError, RealtimeTranscriber


DEFAULT_RECORDINGS_ROOT = Path.home() / ".local/share/voxtray/recordings"
DEFAULT_OUTPUT_ROOT = Path("benchmarks/voxtray")


@dataclass(frozen=True)
class AudioCase:
    name: str
    audio_path: Path
    result_path: Path | None
    duration_seconds: float
    reference_text: str


@dataclass(frozen=True)
class Variant:
    name: str
    max_model_len: int
    max_num_batched_tokens: int = 512
    gpu_memory_utilization: float = 0.68
    enforce_eager: bool = True
    disable_compile_cache: bool = True
    final_timeout_seconds: float = 8.0
    segment_finalize_timeout_seconds: float = 6.0
    segment_max_seconds: int = 90
    accept_partial_timeouts: bool = False


class SimulatedMicrophoneStream:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        chunk_seconds: float,
        speed: float,
    ) -> None:
        self.chunks = chunks
        self.chunk_seconds = max(0.001, chunk_seconds)
        self.speed = max(0.0, speed)
        self.index = 0
        self.started_at = 0.0
        self.closed = False

    def start(self) -> None:
        self.started_at = time.perf_counter()

    def stop(self) -> None:
        self.closed = True

    def get_chunk(self, timeout: float = 0.05) -> bytes | None:
        if self.index >= len(self.chunks):
            time.sleep(max(0.0, min(timeout, 0.01)))
            return None
        if self.speed > 0:
            due_at = self.started_at + ((self.index * self.chunk_seconds) / self.speed)
            remaining = due_at - time.perf_counter()
            if remaining > 0:
                time.sleep(max(0.0, min(timeout, remaining)))
                if time.perf_counter() < due_at:
                    return None
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk

    def drain(self) -> list[bytes]:
        if self.index >= len(self.chunks):
            return []
        remaining = self.chunks[self.index :]
        self.index = len(self.chunks)
        return remaining


class VramSampler:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not shutil_which("nvidia-smi"):
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return max(self.samples) if self.samples else None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=2.0,
                )
                for line in proc.stdout.splitlines():
                    line = line.strip()
                    if line:
                        self.samples.append(int(line))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self._stop.wait(self.interval_seconds)


def shutil_which(command: str) -> str | None:
    path = os.environ.get("PATH", "")
    for directory in path.split(os.pathsep):
        candidate = Path(directory) / command
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def normalize_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def similarity(reference: str, candidate: str) -> float:
    reference_norm = normalize_text(reference)
    candidate_norm = normalize_text(candidate)
    if not reference_norm and not candidate_norm:
        return 1.0
    if not reference_norm or not candidate_norm:
        return 0.0
    return difflib.SequenceMatcher(None, reference_norm, candidate_norm).ratio()


def audio_duration_seconds(audio_path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        return round(float(proc.stdout.strip()), 3)
    except ValueError:
        return 0.0


def load_audio_case(audio_path: Path) -> AudioCase | None:
    result_path = audio_path.with_name("result.json")
    result: dict[str, Any] = {}
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = {}
    if result.get("status") != "success":
        return None
    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    if diagnostics.get("completion_status") != "complete":
        return None

    reference_text = str(result.get("normalized_text") or result.get("raw_text") or "").strip()
    if not reference_text:
        return None

    duration = float(result.get("audio_duration_seconds") or 0.0)
    if duration <= 0:
        duration = audio_duration_seconds(audio_path)
    if duration <= 0:
        return None

    parent = audio_path.parent
    return AudioCase(
        name=f"{parent.parent.parent.name}-{parent.parent.name}-{parent.name}",
        audio_path=audio_path,
        result_path=result_path if result_path.exists() else None,
        duration_seconds=duration,
        reference_text=reference_text,
    )


def discover_corpus(recordings_root: Path, limit: int) -> list[AudioCase]:
    candidates: list[AudioCase] = []
    for audio_path in sorted(recordings_root.rglob("audio.wav")):
        case = load_audio_case(audio_path)
        if case is not None:
            candidates.append(case)
    if not candidates:
        return []

    bins = [
        (0, 25),
        (25, 75),
        (75, 130),
        (130, 240),
        (240, 900),
    ]
    selected: list[AudioCase] = []
    seen_paths: set[Path] = set()
    for low, high in bins:
        in_bin = [
            case
            for case in candidates
            if low <= case.duration_seconds < high and case.audio_path not in seen_paths
        ]
        if not in_bin:
            continue
        target = (low + high) / 2
        case = min(in_bin, key=lambda item: abs(item.duration_seconds - target))
        selected.append(case)
        seen_paths.add(case.audio_path)
        if len(selected) >= limit:
            return selected

    for case in sorted(candidates, key=lambda item: item.duration_seconds):
        if case.audio_path in seen_paths:
            continue
        selected.append(case)
        seen_paths.add(case.audio_path)
        if len(selected) >= limit:
            break
    return selected


def quick_variants() -> list[Variant]:
    return [
        Variant(name="len1024_batch512", max_model_len=1024),
        Variant(name="len1152_batch512", max_model_len=1152),
        Variant(name="len1280_batch512", max_model_len=1280),
        Variant(name="len1536_batch512", max_model_len=1536),
        Variant(name="len1792_batch512", max_model_len=1792),
        Variant(name="len2048_batch512", max_model_len=2048),
        Variant(name="len1536_batch384", max_model_len=1536, max_num_batched_tokens=384),
        Variant(name="len1536_partial", max_model_len=1536, accept_partial_timeouts=True),
    ]


def set_flag(args: list[str], flag: str, value: str) -> list[str]:
    updated: list[str] = []
    skip_next = False
    found = False
    for index, item in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if item == flag:
            updated.extend([flag, value])
            found = True
            if index + 1 < len(args):
                skip_next = True
            continue
        if item.startswith(flag + "="):
            updated.extend([flag, value])
            found = True
            continue
        updated.append(item)
    if not found:
        updated.extend([flag, value])
    return updated


def apply_variant(base_config: VoxtrayConfig, variant: Variant, port: int) -> VoxtrayConfig:
    config = copy.deepcopy(base_config)
    config.server.port = port
    config.server.external_base_url = ""
    config.server.start_timeout_seconds = max(config.server.start_timeout_seconds, 900)
    config.engine.enforce_eager = variant.enforce_eager
    config.engine.disable_compile_cache = variant.disable_compile_cache
    config.realtime.final_timeout_seconds = variant.final_timeout_seconds
    config.realtime.segment_finalize_timeout_seconds = variant.segment_finalize_timeout_seconds
    config.realtime.segment_max_seconds = variant.segment_max_seconds

    extra_args = list(config.engine.extra_args)
    extra_args = set_flag(extra_args, "--gpu-memory-utilization", str(variant.gpu_memory_utilization))
    extra_args = set_flag(extra_args, "--max-model-len", str(variant.max_model_len))
    extra_args = set_flag(extra_args, "--max-num-batched-tokens", str(variant.max_num_batched_tokens))
    extra_args = set_flag(extra_args, "--max-num-seqs", "1")
    extra_args = set_flag(extra_args, "--disable-access-log-for-endpoints", "/v1/models")
    config.engine.extra_args = extra_args
    return config


def build_vllm_command(config: VoxtrayConfig) -> list[str]:
    command = [
        config.engine.command,
        "serve",
        config.model_id,
        "--host",
        config.server.host,
        "--port",
        str(config.server.port),
        "--compilation_config",
        config.engine.compilation_config,
    ]
    if config.engine.enforce_eager:
        command.append("--enforce-eager")
    command.extend(config.engine.extra_args)
    return command


def server_ready(config: VoxtrayConfig, timeout_seconds: float = 2.0) -> bool:
    try:
        response = httpx.get(f"{config.server_base_url}/v1/models", timeout=timeout_seconds)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def wait_for_server(config: VoxtrayConfig, proc: subprocess.Popen[bytes]) -> float:
    start = time.perf_counter()
    deadline = start + config.server.start_timeout_seconds
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with code {proc.returncode}")
        if server_ready(config):
            return time.perf_counter() - start
        time.sleep(1.0)
    raise RuntimeError(f"vLLM not ready after {config.server.start_timeout_seconds}s")


def start_engine(config: VoxtrayConfig, log_path: Path) -> tuple[subprocess.Popen[bytes], float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_vllm_command(config)
    env = os.environ.copy()
    env["VLLM_HOST_IP"] = config.server.host if config.server.host else "127.0.0.1"
    if config.engine.disable_compile_cache:
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    log_file = log_path.open("ab")
    try:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    except Exception:
        log_file.close()
        raise
    try:
        load_seconds = wait_for_server(config, proc)
    except Exception:
        stop_engine(proc)
        raise
    return proc, load_seconds


def stop_engine(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.perf_counter() + 15.0
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def summarize_capture(capture: Any) -> dict[str, Any]:
    if capture is None:
        return {
            "segments": 0,
            "timeouts": 0,
            "retries": 0,
            "recovered_segments": 0,
            "empty_segments": 0,
            "completion_status": "",
        }
    segments = list(getattr(capture, "segments", []) or [])
    timeouts = 0
    retries = 0
    recovered = 0
    empty = 0
    for segment in segments:
        attempts = segment.get("attempts") if isinstance(segment, dict) else []
        if isinstance(attempts, list):
            retries += max(0, len(attempts) - 1)
            for attempt in attempts:
                error = str(attempt.get("error", "")) if isinstance(attempt, dict) else ""
                if "timed out waiting" in error:
                    timeouts += 1
        status = str(segment.get("status", "")) if isinstance(segment, dict) else ""
        if status == "recovered":
            recovered += 1
        if status == "empty":
            empty += 1
    return {
        "segments": len(segments),
        "timeouts": timeouts,
        "retries": retries,
        "recovered_segments": recovered,
        "empty_segments": empty,
        "completion_status": getattr(capture, "completion_status", ""),
    }


def enable_partial_timeout_acceptance(transcriber: RealtimeTranscriber) -> None:
    original_retry = transcriber._retry_segment_audio

    async def _retry_or_accept_partial(
        raw_audio: bytes,
        *,
        timeout_seconds: float,
        segment: dict[str, object],
        on_delta=None,
        allow_empty_retry: bool = False,
    ) -> str:
        error = str(segment.get("error") or "")
        marker = " after receiving partial transcript: "
        if marker in error:
            text = error.split(marker, 1)[1].strip()
            if text:
                transcriber._record_segment_attempt(
                    segment,
                    attempt=2,
                    status="partial_timeout",
                    timeout_seconds=timeout_seconds,
                    text=text,
                )
                segment["status"] = "recovered"
                segment["text_chars"] = len(text)
                segment["recovered"] = True
                segment["accepted_partial_timeout"] = True
                return text
        return await original_retry(
            raw_audio,
            timeout_seconds=timeout_seconds,
            segment=segment,
            on_delta=on_delta,
            allow_empty_retry=allow_empty_retry,
        )

    transcriber._retry_segment_audio = _retry_or_accept_partial  # type: ignore[method-assign]


def base_result_row(
    *,
    variant: Variant,
    case: AudioCase,
    mode: str,
    wall_seconds: float,
    status: str,
    error: str,
    text: str,
    peak_vram_mb: int | None,
    capture: Any,
    post_stop_seconds: float | None = None,
) -> dict[str, Any]:
    capture_summary = summarize_capture(capture)
    return {
        "variant": variant.name,
        "mode": mode,
        "audio": case.name,
        "audio_path": str(case.audio_path),
        "duration_seconds": round(case.duration_seconds, 3),
        "wall_seconds": round(wall_seconds, 3),
        "post_stop_seconds": round(post_stop_seconds, 3)
        if post_stop_seconds is not None
        else None,
        "rtf": round(wall_seconds / case.duration_seconds, 4)
        if case.duration_seconds
        else None,
        "status": status,
        "error": error,
        "text_chars": len(text.strip()),
        "transcript_path": "",
        "reference_chars": len(case.reference_text),
        "similarity": round(similarity(case.reference_text, text), 4),
        "peak_vram_mb": peak_vram_mb,
        "_transcript": text,
        **capture_summary,
    }


def benchmark_file_case(config: VoxtrayConfig, variant: Variant, case: AudioCase) -> dict[str, Any]:
    transcriber = RealtimeTranscriber(config)
    if variant.accept_partial_timeouts:
        enable_partial_timeout_acceptance(transcriber)
    sampler = VramSampler()
    sampler.start()
    start = time.perf_counter()
    error = ""
    text = ""
    try:
        text = transcriber.transcribe_file_blocking(case.audio_path)
        status = "success"
    except RealtimeError as exc:
        status = "error"
        error = str(exc)
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    wall_seconds = time.perf_counter() - start
    peak_vram_mb = sampler.stop()
    return base_result_row(
        variant=variant,
        case=case,
        mode="file",
        wall_seconds=wall_seconds,
        status=status,
        error=error,
        text=text,
        peak_vram_mb=peak_vram_mb,
        capture=transcriber.last_capture,
    )


def benchmark_live_case(
    config: VoxtrayConfig,
    variant: Variant,
    case: AudioCase,
    *,
    live_speed: float,
) -> dict[str, Any]:
    transcriber = RealtimeTranscriber(config)
    if variant.accept_partial_timeouts:
        enable_partial_timeout_acceptance(transcriber)
    raw_audio = transcriber._audio_file_to_pcm16_bytes(case.audio_path)
    bytes_per_chunk = int(config.audio.sample_rate * 2 * (config.audio.chunk_ms / 1000.0))
    bytes_per_chunk = max(2, bytes_per_chunk - (bytes_per_chunk % 2))
    chunks = [
        raw_audio[offset : offset + bytes_per_chunk]
        for offset in range(0, len(raw_audio), bytes_per_chunk)
        if raw_audio[offset : offset + bytes_per_chunk]
    ]
    mic = SimulatedMicrophoneStream(
        chunks,
        chunk_seconds=max(0.001, config.audio.chunk_ms / 1000.0),
        speed=live_speed,
    )
    stop_event = threading.Event()
    stopped_at: list[float] = []

    def stop_after_audio() -> None:
        delay = case.duration_seconds / live_speed if live_speed > 0 else 0.0
        time.sleep(delay)
        stopped_at.append(time.perf_counter())
        stop_event.set()

    stopper = threading.Thread(target=stop_after_audio, daemon=True)
    sampler = VramSampler()
    sampler.start()
    start = time.perf_counter()
    stopper.start()
    error = ""
    text = ""
    try:
        text = transcriber.transcribe_microphone_blocking(
            stop_event,
            mic=mic,  # type: ignore[arg-type]
            close_mic=True,
        )
        status = "success"
    except RealtimeError as exc:
        status = "error"
        error = str(exc)
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    finished_at = time.perf_counter()
    wall_seconds = finished_at - start
    peak_vram_mb = sampler.stop()
    post_stop_seconds = (
        finished_at - stopped_at[0]
        if stopped_at
        else None
    )
    return base_result_row(
        variant=variant,
        case=case,
        mode="live",
        wall_seconds=wall_seconds,
        post_stop_seconds=post_stop_seconds,
        status=status,
        error=error,
        text=text,
        peak_vram_mb=peak_vram_mb,
        capture=transcriber.last_capture,
    )


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    csv_path = output_dir / "results.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary_path = output_dir / "summary.md"
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Voxtray Benchmark\n\n")
        handle.write("| variant | mode | audios | avg rtf | p95 wall | p95 post-stop | min similarity | timeouts | retries | peak vram MB |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for variant_name, variant_rows in by_variant.items():
            rtfs = sorted(float(row["rtf"] or 0.0) for row in variant_rows)
            walls = sorted(float(row["wall_seconds"]) for row in variant_rows)
            post_stops = sorted(
                float(row["post_stop_seconds"])
                for row in variant_rows
                if row.get("post_stop_seconds") is not None
            )
            similarities = [float(row["similarity"]) for row in variant_rows]
            p95_index = min(len(walls) - 1, int(round((len(walls) - 1) * 0.95)))
            post_stop = ""
            if post_stops:
                post_index = min(len(post_stops) - 1, int(round((len(post_stops) - 1) * 0.95)))
                post_stop = f"{post_stops[post_index]:.1f}"
            peak_vram = max(
                [int(row["peak_vram_mb"]) for row in variant_rows if row["peak_vram_mb"] is not None]
                or [0]
            )
            handle.write(
                f"| {variant_name} | {variant_rows[0].get('mode', '')} | {len(variant_rows)} | "
                f"{sum(rtfs) / len(rtfs):.3f} | {walls[p95_index]:.1f} | {post_stop} | "
                f"{min(similarities):.3f} | "
                f"{sum(int(row['timeouts']) for row in variant_rows)} | "
                f"{sum(int(row['retries']) for row in variant_rows)} | "
                f"{peak_vram or ''} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Voxtray/Voxtral variants.")
    parser.add_argument("--config", type=Path, default=Path.home() / ".config/voxtray/config.toml")
    parser.add_argument("--audio", action="append", type=Path, default=[])
    parser.add_argument("--recordings-root", type=Path, default=DEFAULT_RECORDINGS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reuse-running", action="store_true")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--mode", choices=["file", "live"], default="file")
    parser.add_argument("--live-speed", type=float, default=8.0)
    parser.add_argument("--save-transcripts", action="store_true")
    return parser.parse_args()


def select_variants(names: list[str]) -> list[Variant]:
    variants = quick_variants()
    if not names:
        return variants
    by_name = {variant.name: variant for variant in variants}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit(f"unknown variant(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def main() -> int:
    args = parse_args()
    base_config = load_config(args.config)
    variants = select_variants(args.variant)
    if args.audio:
        corpus = []
        for audio_path in args.audio:
            case = load_audio_case(audio_path)
            if case is None:
                raise SystemExit(f"audio has no complete success reference: {audio_path}")
            corpus.append(case)
    else:
        corpus = discover_corpus(args.recordings_root, args.limit)
    if not corpus:
        print("No benchmark corpus found. Pass --recordings-root or provide result.json references.", file=sys.stderr)
        return 2

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_root / run_id
    rows: list[dict[str, Any]] = []
    print(f"Output: {output_dir}", flush=True)
    print("Corpus:", flush=True)
    for case in corpus:
        print(f"  {case.name}: {case.duration_seconds:.1f}s", flush=True)

    for variant in variants:
        config = apply_variant(base_config, variant, args.port)
        engine_proc: subprocess.Popen[bytes] | None = None
        load_seconds = 0.0
        log_path = output_dir / variant.name / "vllm.log"
        print(f"\n== {variant.name} ==", flush=True)
        if args.reuse_running:
            if not server_ready(config):
                raise SystemExit("--reuse-running was set but the configured server is not ready")
        else:
            if server_ready(config, timeout_seconds=0.5):
                raise SystemExit(
                    f"server already responds on {config.server_base_url}; stop it or use --reuse-running"
                )
            engine_proc, load_seconds = start_engine(config, log_path)
            print(f"loaded in {load_seconds:.1f}s", flush=True)
        try:
            for case in corpus:
                if args.mode == "live":
                    row = benchmark_live_case(
                        config,
                        variant,
                        case,
                        live_speed=args.live_speed,
                    )
                else:
                    row = benchmark_file_case(config, variant, case)
                row["engine_load_seconds"] = round(load_seconds, 3)
                transcript = str(row.pop("_transcript", ""))
                if args.save_transcripts:
                    transcript_dir = output_dir / "transcripts"
                    transcript_dir.mkdir(parents=True, exist_ok=True)
                    transcript_path = transcript_dir / f"{variant.name}-{case.name}.txt"
                    transcript_path.write_text(transcript, encoding="utf-8")
                    row["transcript_path"] = str(transcript_path)
                rows.append(row)
                print(
                    f"{case.name}: {row['wall_seconds']:.1f}s "
                    f"post_stop={row['post_stop_seconds']} "
                    f"rtf={row['rtf']} sim={row['similarity']} "
                    f"segments={row['segments']} retries={row['retries']} "
                    f"status={row['status']}",
                    flush=True,
                )
                write_outputs(rows, output_dir)
        finally:
            if engine_proc is not None:
                stop_engine(engine_proc)

    write_outputs(rows, output_dir)
    print(f"\nWrote {output_dir / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
