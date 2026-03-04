from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import sys
from typing import Any
import tomllib

import tomli_w

from .paths import CONFIG_FILE, ensure_app_dirs


VOXTRAL_MODEL_PREFIX = "mistralai/voxtral-"


@dataclass(slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    start_timeout_seconds: int = 180
    external_base_url: str = ""


@dataclass(slots=True)
class AudioConfig:
    device: str = "default"
    sample_rate: int = 16000
    chunk_ms: int = 40


@dataclass(slots=True)
class HistoryConfig:
    max_items: int = 5


@dataclass(slots=True)
class ClipboardConfig:
    backend: str = "auto"


@dataclass(slots=True)
class PostprocessConfig:
    clean_text: bool = True


@dataclass(slots=True)
class RealtimeConfig:
    final_timeout_seconds: float = 12.0
    segment_max_seconds: int = 120
    segment_finalize_timeout_seconds: float = 12.0
    mic_queue_chunks: int = 4096
    stop_tail_ms: int = 240
    first_chunk_grace_ms: int = 220


@dataclass(slots=True)
class EngineConfig:
    command: str = "vllm"
    enforce_eager: bool = False
    disable_compile_cache: bool = True
    compilation_config: str = '{"cudagraph_mode":"PIECEWISE"}'
    extra_args: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VoxtrayConfig:
    model_id: str = "mistralai/Voxtral-Mini-4B-Realtime-2602"
    server: ServerConfig = field(default_factory=ServerConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    clipboard: ClipboardConfig = field(default_factory=ClipboardConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)

    @property
    def server_base_url(self) -> str:
        if self.server.external_base_url:
            return self.server.external_base_url.rstrip("/")
        return f"http://{self.server.host}:{self.server.port}"

    @property
    def websocket_url(self) -> str:
        if self.server.external_base_url:
            base = self.server.external_base_url.rstrip("/")
            if base.startswith("https://"):
                return "wss://" + base[len("https://") :] + "/v1/realtime"
            if base.startswith("http://"):
                return "ws://" + base[len("http://") :] + "/v1/realtime"
            return f"ws://{base}/v1/realtime"
        return f"ws://{self.server.host}:{self.server.port}/v1/realtime"


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _default_dict() -> dict[str, Any]:
    venv_vllm = Path(sys.executable).with_name("vllm")
    default_engine_command = (
        str(venv_vllm)
        if (not shutil.which("vllm") and venv_vllm.exists())
        else "vllm"
    )
    return {
        "model_id": "mistralai/Voxtral-Mini-4B-Realtime-2602",
        "server": {
            "host": "127.0.0.1",
            "port": 8000,
            "start_timeout_seconds": 180,
            "external_base_url": "",
        },
        "audio": {
            "device": "default",
            "sample_rate": 16000,
            "chunk_ms": 40,
        },
        "history": {"max_items": 5},
        "clipboard": {"backend": "auto"},
        "postprocess": {"clean_text": True},
        "realtime": {
            "final_timeout_seconds": 12.0,
            "segment_max_seconds": 120,
            "segment_finalize_timeout_seconds": 12.0,
            "mic_queue_chunks": 4096,
            "stop_tail_ms": 240,
            "first_chunk_grace_ms": 220,
        },
        "engine": {
            "command": default_engine_command,
            "enforce_eager": False,
            "disable_compile_cache": True,
            "compilation_config": '{"cudagraph_mode":"PIECEWISE"}',
            "extra_args": ["--trust-remote-code"],
        },
    }


def _normalize_engine_extra_args(model_id: str, engine_data: dict[str, Any]) -> dict[str, Any]:
    raw_args = engine_data.get("extra_args", [])
    if isinstance(raw_args, list):
        extra_args = [str(arg) for arg in raw_args]
    else:
        extra_args = [str(raw_args)] if raw_args else []

    if model_id.lower().startswith(VOXTRAL_MODEL_PREFIX):
        if "--trust-remote-code" not in extra_args:
            extra_args = ["--trust-remote-code", *extra_args]

    engine_data["extra_args"] = extra_args
    return engine_data


def config_to_dict(config: VoxtrayConfig) -> dict[str, Any]:
    return {
        "model_id": config.model_id,
        "server": {
            "host": config.server.host,
            "port": config.server.port,
            "start_timeout_seconds": config.server.start_timeout_seconds,
            "external_base_url": config.server.external_base_url,
        },
        "audio": {
            "device": config.audio.device,
            "sample_rate": config.audio.sample_rate,
            "chunk_ms": config.audio.chunk_ms,
        },
        "history": {"max_items": config.history.max_items},
        "clipboard": {"backend": config.clipboard.backend},
        "postprocess": {"clean_text": config.postprocess.clean_text},
        "realtime": {
            "final_timeout_seconds": config.realtime.final_timeout_seconds,
            "segment_max_seconds": config.realtime.segment_max_seconds,
            "segment_finalize_timeout_seconds": config.realtime.segment_finalize_timeout_seconds,
            "mic_queue_chunks": config.realtime.mic_queue_chunks,
            "stop_tail_ms": config.realtime.stop_tail_ms,
            "first_chunk_grace_ms": config.realtime.first_chunk_grace_ms,
        },
        "engine": {
            "command": config.engine.command,
            "enforce_eager": config.engine.enforce_eager,
            "disable_compile_cache": config.engine.disable_compile_cache,
            "compilation_config": config.engine.compilation_config,
            "extra_args": config.engine.extra_args,
        },
    }


def _resolve_engine_command(command: str) -> str:
    if command != "vllm":
        return command
    if shutil.which("vllm"):
        return command
    venv_vllm = Path(sys.executable).with_name("vllm")
    if venv_vllm.exists():
        return str(venv_vllm)
    return command


def load_config(path: Path | None = None) -> VoxtrayConfig:
    ensure_app_dirs()
    config_path = path or CONFIG_FILE
    data = _default_dict()
    if config_path.exists():
        with config_path.open("rb") as handle:
            file_data = tomllib.load(handle)
            if isinstance(file_data, dict):
                _deep_update(data, file_data)

    server = ServerConfig(**data["server"])
    audio = AudioConfig(**data["audio"])
    history = HistoryConfig(**data["history"])
    clipboard = ClipboardConfig(**data["clipboard"])
    postprocess = PostprocessConfig(**data["postprocess"])
    realtime = RealtimeConfig(**data["realtime"])
    model_id = str(data["model_id"])
    engine_data = _normalize_engine_extra_args(model_id, dict(data["engine"]))
    engine_data["command"] = _resolve_engine_command(
        str(engine_data.get("command", "vllm"))
    )
    engine = EngineConfig(**engine_data)
    return VoxtrayConfig(
        model_id=model_id,
        server=server,
        audio=audio,
        history=history,
        clipboard=clipboard,
        postprocess=postprocess,
        realtime=realtime,
        engine=engine,
    )


def write_default_config(path: Path | None = None) -> Path:
    ensure_app_dirs()
    config_path = path or CONFIG_FILE
    if config_path.exists():
        return config_path
    with config_path.open("wb") as handle:
        handle.write(tomli_w.dumps(_default_dict()).encode("utf-8"))
    return config_path


def save_config(config: VoxtrayConfig, path: Path | None = None) -> Path:
    ensure_app_dirs()
    config_path = path or CONFIG_FILE
    with config_path.open("wb") as handle:
        handle.write(tomli_w.dumps(config_to_dict(config)).encode("utf-8"))
    return config_path
