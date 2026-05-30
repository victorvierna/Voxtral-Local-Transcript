from pathlib import Path

from voxtray.config import load_config


def test_load_config_injects_realtime_defaults_for_legacy_file(tmp_path: Path):
    cfg_path = tmp_path / "legacy.toml"
    cfg_path.write_text(
        """
model_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"

[server]
host = "127.0.0.1"
port = 8000
start_timeout_seconds = 180
external_base_url = ""

[audio]
device = "default"
sample_rate = 16000
chunk_ms = 40

[history]
max_items = 5

[clipboard]
backend = "auto"

[postprocess]
clean_text = true

[engine]
command = "vllm"
enforce_eager = false
disable_compile_cache = true
compilation_config = '{"cudagraph_mode":"PIECEWISE"}'
extra_args = []
""".strip(),
        encoding="utf-8",
    )

    config = load_config(cfg_path)

    assert config.realtime.final_timeout_seconds == 12.0
    assert config.realtime.segment_max_seconds == 120
    assert config.realtime.segment_finalize_timeout_seconds == 12.0
    assert config.realtime.mic_queue_chunks == 4096
    assert config.realtime.stop_tail_ms == 240
    assert config.realtime.first_chunk_grace_ms == 220
    assert config.transcription.provider == "local_voxtral"
    assert config.mistral_realtime.api_key_env == "MISTRAL_API_KEY"
    assert config.openai_realtime.api_key_env == "OPENAI_API_KEY"
    assert config.openai_realtime.model == "gpt-realtime-whisper"
    assert config.openai_realtime.fallback_model == "whisper-1"
    assert config.openai_realtime.delay == "high"
    assert config.engine.extra_args == ["--trust-remote-code"]
    assert config.assistant.enabled is False
    assert config.assistant.endpoint == "http://127.0.0.1:8777/api/route"
    assert config.assistant.token_env == "HARVIS_API_TOKEN"
    assert config.assistant.token_file == ""
    assert config.assistant.timeout_seconds == 1.5
    assert config.assistant.fail_open_to_clipboard is True
    assert config.assistant.speak_confirmations is True


def test_load_config_accepts_realtime_overrides(tmp_path: Path):
    cfg_path = tmp_path / "custom.toml"
    cfg_path.write_text(
        """
[realtime]
final_timeout_seconds = 60.0
segment_max_seconds = 90
segment_finalize_timeout_seconds = 8.0
mic_queue_chunks = 2048
stop_tail_ms = 320
first_chunk_grace_ms = 180
""".strip(),
        encoding="utf-8",
    )

    config = load_config(cfg_path)

    assert config.realtime.final_timeout_seconds == 60.0
    assert config.realtime.segment_max_seconds == 90
    assert config.realtime.segment_finalize_timeout_seconds == 8.0
    assert config.realtime.mic_queue_chunks == 2048
    assert config.realtime.stop_tail_ms == 320
    assert config.realtime.first_chunk_grace_ms == 180
    assert config.engine.extra_args == ["--trust-remote-code"]


def test_load_config_does_not_force_trust_remote_code_for_non_voxtral_model(tmp_path: Path):
    cfg_path = tmp_path / "custom-model.toml"
    cfg_path.write_text(
        """
model_id = "openai/whisper-large-v3"

[engine]
extra_args = []
""".strip(),
        encoding="utf-8",
    )

    config = load_config(cfg_path)

    assert config.engine.extra_args == []


def test_load_config_accepts_cloud_provider_overrides(tmp_path: Path):
    cfg_path = tmp_path / "cloud.toml"
    cfg_path.write_text(
        """
[transcription]
provider = "mistral_realtime"

[mistral_realtime]
api_key_env = "CUSTOM_MISTRAL_KEY"
model = "voxtral-mini-transcribe-realtime-2602"
sample_rate = 16000
target_delay_ms = 1000

[openai_realtime]
api_key_env = "CUSTOM_OPENAI_KEY"
model = "gpt-realtime-whisper"
fallback_model = "gpt-4o-transcribe"
sample_rate = 24000
turn_detection = "manual"
delay = "low"
language = "es"
prompt = "Proyecto"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(cfg_path)

    assert config.transcription.provider == "mistral_realtime"
    assert config.mistral_realtime.api_key_env == "CUSTOM_MISTRAL_KEY"
    assert config.mistral_realtime.target_delay_ms == 1000
    assert config.openai_realtime.model == "gpt-realtime-whisper"
    assert config.openai_realtime.fallback_model == "gpt-4o-transcribe"
    assert config.openai_realtime.delay == "low"
    assert config.openai_realtime.language == "es"
    assert config.openai_realtime.prompt == "Proyecto"


def test_load_config_accepts_assistant_overrides(tmp_path: Path):
    cfg_path = tmp_path / "assistant.toml"
    cfg_path.write_text(
        """
[assistant]
enabled = true
endpoint = "http://127.0.0.1:9999/api/route"
token_env = "CUSTOM_HARVIS_TOKEN"
token_file = "/tmp/harvis-token"
timeout_seconds = 0.25
fail_open_to_clipboard = false
speak_confirmations = false
""".strip(),
        encoding="utf-8",
    )

    config = load_config(cfg_path)

    assert config.assistant.enabled is True
    assert config.assistant.endpoint == "http://127.0.0.1:9999/api/route"
    assert config.assistant.token_env == "CUSTOM_HARVIS_TOKEN"
    assert config.assistant.token_file == "/tmp/harvis-token"
    assert config.assistant.timeout_seconds == 0.25
    assert config.assistant.fail_open_to_clipboard is False
    assert config.assistant.speak_confirmations is False
