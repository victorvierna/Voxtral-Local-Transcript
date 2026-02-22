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
