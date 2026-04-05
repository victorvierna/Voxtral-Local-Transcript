from pathlib import Path
import subprocess

import pytest

from voxtray.config import EngineConfig, ServerConfig, VoxtrayConfig
from voxtray.engine import EngineError, EngineManager
from voxtray.state import StateStore


def _build_manager(tmp_path: Path) -> tuple[EngineManager, StateStore]:
    state = StateStore(path=tmp_path / "state.json")
    config = VoxtrayConfig(
        server=ServerConfig(start_timeout_seconds=180),
        engine=EngineConfig(command="/tmp/vllm"),
    )
    return EngineManager(config, state), state


def test_ensure_running_resets_unexpected_pid_and_starts_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manager, state = _build_manager(tmp_path)
    state.set_values(engine_pid=12345)

    monkeypatch.setattr("voxtray.engine.pid_is_alive", lambda pid: True)
    monkeypatch.setattr("voxtray.state.pid_is_alive", lambda pid: True)
    monkeypatch.setattr(manager, "is_ready", lambda timeout_seconds=3.0: False)
    monkeypatch.setattr(manager, "_is_expected_engine_pid", lambda pid: False)
    monkeypatch.setattr(
        manager,
        "_wait_until_ready_or_fail",
        lambda pid: (_ for _ in ()).throw(AssertionError("must not wait on invalid pid")),
    )

    calls = {"start": 0}

    def _start_local_engine() -> None:
        calls["start"] += 1

    monkeypatch.setattr(manager, "start_local_engine", _start_local_engine)

    manager.ensure_running()

    assert calls["start"] == 1
    assert state.read()["engine_pid"] is None


def test_ensure_running_restarts_after_existing_pid_readiness_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manager, state = _build_manager(tmp_path)
    state.set_values(engine_pid=22334)

    monkeypatch.setattr("voxtray.engine.pid_is_alive", lambda pid: True)
    monkeypatch.setattr("voxtray.state.pid_is_alive", lambda pid: True)
    monkeypatch.setattr(manager, "is_ready", lambda timeout_seconds=3.0: False)
    monkeypatch.setattr(manager, "_is_expected_engine_pid", lambda pid: True)

    def _wait_fail(pid: int) -> None:
        raise EngineError("vLLM not ready after 900s. See log: /tmp/vllm.log")

    monkeypatch.setattr(manager, "_wait_until_ready_or_fail", _wait_fail)

    stops: list[int] = []
    starts = {"count": 0}
    monkeypatch.setattr(
        manager,
        "_stop_process_group",
        lambda pgid, timeout_seconds: stops.append(pgid),
    )
    monkeypatch.setattr(manager, "start_local_engine", lambda: starts.__setitem__("count", 1))

    manager.ensure_running()

    assert stops == [22334]
    assert starts["count"] == 1
    assert state.read()["engine_pid"] is None


def test_wait_until_ready_timeout_clears_engine_pid(tmp_path: Path):
    state_path = tmp_path / "state.json"
    state = StateStore(path=state_path)
    state.set_values(engine_pid=777)
    config = VoxtrayConfig(
        server=ServerConfig(start_timeout_seconds=0),
        engine=EngineConfig(command="/tmp/vllm"),
    )
    manager = EngineManager(config, state)

    with pytest.raises(EngineError, match="not ready after 0s"):
        manager._wait_until_ready_or_fail(777)

    assert state.read()["engine_pid"] is None


def test_start_local_engine_sets_vllm_host_ip_to_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manager, state = _build_manager(tmp_path)
    manager.config.server.host = "0.0.0.0"
    monkeypatch.setattr("voxtray.engine.VLLM_LOG_FILE", str(tmp_path / "vllm.log"))

    captured: dict[str, object] = {}

    class DummyProcess:
        pid = 4242

    def _fake_popen(*args, **kwargs):
        captured["command"] = args[0]
        captured["env"] = kwargs["env"]
        return DummyProcess()

    monkeypatch.setattr(manager, "is_ready", lambda timeout_seconds=2.0: False)
    monkeypatch.setattr(manager, "_wait_until_ready_or_fail", lambda pid: None)
    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    monkeypatch.setattr("voxtray.state.pid_is_alive", lambda pid: True)

    manager.start_local_engine()

    assert captured["command"] == manager._build_command()
    assert captured["env"]["VLLM_HOST_IP"] == "127.0.0.1"
    assert state.read()["engine_pid"] == 4242
