from pathlib import Path
import signal

from voxtray.controller import Controller
from voxtray.state import StateStore


def _build_controller(tmp_path: Path, monkeypatch) -> Controller:
    ctl = Controller(config_path=tmp_path / "config.toml")
    ctl.state_store = StateStore(path=tmp_path / "state.json")
    ctl.engine.state_store = ctl.state_store
    monkeypatch.setattr(ctl.engine, "is_ready", lambda timeout_seconds=1.5: False)
    monkeypatch.setattr("voxtray.state.pid_is_alive", lambda pid: True)
    monkeypatch.setattr("voxtray.controller.pid_is_alive", lambda pid: True)
    return ctl


def test_status_reports_processing_after_stop_requested(tmp_path, monkeypatch):
    ctl = _build_controller(tmp_path, monkeypatch)
    ctl.state_store.set_values(
        recording_pid=1234,
        recording_stop_requested=True,
        activity_state="transcribing",
    )

    status = ctl.status()

    assert status["recording"] is False
    assert status["processing"] is True
    assert status["activity_state"] == "transcribing"
    assert status["recording_pid"] == 1234
    assert status["recording_stop_requested"] is True


def test_stop_recording_marks_processing_immediately(tmp_path, monkeypatch):
    ctl = _build_controller(tmp_path, monkeypatch)
    ctl.state_store.set_values(recording_pid=1234, recording_stop_requested=False)
    killed: list[tuple[int, signal.Signals]] = []

    def fake_kill(pid: int, sig: signal.Signals) -> None:
        killed.append((pid, sig))

    monkeypatch.setattr("voxtray.controller.os.kill", fake_kill)

    message = ctl.stop_recording(timeout_seconds=0)

    assert message == "stop signal sent to pid=1234, still shutting down"
    assert killed == [(1234, signal.SIGUSR1)]
    state = ctl.state_store.read()
    assert state["recording_pid"] == 1234
    assert state["recording_stop_requested"] is True
    assert state["activity_state"] == "transcribing"

    status = ctl.status()
    assert status["recording"] is False
    assert status["processing"] is True
    assert status["activity_state"] == "transcribing"


def test_toggle_ignores_processing_recording(tmp_path, monkeypatch):
    ctl = _build_controller(tmp_path, monkeypatch)
    ctl.state_store.set_values(
        recording_pid=1234,
        recording_stop_requested=True,
        activity_state="transcribing",
    )

    message = ctl.toggle_recording()

    assert message == "recording is already stopping and transcribing (pid=1234)"


def test_shutdown_waits_for_already_stopping_recording_before_unload(
    tmp_path,
    monkeypatch,
):
    ctl = _build_controller(tmp_path, monkeypatch)
    ctl.state_store.set_values(
        recording_pid=1234,
        recording_stop_requested=True,
        activity_state="transcribing",
    )
    events: list[str] = []
    alive_results = iter([True, True, False])

    def fake_pid_is_alive(pid: int) -> bool:
        events.append(f"alive:{pid}")
        return next(alive_results)

    def fail_kill(pid: int, sig: signal.Signals) -> None:
        raise AssertionError(f"should not resend stop signal to {pid} with {sig}")

    def fake_stop_engine(timeout_seconds: float = 8.0) -> None:
        events.append(f"engine_stop:{timeout_seconds}")

    monkeypatch.setattr("voxtray.controller.pid_is_alive", fake_pid_is_alive)
    monkeypatch.setattr("voxtray.controller.os.kill", fail_kill)
    monkeypatch.setattr(ctl.engine, "stop_if_running", fake_stop_engine)

    message = ctl.shutdown_for_exit()

    assert message == "recording stopped; model unloaded"
    assert events == [
        "alive:1234",
        "alive:1234",
        "alive:1234",
        "engine_stop:8.0",
    ]
    state = ctl.state_store.read()
    assert state["recording_pid"] is None
    assert state["recording_stop_requested"] is False
    assert state["activity_state"] == "idle"
