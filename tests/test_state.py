from pathlib import Path

from voxtray.state import StateStore, UNEXPECTED_RECORDING_EXIT_MESSAGE


def test_state_store_roundtrip(tmp_path: Path):
    state_path = tmp_path / "state.json"
    store = StateStore(path=state_path)

    original = store.read()
    assert original["warm_enabled"] is True
    assert original["recording_stop_requested"] is False
    assert original["activity_state"] == "idle"
    assert original["last_toggle_epoch"] == 0.0
    assert original["last_notice_id"] == ""
    assert original["last_notice_level"] == "info"
    assert original["last_artifact_path"] == ""
    assert original["last_history_id"] == ""
    assert original["last_history_index"] == 0
    assert original["last_clipboard_backend"] == ""
    assert original["last_clipboard_verified"] is False
    assert original["last_clipboard_verification_supported"] is False
    assert original["last_clipboard_error"] == ""
    assert original["last_assistant_command_id"] == ""
    assert original["last_assistant_route"] == ""
    assert original["last_assistant_agent_id"] == ""
    assert original["last_assistant_error"] == ""

    store.set_values(
        warm_enabled=True,
        last_error="x",
        last_notice_id="notice-1",
        last_notice_title="Voxtray",
        last_notice_body="done",
        last_notice_level="warning",
        last_artifact_path="/tmp/artifact",
        last_history_id="history-1",
        last_history_index=1,
        last_clipboard_backend="xclip",
        last_clipboard_verified=True,
        last_clipboard_verification_supported=True,
        last_clipboard_error="",
        last_assistant_command_id="cmd_1",
        last_assistant_route="agent",
        last_assistant_agent_id="Harvis",
        last_assistant_error="",
    )
    updated = store.read()
    assert updated["warm_enabled"] is True
    assert updated["last_error"] == "x"
    assert updated["last_notice_id"] == "notice-1"
    assert updated["last_notice_body"] == "done"
    assert updated["last_notice_level"] == "warning"
    assert updated["last_artifact_path"] == "/tmp/artifact"
    assert updated["last_history_id"] == "history-1"
    assert updated["last_history_index"] == 1
    assert updated["last_clipboard_backend"] == "xclip"
    assert updated["last_clipboard_verified"] is True
    assert updated["last_clipboard_verification_supported"] is True
    assert updated["last_clipboard_error"] == ""
    assert updated["last_assistant_command_id"] == "cmd_1"
    assert updated["last_assistant_route"] == "agent"
    assert updated["last_assistant_agent_id"] == "Harvis"
    assert updated["last_assistant_error"] == ""


def test_read_marks_stale_recording_as_visible_error(tmp_path: Path, monkeypatch):
    state_path = tmp_path / "state.json"
    store = StateStore(path=state_path)
    store.set_values(
        recording_pid=1234,
        recording_stop_requested=False,
        activity_state="recording",
        last_notice_body="Recording started",
    )
    monkeypatch.setattr("voxtray.state.pid_is_alive", lambda pid: False)

    state = store.read()

    assert state["recording_pid"] is None
    assert state["recording_stop_requested"] is False
    assert state["activity_state"] == "error"
    assert state["last_error"] == UNEXPECTED_RECORDING_EXIT_MESSAGE
    assert state["last_notice_title"] == "Voxtray Error"
    assert state["last_notice_body"] == UNEXPECTED_RECORDING_EXIT_MESSAGE
    assert state["last_notice_level"] == "error"
    assert state["last_notice_id"]
