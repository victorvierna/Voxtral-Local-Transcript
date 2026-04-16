from pathlib import Path

from voxtray.state import StateStore


def test_state_store_roundtrip(tmp_path: Path):
    state_path = tmp_path / "state.json"
    store = StateStore(path=state_path)

    original = store.read()
    assert original["warm_enabled"] is True
    assert original["last_toggle_epoch"] == 0.0
    assert original["last_notice_id"] == ""
    assert original["last_notice_level"] == "info"

    store.set_values(
        warm_enabled=True,
        last_error="x",
        last_notice_id="notice-1",
        last_notice_title="Voxtray",
        last_notice_body="done",
        last_notice_level="warning",
    )
    updated = store.read()
    assert updated["warm_enabled"] is True
    assert updated["last_error"] == "x"
    assert updated["last_notice_id"] == "notice-1"
    assert updated["last_notice_body"] == "done"
    assert updated["last_notice_level"] == "warning"
