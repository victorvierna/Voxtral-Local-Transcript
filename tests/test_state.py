from pathlib import Path

from voxtray.state import StateStore


def test_state_store_roundtrip(tmp_path: Path):
    state_path = tmp_path / "state.json"
    store = StateStore(path=state_path)

    original = store.read()
    assert original["warm_enabled"] is True
    assert original["last_toggle_epoch"] == 0.0

    store.set_values(warm_enabled=True, last_error="x")
    updated = store.read()
    assert updated["warm_enabled"] is True
    assert updated["last_error"] == "x"
